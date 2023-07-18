import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .weighted_obs import WeightedOBS
from .layerwrapper import WrappedGPT, WrappedPlusGPT, SkillGPT
from .data import get_loaders 

from pdb import set_trace as st 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity_skill(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += W.numel()
            total_params += subset[name].in_features * subset[name].out_features
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()

            sub_count += W.numel()
            sub_params += subset[name].in_features * subset[name].out_features

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:   # 行批量计算, 列按照prune_m分块
                        tmp = W_metric[:,ii:(ii+prune_m)].float()   # 一次计算prune_m列
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)    # 按照prune_n个最小值进行mask
            else:   # 每个weight矩阵按照sparsity_ratio进行剪枝
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0   # W_mask为True的位置置0


def prune_taylor(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    # 假设 model 是你的模型
    for param in model.parameters():
        param.requires_grad = False
    layer_name = 'layers[31].self_attn.q_proj'
    model.model.layers[31].self_attn.q_proj.weight.requires_grad = True
    grads = []
    for batch in dataloader:
        with torch.cuda.amp.autocast():
            lm_logits = model(batch[0].to(device)).logits
            
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()    # [cat, sat, on, ???] -> [cat, sat, on]
        shift_labels = batch[1][:, 1:].to(device)    # [The, cat, sat, on] -> [cat, sat, on]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        loss.backward()  # 计算梯度
        
        # 统计梯度
        grads.append(model.model.layers[31].self_attn.q_proj.weight.grad.detach())
    
    grads = torch.stack(grads, dim=0)

    print(f"pruning {layer_name}")
    W = model.model.layers[31].self_attn.q_proj.weight.data
    W_metric = torch.abs(W) * grads.mean(axis=0)
    
    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
    W_mask = (W_metric<=thresh)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(W_mask.cpu().numpy(), cmap='Blues', interpolation='nearest')
    plt.title(f'{layer_name} sparsity={args.sparsity_ratio} taylor up')
    plt.savefig(f'figures/taylor_up/{layer_name}_{args.sparsity_ratio}.png')
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_with_skill(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        # subset = {'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']}
        subset = {'mlp.down_proj': find_layers(layer)['mlp.down_proj']}

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = SkillGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()


        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))
            if name == 'self_attn.o_proj':   # 需要组剪枝
                W_metric = W_metric.mean(axis=0).reshape(-1, 128).sum(dim=1)
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric>=thresh).repeat_interleave(128)
            else:
                W_metric = W_metric.mean(axis=0)
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric>=thresh)
            
            if name == 'self_attn.o_proj':   # 需要组剪枝
                model.model.layers[i].self_attn.q_proj.weight.data = model.model.layers[i].self_attn.q_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].self_attn.k_proj.weight.data = model.model.layers[i].self_attn.k_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].self_attn.v_proj.weight.data = model.model.layers[i].self_attn.v_proj.weight.data[torch.where(W_mask)[0]]
                output_weight = model.model.layers[i].self_attn.o_proj.weight.data
                output_bias = ((wrapped_layers[name].skill_inp.type(torch.half) * ~W_mask) @ output_weight.T)
                output_weight = output_weight[:, torch.where(W_mask)[0]]
                shape = model.model.layers[i].self_attn.o_proj.weight.data.shape
                model.model.layers[i].self_attn.num_heads = int(W_metric.numel()*(1 -args.sparsity_ratio))
                model.model.layers[i].self_attn.hidden_size = int(W_metric.numel()*(1 -args.sparsity_ratio)) * 128
                model.model.layers[i].self_attn.o_proj = torch.nn.Linear(in_features=shape[1], out_features=shape[0], bias=True).to(device)
                model.model.layers[i].self_attn.o_proj.weight.data = output_weight
                model.model.layers[i].self_attn.o_proj.bias.data = output_bias
            else:
                model.model.layers[i].mlp.up_proj.weight.data = model.model.layers[i].mlp.up_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].mlp.gate_proj.weight.data = model.model.layers[i].mlp.gate_proj.weight.data[torch.where(W_mask)[0]]
                output_weight = model.model.layers[i].mlp.down_proj.weight.data
                output_bias = ((wrapped_layers[name].skill_inp.type(torch.half) * ~W_mask) @ output_weight.T)
                output_weight = output_weight[:, torch.where(W_mask)[0]]
                shape = model.model.layers[i].mlp.down_proj.weight.data.shape
                model.model.layers[i].mlp.down_proj = torch.nn.Linear(in_features=shape[1], out_features=shape[0], bias=True).to(device)
                model.model.layers[i].mlp.down_proj.weight.data = output_weight
                model.model.layers[i].mlp.down_proj.bias.data = output_bias
            wrapped_layers[name].free()


        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # 将剪枝后的输出作为下一层的输入
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()


        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            wrapped_layers[name].free()
            print(f"layer{i} {name} L2 Norm: {torch.norm(W_metric)}")
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:    # 二分查找剪枝
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]  # [lower, upper]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # per-layer
                    if args.mode == 'per-layer':
                        W = subset[name].weight.data
                        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                        W_mask = (W_metric<=thresh)
                    
                    elif args.mode == 'per-out':
                        # per-out
                        # unstructured pruning
                        indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # 将剪枝后的输出作为下一层的输入
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_wanda_plus(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedPlusGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()


        for name in subset:
            print(f"pruning layer {i} name {name}")
            # if name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.gate_proj', 'mlp.up_proj']:
            # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1))) / torch.sqrt(wrapped_layers[name].scaler_out.reshape((-1,1)))
            # else:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))
            wrapped_layers[name].free()
            # print(f"layer{i} {name} L2 Norm: {torch.norm(W_metric)}")
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:    # 二分查找剪枝
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]  # [lower, upper]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # per-layer
                    if args.mode == 'per-layer':
                        W = subset[name].weight.data
                        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                        W_mask = (W_metric<=thresh)
                    
                    elif args.mode == 'per-out':
                        # per-out
                        # unstructured pruning
                        indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)   # 1表示被剪去

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # 将剪枝后的输出作为下一层的输入
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer   # SparseGPT原始代码将很多放入了CPU
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
 
 
@torch.no_grad()
def prune_weightedobs_v2(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    # inps_2 = inps.copy()
    # outs_2 = outs.copy()

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        # 计算Weighted_obs
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch_1(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch_1(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        
        weighted_layers = {}
        for name in subset:
            S_ij = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            weighted_obs = torch.diag(S_ij.sum(axis=0) / S_ij.sum())    # (in_dim, in_dim)
            wrapped_layers[name].free()
            weighted_layers[name] = WeightedOBS(subset[name], weighted_obs)

        def add_batch_2(name):
            def tmp(_, inp, out):
                weighted_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in weighted_layers:
            handles.append(subset[name].register_forward_hook(add_batch_2(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in weighted_layers:
            print(i, name)
            print('Pruning ...')
            
            weighted_layers[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            weighted_layers[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer   # SparseGPT原始代码将很多放入了CPU
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache() 
    
# @torch.no_grad()
# def prune_weightedobs(args, model, tokenizer, dev, prune_n=0, prune_m=0):
#     ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
#     print('Starting ...')
#     dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)

#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = model.model.layers

#     if "model.embed_tokens" in model.hf_device_map:
#         dev = model.hf_device_map["model.embed_tokens"]

#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
#     )
#     cache = {'i': 0, 'attention_mask': None, "position_ids": None}

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             cache['attention_mask'] = kwargs['attention_mask']
#             cache['position_ids'] = kwargs['position_ids']
#             raise ValueError
#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         try:
#             model(batch[0].to(dev))
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
#     torch.cuda.empty_cache()

#     outs = torch.zeros_like(inps)
#     attention_mask = cache['attention_mask']
#     position_ids = cache['position_ids']

#     print('Ready.')

#     for i in range(len(layers)):
#         layer = layers[i]
#         if f"model.layers.{i}" in model.hf_device_map:
#             dev = model.hf_device_map[f"model.layers.{i}"]
#             print(f"layer {i} device {dev}")
#             inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

#         subset = find_layers(layer)

#         weighted_layers = {}
#         for name in subset:
#             weighted_layers[name] = WeightedOBS(subset[name])

#         def add_batch(name):
#             def tmp(_, inp, out):
#                 weighted_layers[name].add_batch(inp[0].data, out.data)
#             return tmp

#         handles = []
#         for name in weighted_layers:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))

#         for j in range(args.nsamples):
#             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#         for h in handles:
#             h.remove()

#         for name in weighted_layers:
#             print(i, name)
#             print('Pruning ...')
#             S_ij = torch.abs(subset[name].weight.data) * torch.sqrt(weighted_layers[name].scaler_row.reshape((1,-1)))
#             weighted_obs = torch.diag(S_ij.sum(axis=0) / S_ij.sum())    # (in_dim, in_dim)
#             # # 计算 Frobenius 范数
#             # norm_obs = torch.norm(weighted_obs)

#             # # 除以 Frobenius 范数，使得新的对角矩阵的 Frobenius 范数为1
#             # weighted_obs = weighted_obs / norm_obs
            
#             weighted_layers[name].fasterprune(args.sparsity_ratio, weighted_obs, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
#             weighted_layers[name].free()

#         for j in range(args.nsamples):
#             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

#         layers[i] = layer   # SparseGPT原始代码将很多放入了CPU
#         torch.cuda.empty_cache()

#         inps, outs = outs, inps

#     model.config.use_cache = use_cache
#     torch.cuda.empty_cache()