import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT, BiasGPT
from .data import get_loaders 
import math
from pdb import set_trace as st 

# create a dictionary to map the method name to the function
metrics = {
    'L2': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'L2W': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'wanda': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}

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

def check_sparsity_sp(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if 'self_attn' in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()
            
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
    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
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

def compress(layer, i, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device):
    # attn
    if attn_mask is not None:
        retain_heads = torch.count_nonzero(attn_mask)
        attn_mask = attn_mask.repeat_interleave(128)
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
        layer.self_attn.q_proj.out_features = attn_mask.sum().item()
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
        layer.self_attn.k_proj.out_features = attn_mask.sum().item()
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
        layer.self_attn.v_proj.out_features = attn_mask.sum().item()
        output_weight = layer.self_attn.o_proj.weight.data
        output_bias = ((attn_mean_inp * ~attn_mask) @ output_weight.T)
        output_weight = output_weight[:, torch.where(attn_mask)[0]]
        shape = output_weight.shape
        layer.self_attn.num_heads = retain_heads
        layer.self_attn.hidden_size = retain_heads * 128
        layer.self_attn.o_proj = torch.nn.Linear(in_features=shape[1], out_features=shape[0], bias=True).to(device)
        layer.self_attn.o_proj.weight.data = output_weight
        layer.self_attn.o_proj.bias.data = output_bias
        print(f"layer{i}-head稀疏率: {retain_heads / 32 * 100:.1f}%")

    # mlp
    if mlp_mask is not None:
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
        layer.mlp.up_proj.out_features = mlp_mask.sum().item()
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
        layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
        output_weight = layer.mlp.down_proj.weight.data
        output_bias = ((mlp_mean_inp * ~mlp_mask) @ output_weight.T)
        output_weight = output_weight[:, torch.where(mlp_mask)[0]]
        shape = output_weight.shape
        layer.mlp.down_proj = torch.nn.Linear(in_features=shape[1], out_features=shape[0], bias=True).to(device)
        layer.mlp.down_proj.weight.data = output_weight
        layer.mlp.down_proj.bias.data = output_bias
        print(f"layer{i}-mlp稀疏率: {shape[1] / 11008 * 100:.1f}%")
        
    torch.cuda.empty_cache()
    

def prune_bias(args, model, tokenizer, device=torch.device("cuda:0")):
    # 参考数据集+初始准备
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    
    mode = [x for x in args.modes.split()]
    metric = {'self_attn.o_proj': args.metrics.split()[0], 'mlp.down_proj': args.metrics.split()[1]}
    
    layers = model.model.layers
    if mode[0] == "adaptive":
        attn_metric_list = []
        attn_baseline_inp_list = []
    if mode[1] == "adaptive":
        mlp_metric_list = []
        mlp_baseline_inp_list = []
        
    # 拆分为子问题, 每个模块单独统计
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        if args.remove_heads != 0:
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        if not math.isclose(args.sparsity_ratio, 0.0):
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], metric[name])            

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
            if name == 'self_attn.o_proj':   # 需要组剪枝
                W_metric = metrics[metric[name]](wrapped_layers, subset, name)
                W_metric = W_metric.reshape(-1, 128).sum(dim=1) # .sqrt() # sqrt(a²+b²+c²)
                if mode[0] == "uniform":
                    thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // len(layers)].cpu()
                    W_mask = (W_metric>=thresh)
                    compress(layer, i, W_mask, None, wrapped_layers[name].baseline_inp.type(torch.half), None, device)
                else:
                    attn_metric_list.append(W_metric)
                    attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[metric[name]](wrapped_layers, subset, name).sqrt()
                if mode[1] == "uniform":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric>=thresh)
                    compress(layer, i, None, W_mask, None, wrapped_layers[name].baseline_inp.type(torch.half), device)
                else:
                    mlp_metric_list.append(W_metric)
                    mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()

        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # 将剪枝后的输出作为下一层的输入
        torch.cuda.empty_cache()
    
    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
    
    # 对波动性整体对齐
    if args.remove_heads != 0 and mode[0] == "adaptive":
        attn_metric = torch.stack(attn_metric_list)   # [32, 32]
        attn_metric = standarlization(attn_metric)
        sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
        attn_thres = sorted_attn[-int(args.remove_heads)]
        attn_mask = (attn_metric > attn_thres)
    
    if not math.isclose(args.sparsity_ratio, 0.0) and mode[1] == "adaptive":
        mlp_metric = torch.stack(mlp_metric_list)     # [32, 11008]
        mlp_metric = standarlization(mlp_metric)
        sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
        mlp_thres = sorted_mlp[-int(len(sorted_mlp)*args.sparsity_ratio)]
        mlp_mask = (mlp_metric > mlp_thres)
    
    # 根据掩码矩阵逐层剪枝
    for idx in range(len(layers)):
        if args.remove_heads != 0 and mode[0] == "adaptive":
            compress(model.model.layers[idx], idx, attn_mask[idx], None, attn_baseline_inp_list[idx], None, device)
        if not math.isclose(args.sparsity_ratio, 0.0) and mode[1] == "adaptive":
            compress(model.model.layers[idx], idx, None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device)
    
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    if args.use_weight_update:
        pass
    
def prune_wanda_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        if args.remove_heads != 0:
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

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
            
            if name == 'self_attn.o_proj':   # 需要组剪枝
                # W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                W_metric = W_metric.mean(axis=0).reshape(-1, 128).sum(dim=1) # old version
                thresh = torch.sort(W_metric.cuda())[0][args.remove_heads].cpu()
                W_mask = (W_metric>=thresh).repeat_interleave(128)
            else:
                W_metric = W_metric.mean(axis=0) # old version
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric>=thresh)
            
            if name == 'self_attn.o_proj':   # 需要组剪枝
                model.model.layers[i].self_attn.q_proj.weight.data = model.model.layers[i].self_attn.q_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].self_attn.q_proj.out_features = W_mask.sum().item()
                model.model.layers[i].self_attn.k_proj.weight.data = model.model.layers[i].self_attn.k_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].self_attn.k_proj.out_features = W_mask.sum().item()
                model.model.layers[i].self_attn.v_proj.weight.data = model.model.layers[i].self_attn.v_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].self_attn.v_proj.out_features = W_mask.sum().item()
                output_weight = model.model.layers[i].self_attn.o_proj.weight.data
                output_weight = output_weight[:, torch.where(W_mask)[0]]
                model.model.layers[i].self_attn.num_heads = int(W_metric.numel()) - args.remove_heads
                model.model.layers[i].self_attn.hidden_size = int(W_metric.numel() - args.remove_heads) * 128
                model.model.layers[i].self_attn.o_proj.weight.data = output_weight
            else:
                model.model.layers[i].mlp.up_proj.weight.data = model.model.layers[i].mlp.up_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].mlp.up_proj.out_features = W_mask.sum().item()
                model.model.layers[i].mlp.gate_proj.weight.data = model.model.layers[i].mlp.gate_proj.weight.data[torch.where(W_mask)[0]]
                model.model.layers[i].mlp.gate_proj.out_features = W_mask.sum().item()
                output_weight = model.model.layers[i].mlp.down_proj.weight.data
                output_weight = output_weight[:, torch.where(W_mask)[0]]
                model.model.layers[i].mlp.down_proj.weight.data = output_weight
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # 将剪枝后的输出作为下一层的输入
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()