import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT
from .data import get_loaders 


def find_module(module, layers=[nn.Linear], name='', target_name='q_proj'):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.
        target_name (str): The target string to search for in layer names.

    """
    if type(module) in layers and target_name in name:
        return {target_name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_module(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1, target_name=target_name
        ))
    return res

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

def hijack_input(module, list_to_append):
    # 将module的输入存入list_to_append中
    hook = lambda _, inputs: list_to_append.append(inputs[0].data)
    handle = module.register_forward_pre_hook(hook)
    return handle

def get_features(model, tokenizer, module, seed, nsamples, device=torch.device("cuda:0")):
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    features_list = []
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        
        target_layer = find_module(layer, target_name=module)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        inputs = []
        handle = hijack_input(target_layer[module], inputs)
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        handle.remove()
        
        torch.cuda.empty_cache()
        
        features_list.append(torch.cat(inputs, dim=0).mean(dim=0))
        inps, outs = outs, inps # 将剪枝后的输出作为下一层的输入

    return features_list
        