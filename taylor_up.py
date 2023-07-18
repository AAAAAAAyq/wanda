#%% 
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.data import get_loaders
import torch.nn as nn 
from tqdm import tqdm
from lib.layerwrapper import WrappedGPT

seed = 0
nsamples = 128
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Setting seeds for reproducibility
np.random.seed(seed)
torch.random.manual_seed(seed)

#%%
model_name = 'decapoda-research/llama-7b-hf'

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.seqlen = 2048
    return model

print(f"loading llm model {model_name}")
model = get_llm(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

device = torch.device("cuda:0")

#%%
model.config.use_cache = False 
print("loading calibdation data")
dataloader, _ = get_loaders("c4", nsamples=nsamples, seed=seed, seqlen=2048, tokenizer=tokenizer)
print("dataset loading complete")

#%%
def mask_analyisis_magnitude(model, layer_name, layer_id, sparsity_ratio=0.9):
    W = model.model.layers[layer_id].self_attn.q_proj.weight.data
    W_metric = torch.abs(W)
    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
    W_mask = (W_metric>=thresh)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(W_mask.cpu().numpy(), cmap='Blues', interpolation='nearest')
    plt.title(f'{layer_name} \nsparsity={sparsity_ratio} \nmagnitude')
    plt.savefig(f'figures/taylor_up/0_{layer_name}_{sparsity_ratio}.png')

def mask_analyisis_taylor(model, dataloader, layer_name, layer_id, device=torch.device("cuda:0"), sparsity_ratio=0.9):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    # 假设 model 是你的模型
    for param in model.parameters():
        param.requires_grad = False
    model.model.layers[layer_id].self_attn.q_proj.weight.requires_grad = True
    grads = []
    for batch in tqdm(dataloader):
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
        grads.append(model.model.layers[layer_id].self_attn.q_proj.weight.grad.detach())
    
    grads = torch.stack(grads, dim=0)

    print(f"pruning {layer_name}")
    W = model.model.layers[layer_id].self_attn.q_proj.weight.data
    W_metric = torch.abs(W) * grads.mean(axis=0)
    
    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
    W_mask = (W_metric>=thresh)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(W_mask.cpu().numpy(), cmap='Blues', interpolation='nearest')
    plt.title(f'{layer_name} \nsparsity={sparsity_ratio} \ntaylor up')
    plt.savefig(f'figures/taylor_up/1_{layer_name}_{sparsity_ratio}.png')
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
def mask_analyisis_wanda(model, dataloader, layer_name, layer_id, mode, device=torch.device("cuda:0"), sparsity_ratio=0.9):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    for param in model.parameters():
        param.requires_grad = False
        
    wrapped_layer = WrappedGPT(model.model.layers[layer_id].self_attn.q_proj)

    def add_batch():
        def tmp(_, inp, out):
            wrapped_layer.add_batch(inp[0].data, out.data)
        return tmp
    handle = model.model.layers[layer_id].self_attn.q_proj.register_forward_hook(add_batch())
    for batch in tqdm(dataloader):
        with torch.cuda.amp.autocast():
            model(batch[0].to(device))
    handle.remove()

    print(f"pruning {layer_name}")
    W_metric = torch.abs(model.model.layers[layer_id].self_attn.q_proj.weight.data) * torch.sqrt(wrapped_layer.scaler_row.reshape((1,-1)))
    wrapped_layer.free()
    
    W_mask = (torch.zeros_like(W_metric) == 0)  ## initialize a mask to be all False

    # per-layer
    if mode == 'per-layer':
        W = model.model.layers[layer_id].self_attn.q_proj.weight.data
        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
        W_mask = (W_metric>=thresh)

    elif mode == 'per-out':
        # per-out
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
        W_mask.scatter_(1, indices, False)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(W_mask.cpu().numpy(), cmap='Blues', interpolation='nearest')
    plt.title(f'{layer_name} \nsparsity={sparsity_ratio} \nwanda \n{mode}')
    plt.savefig(f'figures/taylor_up/wanda_{mode}_{layer_name}_{sparsity_ratio}.png')


    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
 
 
    
##########################################################
#%%
layer_id = 30
layer_name = f'layer{layer_id}.q_proj'
sparsity_ratio = 0.90
mask_analyisis_taylor(model, dataloader, layer_name, layer_id, device, sparsity_ratio=sparsity_ratio)
mask_analyisis_wanda(model, dataloader, layer_name, layer_id, mode='per-layer', device=device, sparsity_ratio=sparsity_ratio)
mask_analyisis_wanda(model, dataloader, layer_name, layer_id, mode='per-out', device=device, sparsity_ratio=sparsity_ratio)
mask_analyisis_magnitude(model, layer_name, layer_id, sparsity_ratio=sparsity_ratio)
