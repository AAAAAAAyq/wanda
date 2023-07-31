#%% 
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.data import get_loaders
from lib.analysis import hijack_input, find_module
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn.functional as F

seed = 0
nsamples = 128
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    model.seqlen = 128
    return model

print(f"loading llm model {model_name}")
model = get_llm(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

device = torch.device("cuda:0")

#%%
data_type = "wikitext2"
model.config.use_cache = False 
print("loading calibdation data")
dataloader, _ = get_loaders(data_type, nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer)
print("dataset loading complete")
##########################################################

#%%
def get_input_features(model, dataloader, module, layer_id):
    for param in model.parameters():
        param.requires_grad = False
    
    layers = model.model.layers
    layer = layers[layer_id]
    target_layer = find_module(layer, target_name=module)

    features_list = []
    handle = hijack_input(target_layer[module], features_list)
    for batch in tqdm(dataloader):
        with torch.cuda.amp.autocast():
            model(batch[0].to(device))
    handle.remove()
    torch.cuda.empty_cache()
    return features_list
    
def sign_preserving_max_pooling(tensor, kernel_size=16, stride=16, padding=0):
    # Save the sign of the original tensor
    tensor_sign = tensor.sign().flatten()
    # Apply max pooling to the absolute values, and get the indices of max values
    tensor_abs = tensor.abs().reshape(1, 1, tensor.shape[0], tensor.shape[1])
    tensor_abs_pooled, indices = F.max_pool2d(tensor_abs, kernel_size, stride, padding, return_indices=True)
    tensor_abs_pooled = tensor_abs_pooled.reshape(tensor_abs_pooled.shape[2], tensor_abs_pooled.shape[3])

    # Use the indices of max values to select signs
    tensor_sign_pooled = tensor_sign[indices.flatten()].reshape(tensor_abs_pooled.shape)

    # Multiply the pooled absolute values by the pooled signs to restore the original signs
    tensor_pooled = tensor_abs_pooled * tensor_sign_pooled

    return tensor_pooled

def draw_loss_each_channel(data, module, original_shape=(2048, 4096), row_num=8, col_num=4):
    fig, axs = plt.subplots(row_num, col_num, figsize=(20, 15))
    # Calculate the step size for original data
    x_step_size = original_shape[1] // 8

    # Generate the labels for original data
    x_labels = list(range(0, original_shape[1] + 1, x_step_size))
    for i, ax in enumerate(axs.flatten()):
        L2_Loss = torch.sqrt(torch.sum(data[i]**2, dim=0))
        ax.plot(L2_Loss.cpu().numpy())
            
        # Set the labels for x and y axis
        ax.set_xticks(np.arange(0, data[i].shape[1]+1, data[i].shape[1] // 8))
        ax.set_xticklabels(x_labels)
        ax.title.set_text(f'Sample {i} L2 Loss')
        ax.set_xlabel("Feature Index")  # 设置x轴标签
        ax.set_ylabel("L2 Loss")  # 设置y轴标签
        # print(f'finish {i}')

    plt.tight_layout()
    plt.savefig(f"figures/skill_l2_loss/L2_loss_l{layer_id}_{module}_llama_7b.png")
    plt.show()

#%%
def draw_heatmap(data, fig_title, original_shape=(2048, 4096), row_num=8, col_num=4):
    fig, axs = plt.subplots(row_num, col_num, figsize=(20, 20))

    # Calculate the step size for original data
    y_step_size = original_shape[0] // 8
    x_step_size = original_shape[1] // 8

    # Generate the labels for original data
    y_labels = list(range(0, original_shape[0] + 1, y_step_size))
    x_labels = list(range(0, original_shape[1] + 1, x_step_size))

    for i, ax in enumerate(axs.flatten()):
        sns.heatmap(data[i].cpu().numpy(), ax=ax, cmap='RdBu', cbar=False, vmin=-0.1, vmax=0.1)  # , vmin=-0.5, vmax=0.5
                
        # Set the labels for x and y axis
        ax.set_xticks(np.arange(0, data[i].shape[1]+1, data[i].shape[1] // 8))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(0, data[i].shape[0]+1, data[i].shape[0] // 8))
        ax.set_yticklabels(y_labels)
        
        ax.set_xlabel("seqlen")
        ax.set_ylabel("nsamples")

    plt.title(fig_title)
    
    plt.tight_layout()
    plt.show()

# Set up logging
module = "down_proj" # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj" 
layer_id = 5
offset = 1520
num_channel = 16
features_list = get_input_features(model, dataloader, module, layer_id)
channel_feature = torch.cat(features_list)[:,:,offset:num_channel+offset]
channel_features_list = [channel_feature[:,:,i] for i in range(num_channel)]
title = f"layer={layer_id} module={module} offset={offset}"
draw_heatmap(channel_features_list, title, tuple(channel_features_list[0].shape), row_num=4, col_num=4)

    # print(f'finish layer{layer_id}-{module}')

# %%
