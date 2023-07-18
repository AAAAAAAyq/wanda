#%% 
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.data import get_loaders
from lib.analysis import hijack_input, find_module, prepare_calibration_input
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

import torch
import torch.nn.functional as F

seed = 0
nsamples = 1
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
data_type = "c4"
model.config.use_cache = False 
print("loading calibdation data")
dataloader, _ = get_loaders(data_type, nsamples=nsamples, seed=seed, seqlen=2048, tokenizer=tokenizer)
print("dataset loading complete")
##########################################################

#%%

def get_input_features(model, module):
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples)
    
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

def draw_heatmap(data, module, mode, original_shape=(2048, 4096), row_num=8, col_num=4):
    fig, axs = plt.subplots(row_num, col_num, figsize=(15, 20))

    # Calculate the step size for original data
    y_step_size = original_shape[0] // 4
    x_step_size = original_shape[1] // 8

    # Generate the labels for original data
    y_labels = list(range(0, original_shape[0] + 1, y_step_size))
    x_labels = list(range(0, original_shape[1] + 1, x_step_size))

    for i, ax in enumerate(axs.flatten()):
        if mode == 'abs':
            sns.heatmap(data[i].abs().cpu().numpy(), ax=ax, cmap='RdBu', cbar=False, vmin=-12, vmax=12)
        else:
            sns.heatmap(data[i].cpu().numpy(), ax=ax, cmap='RdBu', cbar=False, vmin=-12, vmax=12)
            
        # Set the labels for x and y axis
        ax.set_xticks(np.arange(0, data[i].shape[1]+1, data[i].shape[1] // 8))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(0, data[i].shape[0]+1, data[i].shape[0] // 4))
        ax.set_yticklabels(y_labels)
        
        ax.title.set_text(f'Layer {i+1}')
        # print(f'finish {i}')

    plt.tight_layout()
    plt.savefig(f"figures/input_features/{data_type}_{module}_{mode}_llama_7b.png")
    plt.show()

def analysis_feature(data, logging, module):
    # 各层绝对值>6.0的元素数量统计
    element_counts = [(feature.abs() > abs_threshold).sum().item() for feature in data]
    # for i, count in enumerate(element_counts):
    #     logging.info(f"{module} - Layer {i+1}: {count} elements with absolute value > 6.0")
    # logging.info('-' * 80)
 
    logging.info(f"{module} - Total: {sum(element_counts):,} elements with absolute value > 6.0")    
    indices_list = [torch.where(feature.abs() > abs_threshold) for feature in data]
    indices_list = [(indices[0].tolist(), indices[1].tolist()) for indices in indices_list]
    dim_indices_set = [set(indices[1]) for indices in indices_list]
    merged_dim_indices = [dim for dim_set in dim_indices_set for dim in dim_set]
    dim_counter = Counter(merged_dim_indices)
    # 各层绝对值>6.0的各dim出现层数统计
    # print(dim_counter)

    # Filter out dims that appear in more than 25% of the layers
    filtered_dims = [dim for dim, freq in dim_counter.items() if freq > len(data) * layer_ratio]

    # 各层绝对值>6.0,且dim影响层数超过25%的dim统计
    # print(filtered_dims)
    logging.info(f"{module} - Total: {' '.join(map(str, sorted(filtered_dims)))} dim_indices with absolute value > 6.0 and affected layers > 25%")

    seqlen_frequencies = [Counter(indices[1]) for indices in indices_list]
    # Filter out dim_indices that appear more than threshold times in each layer
    filtered_dim_indices = [[dim for dim, freq in counter.items() if freq > model.seqlen * seq_ratio] for counter in seqlen_frequencies]

    merged_dim_indices_2 = [dim for dim_set in filtered_dim_indices for dim in dim_set]
    dim_counter_2 = Counter(merged_dim_indices_2)
    # 各层绝对值>6.0且seqlen>6%的各dim出现层数统计
    # print(dim_counter_2)
    filtered_dims_2 = [dim for dim, freq in dim_counter_2.items() if freq > len(data) * layer_ratio]
    # 各层绝对值>6.0,且dim影响层数超过25%,且seqlen>6%的dim统计
    # print(filtered_dims_2)
    logging.info(f"{module} - Total: {' '.join(map(str, sorted(filtered_dims_2)))} dim_indices with absolute value > 6.0 and affected layers > 25% and affected seqlen > 6%")
    
#%%
# Set up logging
logging.basicConfig(filename=f'figures/input_features/{data_type}_llama_7b_seed{seed}.txt', level=logging.INFO)

# module = "q_proj" # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj" 
# mode = 'mag'
abs_threshold = 6.0   # 用于筛选绝对值大于6.0的阈值
seq_ratio = 0.06  # 同一层seqlen比例
layer_ratio = 0.25 # 影响层比例

for module in ["q_proj", "o_proj", "down_proj", "up_proj"]:
    features_list = get_input_features(model, module)
    # Assuming features_list is a list of 2D tensors
    pooled_features_list = [sign_preserving_max_pooling(feature) for feature in features_list]

    draw_heatmap(pooled_features_list, module, 'abs', tuple(features_list[0].shape))
    draw_heatmap(pooled_features_list, module, 'mag', tuple(features_list[0].shape))

    analysis_feature(features_list, logging, module)
    print(f'finish {module}')
    logging.info('=' * 80)
