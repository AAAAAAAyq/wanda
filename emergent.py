#%% 
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.analysis import get_features

seed = 0
nsamples = 128

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
module = "gate_proj" # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj" 
features_list = get_features(model, tokenizer, module, seed, nsamples, device)  # layer*[bs,seq,dim]
import pdb;pdb.set_trace()
