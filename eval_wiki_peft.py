import argparse
import os 
import numpy as np
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from models.hf_llama.modeling_llama import LlamaForCausalLM
from lib.eval import eval_ppl
from peft import PeftModel

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(base_model, lora_weights):
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model, 
    #     torch_dtype=torch.float16, 
    #     cache_dir="llm_weights", 
    #     low_cpu_mem_usage=True, 
    #     device_map="auto"
    # )
    
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    model.seqlen = 128
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--prune_method", type=str, choices=["skill", "taylor", "magnitude", "wanda", "wanda++", "sparsegpt", "weightedobs"])
    parser.add_argument("--lora_weights", default="lora_weights", type=str )
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.lora_weights)
    
    model.half()  # seems to fix bugs for some users.
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    ppl = eval_ppl(model, tokenizer, device)    # evaluate the model
    print(f"ppl on wikitext {ppl}")
    # exit()

if __name__ == '__main__':
    main()