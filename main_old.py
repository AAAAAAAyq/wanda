import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune_old import prune_taylor, prune_with_skill, prune_wanda, prune_magnitude, prune_sparsegpt, prune_weightedobs_v2,prune_wanda_sp,prune_bias_unify
from lib.prune_old import check_sparsity, prune_wanda_plus, check_sparsity_skill
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=2048, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--remove_heads', type=int, default=8, help='Remove num_heads')
    parser.add_argument("--sparsity_type", type=str, choices=["structured", "unstructured", "4:8", "2:4"])
    parser.add_argument("--mode", type=str, choices=["per-layer", "per-out"])
    parser.add_argument("--prune_method", type=str, choices=["unify", "skill", "taylor", "magnitude", "wanda_sp", "wanda", "wanda++", "sparsegpt", "weightedobs"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")    # TODO:what is this?
    parser.add_argument('--save', type=str, default=None, help='Path to save results.') # save the log file
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_type != "structured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)   # the core algorithm!!
        elif args.prune_method == "skill":
            prune_with_skill(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda++":
            prune_wanda_plus(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda_sp":
            prune_wanda_sp(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "unify":
            prune_bias_unify(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "taylor":
            prune_taylor(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            exit()
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "weightedobs":
            prune_weightedobs_v2(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    if args.prune_method in ["skill", "wanda_sp", "sparsegpt_sp", "unify"]:
        sparsity_ratio = check_sparsity_skill(model)  # check the sparsity of the model
    else:
        sparsity_ratio = check_sparsity(model)  # check the sparsity of the model
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl = eval_ppl(model, tokenizer, device)    # evaluate the model
    print(f"ppl on wikitext {ppl}")
    exit()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, "log.txt")
    with open(save_filepath, "w") as f:
        print("actual_sparsity\tppl", file=f, flush=True)
        print(f"{sparsity_ratio:.4f}\t{ppl:.4f}", file=f, flush=True)
        
    
    model.config.num_attention_heads = model.model.layers[0].self_attn.num_heads
    model.config.intermediate_size = model.model.layers[0].mlp.up_proj.out_features
    

    if args.save_model:
        # torch.save({
        #     'model': model,
        #     'tokenizer': tokenizer,
        # }, args.save_model)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()