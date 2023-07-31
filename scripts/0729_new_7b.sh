#!/bin/bash

# Set common variables
model="decapoda-research/llama-7b-hf"
# model="/data/anyongqi/wanda/llm_weights/mlp0.5_llama_7b"
sparsity_ratio=0.425
remove_heads=50
cuda_device=$1

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    /data/anyongqi/miniconda3/envs/prune_llm/bin/python3.9 main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --remove_heads $remove_heads \
    --metrics "L2 L2W" \
    --save $2 \
    --modes "adaptive adaptive" \
    --nsamples 1024 \
    --save_model llm_weights/ns1024_seq128_rh${remove_heads}_mlp${sparsity_ratio}_llama_7b/
}

# llama-7b with bias_unify pruning method
echo "Running with bias pruning method"
run_python_command "bias" "out/llama_7b/structured/bias_rh${remove_heads}_mlp${sparsity_ratio}/"

# # llama-7b with wanda_sp pruning method
# echo "Running with wanda_sp pruning method"
# run_python_command "wanda_sp" "out/llama_7b/unstructured/wanda_sp_rh${remove_heads}_mlp${sparsity_ratio}/"
