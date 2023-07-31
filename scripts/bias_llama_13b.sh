#!/bin/bash

# Set common variables
model="decapoda-research/llama-13b-hf"
# model="/data/anyongqi/wanda/llm_weights/mlp0.5_llama-13b"
# sparsity_ratio=0.625
# remove_heads=8
cuda_device=2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    /data/anyongqi/miniconda3/envs/prune_llm/bin/python3.9 main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $4 \
    --remove_heads $5 \
    --sparsity_type $2 \
    --save $3 \
    --mode per-out \
    --nsamples 1024 \
    --save_model llm_weights/ns1024_seq128_rh${5}_mlp${4}_llama-13b/
}

# llama-13b with skill pruning method
echo "Running with skill pruning method"
run_python_command "skill" "structured" "out/llama-13b/structured/skill_mlp0.3/" 0.3 0     # 20%
# run_python_command "skill" "structured" "out/llama-13b/structured/skill_mlp0.375/" 0.375 0     # 25%
# run_python_command "skill" "structured" "out/llama-13b/structured/skill_mlp0.75/" 0.75 0    # 50%
# run_python_command "skill" "structured" "out/llama-13b/structured/skill_rh10_mlp0.625/" 0.625 10   # 50%
# run_python_command "skill" "structured" "out/llama-13b/structured/skill_rh8_mlp0.65/" 0.65 8   # 50%

# llama-13b with wanda_sp pruning method
# echo "Running with wanda_sp pruning method"
# run_python_command "wanda_sp" "structured" "out/llama-13b/structured/wanda_sp_mlp0.3/" 0.3 0     # 20%
# run_python_command "wanda_sp" "structured" "out/llama-13b/structured/wanda_sp_mlp0.375/" 0.375 0     # 25%
# run_python_command "wanda_sp" "structured" "out/llama-13b/structured/wanda_sp_mlp0.75/" 0.75 0    # 50%
# run_python_command "wanda_sp" "structured" "out/llama-13b/structured/wanda_sp_rh10_mlp0.625/" 0.625 10   # 50%
# run_python_command "wanda_sp" "structured" "out/llama-13b/structured/wanda_sp_rh8_mlp0.65/" 0.65 8   # 50%

# # llama-13b with wanda pruning method
# echo "Running with wanda++ pruning method"
# run_python_command "wanda++" "unstructured" "out/llama-13b/unstructured/wanda_plus_$sparsity_ratio/"

# # llama-13b with taylor pruning method
# echo "Running with taylor pruning method"
# run_python_command "taylor" "unstructured" "out/llama-13b/unstructured/taylor/"

# # llama-13b with weightedobs pruning method
# echo "Running with weightedobs pruning method"
# run_python_command "weightedobs" "unstructured" "out/llama-13b/unstructured/weighteobs/"

# # llama-13b with wanda pruning method
# echo "Running with wanda pruning method"
# run_python_command "wanda" "unstructured" "out/llama-13b/unstructured/wanda_$sparsity_ratio/"
# run_python_command "wanda" "2:4" "out/llama-13b/2-4/wanda/"
# run_python_command "wanda" "4:8" "out/llama-13b/4-8/wanda/"
# echo "Finished wanda pruning method"

# # llama-13b with sparsegpt pruning method
# echo "Running with sparsegpt pruning method"
# run_python_command "sparsegpt" "unstructured" "out/llama-13b/unstructured/sparsegpt/"
# run_python_command "sparsegpt" "2:4" "out/llama-13b/2-4/sparsegpt/"
# run_python_command "sparsegpt" "4:8" "out/llama-13b/4-8/sparsegpt/"
# echo "Finished sparsegpt pruning method"

# # llama-13b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama-13b/unstructured/magnitude/"
# run_python_command "magnitude" "2:4" "out/llama-13b/2-4/magnitude/"
# run_python_command "magnitude" "4:8" "out/llama-13b/4-8/magnitude/"
# echo "Finished magnitude pruning method"