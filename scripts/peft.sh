#!/bin/bash

# Set common variables
# model="decapoda-research/llama-7b-hf"
cuda_device=$1
base_model=$2
lora_name=$3
checkpoint=$4

model="llm_weights/$base_model"
lora_weights="alpaca-lora/${lora_name}_${base_model}"

cp "${lora_weights}/checkpoint-${checkpoint}/pytorch_model.bin" "${lora_weights}/adapter_model.bin"

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

echo "Eval peft model in WikiText2"

# Define function to run python command
/data/anyongqi/miniconda3/envs/sp_llm/bin/python3.9 eval_wiki_peft.py \
    --model $model \
    --lora_weights $lora_weights
