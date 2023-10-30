#!/bin/bash

# assign first argument to gpu variable
gpu=$1
output_file=$2

CUDA_VISIBLE_DEVICES=$gpu python examples/create_validation_dataset.py \
  --policy_model_name_or_path huggyllama/llama-7b \
  --policy_model_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/sft_seahorse/ \
  --four_bits True \
  --per_device_batch_size 1 \
  --output_file $2 \
  --dataset_path ./seahorse_data/ \
  --flash_attn False \
  --create_validation_dataset False \
  --infinite_gen True
