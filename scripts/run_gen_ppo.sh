#!/bin/bash

ppo_checkpoint=$1
result_file=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir $ppo_checkpoint \
  --sft_checkpoint_dir /mnt/nfs_csail/models/idanshen/shared/models/sft/sft_seahorse/ \
  --load_in_4_bits True \
  --temp 1.0 \
  --per_device_batch_size 1 \
  --path_to_result $2 \
  --dataset_path ./seahorse_data/ \
  --dataset_name '' \
  --fp16 \
  --flash_attn False
