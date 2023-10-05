#!/bin/bash

ppo_checkpoint=$1
result_file=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir $ppo_checkpoint \
  --load_in_4_bits True \
  --temp 1.0 \
  --greedy True \
  --per_device_batch_size 1 \
  --path_to_result $2 \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons \
  --fp16 \
  --flash_attn
