#!/bin/bash
q_checkpoint=$1
beta=$2
result_file=$3
gpu=$4

CUDA_VISIBLE_DEVICES=$gpu python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  --q_checkpoint_dir $q_checkpoint \
  --num_q_heads 1 \
  --q_head_type linear \
  --load_in_4_bits True \
  --greedy True \
  --temp 1.0 \
  --per_device_batch_size 1 \
  --path_to_result $result_file \
  --beta $beta \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons \
  --fp16 \
  --flash_attn True

