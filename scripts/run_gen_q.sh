#!/bin/bash
q_checkpoint=$1
beta=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/ppo_news_sum_argilla4/ \
  --q_checkpoint_dir $q_checkpoint \
  --num_q_heads 1 \
  --q_head_type projection \
  --load_in_4_bits True \
  --temp 0.7 \
  --per_device_batch_size 1 \
  --path_to_result outputs_fqecql3i_beta0.5.json \
  --beta $beta \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons \
  --fp16 \
  --flash_attn False

