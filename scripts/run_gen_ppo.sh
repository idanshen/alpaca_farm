#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/ppo_news_sum_argilla4/ \
  --load_in_4_bits True \
  --temp 0.7 \
  --per_device_batch_size 4 \
  --path_to_data ./outputs_ppo.json \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons