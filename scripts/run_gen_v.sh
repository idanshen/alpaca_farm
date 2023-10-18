#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/ppo_news_sum_argilla4/ \
  --q_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/fve_news_sum_argilla/fve/ \
  --topk 20 \
  --load_in_4_bits True \
  --temp 0.7 \
  --per_device_batch_size 1 \
  --path_to_result outputs_fve_beta_1.0_topk20.json \
  --beta 1.0 \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons \
  --fp16 \
  --flash_attn False

