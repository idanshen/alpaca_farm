#!/bin/bash

topk=$1
beta=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  --q_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/new_models_for_eval/fve_norm_10/ \
  --topk "$topk" \
  --load_in_4_bits True \
  --temp 0.7 \
  --per_device_batch_size 16 \
  --path_to_result "outputs_fve_tristan_beta$beta_topk$topk.json" \
  --beta "$beta" \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons \
  --fp16 \
  --flash_attn False

