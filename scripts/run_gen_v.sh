#!/bin/bash

vmodel=$1
topk=$2
beta=$3
gpu=$4

CUDA_VISIBLE_DEVICES=$gpu python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  --v_checkpoint_dir "$vmodel" \
  --topk "$topk" \
  --load_in_4_bits True \
  --temp 0.7 \
  --per_device_batch_size 1 \
  --path_to_result "outputs_fve_tristan_beta$beta_topk$topk.json" \
  --beta "$beta" \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons \
  --fp16 \
  --flash_attn False

