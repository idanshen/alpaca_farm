#!/bin/bash

# assign first argument to gpu variable
gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python examples/create_validation_dataset.py \
  --policy_model_name_or_path huggyllama/llama-7b \
  --policy_model_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  --four_bits True \
  --per_device_batch_size 1 \
  --output_file ./openai_summarize_from_feedback_fqe_train.json \
  --dataset_path openai/summarize_from_feedback \
  --dataset_name comparisons \
  --flash_attn True
