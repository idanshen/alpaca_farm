#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python examples/evaluate_outputs_rm.py \
  --reward_model_name_or_path Tristan/gpt2_reward_summarization \
  --output_filepath ./outputs_sft.json \
  --path_to_result ./outputs_rewards_sft.json \
  --load_in_4_bits True \
  --per_device_batch_size 12

