#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python examples/evaluate_outputs_rm.py \
  --reward_model_name_or_path CogComp/bart-faithful-summary-detector \
  --output_filepath $1 \
  --path_to_result $2 \
  --four_bits True \
  --per_device_batch_size 4
  #--fp16
  # CogComp/bart-faithful-summary-detector 
  # Tristan/gpt2_reward_summarization

