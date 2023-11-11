#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python examples/evaluate_outputs_rm.py \
  --reward_model_name_or_path huggyllama/llama-7b \
  --reward_model_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/reward/rlaif_openai_summarize_from_feedback5/ \
  --output_filepath $1 \
  --path_to_result $2 \
  --per_device_batch_size 1 \
  --fp16 True \
  --use_lora True \
  --four_bits True \
  --soft_preference True
  # CogComp/bart-faithful-summary-detector 
  # Tristan/gpt2_reward_summarization

