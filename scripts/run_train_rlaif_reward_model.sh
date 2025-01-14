#!/bin/bash

output_dir=$1
run_name=$2
lr=$3
gpu=$4

config_file="./examples/accelerate_configs/rlhf_ppo_npp_llama.yaml"

CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --config_file "${config_file}" examples/rlaif_reward_modeling.py \
  --output_dir "${output_dir}" \
  --run_name "${run_name}" \
  --num_train_epochs 3 \
  --fp16 True \
  --bf16 False \
  --seed 42 \
  --model_name_or_path "huggyllama/llama-7b" \
  --pretrained_lora_weights "/data/pulkitag/models/idanshen/shared/models/sft/test_5/" \
  --dataset_path openai/summarize_from_feedback \
  --model_max_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 128 \
  --eval_steps 500 \
  --save_strategy "steps" \
  --save_steps 100 \
  --learning_rate "${lr}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "rlaif" \
  --flash_attn True \
  --tf32 True
