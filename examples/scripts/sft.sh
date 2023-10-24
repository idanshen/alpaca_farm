output_dir=$1
run_name=$2
model_name_or_path=$3
gpu=$4

# torchrun --nproc_per_node=8 --master_port=1234 
CUDA_VISIBLE_DEVICES=$gpu python3 examples/supervised.py \
  --model_name_or_path "${model_name_or_path}" \
  --dataset_path "./seahorse_data/" \
  --fp16 True \
  --bf16 False \
  --seed 42 \
  --output_dir "${output_dir}" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --per_device_eval_batch_size 4 \
  --eval_steps 500 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "sft" \
  --run_name "${run_name}" \
  --tf32 True \
  --flash_attn False \
  --model_max_length 2048 \
  --train_splits "train" \
  --eval_splits "validation"
  # --ddp_timeout 1800 \
  # --fsdp "full_shard auto_wrap" \
  # --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \

