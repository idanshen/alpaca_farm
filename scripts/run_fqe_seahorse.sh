gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python3 ./examples/rlhf_fqe.py \
  --run_name fqe_seahorse_q4_flant5 \
  --step_per_device_batch_size 1 \
  --rollout_per_device_batch_size 64 \
  --per_device_eval_batch_size 1 \
  --output_dir fqe_seahorse_q4_flant5 \
  --reward_model_name_or_path ./q_four_flant5/ \
  --policy_model_name_or_path huggyllama/llama-7b \
  --policy_model_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/sft_seahorse/ \
  --init_value_with_reward False \
  --rollout_batch_size 64 \
  --step_batch_size 64 \
  --learning_rate 1e-4 \
  --warmup_steps 10 \
  --total_epochs 1 \
  --flash_attn False \
  --prompt_dict_path /data/pulkitag/models/idanshen/alpaca_farm/examples/prompts/v0_inputs_noinputs.json \
  --save_steps 100 \
  --static_dataset True \
  --query_len 950 \
  --static_dataset_path /data/pulkitag/models/idanshen/shared/data/seahorse_train \
  --static_val_dataset_path /data/pulkitag/models/idanshen/shared/data/validation_dataset.json \
  --evaluation_strategy steps \
  --eval_steps 40 \
  --td_one True \
  --kl_coef 0.02 \
  --q_head_type linear \
  --max_grad_norm 10.0 \
  --wandb_project fqe