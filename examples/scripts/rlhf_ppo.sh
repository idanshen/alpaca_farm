output_dir="./ppo_seahorse_q4_flant5_kl0.05/"
run_name="run_ppo_seahorse_q4_flant5"
reward_model_name_or_path="./q_four_flant5/"
reward_checkpoint=""
policy_model_name_or_path="huggyllama/llama-7b"
policy_checkpoint="/data/pulkitag/models/idanshen/shared/models/sft/sft_seahorse/"
kl_coef=0.0067
dataset_path="./seahorse_data/"
dataset_name=""

config_file="./examples/accelerate_configs/rlhf_ppo_npp_llama.yaml"

CUDA_VISIBLE_DEVICES=0,1 python3 examples/rlhf_ppo.py \
  --run_name "${run_name}" \
  --step_per_device_batch_size 1 \
  --rollout_per_device_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --output_dir "${output_dir}" \
  --reward_model_name_or_path "${reward_model_name_or_path}" \
  --reward_model_checkpoint_dir "${reward_checkpoint}" \
  --policy_model_name_or_path "${policy_model_name_or_path}" \
  --policy_model_checkpoint_dir "${policy_checkpoint}" \
  --dataset_path "${dataset_path}" \
  --dataset_name "${dataset_name}" \
  --init_value_with_reward False \
  --rollout_batch_size 512 \
  --step_batch_size 128 \
  --learning_rate 1e-4 \
  --warmup_steps 5 \
  --kl_coef "${kl_coef}" \
  --total_epochs 1 \
  --flash_attn False \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --eval_steps 10 \
  --save_steps 5 \
  --train_splits "train" \
  --eval_splits "validation" \
  --query_len 950 \
  --wandb_project ppo

  # --step_per_device_batch_size 2 \
  # --rollout_per_device_batch_size 32 \
  # --per_device_eval_batch_size 32 \
  # --rollout_batch_size 512 \
  # --step_batch_size 256 \
