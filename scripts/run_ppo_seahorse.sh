#!/bin/bash

export WANDB_DATA_DIR=/data/pulkitag/models/idanshen/shared/wandb/
export WANDB_CACHE_DIR=/data/pulkitag/models/idanshen/shared/wandb/wandb_cache/

bash examples/scripts/rlhf_ppo.sh \
  ./ppo_seahorse_flan_t5_kl0.0067/ \
  run_ppo_seahorse_flan_t5 \
  /data/pulkitag/models/idanshen/shared/models/reward/seahorse/q_six_final1/ \
  huggyllama/llama-7b \
  /data/pulkitag/models/idanshen/shared/models/sft/sft_seahorse/ \
  0.0067 \
  ./seahorse_data/ \ 
  ""
