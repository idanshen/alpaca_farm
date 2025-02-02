#!/bin/bash

export WANDB_DATA_DIR=/data/pulkitag/models/idanshen/shared/wandb/
export WANDB_CACHE_DIR=/data/pulkitag/models/idanshen/shared/wandb/wandb_cache/

bash examples/scripts/rlhf_ppo.sh \
  ./ppo_seahorse_q4_flant5_kl0.05/ \
  run_ppo_seahorse_q4_flant5 \
  ./q_four_flant5/ \
  "" \
  huggyllama/llama-7b \
  /data/pulkitag/models/idanshen/shared/models/sft/sft_seahorse/ \
  0.05 \
  ./seahorse_data/ \ 
  ""
