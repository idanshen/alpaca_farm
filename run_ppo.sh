#!/bin/bash

export WANDB_DATA_DIR=/data/pulkitag/models/idanshen/shared/wandb/
export WANDB_CACHE_DIR=/data/pulkitag/models/idanshen/shared/wandb/wandb_cache/

bash examples/scripts/rlhf_ppo.sh \
  /data/pulkitag/models/idanshen/shared/models/ppo_news_sum_argilla4/ \
  run_ppo_news_summarization_argilla \
  Tristan/gpt2_reward_summarization \
  huggyllama/llama-7b \
  /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  0.0067 \
  argilla/news-summary \
  comparisons
