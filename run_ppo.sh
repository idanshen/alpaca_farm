#!/bin/bash

bash examples/scripts/rlhf_ppo.sh \
  /data/pulkitag/models/idanshen/shared/models/ppo_news_sum_argilla/ \
  run_ppo_news_summarization_argilla \
  Tristan/gpt2_reward_summarization \
  huggyllama/llama-7b \
  /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  0.2 \
  argilla/news-summary \
  comparisons
