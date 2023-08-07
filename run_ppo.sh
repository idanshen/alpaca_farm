#!/bin/bash

bash examples/scripts/rlhf_ppo.sh \
  ./pretrained_models/ppo_news_summarization \
  run_ppo_news_summarization \
  Tristan/gpt2_reward_summarization \
  huggyllama/llama-7b \
  /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  0.2 \
  argilla/news-summary \
  comparisons
