#!/bin/bash

bash examples/scripts/rlhf_ppo.sh \
  ./pretrained_models/ppo_news_summarization \
  run_ppo_news_summarization \
  Tristan/gpt2_reward_summarization \
  decapoda-research/llama-7b-hf \
  ./pretrained_models/sft10k/ \
  0.2 \
  argilla/news-summary \
  comparisons
