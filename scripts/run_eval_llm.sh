#!/bin/bash

python examples/evaluate_outputs_llm.py \
  --output_filepath1 ./outputs/outputs_sft.json \
  --output_filepath2 ./outputs/outputs_ppo.json \
  --eval_outputs_llm eval_outputs_sft_vs_ppo
# --path_to_result ./outputs_rewards_ppo.json

