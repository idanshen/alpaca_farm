#!/bin/bash

python examples/evaluate_outputs_llm.py \
  --output_filepath1 ./outputs/outputs_fqecql2_beta2.0.json \
  --output_filepath2 ./outputs/outputs_ppo.json \
  --exp_name eval_outputs_ppo_vs_fqecql2beta2.0
# --path_to_result ./outputs_rewards_ppo.json

