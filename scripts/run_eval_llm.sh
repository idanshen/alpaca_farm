#!/bin/bash

output_filepath1=$1
output_filepath2=$2

python examples/evaluate_outputs_llm.py \
  --output_filepath1 $output_filepath1 \
  --output_filepath2 $output_filepath2
# --path_to_result ./outputs_rewards_ppo.json

