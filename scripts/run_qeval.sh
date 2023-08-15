#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python examples/evaluate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  --q_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/ppo_news_sum_argilla_1/ \
  --path_to_data ./output.json \
  --beta 1.0 \
  --exp_name qmodel_eval
