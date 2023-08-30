#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/ppo_news_sum_argilla4/ \
  --q_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/fqe_news_sum_argilla/fqe_cql_2/ \
  --load_in_4_bits True \
  --temp 0.7 \
  --per_device_batch_size 2 \
  --path_to_result outputs_fqecql2_beta0.5.json \
  --beta 0.5 \
  --dataset_path argilla/news-summary \
  --dataset_name comparisons \
  --fp16

