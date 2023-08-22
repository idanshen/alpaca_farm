#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python examples/generate_qmodel.py \
  --decoder_name_or_path huggyllama/llama-7b \
  --decoder_checkpoint_dir /data/pulkitag/models/idanshen/shared/models/sft/test_5/ \
  --load_in_4_bits True \
  --temp 0.7 \
  --per_device_batch_size 12 \
  --path_to_data ./outputs_sft.json
