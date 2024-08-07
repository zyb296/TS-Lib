#!/bin/bash

# root_dir=$(pwd)  # 项目工作路径
# echo "root_dir: $root_dir"

python -u run_PatchTST.py \
  --task_name classification \
  --root_path ./dataset \
  --model PatchTST \
  --log_dir ./log/patchTST \
  --data sleep_data \
  --num_class 3 \
  --patch_len 8 \
  --stride 8 \
  --seq_len 180 \
  --enc_in 2 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 512 \
   --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 20