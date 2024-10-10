# export CUDA_VISIBLE_DEVICES=2
# 设置 PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

model_name=TimesNet
file_path=./run

# python -m debugpy --listen 5678 --wait-for-client $file_path/run_TimesNet.py \
python -u $file_path/run_TimesNet.py \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --batch_size 32 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \