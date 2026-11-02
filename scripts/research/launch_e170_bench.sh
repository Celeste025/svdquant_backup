#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
mkdir -p results/research/E170 results/logs
"$python_bin" scripts/research/profile_e169_480p_paths.py \
  --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 \
  --warmup 5 --repeats 12 \
  --output results/research/E170/480p_124f_q128_skip0_fp90_int10.json
"$python_bin" scripts/research/profile_e163_three_paths.py \
  --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 \
  --warmup 5 --repeats 12 \
  --output results/research/E170/768p_124f_q128_skip0_fp90_int10.json
