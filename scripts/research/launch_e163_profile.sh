#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
mkdir -p results/research/E163 results/logs
export CUDA_VISIBLE_DEVICES=2
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for frames in 124 362; do
  "$python_bin" scripts/research/profile_e163_three_paths.py \
    --frames "$frames" --query-block 128 --skip-fraction 0 \
    --nvfp4-fraction 0.9 --warmup 5 --repeats 12 \
    --output "results/research/E163/q128_skip0_fp90_int10_${frames}f_four_path.json"
done
