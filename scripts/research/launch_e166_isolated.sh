#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=2
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for frames in 124 362; do
  "$python_bin" scripts/research/profile_e163_three_paths.py \
    --frames "$frames" --query-block 128 --skip-fraction 0 \
    --nvfp4-fraction 0.9 --warmup 1 --repeats 3 \
    --output "results/research/E166/q128_skip0_fp90_int10_${frames}f_isolated.json"
done
