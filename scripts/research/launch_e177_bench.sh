#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for size in 480 768; do
  "$python_bin" scripts/research/check_e172_outputs.py \
    --input-size "$size" --frames 124 --query-block 128 --skip-fraction 0 \
    --nvfp4-fraction 0.9 --output "results/research/E177/${size}p_after_check.json"
  if [ "$size" = 480 ]; then profiler=profile_e169_480p_paths.py; else profiler=profile_e163_three_paths.py; fi
  "$python_bin" "scripts/research/$profiler" \
    --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 \
    --warmup 5 --repeats 12 --output "results/research/E177/${size}p_after.json"
done
