#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for size in 480 768; do
  "$python_bin" scripts/research/bench_e171_static_materialized.py \
    --input-size "$size" --frames 124 --query-block 128 --skip-fraction 0 \
    --nvfp4-fraction 0.9 --warmup 5 --repeats 30 \
    --output "results/research/E171/${size}p_124f.json"
done
