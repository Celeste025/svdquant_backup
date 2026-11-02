#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
active=third_party/anemoi-code/anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so
saved=results/local/E172/optimized_sm120.so
cp "$active" "$saved"
trap 'cp "$saved" "$active"' EXIT
cp results/local/E170/source_before/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so "$active"
export CUDA_VISIBLE_DEVICES=1
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for size in 480 768; do
  "$python_bin" scripts/research/check_e172_outputs.py \
    --input-size "$size" --frames 124 --query-block 128 --skip-fraction 0 \
    --nvfp4-fraction 0.9 --output "results/research/E172/${size}p_original_check.json"
done
