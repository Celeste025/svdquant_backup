#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wjq/workspace/svdquant-exp
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
baseline=third_party/sage3_h3_maskfix
combined=/data1/models/svdquant-wjq/research/20261025/E130/optin_rawk_readonly
export CUDA_VISIBLE_DEVICES=${1:-2}
export PYTHONUNBUFFERED=1
mkdir -p results/research/E130 results/logs

"$worker" -u scripts/research/bench_e130_combined_lengths.py \
  --sage3-dir "$baseline" --mode baseline \
  --output results/research/E130/combined_lengths_baseline_round1.json
"$worker" -u scripts/research/bench_e130_combined_lengths.py \
  --sage3-dir "$combined" --mode combined \
  --output results/research/E130/combined_lengths_optimized_round1.json
"$worker" -u scripts/research/bench_e130_combined_lengths.py \
  --sage3-dir "$combined" --mode combined \
  --lengths 22677 16384 9482 8192 4096 2048 1024 \
  --output results/research/E130/combined_lengths_optimized_round2.json
"$worker" -u scripts/research/bench_e130_combined_lengths.py \
  --sage3-dir "$baseline" --mode baseline \
  --lengths 22677 16384 9482 8192 4096 2048 1024 \
  --output results/research/E130/combined_lengths_baseline_round2.json
