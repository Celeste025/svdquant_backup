#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
mkdir -p results/local/E188 results/research/E188
active=third_party/anemoi-code/anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so
saved=results/local/E188/current_optimized.so
cp "$active" "$saved"
trap 'cp "$saved" "$active"' EXIT
export CUDA_VISIBLE_DEVICES=1
unset ANEMOI_PHASE_LSE || true
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for version in original_a optimized original_b; do
  if [ "$version" = optimized ]; then cp "$saved" "$active"; else cp results/local/E170/source_before/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so "$active"; fi
  "$python_bin" scripts/research/bench_e188_trajectory_reuse.py --warmup 10 --repeats 30 --output "results/research/E188/${version}.json"
done
echo E188_ABA_COMPLETE
