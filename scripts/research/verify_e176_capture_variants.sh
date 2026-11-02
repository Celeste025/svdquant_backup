#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
mkdir -p results/local/E176 results/research/E176
active=third_party/anemoi-code/anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so
saved=results/local/E176/accepted_e173.so
cp "$active" "$saved"
trap 'cp "$saved" "$active"' EXIT
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
export CUDA_VISIBLE_DEVICES=1
for version in optimized original; do
  if [ "$version" = original ]; then cp results/local/E170/source_before/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so "$active"; fi
  "$python_bin" scripts/research/check_e172_outputs.py --input-size 480 --frames 124 --query-block 64 --skip-fraction 0.75 --nvfp4-fraction 0.6 --output "results/research/E176/480_q64_${version}.json"
  "$python_bin" scripts/research/check_e172_outputs.py --input-size 768 --frames 124 --query-block 64 --skip-fraction 0 --nvfp4-fraction 0.9 --output "results/research/E176/768_q64_${version}.json"
  "$python_bin" scripts/research/check_e172_outputs.py --input-size 768 --frames 362 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 --output "results/research/E176/768_long_${version}.json"
done
cmp results/research/E176/480_q64_original_hashes.json results/research/E176/480_q64_optimized_hashes.json
cmp results/research/E176/768_q64_original_hashes.json results/research/E176/768_q64_optimized_hashes.json
cmp results/research/E176/768_long_original_hashes.json results/research/E176/768_long_optimized_hashes.json
echo ALL_VARIANTS_BITWISE_EQUAL
