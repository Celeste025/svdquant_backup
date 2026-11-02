#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
mkdir -p results/local/E186 results/research/E186/aba
active=third_party/anemoi-code/anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so
saved=results/local/E186/candidate_unroll4.so
cp "$active" "$saved"
trap 'cp "$saved" "$active"' EXIT
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
export CUDA_VISIBLE_DEVICES=1
for version in original_a optimized original_b; do
  if [ "$version" = optimized ]; then cp "$saved" "$active"; else cp results/local/E170/source_before/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so "$active"; fi
  "$python_bin" scripts/research/validate_e169_480p_paths.py --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 --warmup 10 --repeats 40 --output "results/research/E186/aba/480p_${version}.json"
  "$python_bin" scripts/research/validate_e163_four_paths.py --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 --warmup 10 --repeats 40 --output "results/research/E186/aba/768p_${version}.json"
done
echo E186_ABA_COMPLETE
