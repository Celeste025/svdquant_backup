#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
mkdir -p results/research/E193 results/local/E193
active=third_party/anemoi-code/anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so
saved=results/local/E193/bank_candidate.so
cp "$active" "$saved"
trap 'cp "$saved" "$active"' EXIT
export CUDA_VISIBLE_DEVICES=1
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for version in baseline_a candidate baseline_b; do
 if [ "$version" = candidate ]; then cp "$saved" "$active"; else cp results/local/E192/source_before/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so "$active"; fi
 "$python_bin" scripts/research/validate_e169_480p_paths.py --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 --warmup 10 --repeats 50 --output "results/research/E193/480p_${version}.json"
 "$python_bin" scripts/research/validate_e163_four_paths.py --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 --warmup 10 --repeats 50 --output "results/research/E193/768p_${version}.json"
done
echo E193_BANK_ABA_COMPLETE
