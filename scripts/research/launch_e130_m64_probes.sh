#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=2
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
baseline=/data1/models/svdquant-wjq/research/20261025/E130/optin_rawk_readonly
candidate=/tmp/e130-m64-dsfix

for probe in zero_qk ones_v zero_qk_ones_v; do
    for arm in baseline candidate; do
        source_dir=$baseline
        if [[ $arm == candidate ]]; then
            source_dir=$candidate
        fi
        "$worker" -u scripts/research/capture_e130_sage3_output.py \
            --sage3-dir "$source_dir" --tokens 256 --probe "$probe" \
            --output "/tmp/e130_m64_${arm}_${probe}_256.pt"
    done
    "$worker" scripts/research/compare_e130_saved_outputs.py \
        --reference "/tmp/e130_m64_baseline_${probe}_256.pt" \
        --candidate "/tmp/e130_m64_candidate_${probe}_256.pt" \
        --output "results/research/E130/m64_dsfix_${probe}_256.json"
done
