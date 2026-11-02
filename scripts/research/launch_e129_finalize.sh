#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
cd "$root"
logs="$root/results/logs"
while true; do
    all_complete=1
    for gpu in 0 1 2 3; do
        if rg -q "GROUP_COMPLETE gpu=$gpu " "$logs/e129_gpu$gpu.log"; then
            continue
        fi
        all_complete=0
        if ! tmux has-session -t "e129_gpu$gpu" 2>/dev/null; then
            echo "GPU $gpu worker stopped before completion; inspect $logs/e129_gpu$gpu.log" >&2
            exit 1
        fi
    done
    if (( all_complete )); then
        break
    fi
    sleep 60
done

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$root/scripts:$root/scripts/research:$root/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
"$worker" -u scripts/research/finalize_e129_h3_svdquant_attention.py
for case_id in p001 p011 p016 p020 p025 p042 p051 p068 p076 p093 p103 p116 \
               vbench_000 vbench_001 vbench_002 vbench_003 vbench_004; do
    echo "METRIC_START $case_id $(date --iso-8601=seconds)"
    "$worker" -u scripts/research/evaluate_e129_h3_svdquant_videos.py --case "$case_id"
    echo "METRIC_DONE $case_id $(date --iso-8601=seconds)"
done
"$worker" -u scripts/research/summarize_e129_h3_svdquant_videos.py
echo "E129_FINAL_COMPLETE $(date --iso-8601=seconds)"
