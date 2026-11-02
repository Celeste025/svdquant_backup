#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
cd "$root"
report="$root/results/research/E129"
logs="$root/results/logs"
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python

for pair in '0:vbench_000' '3:vbench_003'; do
    gpu=${pair%%:*}
    case_id=${pair#*:}
    pairing="$report/$case_id/initial_pairing.json"
    denoise="$report/$case_id/svdquant_bf16_attention_denoise.json"
    while [[ ! -f "$pairing" ]]; do
        if [[ -f "$denoise" ]] && rg -q '"status": "failed"' "$denoise"; then
            echo "Denoising failed: $case_id; see $denoise" >&2
            exit 1
        fi
        if ! ps -eo args | rg -q "[r]un_e129_h3_svdquant_attention.py --suite e126 --case $case_id --phase denoise"; then
            echo "Denoising stopped before pairing completed: $case_id" >&2
            exit 1
        fi
        sleep 60
    done
    while ps -eo args | rg -q "[r]un_e129_h3_svdquant_attention.py --suite e126 --case $case_id --phase denoise"; do
        sleep 5
    done
    tmux kill-session -t "e129_gpu$gpu"
    tmux new-session -d -s "e129_decode_g$gpu" \
        "bash scripts/research/launch_e129_h3_svdquant_gpu.sh $gpu single e126:$case_id > results/logs/e129_decode_gpu$gpu.log 2>&1"
    echo "DECODE_LAUNCHED gpu=$gpu case=$case_id $(date --iso-8601=seconds)"
done

for pair in '0:vbench_000' '1:p016' '3:vbench_003'; do
    gpu=${pair%%:*}
    case_id=${pair#*:}
    media="$report/$case_id/svdquant_bf16_attention_video.json"
    while [[ ! -f "$media" ]]; do
        session="e129_race_g1"
        if [[ "$gpu" != 1 ]]; then
            session="e129_decode_g$gpu"
        fi
        if ! tmux has-session -t "$session" 2>/dev/null; then
            echo "Worker $session stopped before video completion: $case_id" >&2
            exit 1
        fi
        sleep 60
    done
done

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$root/scripts:$root/scripts/research:$root/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
"$worker" -u scripts/research/finalize_e129_h3_svdquant_attention.py \
    --cases p016 vbench_000 vbench_003
for case_id in p016 vbench_000 vbench_003; do
    echo "METRIC_START $case_id $(date --iso-8601=seconds)"
    "$worker" -u scripts/research/evaluate_e129_h3_svdquant_videos.py --case "$case_id"
    echo "METRIC_DONE $case_id $(date --iso-8601=seconds)"
done
"$worker" -u scripts/research/summarize_e129_h3_svdquant_videos.py
echo "E129_SELECTED_COMPLETE $(date --iso-8601=seconds)"
