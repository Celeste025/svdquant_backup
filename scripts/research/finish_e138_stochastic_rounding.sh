#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/scripts:$PWD/scripts/research:$PWD/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
deadline=$((SECONDS+2400))
for case_id in race clap0; do
    while [[ ! -f results/research/E138/$case_id/nearest_video.json || ! -f results/research/E138/$case_id/activation_sr_video.json ]]; do
        (( SECONDS < deadline )) || { echo "E138 evaluation deadline reached for $case_id"; exit 2; }
        sleep 20
    done
    "$py" -u scripts/research/evaluate_e138_videos.py --case "$case_id"
done
