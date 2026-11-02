#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/scripts:$PWD/scripts/research:$PWD/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
deadline=$((SECONDS+2400))
for case_id in race clap0; do
    for arm in s70_int8_30 s70_fp4_20_int8_10; do
        while [[ ! -f results/research/E139/$case_id/${arm}_video.json ]]; do
            (( SECONDS < deadline )) || { echo "E139 finalizer deadline"; exit 2; }
            sleep 15
        done
    done
done
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/finish_e139_draftattention2.py
