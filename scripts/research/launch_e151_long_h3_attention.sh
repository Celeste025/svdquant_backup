#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for frames in 124 362; do
    "$python_bin" -u scripts/research/capture_e151_long_h3_qkv.py --frames "$frames" >> results/logs/E151_long_h3_attention.log 2>&1
done
"$python_bin" -u scripts/research/bench_e151_long_h3_attention.py >> results/logs/E151_long_h3_attention.log 2>&1
