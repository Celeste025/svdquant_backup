#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
gpu="$1"
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
run() { CUDA_VISIBLE_DEVICES="$gpu" /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/run_e201_prob_reuse_video.py --rule "$1" --skip "$2" > "results/logs/E201_${1}_skip${2}.log" 2>&1; }
if [[ "$gpu" == 1 ]]; then run reuse_qk 0.0; run reuse_qk 0.75; else run reuse_prob 0.0; run reuse_prob 0.75; fi
: > "results/logs/E201_gpu${gpu}.done"
