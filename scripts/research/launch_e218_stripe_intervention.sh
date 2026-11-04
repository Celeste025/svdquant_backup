#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export CUDA_VISIBLE_DEVICES="$1"
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/run_e218_stripe_intervention.py --prompt "$2" > "results/logs/E218_p${2}.log" 2>&1
