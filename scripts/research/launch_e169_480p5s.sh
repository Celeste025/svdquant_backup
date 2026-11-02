#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
mkdir -p results/research/E169 results/logs
"$python_bin" -u scripts/research/capture_e169_h3_480p5s_qkv.py --frames 124
"$python_bin" -u scripts/research/profile_e169_480p_paths.py \
  --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 \
  --warmup 5 --repeats 12 \
  --output results/research/E169/q128_skip0_fp90_int10_124f.json
"$python_bin" -u scripts/research/validate_e169_480p_paths.py \
  --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 \
  --warmup 1 --repeats 2 --kernel-profile \
  --output results/research/E169/q128_skip0_fp90_int10_124f_kernels.json
