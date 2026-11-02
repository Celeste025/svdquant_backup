#!/usr/bin/env bash
set -euo pipefail

cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=0
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
log_file=results/logs/E144_text_detail_draft.log

"$python_bin" scripts/research/run_e144_text_detail_draft.py --phase denoise >"$log_file" 2>&1
"$python_bin" scripts/research/run_e144_text_detail_draft.py --phase decode >>"$log_file" 2>&1
