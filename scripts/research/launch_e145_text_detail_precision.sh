#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU required}
arm=${2:?arm required}
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES="$gpu"
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
log_file="results/logs/E145_p025_${arm}.log"

"$python_bin" scripts/research/run_e145_text_detail_pure_precision.py --arm "$arm" --phase denoise >"$log_file" 2>&1
"$python_bin" scripts/research/run_e145_text_detail_pure_precision.py --arm "$arm" --phase decode >>"$log_file" 2>&1
