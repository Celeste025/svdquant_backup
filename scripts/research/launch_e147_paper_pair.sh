#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export DIFFSYNTH_SKIP_DOWNLOAD=True
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
"$python_bin" scripts/research/run_e147_paper_pair.py --arm s743_fp4193_int864 --phase denoise > results/logs/E147_paper_pair.log 2>&1
"$python_bin" scripts/research/run_e147_paper_pair.py --arm s743_fp4193_int864 --phase decode >> results/logs/E147_paper_pair.log 2>&1
