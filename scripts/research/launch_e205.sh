#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098 TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=2
export PATH="$CUDA_HOME/bin:/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
for d in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do export CPATH="$CPATH:$d"; done
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/validate_e205_optimized_capture.py
