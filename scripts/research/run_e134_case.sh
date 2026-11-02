#!/usr/bin/env bash
set -Eeuo pipefail
gpu=${1:?GPU}
case_id=${2:?case}
arm=${3:?arm}
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES="$gpu"
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH="$PWD/scripts:$PWD/scripts/research:$PWD/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for phase in denoise decode; do
  echo "E134 $case_id/$arm START $phase $(date --iso-8601=seconds)"
  "$worker" -u scripts/research/run_e134_svdquant_bf16_window.py \
    --case "$case_id" --arm "$arm" --phase "$phase"
  echo "E134 $case_id/$arm DONE $phase $(date --iso-8601=seconds)"
done
