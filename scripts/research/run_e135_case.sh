#!/usr/bin/env bash
set -Eeuo pipefail
gpu=${1:?GPU}
case_id=${2:?case}
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
  echo "E135 $case_id START $phase $(date --iso-8601=seconds)"
  "$worker" -u scripts/research/run_e135_h3_bf16_w4a4_window.py \
    --case "$case_id" --phase "$phase"
  echo "E135 $case_id DONE $phase $(date --iso-8601=seconds)"
done
