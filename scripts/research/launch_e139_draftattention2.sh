#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
gpu=${1:?GPU} case_id=${2:?case} arm=${3:?arm}
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
(( free > 60000 )) || { echo "GPU $gpu insufficient free memory"; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu" DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH="$PWD/third_party/anemoi-code:$PWD/scripts:$PWD/scripts/research:$PWD/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
worker=${E139_WORKER:-scripts/research/run_e139_draftattention2.py}
if [[ "$case_id" == smoke ]]; then
    "$py" -u "$worker" --phase smoke
    exit
fi
"$py" -u "$worker" --phase denoise --case "$case_id" --arm "$arm"
"$py" -u "$worker" --phase decode --case "$case_id" --arm "$arm"
