#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
gpu=${1:?GPU} sample=${2:?sample}
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
(( free > 60000 )) || exit 2
export CUDA_VISIBLE_DEVICES="$gpu" DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH="$PWD/scripts:$PWD/scripts/research:$PWD/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
"$py" -u scripts/research/capture_e142_attention.py --sample "$sample"
"$py" -u scripts/research/bench_e142_attention_tradeoff.py --sample "$sample"
