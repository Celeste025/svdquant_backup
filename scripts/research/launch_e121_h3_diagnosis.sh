#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
gpu=${1:?usage: launch_e121_h3_diagnosis.sh GPU capture|compare}
phase=${2:?usage: launch_e121_h3_diagnosis.sh GPU capture|compare}
[[ $gpu =~ ^[0-7]$ ]] || exit 2
[[ $phase == capture || $phase == compare ]] || exit 2
cd "$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH="$root/scripts:$root/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u \
    scripts/research/diagnose_e121_long_attention.py --phase "$phase" "${@:3}"
