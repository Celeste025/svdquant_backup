#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
arm=${1:?usage: launch_e125_h3_clap_attention.sh ARM GPU_ID}
gpu=${2:?usage: launch_e125_h3_clap_attention.sh ARM GPU_ID}
case "$arm" in bf16|sage3_fixed|split_mixed) ;; *) echo "Invalid arm: $arm" >&2; exit 2 ;; esac
case "$gpu" in 5|6|7) ;; *) echo "GPU must be 5, 6, or 7" >&2; exit 2 ;; esac
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if (( free_mib < 55000 )); then
    echo "GPU $gpu has only $free_mib MiB free" >&2
    exit 2
fi
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
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python

echo "START gpu=$gpu arm=$arm $(date --iso-8601=seconds)"
"$worker" -u scripts/research/run_e125_h3_clap_attention.py --arm "$arm" --phase denoise
"$worker" -u scripts/research/run_e125_h3_clap_attention.py --arm "$arm" --phase decode
echo "COMPLETE gpu=$gpu arm=$arm $(date --iso-8601=seconds)"
