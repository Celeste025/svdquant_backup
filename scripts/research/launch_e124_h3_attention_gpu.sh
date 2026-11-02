#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
gpu=${1:?usage: launch_e124_h3_attention_gpu.sh GPU_ID_4_TO_7}
case "$gpu" in
    4) ids=(1 11 16) ;;
    5) ids=(20 25 42) ;;
    6) ids=(51 68 76) ;;
    7) ids=(93 103 116) ;;
    *) echo "GPU must be one of 4, 5, 6, 7" >&2; exit 2 ;;
esac
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

for id in "${ids[@]}"; do
    for arm in bf16 sage3_fixed split_mixed; do
        echo "START gpu=$gpu prompt=$id arm=$arm $(date --iso-8601=seconds)"
        "$worker" -u scripts/research/run_e124_h3_attention_prompts.py \
            --prompt-id "$id" --arm "$arm" --phase denoise
        "$worker" -u scripts/research/run_e124_h3_attention_prompts.py \
            --prompt-id "$id" --arm "$arm" --phase decode
        echo "DONE gpu=$gpu prompt=$id arm=$arm $(date --iso-8601=seconds)"
    done
done
echo "GROUP_COMPLETE gpu=$gpu $(date --iso-8601=seconds)"
