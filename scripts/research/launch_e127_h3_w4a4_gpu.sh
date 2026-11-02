#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
gpu=${1:?usage: launch_e127_h3_w4a4_gpu.sh GPU_ID_0_TO_3 preflight|run}
mode=${2:?usage: launch_e127_h3_w4a4_gpu.sh GPU_ID_0_TO_3 preflight|run fake|native}
execution=${3:-fake}
case "$gpu" in 0|1|2|3) ;; *) echo "GPU must be 0..3" >&2; exit 2 ;; esac
case "$mode" in preflight|run) ;; *) echo "mode must be preflight or run" >&2; exit 2 ;; esac
case "$execution" in fake|native) ;; *) echo "execution must be fake or native" >&2; exit 2 ;; esac
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
export PYTHONPATH="$root/scripts:$root/scripts/research:$root/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python

if [[ "$mode" == preflight ]]; then
    if [[ "$gpu" == 0 ]]; then
        suite=e124 case_id=p001 arm=w4a4_bf16_attention
    elif [[ "$gpu" == 1 ]]; then
        suite=e126 case_id=vbench_000 arm=w4a4_sage3_fixed
    else
        echo "Preflight uses GPU 0 (E124) or 1 (E126)" >&2
        exit 2
    fi
    "$worker" -u scripts/research/run_e127_h3_w4a4_attention.py \
        --suite "$suite" --case "$case_id" --arm "$arm" --phase preflight --execution "$execution"
    exit
fi

case "$gpu" in
    0) tasks=(e126:vbench_000 e126:vbench_004 e124:p001) ;;
    1) tasks=(e126:vbench_001 e124:p011 e124:p020 e124:p025) ;;
    2) tasks=(e126:vbench_002 e124:p016 e124:p042 e124:p051 e124:p068) ;;
    3) tasks=(e126:vbench_003 e124:p076 e124:p093 e124:p103 e124:p116) ;;
esac
for task in "${tasks[@]}"; do
    suite=${task%%:*}
    case_id=${task#*:}
    for arm in w4a4_bf16_attention w4a4_sage3_fixed; do
        echo "START gpu=$gpu suite=$suite case=$case_id arm=$arm $(date --iso-8601=seconds)"
        "$worker" -u scripts/research/run_e127_h3_w4a4_attention.py \
            --suite "$suite" --case "$case_id" --arm "$arm" --phase denoise --execution "$execution"
        "$worker" -u scripts/research/run_e127_h3_w4a4_attention.py \
            --suite "$suite" --case "$case_id" --arm "$arm" --phase decode --execution "$execution"
        echo "DONE gpu=$gpu suite=$suite case=$case_id arm=$arm $(date --iso-8601=seconds)"
    done
done
echo "GROUP_COMPLETE gpu=$gpu $(date --iso-8601=seconds)"
