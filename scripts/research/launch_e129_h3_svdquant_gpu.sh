#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
gpu=${1:?usage: launch_e129_h3_svdquant_gpu.sh GPU_ID_0_TO_3 preflight|run|single [suite:case]}
mode=${2:?usage: launch_e129_h3_svdquant_gpu.sh GPU_ID_0_TO_3 preflight|run|single [suite:case]}
case "$gpu" in 0|1|2|3) ;; *) echo "GPU must be 0..3" >&2; exit 2 ;; esac
case "$mode" in preflight|run|single) ;; *) echo "mode must be preflight, run or single" >&2; exit 2 ;; esac
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
        suite=e124 case_id=p016
    elif [[ "$gpu" == 1 ]]; then
        suite=e126 case_id=vbench_000
    else
        echo "Preflight uses GPU 0 (E124) or 1 (E126)" >&2
        exit 2
    fi
    "$worker" -u scripts/research/run_e129_h3_svdquant_attention.py \
        --suite "$suite" --case "$case_id" --phase preflight
    exit
fi

if [[ "$mode" == single ]]; then
    task=${3:?single mode needs e124:p016, e126:vbench_000 or e126:vbench_003}
    case "$task" in e124:p016|e126:vbench_000|e126:vbench_003) ;;
        *) echo "single mode only permits the three selected cases" >&2; exit 2 ;;
    esac
    suite=${task%%:*}
    case_id=${task#*:}
    echo "START_SINGLE gpu=$gpu suite=$suite case=$case_id $(date --iso-8601=seconds)"
    "$worker" -u scripts/research/run_e129_h3_svdquant_attention.py \
        --suite "$suite" --case "$case_id" --phase denoise
    "$worker" -u scripts/research/run_e129_h3_svdquant_attention.py \
        --suite "$suite" --case "$case_id" --phase decode
    echo "SINGLE_COMPLETE gpu=$gpu suite=$suite case=$case_id $(date --iso-8601=seconds)"
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
    echo "START gpu=$gpu suite=$suite case=$case_id $(date --iso-8601=seconds)"
    "$worker" -u scripts/research/run_e129_h3_svdquant_attention.py \
        --suite "$suite" --case "$case_id" --phase denoise
    "$worker" -u scripts/research/run_e129_h3_svdquant_attention.py \
        --suite "$suite" --case "$case_id" --phase decode
    echo "DONE gpu=$gpu suite=$suite case=$case_id $(date --iso-8601=seconds)"
done
echo "GROUP_COMPLETE gpu=$gpu $(date --iso-8601=seconds)"
