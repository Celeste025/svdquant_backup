#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
mode=${1:?usage: launch_e126_h3_vbench5_gpu.sh preflight|compare|run GPU_ID [CASE]}
gpu=${2:?usage: launch_e126_h3_vbench5_gpu.sh preflight|compare|run GPU_ID [CASE]}
case "$mode" in preflight|compare|run) ;; *) echo "Invalid mode: $mode" >&2; exit 2 ;; esac
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

if [[ "$mode" == preflight ]]; then
    "$worker" -u scripts/research/run_e126_h3_vbench5_attention.py \
        --case vbench_000 --phase preflight
    exit
fi
if [[ "$mode" == compare ]]; then
    case_id=${3:?compare mode requires a case ID}
    "$worker" -u scripts/research/run_e126_h3_vbench5_attention.py \
        --case "$case_id" --phase compare_first
    exit
fi

case "$gpu" in
    5) tasks=(vbench_000:bf16 vbench_001:sage3_fixed vbench_002:split_mixed
              vbench_003:bf16 vbench_004:sage3_fixed) ;;
    6) tasks=(vbench_000:sage3_fixed vbench_001:split_mixed vbench_002:bf16
              vbench_003:sage3_fixed vbench_004:split_mixed) ;;
    7) tasks=(vbench_000:split_mixed vbench_001:bf16 vbench_002:sage3_fixed
              vbench_003:split_mixed vbench_004:bf16) ;;
esac
for task in "${tasks[@]}"; do
    case_id=${task%%:*}
    arm=${task#*:}
    echo "START gpu=$gpu case=$case_id arm=$arm $(date --iso-8601=seconds)"
    "$worker" -u scripts/research/run_e126_h3_vbench5_attention.py \
        --case "$case_id" --arm "$arm" --phase denoise
    "$worker" -u scripts/research/run_e126_h3_vbench5_attention.py \
        --case "$case_id" --arm "$arm" --phase decode
    echo "DONE gpu=$gpu case=$case_id arm=$arm $(date --iso-8601=seconds)"
done
echo "GROUP_COMPLETE gpu=$gpu $(date --iso-8601=seconds)"
