#!/usr/bin/env bash
# E099: one 480x832/124-frame/20-step H3 rank-64 SVDQuant scale audit.
set -euo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
gpu=${1:-6}
output=/data1/models/svdquant-wjq/research/20261015/E099_h3_svd_scale_p1
state=$root/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt

if [[ ! $gpu =~ ^[567]$ ]]; then
    echo 'This shared-GPU run is restricted to GPU 5, 6, or 7.' >&2
    exit 2
fi
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if (( free_mib < 45000 )); then
    echo "GPU $gpu has only $free_mib MiB free; refusing to crowd the co-tenant." >&2
    exit 2
fi
mkdir -p "$output"
echo "E099 start $(date -Is), physical GPU $gpu, free ${free_mib} MiB"
cd "$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH="$root/scripts:$root/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"
"$py" scripts/research/run_h3_svd_nvfp4_scale_audit.py \
    --state "$state" --output-dir "$output" --prompt-id 1
echo "E099 complete $(date -Is)"
