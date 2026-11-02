#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
gpu=${1:?usage: launch_e130_raw_k_dit.sh GPU_ID SAGE3_SOURCE_DIR}
sage_source=${2:?usage: launch_e130_raw_k_dit.sh GPU_ID SAGE3_SOURCE_DIR}
worker=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
[[ $gpu =~ ^[0-7]$ ]] || { echo "invalid GPU: $gpu" >&2; exit 2; }
[[ -f $sage_source/fp4attn_cuda.cpython-312-x86_64-linux-gnu.so &&
   -f $sage_source/fp4quant_cuda.cpython-312-x86_64-linux-gnu.so ]] || {
    echo "Sage3 extensions missing: $sage_source" >&2; exit 2;
}
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if (( free_mib < 55000 )); then
    echo "GPU $gpu has only $free_mib MiB free" >&2
    exit 2
fi

cd "$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export E130_SAGE3_SOURCE="$sage_source"
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH="$root/scripts:$root/scripts/research:$root/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"

"$worker" -u scripts/research/e122_profile_h3_attention_linear.py \
    --resident --steps 0 10 \
    --arms bf16_sdpa sage3_fixed sage3_api_qpack_delta_tiled sage3_api_raw_k_delta \
    --output results/research/E130/h3_resident_raw_k_steps0_10_gpu${gpu}.json
