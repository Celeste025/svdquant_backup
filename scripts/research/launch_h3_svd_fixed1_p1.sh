#!/usr/bin/env bash
# E100: paired BF16 and fixed-second-level-scale SVDQuant H3 generation.
set -euo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
gpu=${1:-5}
output=/data1/models/svdquant-wjq/research/20261015/E100_h3_svd_fixed1_p1
state=$root/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt

if [[ ! $gpu =~ ^[567]$ ]]; then
    echo 'Shared-GPU run is restricted to GPU 5, 6, or 7.' >&2
    exit 2
fi
if [[ ! -f $state ]]; then
    echo "Missing SVDQuant state: $state" >&2
    exit 2
fi

check_headroom() {
    local free_mib
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    if (( free_mib < 45000 )); then
        echo "GPU $gpu has only $free_mib MiB free; refusing to crowd the co-tenant." >&2
        exit 2
    fi
    echo "GPU $gpu free: $free_mib MiB"
}

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

for case in bf16 svdquant_fixed1; do
    case_dir=$output/$case
    target=$case_dir/${case}_id1_seed22529_480x832_124f_20steps.mp4
    if [[ -e $target || -e $case_dir/$case.json ]]; then
        echo "Existing output found at $case_dir; refusing to overwrite." >&2
        exit 2
    fi
    check_headroom
    mkdir -p "$case_dir"
    echo "START $case $(date -Is)"
    "$py" -u scripts/infer_minimax_h3_svdquant_standard.py \
        --case "$case" --state "$state" --output-dir "$case_dir" \
        --prompt-id 1 --height 480 --width 832 --frames 124 --steps 20
    "$py" - "$target" <<'PY'
import av
import sys

with av.open(sys.argv[1]) as container:
    stream = container.streams.video[0]
    frames = sum(1 for _ in container.decode(stream))
    print(f"validated video: {stream.width}x{stream.height}, {frames} frames")
    assert (stream.width, stream.height, frames) == (832, 480, 124)
PY
    echo "DONE $case $(date -Is)"
done
