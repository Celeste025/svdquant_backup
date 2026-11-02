#!/usr/bin/env bash
# E100: paired teacher-forced DiT error on the BF16 prompt-1 trajectory.
set -euo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
gpu=${1:-5}
media=/data1/models/svdquant-wjq/research/20261015/E100_h3_svd_fixed1_p1
report=$root/results/research/E100
state=$root/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt
calls=$media/bf16_teacher_calls.pt

if [[ ! $gpu =~ ^[567]$ ]]; then
    echo 'Shared-GPU run is restricted to GPU 5, 6, or 7.' >&2
    exit 2
fi
for case in bf16 svdquant_fixed1; do
    if [[ ! -f $media/$case/$case.json ]]; then
        echo "Missing completed $case generation; not starting error evaluation." >&2
        exit 2
    fi
done

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
mkdir -p "$report"

for variant in svdquant svdquant_fixed1; do
    output=$report/${variant}_p1_480x832_124f_20steps.json
    if [[ -e $output ]]; then
        echo "Existing report found at $output; refusing to overwrite." >&2
        exit 2
    fi
    check_headroom
    echo "START paired error $variant $(date -Is)"
    if [[ $variant == svdquant ]]; then
        if [[ -e $calls ]]; then
            echo "Existing calls file found at $calls; refusing to overwrite." >&2
            exit 2
        fi
        "$py" -u scripts/eval_minimax_h3_paired_step_error.py \
            --variant "$variant" --state "$state" --prompt-id 1 \
            --height 480 --width 832 --frames 124 --steps 20 \
            --save-calls "$calls" --output "$output"
    else
        "$py" -u scripts/eval_minimax_h3_paired_step_error.py \
            --variant "$variant" --state "$state" --prompt-id 1 \
            --height 480 --width 832 --frames 124 --steps 20 \
            --calls-file "$calls" --output "$output"
    fi
    echo "DONE paired error $variant $(date -Is)"
done
