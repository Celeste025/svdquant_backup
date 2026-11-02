#!/usr/bin/env bash
# E101: teacher-forced complete-DiT error for matched dynamic/fixed activations.
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
state=$root/results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt
media=/data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases
report=$root/results/research/E101
case_id=${1:?usage: launch_h3_fixed1_three_cases_error.sh race|clap0|clap1 GPU}
gpu=${2:?usage: launch_h3_fixed1_three_cases_error.sh race|clap0|clap1 GPU}

case "$case_id" in
    race) prompt_id=16; seed=20026; extra=() ;;
    clap0) prompt_id=161; seed=38001610; extra=(--prompt-text 'A person is clapping' --seed "$seed") ;;
    clap1) prompt_id=161; seed=38001611; extra=(--prompt-text 'A person is clapping' --seed "$seed") ;;
    *) echo "Unknown case: $case_id" >&2; exit 2 ;;
esac
[[ $gpu =~ ^[0-7]$ ]] || { echo "Invalid GPU $gpu" >&2; exit 2; }
[[ -f $state ]] || { echo "Missing state: $state" >&2; exit 2; }
[[ -f $media/$case_id/svdquant_fixed1/svdquant_fixed1.json ]] || {
    echo "Fixed-scale generation for $case_id is not complete" >&2; exit 2;
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

mkdir -p "$report/$case_id" "$media/$case_id"
calls=$media/$case_id/bf16_teacher_calls.pt
for variant in svdquant svdquant_fixed1; do
    target=$report/$case_id/$variant.json
    if [[ -f $target ]]; then
        echo "Skipping completed $target"
        continue
    fi
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    if (( free_mib < 45000 )); then
        echo "GPU $gpu has only $free_mib MiB free; refusing to crowd co-tenant" >&2
        exit 2
    fi
    call_arg=(--calls-file "$calls")
    if [[ ! -f $calls ]]; then call_arg=(--save-calls "$calls"); fi
    echo "START paired error $case_id $variant on GPU $gpu, free $free_mib MiB, $(date -Is)"
    "$py" -u scripts/eval_minimax_h3_paired_step_error.py \
        --variant "$variant" --state "$state" --prompt-id "$prompt_id" "${extra[@]}" \
        --height 576 --width 1024 --frames 124 --steps 20 \
        "${call_arg[@]}" --output "$target"
    echo "DONE paired error $case_id $variant $(date -Is)"
done
