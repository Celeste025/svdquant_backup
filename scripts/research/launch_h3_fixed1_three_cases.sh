#!/usr/bin/env bash
# E101: three fixed-scale H3 cases, one case per GPU/tmux worker.
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
state=$root/results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt
output=/data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases
case_id=${1:?usage: launch_h3_fixed1_three_cases.sh race|clap0|clap1 GPU}
gpu=${2:?usage: launch_h3_fixed1_three_cases.sh race|clap0|clap1 GPU}

case "$case_id" in
    race) prompt_id=16; seed=20026; variants=(svdquant_fixed1); extra=() ;;
    clap0) prompt_id=161; seed=38001610; variants=(svdquant svdquant_fixed1); extra=(--prompt-text 'A person is clapping' --seed "$seed") ;;
    clap1) prompt_id=161; seed=38001611; variants=(svdquant svdquant_fixed1); extra=(--prompt-text 'A person is clapping' --seed "$seed") ;;
    *) echo "Unknown case: $case_id" >&2; exit 2 ;;
esac
[[ $gpu =~ ^[0-7]$ ]] || { echo "Invalid GPU $gpu" >&2; exit 2; }
[[ -f $state ]] || { echo "Missing state: $state" >&2; exit 2; }

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

for variant in "${variants[@]}"; do
    case_dir=$output/$case_id/$variant
    target=$case_dir/${variant}_id${prompt_id}_seed${seed}_576x1024_124f_20steps.mp4
    if [[ -e $target || -e $case_dir/$variant.json ]]; then
        echo "Existing output at $case_dir; refusing to overwrite" >&2
        exit 2
    fi
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    if (( free_mib < 45000 )); then
        echo "GPU $gpu has only $free_mib MiB free; refusing to crowd co-tenant" >&2
        exit 2
    fi
    mkdir -p "$case_dir"
    echo "START $case_id $variant on GPU $gpu, free $free_mib MiB, $(date -Is)"
    "$py" -u scripts/infer_minimax_h3_svdquant_standard.py \
        --case "$variant" --state "$state" --output-dir "$case_dir" \
        --prompt-id "$prompt_id" "${extra[@]}" --height 576 --width 1024 --frames 124 --steps 20
    "$py" - "$target" <<'PY'
import av
import sys
with av.open(sys.argv[1]) as container:
    stream = container.streams.video[0]
    frames = sum(1 for _ in container.decode(stream))
    assert (stream.width, stream.height, frames) == (1024, 576, 124)
    assert len(container.streams.audio) == 1
    print(f"validated {stream.width}x{stream.height} {frames} frames and audio")
PY
    echo "DONE $case_id $variant $(date -Is)"
done
