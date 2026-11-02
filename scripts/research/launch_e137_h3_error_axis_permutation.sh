#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
case_id=${1:?usage: launch_e137_h3_error_axis_permutation.sh race|clap0 GPU [all|spatial_shared|temporal_frame]}
gpu=${2:?usage: launch_e137_h3_error_axis_permutation.sh race|clap0 GPU [all|spatial_shared|temporal_frame]}
selected_arm=${3:-all}
[[ $case_id == race || $case_id == clap0 ]] || { echo "invalid case: $case_id" >&2; exit 2; }
[[ $gpu =~ ^[0-7]$ ]] || { echo "invalid GPU: $gpu" >&2; exit 2; }
[[ $selected_arm == all || $selected_arm == spatial_shared || $selected_arm == temporal_frame ]] || {
    echo "invalid arm: $selected_arm" >&2; exit 2;
}
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if (( free_mib < 45000 )); then
    echo "GPU $gpu has only $free_mib MiB free; refusing paired-model run" >&2
    exit 2
fi

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
export E137_ATTEMPT=${E137_ATTEMPT:-attempt2}

echo "E137 $case_id start $(date -Is), GPU $gpu, free $free_mib MiB"
"$py" -u scripts/research/run_e137_h3_error_axis_permutation.py --phase preflight --case "$case_id"
for arm in spatial_shared temporal_frame; do
    [[ $selected_arm == all || $selected_arm == $arm ]] || continue
    denoise_report="results/research/E137/${E137_ATTEMPT}/$case_id/${arm}_denoise.json"
    video_report="results/research/E137/${E137_ATTEMPT}/$case_id/${arm}_video.json"
    if [[ -f $video_report ]]; then
        echo "Already completed $case_id/$arm; preserving media"
        continue
    fi
    if [[ -f $denoise_report ]]; then
        status=$("$py" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$denoise_report")
        [[ $status == denoise_complete ]] || { echo "Existing $denoise_report has status $status" >&2; exit 2; }
    else
        "$py" -u scripts/research/run_e137_h3_error_axis_permutation.py --phase denoise --case "$case_id" --arm "$arm"
    fi
    "$py" -u scripts/research/run_e137_h3_error_axis_permutation.py --phase decode --case "$case_id" --arm "$arm"
done
echo "E137 $case_id complete $(date -Is)"
