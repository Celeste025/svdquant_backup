#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
case_id=${1:?usage: launch_e105_h3_output_interventions.sh race|clap0 GPU [all|arm]}
gpu=${2:?usage: launch_e105_h3_output_interventions.sh race|clap0 GPU [all|arm]}
selected_arm=${3:-all}
[[ $case_id == race || $case_id == clap0 ]] || { echo "invalid case: $case_id" >&2; exit 2; }
[[ $gpu =~ ^[0-7]$ ]] || { echo "invalid GPU: $gpu" >&2; exit 2; }
[[ $selected_arm == all || $selected_arm == tail_restore || $selected_arm == rademacher_flat || $selected_arm == gaussian ]] || { echo "invalid arm: $selected_arm" >&2; exit 2; }
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

echo "E105 $case_id start $(date -Is), GPU $gpu, free $free_mib MiB"
"$py" -u scripts/research/run_e105_h3_output_interventions.py --phase preflight --case "$case_id"
for arm in tail_restore rademacher_flat gaussian; do
    [[ $selected_arm == all || $selected_arm == $arm ]] || continue
    denoise_report=results/research/E105/$case_id/${arm}_denoise.json
    video_report=results/research/E105/$case_id/${arm}_video.json
    if [[ -f $video_report ]]; then
        echo "Already completed $case_id/$arm; preserving media"
        continue
    fi
    if [[ -f $denoise_report ]]; then
        status=$("$py" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$denoise_report")
        if [[ $status != denoise_complete ]]; then
            echo "Existing $denoise_report has status $status; inspect it before retry" >&2
            exit 2
        fi
    else
        echo "DENOISE $case_id/$arm $(date -Is)"
        "$py" -u scripts/research/run_e105_h3_output_interventions.py \
            --phase denoise --case "$case_id" --arm "$arm"
    fi
    echo "DECODE $case_id/$arm $(date -Is)"
    "$py" -u scripts/research/run_e105_h3_output_interventions.py \
        --phase decode --case "$case_id" --arm "$arm"
done
echo "E105 $case_id complete $(date -Is)"
