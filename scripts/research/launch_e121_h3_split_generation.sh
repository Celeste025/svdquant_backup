#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
case_id=${1:-race}
gpu=${2:?usage: launch_e121_h3_split_generation.sh race GPU preflight|all|warmbf16}
mode=${3:-all}
[[ $case_id == race || $case_id == clap0 ]] || { echo "invalid case" >&2; exit 2; }
[[ $gpu =~ ^[0-7]$ ]] || { echo "invalid GPU" >&2; exit 2; }
[[ $mode == preflight || $mode == all || $mode == warmbf16 ]] || { echo "invalid mode" >&2; exit 2; }
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if (( free_mib < 55000 )); then
    echo "GPU $gpu has only $free_mib MiB free; refusing H3 run" >&2
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

script=scripts/research/run_e121_h3_split_generation.py
preflight=results/research/E121/$case_id/attention_preflight_fixed.json
if [[ ! -f $preflight ]]; then
    echo "E121 $case_id preflight start $(date -Is) GPU $gpu"
    "$python_bin" -u "$script" --phase preflight --case "$case_id"
fi
[[ $mode == preflight ]] && exit 0

if [[ $mode == warmbf16 ]]; then
    arms=(bf16_warm)
else
    arms=(bf16 sage3_fixed split_mixed)
fi
for arm in "${arms[@]}"; do
    denoise=results/research/E121/$case_id/${arm}_denoise.json
    video=results/research/E121/$case_id/${arm}_video.json
    if [[ -f $video ]]; then
        echo "Already completed $case_id/$arm; preserving media"
        continue
    fi
    if [[ -f $denoise ]]; then
        status=$("$python_bin" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$denoise")
        if [[ $status != denoise_complete ]]; then
            echo "Existing $denoise status=$status; inspect before retry" >&2
            exit 2
        fi
    else
        echo "E121 $case_id/$arm denoise start $(date -Is) GPU $gpu"
        "$python_bin" -u "$script" --phase denoise --case "$case_id" --arm "$arm"
    fi
    echo "E121 $case_id/$arm decode start $(date -Is) GPU $gpu"
    "$python_bin" -u "$script" --phase decode --case "$case_id" --arm "$arm"
done
echo "E121 $case_id complete $(date -Is)"
