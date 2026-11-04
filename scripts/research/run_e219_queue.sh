#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
gpu="$1"; shift
export CUDA_VISIBLE_DEVICES="$gpu" DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for job in "$@"; do
 policy="${job%:*}"; pid="${job#*:}"; log="results/logs/E219_${policy}_p${pid}.log"
 { date -Is; "$py" -u scripts/research/run_e219_four_video.py --prompt-id "$pid" --policy "$policy" --phase denoise; "$py" -u scripts/research/run_e219_four_video.py --prompt-id "$pid" --policy "$policy" --phase decode; "$py" -u scripts/research/evaluate_e219.py --prompt-id "$pid" --policy "$policy"; date -Is; } >"$log" 2>&1 || { printf 'FAILED %s gpu %s\n' "$job" "$gpu" >> "$log"; continue; }
done
