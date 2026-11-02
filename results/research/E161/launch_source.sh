#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES="$1" MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3 MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036 DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
shift
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for case_id in "$@"; do
 "$python_bin" scripts/research/run_e161_periodic_route_whitebox.py --prompt-id "$case_id" --mode reuse
 "$python_bin" scripts/research/run_e161_video_decode.py --pid "$case_id" --mode reuse
 "$python_bin" scripts/research/evaluate_e161_videos.py --pid "$case_id"
done
