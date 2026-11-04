#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
gpu="$1";pid="$2";py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
export CUDA_VISIBLE_DEVICES="$gpu" DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
case=$(printf 'p%03d' "$pid")
trap 'printf "FAILED gpu=%s prompt=%s line=%s\n" "$gpu" "$pid" "$LINENO" > "results/logs/E202_first_p${pid}.failed"' ERR
if [[ ! -f "results/research/E202/first/${case}/bf16_video.json" ]]; then
 "$py" -u scripts/research/run_e202_prob_reuse_whitebox.py --prompt-id "$pid" --policy first --phase decode > "results/logs/E202_p${pid}_first_decode_retry.log" 2>&1
fi
"$py" -u scripts/research/evaluate_e202_first_only.py --prompt-id "$pid" > "results/logs/E202_p${pid}_first_eval.log" 2>&1
: > "results/logs/E202_first_p${pid}.done"
