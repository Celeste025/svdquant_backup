#!/usr/bin/env bash
# Generate two paired, free-running MiniMax-H3 samples for E086 visual review.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python"
BASE="$ROOT/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
FT="/data1/models/svdquant-wjq/results/E086/checkpoint/quant_state.pt"
OUT="/data1/models/svdquant-wjq/results/E086/videos"
LOG="/data1/models/svdquant-wjq/results/logs/e086_video_compare_$(date +%Y%m%d_%H%M%S).log"
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

run_pair() {
  local prompt_id="$1"
  local gpu="$2"
  local out="$OUT/p${prompt_id}"
  mkdir -p "$out"
  echo "E086 visual pair p${prompt_id} GPU${gpu} start=$(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/infer_minimax_h3_svdquant_standard.py" \
    --case svdquant --state "$BASE" --output-dir "$out" --prompt-id "$prompt_id"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/infer_minimax_h3_svdquant_standard.py" \
    --case svdquant_ft --state "$FT" --output-dir "$out" --prompt-id "$prompt_id"
  echo "E086 visual pair p${prompt_id} complete=$(date -Is)"
}

run_pair 2 1 & p2_pid=$!
run_pair 16 3 & p16_pid=$!
wait "$p2_pid"
wait "$p16_pid"
echo "E086 visual comparison complete $(date -Is)"
