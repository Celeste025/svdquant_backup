#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
GPU="$1"
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
PY=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
mkdir -p results/logs results/research/E198
run() {
  local rule="$1" skip="$2" tag
  if [[ "$rule" == bf16 ]]; then tag=bf16; else tag="${rule}_skip${skip}"; fi
  printf 'START %s %s\n' "$tag" "$(date -Is)" >> "results/logs/E198_gpu${GPU}_queue.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u scripts/research/run_e198_oracle_video.py --rule "$rule" --skip "0.$([[ "$skip" == 75 ]] && printf 75 || printf 0)" > "results/logs/E198_${tag}.log" 2>&1
  printf 'DONE %s %s\n' "$tag" "$(date -Is)" >> "results/logs/E198_gpu${GPU}_queue.log"
}
case "$GPU" in
  1) run bf16 0; run draftmap 75; run draftmap 0; run qk_abs_mean 75; run qk_abs_mean 0 ;;
  2) run prob_mean 75; run prob_mean 0; run prob_rowmax_sum 75; run prob_rowmax_sum 0; run random 75 ;;
  3) run prob_top20_sum 75; run prob_top20_sum 0; run qk_abs_top20_sum 75; run qk_abs_top20_sum 0; run random 0 ;;
  *) exit 2 ;;
esac
: > "results/logs/E198_gpu${GPU}.done"
