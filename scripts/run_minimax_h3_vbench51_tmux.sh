#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
PY=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
LOG="$ROOT/results/logs/minimax-h3-vbench51-gpu7.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=7 DIFFSYNTH_SKIP_DOWNLOAD=True \
  PYTHONPATH="$ROOT/scripts:/home/wjq/workspace/DiffSynth-Studio" TOKENIZERS_PARALLELISM=false \
  "$PY" scripts/run_minimax_h3_vbench51.py --gpu 7 2>&1 | tee -a "$LOG"
