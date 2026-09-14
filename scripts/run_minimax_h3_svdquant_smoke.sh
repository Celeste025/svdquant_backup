#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wjq/workspace/svdquant-exp
PY=/home/wjq/.venvs/minimax-h3-svdquant/bin/python
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH="$ROOT/scripts:$ROOT/third_party/deepcompressor:${PYTHONPATH:-}"

cd /home/wjq/workspace/DiffSynth-Studio

"$PY" "$ROOT/scripts/collect_minimax_h3_calib.py"
"$PY" "$ROOT/scripts/ptq_minimax_h3_svdquant_smoke.py" --max-blocks 1 \
  --output-dir "$ROOT/results/checkpoints/minimax_h3_svdquant_smoke_block0"
"$PY" "$ROOT/scripts/ptq_minimax_h3_svdquant_smoke.py" --max-blocks 50 \
  --output-dir "$ROOT/results/checkpoints/minimax_h3_svdquant_smoke"
for case_name in bf16 w4a4 svdquant; do
  "$PY" "$ROOT/scripts/infer_minimax_h3_svdquant_smoke.py" --case "$case_name"
done
