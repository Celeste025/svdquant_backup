#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
PY=/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python
LOG="$ROOT/results/logs/rcm-int4-svdquant-vbench51-gpu6.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=6 DEEPCOMPRESSOR_WAN_GATED=0 \
  PYTHONPATH="$ROOT/third_party/deepcompressor" TOKENIZERS_PARALLELISM=false \
  PATH="/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin:$PATH" \
  "$PY" scripts/run_rcm_int4_vbench51.py 2>&1 | tee -a "$LOG"
