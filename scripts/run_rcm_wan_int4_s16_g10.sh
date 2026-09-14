#!/usr/bin/env bash
# rCM-Wan 4-step, 77-frame INT4 SVDQuant control experiment.
set -euo pipefail

ROOT=/home/wjq/workspace/svdquant-exp
DIFFUSION="$ROOT/third_party/deepcompressor/examples/diffusion"
DATA=/data1/models/svdquant-wjq
# This is the historical DeepCompressor/Wan PTQ runtime (Diffusers 0.33),
# not the newer recovered H3 runtime (Diffusers 0.40).
PY=/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python
CACHE="$DATA/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16"
MODEL="$DATA/models/Wan2.1-T2V-1.3B-Diffusers"
RCM_TRANSFORMER="$DATA/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer"
CKPT="$DATA/ckpts/rcm-wan2.1-1.3b-int4-s16-g10"
RUN="$DATA/runs/rcm-wan2.1-1.3b-int4-s16-g10"
LOG="$ROOT/results/logs/rcm-wan-int4-s16-g10-ptq.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export DEEPCOMPRESSOR_TRANSFORMER_ONLY=1
# The historical rCM cache was collected for ungated SVDQuant.  Do not enable
# the later experimental Wan gated OutputsError wrapper for this control run.
export DEEPCOMPRESSOR_WAN_GATED=0
export PYTHONPATH="$ROOT/third_party/deepcompressor:${PYTHONPATH:-}"
export PATH="/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin:$PATH"
export RCM_TRANSFORMER_PATH="$RCM_TRANSFORMER"
mkdir -p "$RUN" "$(dirname "$LOG")"
test "$(find "$CACHE/caches" -maxdepth 1 -name '*.pt' | wc -l)" = 64
test ! -e "$CKPT"

cd "$DIFFUSION"
"$PY" "$ROOT/scripts/ptq_rcm_wan.py" \
  configs/model/wan2.1-1.3b.yaml \
  configs/svdquant/int4.yaml \
  configs/svdquant/rcm_wan_int4_s16_g10.yaml \
  --pipeline-path="$MODEL" --calib-path="$CACHE" --calib-num-samples=64 \
  --output-root="$RUN" --cache-root="$RUN" --save-model="$CKPT" \
  --skip-eval --skip-gen --eval-num-gpus=1 2>&1 | tee "$LOG"

"$PY" "$ROOT/scripts/verify_rcm_int4_checkpoint.py" --checkpoint "$CKPT" --model "$MODEL" --transformer "$RCM_TRANSFORMER" --cache "$CACHE/caches/0000-00000-0.pt"
