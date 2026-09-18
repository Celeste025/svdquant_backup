#!/usr/bin/env bash
# Run one self-contained rCM real-NVFP4 W4A4 SVDQuant recipe.
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then echo "usage: $0 GRID RANK" >&2; exit 2; fi
GRID="$1"; RANK="$2"
case "${GRID}-${RANK}" in 20-32|10-64) ;; *) echo "unsupported recipe ${GRID}-${RANK}" >&2; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSION="$ROOT/third_party/deepcompressor/examples/diffusion"
DATA="${SVDQUANT_DATA_ROOT:-/data1/models/svdquant-wjq}"
PY="${SVDQUANT_PTQ_PYTHON:-$DATA/conda-envs/svdquant-ptq/bin/python}"
CACHE="$DATA/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16"
MODEL="$DATA/models/Wan2.1-T2V-1.3B-Diffusers"
RCM_TRANSFORMER="$DATA/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer"
TAG="rcm-wan2.1-1.3b-real-nvfp4-s16-g${GRID}-r${RANK}"
CKPT="$DATA/ckpts/$TAG"
RUN="$DATA/runs/$TAG"
OVERLAY="configs/svdquant/rcm_wan_real_nvfp4_s16_g${GRID}_r${RANK}.yaml"
LOG="$ROOT/results/logs/${TAG}-ptq.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export DEEPCOMPRESSOR_TRANSFORMER_ONLY=1 DEEPCOMPRESSOR_WAN_GATED=0
export PYTHONPATH="$ROOT/third_party/deepcompressor:${PYTHONPATH:-}"
export PATH="$DATA/conda-envs/svdquant-ptq/bin:$PATH"
export RCM_TRANSFORMER_PATH="$RCM_TRANSFORMER"
mkdir -p "$RUN" "$(dirname "$LOG")"
test "$(find "$CACHE/caches" -maxdepth 1 -name '*.pt' | wc -l)" = 64
test -d "$MODEL" && test -d "$RCM_TRANSFORMER"
if [[ -e "$CKPT" ]]; then echo "refusing to overwrite existing checkpoint: $CKPT" >&2; exit 1; fi
if [[ $(df --output=avail "$DATA" | tail -1) -lt 100000000 ]]; then echo "less than 100GB free on $DATA" >&2; exit 1; fi

cd "$DIFFUSION"
"$PY" "$ROOT/scripts/ptq_rcm_wan.py" \
  configs/model/wan2.1-1.3b.yaml configs/svdquant/real_nvfp4.yaml configs/svdquant/wan_s16.yaml "$OVERLAY" \
  --pipeline-path="$MODEL" --calib-path="$CACHE" --calib-num-samples=64 \
  --output-root="$RUN" --cache-root="$RUN" --save-model="$CKPT" \
  --skip-eval --skip-gen --eval-num-gpus=1 2>&1 | tee "$LOG"
"$PY" "$ROOT/scripts/verify_rcm_real_nvfp4_checkpoint.py" --checkpoint "$CKPT" --model "$MODEL" --transformer "$RCM_TRANSFORMER" \
  --cache "$CACHE/caches/0000-00000-0.pt" --rank "$RANK" --smooth-grids "$GRID" 2>&1 | tee -a "$LOG"
