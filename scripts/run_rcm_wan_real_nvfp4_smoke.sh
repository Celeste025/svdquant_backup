#!/usr/bin/env bash
# Full-graph, reduced-cardinality PTQ smoke for an rCM real-NVFP4 recipe.
set -Eeuo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 GRID RANK" >&2; exit 2; fi
GRID="$1"; RANK="$2"
case "${GRID}-${RANK}" in 20-32|10-64) ;; *) echo "unsupported recipe ${GRID}-${RANK}" >&2; exit 2;; esac
ROOT=/home/wjq/workspace/svdquant-exp
DIFFUSION="$ROOT/third_party/deepcompressor/examples/diffusion"
DATA=/data1/models/svdquant-wjq
PY="$DATA/conda-envs/svdquant-ptq/bin/python"
CACHE="$DATA/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16"
MODEL="$DATA/models/Wan2.1-T2V-1.3B-Diffusers"
RCM_TRANSFORMER="$DATA/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer"
TAG="rcm-wan2.1-1.3b-real-nvfp4-s16-g${GRID}-r${RANK}-smoke"
CKPT="$DATA/ckpts/$TAG"; RUN="$DATA/runs/$TAG"; LOG="$ROOT/results/logs/${TAG}.log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" DEEPCOMPRESSOR_TRANSFORMER_ONLY=1 DEEPCOMPRESSOR_WAN_GATED=0
export PYTHONPATH="$ROOT/third_party/deepcompressor:${PYTHONPATH:-}" PATH="$DATA/conda-envs/svdquant-ptq/bin:$PATH" RCM_TRANSFORMER_PATH="$RCM_TRANSFORMER"
mkdir -p "$RUN" "$(dirname "$LOG")"
test "$(find "$CACHE/caches" -maxdepth 1 -name '*.pt' | wc -l)" = 64
verify() {
  "$PY" "$ROOT/scripts/verify_rcm_real_nvfp4_checkpoint.py" --checkpoint "$CKPT" --model "$MODEL" --transformer "$RCM_TRANSFORMER" \
    --cache "$CACHE/caches/0000-00000-0.pt" --rank "$RANK" --smooth-grids "$GRID" --lowrank-iters 1 --calib-samples 1 --smoke 2>&1 | tee -a "$LOG"
}
if [[ -f "$CKPT/manifest.json" ]]; then
  echo "verified smoke checkpoint already exists: $CKPT"
  exit 0
fi
if [[ -d "$CKPT" ]] && [[ -f "$CKPT/model.pt" && -f "$CKPT/scale.pt" && -f "$CKPT/wgts.pt" ]]; then
  echo "reusing completed but unverified smoke checkpoint: $CKPT"
  verify
  exit 0
fi
test ! -e "$CKPT"
cd "$DIFFUSION"
"$PY" "$ROOT/scripts/ptq_rcm_wan.py" configs/model/wan2.1-1.3b.yaml configs/svdquant/real_nvfp4.yaml configs/svdquant/wan_s16.yaml \
  "configs/svdquant/rcm_wan_real_nvfp4_s16_g${GRID}_r${RANK}.yaml" "configs/svdquant/rcm_wan_real_nvfp4_s16_g${GRID}_r${RANK}_smoke.yaml" \
  --pipeline-path="$MODEL" --calib-path="$CACHE" --calib-num-samples=1 --output-root="$RUN" --cache-root="$RUN" --save-model="$CKPT" \
  --skip-eval --skip-gen --eval-num-gpus=1 2>&1 | tee "$LOG"
verify
