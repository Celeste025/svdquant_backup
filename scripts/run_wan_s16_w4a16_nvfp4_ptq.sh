#!/usr/bin/env bash
# Wan2.1-1.3B W4A16 SVDQuant: real_nvfp4 weights, activations BF16 (ipts.dtype null).
# Uses batch1.yaml for tight VRAM; reuse s16 calib caches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSION="${REPO_ROOT}/third_party/deepcompressor/examples/diffusion"
DATA_ROOT="${DATA_ROOT:-/ssd/2/wenjinqi.wjq}"
ENV_PREFIX="${DATA_ROOT}/conda-envs/svdquant-ptq"
LOG_DIR="${DATA_ROOT}/runs/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEEPCOMPRESSOR_TRANSFORMER_ONLY="${DEEPCOMPRESSOR_TRANSFORMER_ONLY:-1}"
export DEEPCOMPRESSOR_WAN_GATED="${DEEPCOMPRESSOR_WAN_GATED:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${DATA_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TMPDIR="${DATA_ROOT}/tmp"
export PATH="${ENV_PREFIX}/bin:${PATH}"

CALIB_DIR="${DATA_ROOT}/datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16"
RUN_ROOT="${DATA_ROOT}/runs/wan_s16_w4a16_nvfp4"
SAVE_DIR="${DATA_ROOT}/ckpts/wan2.1-1.3b-w4a16-nvfp4-s16"
mkdir -p "${RUN_ROOT}" "${SAVE_DIR}" "${LOG_DIR}"

# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
cd "${DIFFUSION}"

if [[ ! -d "${CALIB_DIR}/caches" ]]; then
  echo "ERROR: calib caches missing at ${CALIB_DIR}" >&2
  exit 1
fi

echo "=== $(date -Iseconds) Wan2.1-1.3B s16 W4A16 real_nvfp4 PTQ ==="
echo "CALIB_DIR=${CALIB_DIR}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DEEPCOMPRESSOR_TRANSFORMER_ONLY=${DEEPCOMPRESSOR_TRANSFORMER_ONLY}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv || true

python "${REPO_ROOT}/scripts/check_wan_attn_struct.py"

python "${REPO_ROOT}/scripts/_deepcompressor_entry.py" deepcompressor.app.diffusion.ptq \
  configs/model/wan2.1-1.3b.yaml \
  configs/svdquant/__default__.yaml \
  configs/svdquant/real_nvfp4_w4a16.yaml \
  configs/svdquant/wan_s16.yaml \
  configs/svdquant/batch1.yaml \
  --calib-path "${CALIB_DIR}" \
  --calib-num-samples 64 \
  --output-root "${RUN_ROOT}" \
  --cache-root "${RUN_ROOT}" \
  --save-model "${SAVE_DIR}" \
  --skip-eval \
  --skip-gen \
  --eval-num-gpus 1 \
  2>&1 | tee "${LOG_DIR}/wan_s16_w4a16_nvfp4_ptq.log"
