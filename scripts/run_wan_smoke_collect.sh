#!/usr/bin/env bash
# Wan2.1-1.3B smoke collect: 8 VBench prompts x 50 steps, model-level I/O caches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSION="${REPO_ROOT}/third_party/deepcompressor/examples/diffusion"
DATA_ROOT="${DATA_ROOT:-/ssd/2/wenjinqi.wjq}"
ENV_PREFIX="${DATA_ROOT}/conda-envs/svdquant-ptq"
LOG_DIR="${DATA_ROOT}/runs/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEEPCOMPRESSOR_TRANSFORMER_ONLY="${DEEPCOMPRESSOR_TRANSFORMER_ONLY:-0}"
export DEEPCOMPRESSOR_WAN_FLOW_SHIFT="${DEEPCOMPRESSOR_WAN_FLOW_SHIFT:-3.0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${DATA_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TMPDIR="${DATA_ROOT}/tmp"
export PATH="${ENV_PREFIX}/bin:${PATH}"

mkdir -p "${DATA_ROOT}"/{datasets,runs,ckpts,compare,tmp} "${LOG_DIR}"
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${DIFFUSION}"
echo "=== $(date -Iseconds) Wan2.1-1.3B smoke collect ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || true

python "${REPO_ROOT}/scripts/_deepcompressor_entry.py" deepcompressor.app.diffusion.dataset.collect.calib \
  configs/model/wan2.1-1.3b.yaml \
  configs/collect/vbench.yaml \
  --eval-num-gpus 1 \
  --eval-batch-size 1 \
  2>&1 | tee "${LOG_DIR}/wan_smoke_collect.log"
