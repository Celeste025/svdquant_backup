#!/usr/bin/env bash
# Flux.1-dev calib (64 prompts) + INT4 SVDQuant with official fast.yaml (num_samples=64, grids=10).
# Correctness-first: no nunchaku, no gptq.
#
# Usage (inside tmux):
#   bash scripts/run_flux_dev_fast_ptq.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSION="${REPO_ROOT}/third_party/deepcompressor/examples/diffusion"
DATA_ROOT="${DATA_ROOT:-/ssd/2/wenjinqi.wjq}"
ENV_PREFIX="${DATA_ROOT}/conda-envs/svdquant-ptq"
LOG_DIR="${DATA_ROOT}/runs/logs"
# Prefer idle GPUs; avoid GPU 7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEEPCOMPRESSOR_TRANSFORMER_ONLY="${DEEPCOMPRESSOR_TRANSFORMER_ONLY:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${DATA_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export TMPDIR="${DATA_ROOT}/tmp"
export PIP_CACHE_DIR="${DATA_ROOT}/pip-cache"

mkdir -p "${DATA_ROOT}"/{datasets,runs,ckpts,compare,tmp,pip-cache} "${LOG_DIR}"

# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${DIFFUSION}"

CALIB_ROOT="${DATA_ROOT}/datasets"
# collect writes .../s{num_samples}
CALIB_DIR="${CALIB_ROOT}/torch.bfloat16/flux.1-dev/fmeuler50-g3.5/qdiff/s64"
RUN_ROOT="${DATA_ROOT}/runs"
SAVE_DIR="${DATA_ROOT}/ckpts/flux.1-dev-int4-fast"

echo "=== $(date -Iseconds) Flux-dev FAST PTQ ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME}"
echo "CALIB_DIR=${CALIB_DIR}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || true
df -h "${DATA_ROOT}" | cat

# ----- Step 1: calibration collection (64 prompts) -----
if [[ -d "${CALIB_DIR}/caches" ]] && [[ "$(find "${CALIB_DIR}/caches" -name '*.pt' 2>/dev/null | wc -l)" -gt 100 ]]; then
  echo "[1/2] Calib caches already present at ${CALIB_DIR}, skip collect"
else
  echo "[1/2] Collecting calib (64 prompts)..."
  python "${REPO_ROOT}/scripts/_deepcompressor_entry.py" deepcompressor.app.diffusion.dataset.collect.calib \
    configs/model/flux.1-dev.yaml \
    configs/collect/qdiff.yaml \
    --collect-root "${CALIB_ROOT}" \
    --collect-num-samples 64 \
    2>&1 | tee "${LOG_DIR}/collect_s64.log"
fi

# ----- Step 2: INT4 + fast + lowmem (no gptq) -----
echo "[2/2] PTQ INT4 + fast.yaml + lowmem.yaml ..."
python "${REPO_ROOT}/scripts/_deepcompressor_entry.py" deepcompressor.app.diffusion.ptq \
  configs/model/flux.1-dev.yaml \
  configs/svdquant/int4.yaml \
  configs/svdquant/fast.yaml \
  configs/svdquant/lowmem.yaml \
  --calib-path "${CALIB_DIR}" \
  --calib-num-samples 64 \
  --output-root "${RUN_ROOT}" \
  --cache-root "${RUN_ROOT}" \
  --save-model "${SAVE_DIR}" \
  --eval-benchmarks MJHQ \
  --eval-num-samples 8 \
  --eval-num-gpus 1 \
  2>&1 | tee "${LOG_DIR}/ptq_int4_fast.log"

echo "=== DONE $(date -Iseconds) ==="
echo "calib: ${CALIB_DIR}"
echo "ckpt:  ${SAVE_DIR}"
echo "logs:  ${LOG_DIR}"
