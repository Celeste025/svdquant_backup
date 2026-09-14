#!/usr/bin/env bash
# Flux.1-dev INT4+fast PTQ with torch.svd_lowrank patch (separate from official-SVD run).
# Does NOT touch the running official job / its output dirs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSION="${REPO_ROOT}/third_party/deepcompressor/examples/diffusion"
DATA_ROOT="${DATA_ROOT:-/ssd/2/wenjinqi.wjq}"
ENV_PREFIX="${DATA_ROOT}/conda-envs/svdquant-ptq"
LOG_DIR="${DATA_ROOT}/runs/logs"
# Prefer idle GPU; avoid 5 (official job) and 7 (user constraint)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEEPCOMPRESSOR_TRANSFORMER_ONLY="${DEEPCOMPRESSOR_TRANSFORMER_ONLY:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${DATA_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
unset TRANSFORMERS_CACHE
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export TMPDIR="${DATA_ROOT}/tmp"
export PIP_CACHE_DIR="${DATA_ROOT}/pip-cache"
export PATH="${ENV_PREFIX}/bin:${PATH}"

mkdir -p "${DATA_ROOT}"/{datasets,runs,ckpts,compare,tmp,pip-cache} "${LOG_DIR}"

# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${DIFFUSION}"

CALIB_DIR="${DATA_ROOT}/datasets/torch.bfloat16/flux.1-dev/fmeuler50-g3.5/qdiff/s64"
RUN_ROOT="${DATA_ROOT}/runs/svd_lowrank"
SAVE_DIR="${DATA_ROOT}/ckpts/flux.1-dev-int4-fast-svd-lowrank"
mkdir -p "${RUN_ROOT}" "${SAVE_DIR}"

echo "=== $(date -Iseconds) Flux-dev FAST PTQ (svd_lowrank patch) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CALIB_DIR=${CALIB_DIR}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SAVE_DIR=${SAVE_DIR}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || true

if [[ ! -d "${CALIB_DIR}/caches" ]]; then
  echo "ERROR: calib caches missing at ${CALIB_DIR}" >&2
  exit 1
fi
echo "[1/2] Reuse existing calib caches (skip collect)"

echo "[2/2] PTQ INT4 + fast + lowmem + svd_lowrank patch ..."
python "${REPO_ROOT}/scripts/run_ptq_svd_lowrank.py" deepcompressor.app.diffusion.ptq \
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
  2>&1 | tee "${LOG_DIR}/ptq_int4_fast_svd_lowrank.log"
