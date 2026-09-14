#!/usr/bin/env bash
# Flux.1-dev few-sample ablation: collect 16 prompts x 50 steps, PTQ with 16 caches, grid10.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSION="${REPO_ROOT}/third_party/deepcompressor/examples/diffusion"
DATA_ROOT="${DATA_ROOT:-/ssd/2/wenjinqi.wjq}"
ENV_PREFIX="${DATA_ROOT}/conda-envs/svdquant-ptq"
LOG_DIR="${DATA_ROOT}/runs/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEEPCOMPRESSOR_TRANSFORMER_ONLY="${DEEPCOMPRESSOR_TRANSFORMER_ONLY:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${DATA_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TMPDIR="${DATA_ROOT}/tmp"
export PATH="${ENV_PREFIX}/bin:${PATH}"

CALIB_DIR="${DATA_ROOT}/datasets/torch.bfloat16/flux.1-dev/fmeuler50-g3.5/qdiff/s16"
RUN_ROOT="${DATA_ROOT}/runs/flux_s16"
SAVE_DIR="${DATA_ROOT}/ckpts/flux.1-dev-int4-s16"
COMPARE_DIR="${DATA_ROOT}/compare/flux_bf16_vs_w4a4_s16"
mkdir -p "${RUN_ROOT}" "${SAVE_DIR}" "${COMPARE_DIR}" "${LOG_DIR}" "${DATA_ROOT}"/{datasets,tmp}

# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
cd "${DIFFUSION}"

echo "=== $(date -Iseconds) Flux.1-dev s16 collect+PTQ (GPU ${CUDA_VISIBLE_DEVICES}) ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || true

if [[ ! -d "${CALIB_DIR}/caches" ]] || [[ "$(find "${CALIB_DIR}/caches" -name '*.pt' 2>/dev/null | wc -l)" -lt 800 ]]; then
  echo "[1/3] Collect 16 prompts x 50 steps -> ${CALIB_DIR}"
  python "${REPO_ROOT}/scripts/_deepcompressor_entry.py" deepcompressor.app.diffusion.dataset.collect.calib \
    configs/model/flux.1-dev.yaml \
    configs/collect/qdiff_s16.yaml \
    --eval-num-gpus 1 \
    --eval-batch-size 1 \
    2>&1 | tee "${LOG_DIR}/flux_s16_collect.log"
else
  echo "[1/3] Reuse existing calib at ${CALIB_DIR}"
fi

n_caches="$(find "${CALIB_DIR}/caches" -name '*.pt' 2>/dev/null | wc -l)"
echo "caches=${n_caches}"
if [[ "${n_caches}" -lt 800 ]]; then
  echo "ERROR: expected ~800 caches (16x50), got ${n_caches}" >&2
  exit 1
fi

echo "[2/3] PTQ INT4 + grid10 + calib-num-samples 16 ..."
python "${REPO_ROOT}/scripts/_deepcompressor_entry.py" deepcompressor.app.diffusion.ptq \
  configs/model/flux.1-dev.yaml \
  configs/svdquant/int4.yaml \
  configs/svdquant/flux_s16.yaml \
  configs/svdquant/lowmem.yaml \
  --calib-path "${CALIB_DIR}" \
  --calib-num-samples 16 \
  --output-root "${RUN_ROOT}" \
  --cache-root "${RUN_ROOT}" \
  --save-model "${SAVE_DIR}" \
  --skip-eval \
  --skip-gen \
  --eval-num-gpus 1 \
  2>&1 | tee "${LOG_DIR}/flux_s16_ptq.log"

echo "[3/3] One-image BF16 vs W4A4 compare ..."
cd "${REPO_ROOT}"
python -u scripts/infer_bf16_vs_w4a4_one.py \
  --ckpt "${SAVE_DIR}" \
  --out-dir "${COMPARE_DIR}" \
  2>&1 | tee -a "${LOG_DIR}/flux_s16_ptq.log"

echo "=== $(date -Iseconds) Flux s16 DONE ==="
echo "ckpt=${SAVE_DIR}"
echo "compare=${COMPARE_DIR}"
