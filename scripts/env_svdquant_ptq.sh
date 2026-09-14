#!/usr/bin/env bash
# Activate SVDQuant PTQ env (correctness-first; no nunchaku kernels required).
# Usage: source /home/wjq/workspace/svdquant-exp/scripts/env_svdquant_ptq.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="svdquant-ptq"
DC_ROOT="${REPO_ROOT}/third_party/deepcompressor"
DIFFUSION_EXAMPLES="${DC_ROOT}/examples/diffusion"
DATA_ROOT="${SVDQUANT_DATA_ROOT:-/data1/models/svdquant-wjq}"
PTQ_ENV="${DATA_ROOT}/conda-envs/svdquant-ptq"
HOME_CONDA="${HOME}/miniconda3"

if [[ -x "${PTQ_ENV}/bin/python" ]]; then
  export PATH="${PTQ_ENV}/bin:${PATH}"
elif [[ -f "${HOME_CONDA}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME_CONDA}/etc/profile.d/conda.sh"
  if conda env list | grep -qE '(^|\s)svdquant-ptq\s'; then
    conda activate "${ENV_NAME}"
  else
    echo "No svdquant-ptq environment found at ${PTQ_ENV}" >&2
    return 1 2>/dev/null || exit 1
  fi
else
  echo "No svdquant-ptq environment found at ${PTQ_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -d "${DATA_ROOT}" ]] && touch "${DATA_ROOT}/.write_test" 2>/dev/null; then
  rm -f "${DATA_ROOT}/.write_test"
else
  DATA_ROOT="${REPO_ROOT}/_data"
  mkdir -p "${DATA_ROOT}"
  echo "WARNING: /ssd/2/wenjinqi.wjq not writable; using ${DATA_ROOT}" >&2
fi

export HF_HOME="${DATA_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export PIP_CACHE_DIR="${DATA_ROOT}/pip-cache"
export TMPDIR="${DATA_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${DATA_ROOT}/datasets" "${DATA_ROOT}/runs" \
  "${DATA_ROOT}/ckpts" "${DATA_ROOT}/compare" "${PIP_CACHE_DIR}" "${TMPDIR}"

# Prefer idle GPUs; leave some free for others (adjust after nvidia-smi).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,5,6,7}"

cd "${DIFFUSION_EXAMPLES}" || true
echo "Activated svdquant-ptq"
echo "  python: $(command -v python)"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  HF_ENDPOINT=${HF_ENDPOINT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  cwd: $(pwd)"
