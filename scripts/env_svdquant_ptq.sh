#!/usr/bin/env bash
# Activate SVDQuant PTQ env (correctness-first; no nunchaku kernels required).
# Usage: source scripts/env_svdquant_ptq.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"
DATA_ROOT="${SVDQUANT_DATA_ROOT:-${WORKSPACE_ROOT}/app_data}"
ENV_ROOT="${SVDQUANT_ENV_ROOT:-${WORKSPACE_ROOT}/app_source/envs}"
PTQ_ENV="${SVDQUANT_PTQ_ENV:-${ENV_ROOT}/svdquant-ptq}"
DIFFUSION_EXAMPLES="${REPO_ROOT}/third_party/deepcompressor/examples/diffusion"

if [[ ! -x "${PTQ_ENV}/bin/python" ]]; then
  echo "No PTQ environment found at ${PTQ_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${PTQ_ENV}/bin:${PATH}"
export SVDQUANT_DATA_ROOT="${DATA_ROOT}"

export DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-${REPO_ROOT}/third_party/DiffSynth-Studio}"
export VBENCH_ROOT="${VBENCH_ROOT:-${REPO_ROOT}/third_party/ViDiT-Q/eval/video/Vbench}"
export VBENCH_CACHE_ROOT="${VBENCH_CACHE_ROOT:-${DATA_ROOT}/cache/vbench}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${HF_HOME}/diffusers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-${DATA_ROOT}/models/FLUX.1-dev}"
mkdir -p "${HF_HOME}" "${VBENCH_CACHE_ROOT}" "${DATA_ROOT}/artifacts" "${DATA_ROOT}/datasets" \
  "${DATA_ROOT}/runs/logs" "${DATA_ROOT}/videos" "${PIP_CACHE_DIR}" "${TMPDIR}"

cd "${DIFFUSION_EXAMPLES}" || return 1 2>/dev/null || exit 1
echo "Activated svdquant-ptq"
echo "  python: $(command -v python)"
echo "  SVDQUANT_DATA_ROOT=${SVDQUANT_DATA_ROOT}"
echo "  HF_HOME=${HF_HOME}"
echo "  DIFFSYNTH_ROOT=${DIFFSYNTH_ROOT}"
echo "  VBENCH_ROOT=${VBENCH_ROOT}"
echo "  VBENCH_CACHE_ROOT=${VBENCH_CACHE_ROOT}"
echo "  cwd: $(pwd)"
