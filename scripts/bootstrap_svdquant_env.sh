#!/usr/bin/env bash
# Bootstrap micromamba + svdquant-ptq env (run in your interactive terminal if agent network is slow).
# Usage: bash scripts/bootstrap_svdquant_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"
ENV_NAME="svdquant-ptq"
DC_ROOT="${REPO_ROOT}/third_party/deepcompressor"
STATUS="${REPO_ROOT}/_setup/env_status.txt"

mkdir -p "${MAMBA_ROOT}/bin" "${REPO_ROOT}/_setup"

df -h "${REPO_ROOT}" | tail -1

if [[ ! -x "${MAMBA_BIN}" ]]; then
  echo "[1/5] Installing micromamba into ${MAMBA_ROOT}"
  tmp="$(mktemp -d)"
  # ~30MB binary; prefer official API, fall back to gh release
  if ! curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xjv -C "${tmp}" bin/micromamba; then
    curl -fsSL -o "${tmp}/micromamba" \
      "https://github.com/mamba-org/micromamba-releases/releases/download/2.0.5-0/micromamba-linux-64"
    chmod +x "${tmp}/micromamba"
    mkdir -p "${tmp}/bin"
    mv "${tmp}/micromamba" "${tmp}/bin/micromamba"
  fi
  mv "${tmp}/bin/micromamba" "${MAMBA_BIN}"
  rm -rf "${tmp}"
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
eval "$("${MAMBA_BIN}" shell hook -s bash)"

echo "[2/5] Creating conda env ${ENV_NAME} (python 3.12)"
if ! "${MAMBA_BIN}" env list | grep -q "${ENV_NAME}"; then
  "${MAMBA_BIN}" create -y -n "${ENV_NAME}" python=3.12 pip poetry ninja \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    -c conda-forge || \
  "${MAMBA_BIN}" create -y -n "${ENV_NAME}" python=3.12 pip poetry ninja -c conda-forge
fi

micromamba activate "${ENV_NAME}"
python -V
pip install -U pip

echo "[3/5] Installing PyTorch CUDA wheels"
if ! python -c "import torch" 2>/dev/null; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

echo "[4/5] Installing deepcompressor"
cd "${DC_ROOT}"
poetry config virtualenvs.create false
set +e
poetry install --no-interaction
POETRY_RC=$?
set -e
if [[ ${POETRY_RC} -ne 0 ]]; then
  echo "poetry install had errors; trying pip editable install"
  pip install -e .
fi

echo "[5/5] Verify"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
import deepcompressor
print("deepcompressor OK", getattr(deepcompressor, "__file__", ""))
print("cuda_available", torch.cuda.is_available())
PY

{
  echo "MAMBA_ROOT=${MAMBA_ROOT}"
  echo "ENV_NAME=${ENV_NAME}"
  echo "activate: source ${REPO_ROOT}/scripts/env_svdquant_ptq.sh"
  echo "python: $(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  du -sh "${MAMBA_ROOT}"
  df -h "${REPO_ROOT}" | tail -1
} | tee "${STATUS}"

echo "DONE. Activate with: source ${REPO_ROOT}/scripts/env_svdquant_ptq.sh"
