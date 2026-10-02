#!/usr/bin/env bash
# Bootstrap micromamba + svdquant-ptq env (run in your interactive terminal if agent network is slow).
# Usage: bash scripts/bootstrap_svdquant_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"
DATA_ROOT="${SVDQUANT_DATA_ROOT:-${WORKSPACE_ROOT}/app_data}"
ENV_ROOT="${SVDQUANT_ENV_ROOT:-${WORKSPACE_ROOT}/app_source/envs}"
PTQ_ENV="${SVDQUANT_PTQ_ENV:-${ENV_ROOT}/svdquant-ptq}"
MAMBA_ROOT="${SVDQUANT_MAMBA_ROOT:-${ENV_ROOT}/.micromamba}"
MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"
DC_ROOT="${REPO_ROOT}/third_party/deepcompressor"
STATUS="${DATA_ROOT}/runs/setup/env_status_ptq.txt"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

mkdir -p "${MAMBA_ROOT}/bin" "${ENV_ROOT}" "$(dirname "${STATUS}")"

df -h "${DATA_ROOT}"

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

if [[ ! -x "${PTQ_ENV}/bin/python" ]]; then
  echo "[2/5] Creating Python 3.12 environment at ${PTQ_ENV}"
  "${MAMBA_BIN}" create -y -p "${PTQ_ENV}" python=3.12 pip poetry ninja \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    -c conda-forge || \
  "${MAMBA_BIN}" create -y -p "${PTQ_ENV}" python=3.12 pip poetry ninja -c conda-forge
fi

micromamba activate "${PTQ_ENV}"
python -V
pip install --upgrade pip

echo "[3/5] Installing PyTorch CUDA wheels"
if ! python -c "import torch" 2>/dev/null; then
  pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
fi

echo "[4/5] Installing PTQ and VBench dependencies"
# ImageReward is a Git dependency that does not build on Python 3.12; it is not
# needed for the smoke inference path, so install the local package without deps.
pip install --no-deps -e "${DC_ROOT}"
pip install modelscope==1.39.1 diffusers==0.33.1 transformers==4.49.0 \
  accelerate safetensors numpy opencv-python 'imageio[ffmpeg]' av==18.1.0 einops \
  datasets omniconfig sentencepiece protobuf ftfy timm rotary-embedding-torch
pip install -r "${REPO_ROOT}/third_party/ViDiT-Q/eval/video/requirements.txt"

echo "[5/5] Verify"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
import deepcompressor
print("deepcompressor OK", getattr(deepcompressor, "__file__", ""))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name())
    print("cuda_tensor", torch.ones(1, device="cuda").item())
PY

{
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "PTQ_ENV=${PTQ_ENV}"
  echo "activate: source ${REPO_ROOT}/scripts/env_svdquant_ptq.sh"
  echo "python: $(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  du -sh "${PTQ_ENV}"
  df -h "${DATA_ROOT}"
} | tee "${STATUS}"

echo "DONE. Activate with: source ${REPO_ROOT}/scripts/env_svdquant_ptq.sh"
