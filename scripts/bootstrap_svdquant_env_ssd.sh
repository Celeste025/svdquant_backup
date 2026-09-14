#!/usr/bin/env bash
# Finish PTQ env on /ssd/2 (home disk is full). Correctness-only: torch + deepcompressor, no nunchaku.
# Run in your terminal or: tmux new -s svdquant-env 'bash scripts/bootstrap_svdquant_env_ssd.sh'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/ssd/2/wenjinqi.wjq"
ENV_PREFIX="${DATA_ROOT}/conda-envs/svdquant-ptq"
DC_ROOT="${REPO_ROOT}/third_party/deepcompressor"
STATUS="${REPO_ROOT}/_setup/env_status.txt"
LOG="${REPO_ROOT}/_setup/bootstrap_ssd.log"

mkdir -p "${DATA_ROOT}"/{hf,datasets,runs,ckpts,compare,conda-envs,pip-cache,tmp} \
  "${REPO_ROOT}/_setup"
export PIP_CACHE_DIR="${DATA_ROOT}/pip-cache"
export TMPDIR="${DATA_ROOT}/tmp"

exec > >(tee -a "${LOG}") 2>&1

echo "=== bootstrap SSD env $(date) ==="
df -h /home/wenjinqi.wjq /ssd/2 | cat

# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "[1/4] conda create -p ${ENV_PREFIX} python=3.12"
  conda create -y -p "${ENV_PREFIX}" python=3.12 pip \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    || conda create -y -p "${ENV_PREFIX}" python=3.12 pip
fi

conda activate "${ENV_PREFIX}"
python -V
pip install -U pip poetry ninja

echo "[2/4] torch cu126"
python -c "import torch" 2>/dev/null || \
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 || \
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[3/4] deepcompressor"
cd "${DC_ROOT}"
poetry config virtualenvs.create false
set +e
poetry install --no-interaction
POETRY_RC=$?
set -e
if [[ ${POETRY_RC} -ne 0 ]]; then
  echo "poetry failed (${POETRY_RC}); pip install -e ."
  pip install -e . || true
  # minimal deps for PTQ correctness (skip image_reward)
  pip install "diffusers>=0.32.0" "transformers>=4.46.0" "accelerate>=0.26.0" \
    omniconfig einops tqdm datasets sentencepiece protobuf opencv-python \
    clean-fid ftfy || true
fi

echo "[4/4] verify"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
import deepcompressor
print("deepcompressor OK")
PY

# free home micromamba leftover if SSD env works
if [[ -x "${ENV_PREFIX}/bin/python" ]] && [[ -d "${REPO_ROOT}/.micromamba" ]]; then
  echo "Removing redundant workspace .micromamba to free home disk..."
  rm -rf "${REPO_ROOT}/.micromamba"
fi

{
  echo "timestamp: $(date -Iseconds)"
  echo "env_ready: yes"
  echo "env_prefix: ${ENV_PREFIX}"
  echo "activate: source ${REPO_ROOT}/scripts/env_svdquant_ptq.sh"
  echo "or: source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_PREFIX}"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  df -h /home/wenjinqi.wjq /ssd/2 | cat
  du -sh "${ENV_PREFIX}"
} | tee "${STATUS}"

echo "DONE"
