#!/usr/bin/env bash
# Download FLUX.1-dev (and optionally official W4A4 transformer) onto SSD.
#
# 1) Activate env and login (interactive, you do this):
#      source /home/wenjinqi.wjq/workspace/svdquant_backup/scripts/env_svdquant_ptq.sh
#      hf auth login
#    (or: hf auth login --token $HF_TOKEN)
#
# 2) Run download (tmux recommended):
#      tmux new -s svdquant-hf
#      bash scripts/download_flux_dev.sh
#
# Env overrides:
#   HF_ENDPOINT   default https://hf-mirror.com
#   DATA_ROOT     default /ssd/2/wenjinqi.wjq
#   DOWNLOAD_OFFICIAL_W4A4=1  also fetch mit-han-lab/svdq-int4-flux.1-dev

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/ssd/2/wenjinqi.wjq}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${DATA_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export TMPDIR="${DATA_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${TMPDIR}" "${DATA_ROOT}/ckpts" "${DATA_ROOT}/compare"

LOG="${REPO_ROOT}/_setup/download_flux.log"
mkdir -p "${REPO_ROOT}/_setup"
exec > >(tee -a "${LOG}") 2>&1

echo "=== download_flux_dev $(date -Iseconds) ==="
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
df -h "${DATA_ROOT}" /home/wenjinqi.wjq | cat

# Prefer SSD env python
PY="${DATA_ROOT}/conda-envs/svdquant-ptq/bin/python"
if [[ ! -x "${PY}" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${DATA_ROOT}/conda-envs/svdquant-ptq"
  PY="$(command -v python)"
fi

echo "python: ${PY}"
"${PY}" -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

# Check auth (gated repo needs login)
if ! "${PY}" - <<'PY'
from huggingface_hub import get_token, whoami
token = get_token()
if not token:
    raise SystemExit("NO_TOKEN")
print("logged_in_as:", whoami(token=token).get("name"))
PY
then
  echo ""
  echo "ERROR: 未检测到 Hugging Face token。"
  echo "请先在本终端执行："
  echo "  source ${REPO_ROOT}/scripts/env_svdquant_ptq.sh"
  echo "  hf auth login"
  echo "完成网页授权/粘贴 token 后再重新运行本脚本。"
  exit 2
fi

echo "[1/2] Downloading black-forest-labs/FLUX.1-dev (diffusers layout)..."
"${PY}" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = "black-forest-labs/FLUX.1-dev"
path = snapshot_download(
    repo_id=repo,
    resume_download=True,
    local_dir=None,  # use HF hub cache under HF_HOME
    local_dir_use_symlinks=False,
)
print("FLUX.1-dev cached at:", path)
print("HF_HOME=", os.environ.get("HF_HOME"))
PY

if [[ "${DOWNLOAD_OFFICIAL_W4A4:-0}" == "1" ]]; then
  echo "[2/2] Downloading mit-han-lab/svdq-int4-flux.1-dev (optional reference)..."
  "${PY}" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="mit-han-lab/svdq-int4-flux.1-dev",
    resume_download=True,
)
print("official W4A4 cached at:", path)
PY
else
  echo "[2/2] Skip official W4A4 (set DOWNLOAD_OFFICIAL_W4A4=1 to enable)"
fi

# Smoke: resolve pipeline path without loading full weights to GPU
echo "Smoke: Diffusers from_pretrained resolve..."
"${PY}" - <<'PY'
import torch
from diffusers import FluxPipeline
# Only fetch/config resolve; keep on CPU for smoke
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
print("pipeline ok; transformer type:", type(pipe.transformer).__name__)
del pipe
PY

df -h "${DATA_ROOT}" | cat
du -sh "${HF_HOME}"/* 2>/dev/null | sort -hr | head -20 || true
echo "DONE $(date -Iseconds)"
echo "Next: run calibration / PTQ under examples/diffusion with DATA_ROOT=${DATA_ROOT}"
