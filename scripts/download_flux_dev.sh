#!/usr/bin/env bash
# Download FLUX.1-dev into the configured SVDQuant data root.
# Run after activating the PTQ environment and completing Hugging Face login.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${SVDQUANT_DATA_ROOT:-${WORKSPACE_ROOT}/app_data}}"
ENV_ROOT="${SVDQUANT_ENV_ROOT:-${WORKSPACE_ROOT}/app_source/envs}"
PTQ_ENV="${SVDQUANT_PTQ_ENV:-${ENV_ROOT}/svdquant-ptq}"
PY="${PTQ_PYTHON:-${PTQ_ENV}/bin/python}"
export SVDQUANT_DATA_ROOT="${DATA_ROOT}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${HF_HOME}/diffusers}"
export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-${DATA_ROOT}/models/FLUX.1-dev}"
SOURCE="${FLUX_DOWNLOAD_SOURCE:-modelscope}"
while (($#)); do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done
if [[ "${SOURCE}" != "modelscope" && "${SOURCE}" != "huggingface" ]]; then
  echo "--source must be modelscope or huggingface" >&2
  exit 2
fi
if [[ -n "${SVDQUANT_HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="${SVDQUANT_HF_ENDPOINT}"
fi
mkdir -p "${HF_HOME}" "${TMPDIR}" "${DATA_ROOT}/runs/logs" "${FLUX_MODEL_PATH}"

if [[ ! -x "${PY}" ]]; then
  echo "PTQ Python not found at ${PY}. Run scripts/bootstrap_svdquant_env.sh first." >&2
  exit 1
fi

LOG="${DATA_ROOT}/runs/logs/download_flux.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== download_flux_dev $(date -Iseconds) ==="
echo "SOURCE=${SOURCE}"
echo "HF_HOME=${HF_HOME}"
echo "FLUX_MODEL_PATH=${FLUX_MODEL_PATH}"
df -h "${DATA_ROOT}" "${HOME}"
echo "python: ${PY}"

if [[ "${SOURCE}" == "huggingface" ]]; then
  "${PY}" -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
  if ! "${PY}" - <<'PY'
from huggingface_hub import get_token
if not get_token():
    raise SystemExit("NO_TOKEN")
print("Hugging Face token detected")
PY
  then
    echo "ERROR: 未检测到 Hugging Face token。请先在本终端执行 hf auth login。"
    exit 2
  fi
  echo "[1/2] Downloading black-forest-labs/FLUX.1-dev from Hugging Face..."
  "${PY}" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    resume_download=True,
    local_dir=os.environ["FLUX_MODEL_PATH"],
    local_dir_use_symlinks=False,
)
print("FLUX.1-dev saved at:", path)
PY
else
  echo "[1/2] Downloading black-forest-labs/FLUX.1-dev from ModelScope..."
  "${PY}" - <<'PY'
import os
from modelscope import snapshot_download

path = snapshot_download(
    "black-forest-labs/FLUX.1-dev",
    local_dir=os.environ["FLUX_MODEL_PATH"],
    token=os.environ.get("MODELSCOPE_API_TOKEN") or None,
)
print("FLUX.1-dev saved at:", path)
PY
fi

if [[ "${DOWNLOAD_OFFICIAL_W4A4:-0}" == "1" ]]; then
  echo "[2/2] Downloading mit-han-lab/svdq-int4-flux.1-dev..."
  "${PY}" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="mit-han-lab/svdq-int4-flux.1-dev", resume_download=True)
print("official W4A4 cached at:", path)
PY
else
  echo "[2/2] Skip official W4A4 (set DOWNLOAD_OFFICIAL_W4A4=1 to enable)"
fi

echo "Smoke: resolving local Diffusers pipeline..."
"${PY}" - <<'PY'
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline

model_path = Path(os.environ["FLUX_MODEL_PATH"])
required = (
    "model_index.json", "scheduler", "text_encoder", "text_encoder_2",
    "tokenizer", "tokenizer_2", "transformer", "vae",
)
missing = [name for name in required if not (model_path / name).exists()]
if missing:
    raise RuntimeError(f"incomplete local FLUX Diffusers layout: {missing}")
pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
print("pipeline ok; transformer type:", type(pipe.transformer).__name__)
del pipe
PY

df -h "${DATA_ROOT}"
echo "DONE $(date -Iseconds)"
