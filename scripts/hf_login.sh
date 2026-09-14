#!/usr/bin/env bash
# Interactive HF login helper (run in YOUR terminal — needs browser/token paste).
#
# Usage:
#   source /home/wjq/workspace/svdquant-exp/scripts/env_svdquant_ptq.sh
#   bash /home/wjq/workspace/svdquant-exp/scripts/hf_login.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_svdquant_ptq.sh"

echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
echo ""
echo "即将执行: hf auth login"
echo "注意：FLUX.1-dev 是 gated 模型，账号需先在网页同意许可："
echo "  https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo ""
echo "也可手动："
echo "  hf auth login"
echo "  # 或非交互: hf auth login --token YOUR_TOKEN --add-to-git-credential"
echo ""

if command -v hf >/dev/null 2>&1; then
  hf auth login
elif python -c "import huggingface_hub" 2>/dev/null; then
  # fallback for older hub without `hf` entrypoint
  python -m huggingface_hub.commands.huggingface_cli login 2>/dev/null \
    || python -c "from huggingface_hub import login; login()"
else
  pip install -U "huggingface_hub[cli]"
  hf auth login
fi

python - <<'PY'
from huggingface_hub import get_token, whoami
token = get_token()
assert token, "login failed: no token stored"
info = whoami(token=token)
print("OK logged in as:", info.get("name") or info)
PY

echo ""
echo "登录完成。下一步下载："
echo "  tmux new -s svdquant-hf"
echo "  bash ${REPO_ROOT}/scripts/download_flux_dev.sh"
