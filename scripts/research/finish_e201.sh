#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
while :; do
 if grep -l '"status": "failed"' results/research/E201/*.json >/dev/null 2>&1; then printf 'E201 generation failed; inspect JSON/logs.\n' >&2; exit 1; fi
 if [[ -f results/logs/E201_gpu1.done && -f results/logs/E201_gpu2.done ]]; then break; fi
 sleep 60
done
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=1 /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/evaluate_e201_reuse.py
