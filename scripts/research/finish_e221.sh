#!/usr/bin/env bash
set -eu
cd /home/wjq/workspace/svdquant-exp
while :; do
 if rg -l '^FAILED ' results/logs/E221_*.log >/dev/null 2>&1; then printf 'E221 has failed jobs; inspect logs.\n'; exit 1; fi
 count=$(find results/research/E221 -name metrics.json | wc -l)
 if [ "$count" -eq 12 ]; then break; fi
 sleep 30
done
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/summarize_e221.py
