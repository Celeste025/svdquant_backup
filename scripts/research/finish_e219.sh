#!/usr/bin/env bash
set -eu
cd /home/wjq/workspace/svdquant-exp
while :; do
 if rg -l 'Traceback|^FAILED ' results/logs/E219_protect*.log >/dev/null 2>&1; then printf 'E219 has failed jobs; inspect logs.\n'; exit 1; fi
 count=$(find results/research/E219/protect128 results/research/E219/protect64 -name metrics.json | wc -l)
 if [ "$count" -eq 8 ]; then break; fi
 sleep 30
done
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/summarize_e219.py
