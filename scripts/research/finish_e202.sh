#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
while :; do
 for pid in 25 48 44 7; do
  if [[ -f "results/logs/E202_p${pid}.failed" ]]; then cat "results/logs/E202_p${pid}.failed" >&2; exit 1; fi
 done
 if [[ -f results/logs/E202_p25.done && -f results/logs/E202_p48.done && -f results/logs/E202_p44.done && -f results/logs/E202_p7.done ]]; then break; fi
 sleep 120
done
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/summarize_e202_prob_reuse.py
