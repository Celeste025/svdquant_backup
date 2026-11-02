#!/usr/bin/env bash
set -u
cd /home/wjq/workspace/svdquant-exp
out=results/research/h3_r64_draft_schedules/check_after_30min.txt
{
 date --iso-8601=seconds
 tmux ls 2>&1 || true
 for gpu in 1 2 3; do
  printf '\nGPU %s failures:\n' "$gpu"
  if [ -f "results/logs/h3_schedule_failures_gpu${gpu}.csv" ]; then cat "results/logs/h3_schedule_failures_gpu${gpu}.csv"; else printf 'none recorded\n'; fi
  printf 'GPU %s recent status:\n' "$gpu"
  tmux capture-pane -pt "h3_schedules_gpu${gpu}" -S -15 2>&1 || true
 done
 printf '\nCompleted metrics files:\n'
 find results/research/h3_r64_draft_schedules -path '*/p*/metrics.json' -print | sort
 printf '\nGPU status:\n'
 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} > "$out"
