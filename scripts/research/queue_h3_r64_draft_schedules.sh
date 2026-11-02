#!/usr/bin/env bash
set -u
cd /home/wjq/workspace/svdquant-exp
gpu=${1:?gpu}; shift
for pid in "$@"; do
 for variant in c1 c2 c3 c4 c5 c6 c7; do
  name="${variant}_p$(printf '%03d' "$pid")"; log="results/logs/h3_schedule_${name}.log"
  for phase in denoise decode evaluate; do
   printf '%s GPU=%s variant=%s pid=%s phase=%s START\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" "$pid" "$phase" >> "$log"
   if bash scripts/research/launch_h3_r64_draft_schedules.sh "$gpu" "$variant" "$pid" "$phase" >> "$log" 2>&1; then
    printf '%s %s OK\n' "$(date --iso-8601=seconds)" "$phase" >> "$log"
    if [ "$phase" = evaluate ]; then /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/summarize_h3_r64_draft_schedules.py >> "$log" 2>&1; fi
   else
    rc=$?; printf '%s %s FAILED rc=%s\n' "$(date --iso-8601=seconds)" "$phase" "$rc" >> "$log"
    printf '%s,%s,%s,%s,%s\n' "$(date --iso-8601=seconds)" "$variant" "$pid" "$phase" "$rc" >> "results/logs/h3_schedule_failures_gpu${gpu}.csv"
    break
   fi
  done
 done
done
printf '%s queue completed gpu=%s\n' "$(date --iso-8601=seconds)" "$gpu" >> "results/logs/h3_schedule_queue_gpu${gpu}.log"
