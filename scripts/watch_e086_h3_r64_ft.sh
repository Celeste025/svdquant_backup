#!/usr/bin/env bash
# Lightweight disk/process watchdog.  It never deletes, retries, or changes a
# recipe; it records an actionable terminal status for the supervising agent.
set -euo pipefail

RUN_ROOT="${1:-/data1/models/svdquant-wjq/results/E086}"
SESSION="${2:-e086_h3_r64_ft}"
STATUS="$RUN_ROOT/watchdog_status.json"
POLL_SECONDS="${POLL_SECONDS:-300}"
STALE_SECONDS="${STALE_SECONDS:-900}"
while true; do
  now="$(date +%s)"
  data_avail_kib="$(df -Pk /data1 | awk 'NR==2 {print $4}')"
  root_avail_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
  progress="$RUN_ROOT/checkpoint/progress.pt"
  progress_mtime=0
  test -f "$progress" && progress_mtime="$(stat -c %Y "$progress")"
  # The fine-tuning worker deliberately stops updating progress.pt once its
  # complete checkpoint has been written.  Evaluation can then run for hours,
  # guarded by the launcher's hard deadline, so treating that frozen training
  # file as a stalled whole experiment is a false positive.
  phase="training"
  if test -f "$RUN_ROOT/checkpoint/status.json" && grep -q '"status": "complete"' "$RUN_ROOT/checkpoint/status.json"; then
    phase="evaluation_or_analysis"
  fi
  if (( data_avail_kib < 20 * 1024 * 1024 )); then
    printf '{"status":"blocked_low_data_space","unix":%s,"data_avail_kib":%s,"root_avail_kib":%s}\n' "$now" "$data_avail_kib" "$root_avail_kib" > "$STATUS"
    exit 2
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    if test -f "$RUN_ROOT/status.json" && grep -q '"status": "complete"' "$RUN_ROOT/status.json"; then
      printf '{"status":"complete","unix":%s,"data_avail_kib":%s,"root_avail_kib":%s}\n' "$now" "$data_avail_kib" "$root_avail_kib" > "$STATUS"
      exit 0
    fi
    printf '{"status":"supervisor_action_required","unix":%s,"data_avail_kib":%s,"root_avail_kib":%s,"progress_mtime":%s}\n' "$now" "$data_avail_kib" "$root_avail_kib" "$progress_mtime" > "$STATUS"
    exit 3
  fi
  if test "$phase" = training && (( progress_mtime > 0 && now - progress_mtime > STALE_SECONDS )); then
    printf '{"status":"supervisor_action_required_stalled_progress","unix":%s,"data_avail_kib":%s,"root_avail_kib":%s,"progress_mtime":%s}\n' "$now" "$data_avail_kib" "$root_avail_kib" "$progress_mtime" > "$STATUS"
    exit 4
  fi
  printf '{"status":"running","phase":"%s","unix":%s,"data_avail_kib":%s,"root_avail_kib":%s,"progress_mtime":%s}\n' "$phase" "$now" "$data_avail_kib" "$root_avail_kib" "$progress_mtime" > "$STATUS"
  sleep "$POLL_SECONDS"
done
