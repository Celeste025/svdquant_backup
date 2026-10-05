#!/usr/bin/env bash
# Launch the sequential queue detached.  Attach with: tmux attach -t rcm-h3-svdquant
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
SESSION=rcm-h3-svdquant
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec env CUDA_VISIBLE_DEVICES=5 '$ROOT/scripts/run_rcm_h3_svdquant_queue.sh'"
tmux set-option -t "$SESSION" remain-on-exit on
echo "started tmux session $SESSION"
