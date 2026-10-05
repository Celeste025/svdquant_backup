#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
SESSION=rcm-real-nvfp4-vbench51
GPU="${CUDA_VISIBLE_DEVICES:-4}"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session already exists: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" -c "$ROOT" "exec env CUDA_VISIBLE_DEVICES=$GPU bash $ROOT/scripts/run_rcm_real_nvfp4_vbench51_queue.sh"
tmux set-option -t "$SESSION" remain-on-exit on
echo "started tmux session $SESSION"
