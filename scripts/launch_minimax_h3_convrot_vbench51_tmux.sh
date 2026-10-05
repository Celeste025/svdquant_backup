#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
GPU="${CUDA_VISIBLE_DEVICES:-6}"
SESSION=minimax-h3-convrot-vbench51
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session already exists: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" -c "$ROOT" "exec env CUDA_VISIBLE_DEVICES=$GPU bash $ROOT/scripts/run_minimax_h3_convrot_vbench51_queue.sh"
tmux set-option -t "$SESSION" remain-on-exit on
echo "started tmux session $SESSION on gpu $GPU"
