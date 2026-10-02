#!/usr/bin/env bash
# Capture MiniMax-H3 DiT linear-layer activations (bf16) for outlier inspection.
# Waits for an idle GPU before starting, so it can be queued while other jobs run.
# Usage: bash scripts/run_h3_activation_survey_tmux.sh [gpu] [session] [free-gib]
set -euo pipefail

PROJECT_ROOT="/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup"
PYTHON="/home/admin/workspace/aop_lab/app_source/envs/minimax-h3/bin/python"
DATA_ROOT="/home/admin/workspace/aop_lab/app_data"
OUTPUT_DIR="$DATA_ROOT/runs/h3-activation-outliers"
LOG="$OUTPUT_DIR/survey.log"

GPU="${1:-0}"
SESSION="${2:-h3-act-survey}"
NEED_FREE_MIB="${3:-4096}"

[ -x "$PYTHON" ] || { echo "missing Python: $PYTHON" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

command="SURVEY_EXTRA_ARGS='${SURVEY_EXTRA_ARGS:-}' bash '$PROJECT_ROOT/scripts/wait_gpu_then_h3_activation_survey.sh' $GPU $NEED_FREE_MIB"

tmux new-session -d -s "$SESSION" "$command"
echo "queued $SESSION on GPU $GPU (phase-1 needs >= $NEED_FREE_MIB MiB free); log=$LOG"
