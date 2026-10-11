#!/usr/bin/env bash
# E086: rank-64 H3 SVDQuant low-rank adaptation and timestep analysis.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python"
BASE="$ROOT/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
CACHE="$ROOT/results/calib/minimax_h3_svdquant_standard_8p64s"
OUT="/data1/models/svdquant-wjq/results/E086"
REPORT="$ROOT/results/research/E086"
FT="$OUT/checkpoint/quant_state.pt"
LOG="/data1/models/svdquant-wjq/results/logs/e086_h3_r64_ft_$(date +%Y%m%d_%H%M%S).log"
# A recovery must honour the initial run's budget rather than silently grant
# itself another 9h45m.  These are deliberately opt-in environment overrides
# used by the resume tmux command below.
START="${E086_START_UNIX:-$(date +%s)}"
DEADLINE="${E086_DEADLINE_UNIX:-$((START + 35100))}"  # 9h45m hard wall-clock cap.
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
mkdir -p "$OUT" "$OUT/train" "$OUT/test" "$REPORT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

remaining() { echo $((DEADLINE - $(date +%s))); }
run_deadline() {
  local seconds
  seconds="$(remaining)"
  if (( seconds <= 0 )); then echo "E086 deadline reached"; return 124; fi
  timeout --signal=TERM --kill-after=120 "$seconds" "$@"
}
fail() {
  "$PY" - "$OUT/status.json" "$1" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'failed_or_budget_exhausted','reason':sys.argv[2],'ended_unix':time.time()},indent=2))
PY
  exit 1
}

echo "E086 start=$(date -d "@$START" -Is) deadline=$(date -d "@$DEADLINE" -Is)"
if test "${E086_RESUME:-0}" = 1 && test -f "$FT" && test -f "$OUT/checkpoint/status.json" && grep -q '"valid_ft_checkpoint": true' "$OUT/checkpoint/status.json"; then
  echo "reusing verified fine-tune checkpoint"
else
  run_deadline env CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/scripts/finetune_minimax_h3_r64_lowrank.py" \
    --base-state "$BASE" --cache-dir "$CACHE" --output-dir "$OUT/checkpoint" --max-passes 3 --deadline-unix "$((START + 17100))" \
    || fail "fine_tune_worker_failed"
fi
test -f "$FT" || fail "no_complete_finetune_checkpoint"

# Calibration arms are independent and use the same cached BF16 calls.
eval_train_arm() {
  local arm="$1" gpu="$2" state="$BASE"
  if test -s "$OUT/train/$arm.json"; then
    echo "reusing completed training arm $arm"
    return 0
  fi
  test "$arm" = svdquant_ft && state="$FT"
  run_deadline env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_minimax_h3_calib_step_error.py" \
    --cache-dir "$CACHE" --state "$state" --variant "$arm" --output "$OUT/train/$arm.json"
}
eval_train_arm plain_w4a4 4 & train_plain_pid=$!
eval_train_arm svdquant 5 & train_svd_pid=$!
wait "$train_plain_pid" || fail "train_eval_plain_w4a4_failed"
wait "$train_svd_pid" || fail "train_eval_svdquant_failed"
eval_train_arm svdquant_ft 4 || fail "train_eval_svdquant_ft_failed"

# Each prompt stays on one GPU, so its plain capture is deterministically reused
# by both quantized arms; different prompts are independent and run in pairs.
eval_prompt() {
  local prompt_id="$1"
  local gpu="$2"
  local calls="$OUT/test/p${prompt_id}_bf16_calls.pt"
  if test ! -s "$OUT/test/p${prompt_id}_plain_w4a4.json"; then
    run_deadline env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_minimax_h3_paired_step_error.py" \
      --state "$BASE" --variant plain_w4a4 --prompt-id "$prompt_id" --save-calls "$calls" --output "$OUT/test/p${prompt_id}_plain_w4a4.json"
  fi
  if test ! -s "$OUT/test/p${prompt_id}_svdquant.json"; then
    run_deadline env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_minimax_h3_paired_step_error.py" \
      --state "$BASE" --variant svdquant --prompt-id "$prompt_id" --calls-file "$calls" --output "$OUT/test/p${prompt_id}_svdquant.json"
  fi
  if test ! -s "$OUT/test/p${prompt_id}_svdquant_ft.json"; then
    run_deadline env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_minimax_h3_paired_step_error.py" \
      --state "$FT" --variant svdquant_ft --prompt-id "$prompt_id" --calls-file "$calls" --output "$OUT/test/p${prompt_id}_svdquant_ft.json"
  fi
}
eval_prompt 2 4 & test_p2_pid=$!
eval_prompt 16 5 & test_p16_pid=$!
wait "$test_p2_pid" || fail "test_p2_failed"
wait "$test_p16_pid" || fail "test_p16_failed"
eval_prompt 26 4 & test_p26_pid=$!
eval_prompt 42 5 & test_p42_pid=$!
wait "$test_p26_pid" || fail "test_p26_failed"
wait "$test_p42_pid" || fail "test_p42_failed"
eval_prompt 51 4 || fail "test_p51_failed"
## Legacy serial form intentionally removed: it exceeded the total budget.
: <<'SERIAL_EVAL_REMOVED'
for arm in plain_w4a4 svdquant svdquant_ft; do
  state="$BASE"; test "$arm" = svdquant_ft && state="$FT"
  gpu=4; test "$arm" = svdquant && gpu=5
  run_deadline env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_minimax_h3_calib_step_error.py" \
    --cache-dir "$CACHE" --state "$state" --variant "$arm" --output "$OUT/train/$arm.json" \
    || fail "train_eval_${arm}_failed"
done

# Each prompt captures BF16 once under the plain arm, then all quantized arms
# consume that exact serialized call sequence.
for pid in 2 16 26 42 51; do
  calls="$OUT/test/p${pid}_bf16_calls.pt"
  run_deadline env CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/scripts/eval_minimax_h3_paired_step_error.py" \
    --state "$BASE" --variant plain_w4a4 --prompt-id "$pid" --save-calls "$calls" --output "$OUT/test/p${pid}_plain_w4a4.json" \
    || fail "test_plain_p${pid}_failed"
  run_deadline env CUDA_VISIBLE_DEVICES=5 "$PY" "$ROOT/scripts/eval_minimax_h3_paired_step_error.py" \
    --state "$BASE" --variant svdquant --prompt-id "$pid" --calls-file "$calls" --output "$OUT/test/p${pid}_svdquant.json" \
    || fail "test_svdquant_p${pid}_failed"
  run_deadline env CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/scripts/eval_minimax_h3_paired_step_error.py" \
    --state "$FT" --variant svdquant_ft --prompt-id "$pid" --calls-file "$calls" --output "$OUT/test/p${pid}_svdquant_ft.json" \
    || fail "test_ft_p${pid}_failed"
done
SERIAL_EVAL_REMOVED

run_deadline "$PY" "$ROOT/scripts/analyze_minimax_h3_ft_timestep_nmse.py" --root "$OUT" --output "$OUT/analysis" \
  || fail "analysis_failed"
cp "$OUT/analysis/analysis.json" "$OUT/analysis/all_rows.csv" "$OUT/analysis/video_nmse_curves.csv" \
  "$OUT/analysis/ft_vs_plain_benefit.csv" "$OUT/analysis/manifest.json" "$REPORT/"
"$PY" - "$OUT/status.json" "$START" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'complete','started_unix':int(sys.argv[2]),'ended_unix':time.time()},indent=2))
PY
echo "E086 complete $(date -Is)"
