#!/usr/bin/env bash
# Queue the H3 activation survey until a GPU has enough free VRAM.
#
# Phase 1 (up to 4 h): wait for a mostly idle card and run with the default
# DiffSynth offload budget.  Phase 2: accept a partly free card, but cap the
# device-wide VRAM ceiling so we cannot push a co-tenant into OOM.
set -uo pipefail

PROJECT_ROOT="/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup"
PYTHON="/home/admin/workspace/aop_lab/app_source/envs/minimax-h3/bin/python"
DATA_ROOT="/home/admin/workspace/aop_lab/app_data"
OUTPUT_DIR="$DATA_ROOT/runs/h3-activation-outliers"
LOG="$OUTPUT_DIR/survey.log"

GPU="$1"
PREFERRED_FREE_MIB="${2:-24576}"
MIN_FREE_MIB="${3:-12288}"
POLL_SECONDS="${4:-60}"

total_mib="$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
echo "queued: GPU $GPU total=${total_mib} MiB, want >=${PREFERRED_FREE_MIB} MiB free" | tee -a "$LOG"

# Runs in the foreground: the pipeline is not a subshell replacement, so callers
# must stop the waiter themselves after this returns exactly once.
run_survey() {
  local extra="$1 ${SURVEY_EXTRA_ARGS:-}"
  cd "$PROJECT_ROOT" || return 1
  env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false \
    SVDQUANT_DATA_ROOT="$DATA_ROOT" \
    DIFFSYNTH_ROOT="$PROJECT_ROOT/third_party/DiffSynth-Studio" \
    PYTHONPATH="$PROJECT_ROOT/scripts:$PROJECT_ROOT/third_party/deepcompressor" \
    "$PYTHON" scripts/survey_h3_activation_outliers.py --output-dir "$OUTPUT_DIR" $extra 2>&1 | tee -a "$LOG"
}

for phase in preferred minimum; do
  if [ "$phase" = preferred ]; then
    needed="$PREFERRED_FREE_MIB"
    polls=$((4 * 3600 / POLL_SECONDS))
  else
    needed="$MIN_FREE_MIB"
    polls=$((12 * 3600 / POLL_SECONDS))
  fi
  for _ in $(seq 1 "$polls"); do
    used_mib="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    free_mib=$((total_mib - used_mib))
    if [ "$free_mib" -ge "$needed" ]; then
      if [ "$phase" = preferred ]; then
        echo "starting ($phase): GPU $GPU used=${used_mib} MiB free=${free_mib} MiB (default vram limit) at $(date -Is)" | tee -a "$LOG"
        run_survey ""
        echo "survey finished at $(date -Is)" | tee -a "$LOG"
        exit 0
      fi
      limit_gib="$(awk "BEGIN{printf \"%.1f\", ($used_mib + 8192) / 1024}")"
      echo "starting ($phase): GPU $GPU used=${used_mib} MiB free=${free_mib} MiB vram_limit=${limit_gib} GiB at $(date -Is)" | tee -a "$LOG"
      run_survey "--vram-limit-gib $limit_gib"
      echo "survey finished at $(date -Is)" | tee -a "$LOG"
      exit 0
    fi
    echo "waiting ($phase): GPU $GPU free=${free_mib} MiB at $(date -Is)" | tee -a "$LOG"
    sleep "$POLL_SECONDS"
  done
done

echo "gave up after waiting for GPU $GPU" | tee -a "$LOG"
