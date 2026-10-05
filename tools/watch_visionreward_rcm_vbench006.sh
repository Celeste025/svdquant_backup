#!/usr/bin/env bash
# Wait for a genuinely usable GPU, then run the durable VisionReward pipeline.
set -Eeuo pipefail

repo_dir=/home/wjq/workspace/svdquant-exp
out_dir="$repo_dir/results/visionreward/rcm_vbench006"
# The real smoke load reaches roughly 25 GiB; 36 GiB leaves room for the
# checkpoint transfer peak and video/KV cache while allowing busy compute GPUs.
min_free_mib=36864
max_util_pct=100
interval_seconds=300
mkdir -p "$out_dir"

while true; do
  if [ -f "$out_dir/results.json" ]; then
    echo "$(date -Is) results already present; watcher exits"
    exit 0
  fi
  candidate="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits | awk -F, -v free_needed="$min_free_mib" -v max_util="$max_util_pct" '
      {gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4);
       free=$3-$2; if (free >= free_needed && $4 <= max_util && free > best) {best=free; gpu=$1}}
      END {if (gpu != "") print gpu}' )"
  if [ -n "$candidate" ]; then
    echo "$(date -Is) launching on GPU $candidate"
    TARGET_GPU="$candidate" "$repo_dir/tools/run_visionreward_rcm_vbench006.sh"
    exit $?
  fi
  echo "$(date -Is) no GPU with >=${min_free_mib}MiB free and <=${max_util_pct}% utilization"
  sleep "$interval_seconds"
done
