#!/bin/bash
# SageAttn3 (fixed wrapper) 8-case comparison: 1-GPU smoke first, then 4-GPU shard run.
BASE=/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100
LOGS=/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup/results/logs
PY=/home/admin/workspace/aop_lab/app_source/envs/minimax-h3/bin/python
WORKER=/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup/scripts/minimax_h3_vbench2_worker.py

mode="$1"

if [ "$mode" = "smoke" ]; then
  cd /tmp && CUDA_VISIBLE_DEVICES=0 "$PY" "$WORKER" \
    --manifest "$BASE/manifest.json" \
    --output "$BASE/sageattn3_fixed_smoke" \
    --variant svdquant --attention sageattn3 \
    --gpu-physical 0 --reserve-gib 16 --smoke \
    > "$LOGS/h3-sageattn3-fixed-smoke.log" 2>&1
  echo "smoke exit=$?"
elif [ "$mode" = "cases8" ]; then
  for i in 0 1 2 3; do
    skip=$((i*2))
    cd /tmp && CUDA_VISIBLE_DEVICES=$i "$PY" "$WORKER" \
      --manifest "$BASE/manifest.json" \
      --output "$BASE/sageattn3_cases8/gpu$i" \
      --variant svdquant --attention sageattn3 \
      --gpu-physical $i --reserve-gib 16 --skip $skip --limit 2 \
      > "$LOGS/h3-sageattn3-cases8-gpu$i.log" 2>&1 &
  done
  wait
  echo "cases8 all done"
else
  echo "usage: $0 {smoke|cases8}" >&2
  exit 1
fi
