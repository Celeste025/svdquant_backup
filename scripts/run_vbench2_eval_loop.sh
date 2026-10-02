#!/usr/bin/env bash
# Per-dimension sequential eval loop for one H3 vbench2 variant.
set -u
GPU=$1; VAR=$2
MD=/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100
VB2=/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup/third_party/VBench/VBench-2.0
PY=/home/admin/workspace/aop_lab/app_source/envs/vbench2/bin/python
RUNNER=/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup/scripts/run_minimax_h3_vbench2_eval.py
DIMS="Camera_Motion Multi-View_Consistency Human_Identity Human_Anatomy Material Human_Clothes Composition Dynamic_Attribute Dynamic_Spatial_Relationship Motion_Rationality Mechanics Thermotics Complex_Landscape Complex_Plot Human_Interaction Motion_Order_Understanding Instance_Preservation"
cd $VB2
for DIM in $DIMS; do
  if [ -f "$MD/eval_results/$VAR/${VAR}_${DIM}_eval_results.json" ]; then
    echo "[$VAR] skip $DIM"
    continue
  fi
  echo "[$VAR] start $DIM $(date '+%H:%M:%S')"
  env CUDA_VISIBLE_DEVICES=$GPU VBENCH2_CACHE_DIR=/home/admin/workspace/aop_lab/app_data/cache/vbench2 \
      HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONPATH=$VB2 \
      $PY $RUNNER --variant $VAR --dimensions $DIM 2>&1 | tail -5
  if [ ! -f "$MD/eval_results/$VAR/${VAR}_${DIM}_eval_results.json" ]; then
    echo "[$VAR] FAILED $DIM"
  fi
done
echo "[$VAR] EVAL_ALL_DONE"
