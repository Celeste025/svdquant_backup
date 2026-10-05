#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
source results/research/E024/runtime_env.sh
export CUDA_VISIBLE_DEVICES=1 FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER FASTVIDEO_DISABLE_ATTENTION_COMPILE=0
exec > "$FASTWAN_TASK/logs/smoke_resume1.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$FASTWAN_TASK/smoke_resume1_exit_code"; date -u +%FT%TZ > "$FASTWAN_TASK/smoke_resume1_finished_utc"' EXIT
DEADLINE=1790969580
REMAINING=$(( DEADLINE - $(date +%s) ))
[[ "$REMAINING" -gt 0 ]]
timeout --signal=TERM --kill-after=10s "$REMAINING" /data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin/python scripts/research/check_fastwan_official_environment.py --phase smoke --output results/research/E024/environment_gpu_smoke_resume1.json --deadline-unix "$DEADLINE"
