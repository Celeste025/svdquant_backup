#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
source results/research/E024/runtime_env.sh
export CUDA_VISIBLE_DEVICES=1 FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER FASTVIDEO_DISABLE_ATTENTION_COMPILE=0
exec > "$FASTWAN_TASK/logs/smoke.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$FASTWAN_TASK/smoke_exit_code"; date -u +%FT%TZ > "$FASTWAN_TASK/smoke_finished_utc"' EXIT
DEADLINE=$(( $(date +%s) + 600 ))
timeout --signal=TERM --kill-after=10s 600s /data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin/python scripts/research/check_fastwan_official_environment.py --phase smoke --output results/research/E024/environment_gpu_smoke.json --deadline-unix "$DEADLINE"
