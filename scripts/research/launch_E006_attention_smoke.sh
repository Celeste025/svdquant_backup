#!/usr/bin/env bash
set -u
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export FLASHINFER_WORKSPACE_BASE=/data1/models/svdquant-wjq/research/cache/flashinfer
export TMPDIR=/data1/models/svdquant-wjq/research/cache/tmp
export TORCH_EXTENSIONS_DIR=/data1/models/svdquant-wjq/research/cache/torch_extensions
export MAX_JOBS=4
export PATH=/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin:/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:$PATH
timeout 3600 /data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python -u scripts/research/smoke_nvfp4_attention.py --output results/research/E006_attention_smoke.json > results/logs/E006_attention_smoke.log 2>&1
experiment_exit=$?
printf '\nEXIT_CODE=%s\n' "$experiment_exit" >> results/logs/E006_attention_smoke.log
exit "$experiment_exit"
