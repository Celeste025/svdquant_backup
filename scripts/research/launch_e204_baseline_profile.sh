#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098 TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=2
export PATH="$CUDA_HOME/bin:/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
for d in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do export CPATH="$CPATH:$d"; done
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
/usr/local/cuda-13.0/bin/ncu --profile-from-start off --kernel-name regex:mixed_attention_sm120_q128_int8_dense_importance_kernel --launch-count 1 --section SpeedOfLight --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/research/E204/qk_max /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/profile_e204_fused_probability.py --baseline-ncu
/usr/local/cuda-13.0/bin/ncu --import results/research/E204/qk_max.ncu-rep --csv --page details > results/research/E204/qk_max_details.csv
