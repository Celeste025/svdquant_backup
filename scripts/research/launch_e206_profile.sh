#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098 TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=2
export PATH="$CUDA_HOME/bin:/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
for d in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do export CPATH="$CPATH:$d"; done
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
mkdir -p results/research/E206
/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/profile_e206_probability.py
/usr/local/cuda-13.0/bin/ncu --profile-from-start off --kernel-name regex:probability_capture_kernel --launch-count 1 --section SpeedOfLight --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/research/E206/fused /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/profile_e206_probability.py --ncu
/usr/local/cuda-13.0/bin/ncu --import results/research/E206/fused.ncu-rep --csv --page details > results/research/E206/ncu_details.csv
