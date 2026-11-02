#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
mkdir -p results/research/E192
export CUDA_VISIBLE_DEVICES=1
ncu=/usr/local/cuda-13.0/bin/ncu
"$ncu" --kernel-name 'regex:mixed_attention_sm120_q128_nv_int8_fp16_kernel' \
  --launch-count 1 --section SpeedOfLight --section SourceCounters --section LaunchStats --section Occupancy \
  --force-overwrite --export results/research/E192/90_10_480p_source \
  /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python \
  scripts/research/validate_e169_480p_paths.py \
  --frames 124 --query-block 128 --skip-fraction 0 --nvfp4-fraction 0.9 \
  --warmup 0 --repeats 1 --output results/research/E192/profile_call.json
"$ncu" --import results/research/E192/90_10_480p_source.ncu-rep --page details > results/research/E192/details.txt
"$ncu" --import results/research/E192/90_10_480p_source.ncu-rep --page source --print-source cuda,sass > results/research/E192/source.txt
