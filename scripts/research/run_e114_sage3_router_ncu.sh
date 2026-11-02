#!/usr/bin/env bash
set -euo pipefail

cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=3
mkdir -p results/research/E114
/usr/local/cuda/bin/ncu \
    --target-processes all \
    --kernel-name 'regex:route_pack_v' \
    --launch-skip 8 \
    --launch-count 1 \
    --set full \
    --force-overwrite \
    --export results/research/E114/router_h56_k9482_full \
    /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python \
    scripts/research/profile_e114_sage3_router.py
