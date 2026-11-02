#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_e114_sage3_native_core_ncu.sh official_fp4|copied_fp4|native_mixed}
case "$mode" in
    official_fp4|copied_fp4|native_mixed) ;;
    *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/research/E114
/usr/local/cuda/bin/ncu \
    --target-processes all \
    --kernel-name 'regex:compute_attn_ws' \
    --launch-skip 8 \
    --launch-count 1 \
    --set full \
    --force-overwrite \
    --export "results/research/E114/${mode}_h56_k9482_full" \
    /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python \
    scripts/research/profile_e114_sage3_native_core.py --mode "$mode"
