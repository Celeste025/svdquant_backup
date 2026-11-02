#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wjq/workspace/svdquant-exp
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
while tmux has-session -t e113_native_clap 2>/dev/null; do
    sleep 30
done
for case_id in race clap0; do
    for arm in sage3 native_fp4 native_mixed; do
        test -f "results/research/E113/${case_id}/${arm}_video.json" || {
            echo "Incomplete ${case_id}/${arm}; refusing summary" >&2
            exit 2
        }
    done
done
echo "All six attention videos complete; building paired reports $(date -Is)"
export CUDA_VISIBLE_DEVICES=3
for case_id in race clap0; do
    "$python_bin" -u scripts/research/summarize_e113_native_videos.py --case "$case_id"
    "$python_bin" -u scripts/research/evaluate_e113_native_lpips.py --case "$case_id"
done
echo "E113 native generation, montage, and LPIPS complete $(date -Is)"
