#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wjq/workspace/svdquant-exp
while tmux has-session -t e113_native_race 2>/dev/null; do
    sleep 30
done
for arm in sage3 native_fp4 native_mixed; do
    test -f "results/research/E113/race/${arm}_video.json" || {
        echo "Race did not finish all three arms; clap launch withheld" >&2
        exit 2
    }
done
echo "Race media complete; starting clap $(date -Is)" >&2
bash scripts/research/launch_e113_h3_mixed_pv_video.sh clap0 3 all
