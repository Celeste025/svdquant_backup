#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup"
PYTHON="/home/admin/workspace/aop_lab/app_source/envs/svdquant-ptq/bin/python"
RUNNER="$PROJECT_ROOT/scripts/run_visionreward_q64_all.py"
OUTPUT_ROOT="/home/admin/workspace/aop_lab/app_data/runs/visionreward-q64"

for command in tmux nvidia-smi; do
    command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 1; }
done
[ -x "$PYTHON" ] || { echo "missing Python: $PYTHON" >&2; exit 1; }

for gpu in 0 2 3; do
    used_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [ "$used_mib" -ge 1024 ]; then
        echo "GPU $gpu is using ${used_mib} MiB; Q64 requires an idle GPU" >&2
        exit 1
    fi
done

for session in visionreward-q64-gpu0 visionreward-q64-gpu2 visionreward-q64-gpu3; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 1
    fi
done

"$PYTHON" "$RUNNER" --prepare --output-root "$OUTPUT_ROOT"

physical_gpus=(0 2 3)
for shard in 0 1 2; do
    gpu="${physical_gpus[$shard]}"
    session="visionreward-q64-gpu${gpu}"
    log="$OUTPUT_ROOT/logs/shard${shard}.log"
    command="cd '$PROJECT_ROOT' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 '$PYTHON' '$RUNNER' --worker --gpu $gpu --shard $shard --num-shards 3 --output-root '$OUTPUT_ROOT' >> '$log' 2>&1"
    tmux new-session -d -s "$session" "$command"
    echo "started $session: shard=$shard physical_gpu=$gpu log=$log"
done

echo "Q64 workers started. Attach with: tmux attach -t visionreward-q64-gpu0"
