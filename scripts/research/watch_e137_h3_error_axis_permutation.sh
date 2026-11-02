#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
budget_seconds=${1:-7200}
started=$(date +%s)
deadline=$((started + budget_seconds))
log_dir="$root/results/logs"
mkdir -p "$log_dir"

find_idle_gpu() {
    local gpu free apps
    for gpu in 0 1 2 3 4 5 6 7; do
        free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
        apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
        if (( free >= 60000 )) && ! [[ $apps =~ [0-9] ]]; then
            printf '%s\n' "$gpu"
            return 0
        fi
    done
    return 1
}

run_arm() {
    local case_id=$1 arm=$2 gpu=$3 attempt=$4
    printf 'E137 launching %s/%s (%s) on GPU %s at %s\n' "$case_id" "$arm" "$attempt" "$gpu" "$(date -Is)" | tee -a "$log_dir/e137_axis_watch.log"
    E137_ATTEMPT="$attempt" "$root/scripts/research/launch_e137_h3_error_axis_permutation.sh" "$case_id" "$gpu" "$arm" \
        2>&1 | tee "$log_dir/e137_axis_${case_id}_${arm}_${attempt}.log"
}

# race/spatial_shared completed validly in attempt2.  Queue only missing arms.
tasks=("race temporal_frame" "clap0 spatial_shared" "clap0 temporal_frame")
attempt_no=3
for task in "${tasks[@]}"; do
    read -r case_id arm <<<"$task"
    while :; do
        now=$(date +%s)
        remaining=$((deadline - now))
        if (( remaining <= 0 )); then
            printf 'E137 budget expired before %s at %s\n' "$case_id" "$(date -Is)" | tee -a "$log_dir/e137_axis_watch.log"
            exit 0
        fi
        if (( remaining < 1050 )); then
            printf 'E137 leaves %s seconds, not starting another 20-step arm\n' "$remaining" | tee -a "$log_dir/e137_axis_watch.log"
            exit 0
        fi
        if gpu=$(find_idle_gpu); then
            attempt="attempt${attempt_no}_${case_id}_${arm}"
            if run_arm "$case_id" "$arm" "$gpu" "$attempt"; then
                /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python \
                    "$root/scripts/research/summarize_e137_axis_permutation.py" \
                    --attempt "$attempt" --case "$case_id" \
                    > "$log_dir/e137_axis_${case_id}_${arm}_${attempt}_summary.log"
                break
            fi
            printf 'E137 %s/%s stopped without a result; retaining %s and retrying later\n' "$case_id" "$arm" "$attempt" | tee -a "$log_dir/e137_axis_watch.log"
            attempt_no=$((attempt_no + 1))
        fi
        printf 'E137 waits for an idle >=60GiB GPU; %s seconds remain\n' "$remaining" >> "$log_dir/e137_axis_watch.log"
        sleep 30
    done
done
printf 'E137 all requested cases complete at %s\n' "$(date -Is)" | tee -a "$log_dir/e137_axis_watch.log"
