#!/usr/bin/env bash
# Capture missing E098 details only after a truly process-free GPU appears.
set -euo pipefail

cd /home/wjq/workspace/svdquant-exp
raw_dir=/data1/models/svdquant-wjq/research/20261014/E098
py_bin=/home/wjq/.conda/envs/convrot-wan/bin/python3.12
deadline=$((SECONDS + 7200))

gpu_is_free() {
    local gpu_id=$1
    local active_used active_percent apps
    apps=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$apps" ]] || return 1
    IFS=, read -r active_percent active_used < <(
        nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
    )
    active_percent=${active_percent//[[:space:]]/}
    active_used=${active_used//[[:space:]]/}
    [[ $active_percent =~ ^[0-9]+$ && $active_used =~ ^[0-9]+$ ]] || return 1
    (( active_percent < 5 && active_used < 1024 ))
}

gpu_id=
while (( SECONDS < deadline )); do
    for candidate in {0..7}; do
        if gpu_is_free "$candidate"; then
            sleep 10
            if gpu_is_free "$candidate"; then
                gpu_id=$candidate
                break
            fi
        fi
    done
    [[ -n $gpu_id ]] && break
    sleep 60
done

if [[ -z $gpu_id ]]; then
    echo 'TIMEOUT: no process-free GPU appeared within two hours; retained existing reports are still valid.'
    exit 0
fi
echo "START: GPU $gpu_id is process-free; $(date -Is)"

ncu --kernel-name regex:qk_int_sv_f8_attn_kernel \
    --launch-skip 8 --launch-count 1 \
    --section SourceCounters --metrics smsp__pcsamp_sample_count \
    -o "$raw_dir/h3_sage2pp_core_source" \
    "$py_bin" scripts/research/profile_e098_sage_core_ncu.py --arm sage2pp --gpu "$gpu_id"

nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,osrt \
    --sample=none --cpuctxsw=none --cuda-memory-usage=true \
    --force-overwrite=false \
    -o "$raw_dir/h3_sage2pp_sage3_official_api_timeline" \
    "$py_bin" scripts/research/profile_e098_sage_nsys.py --gpu "$gpu_id" --repeats 3

python scripts/research/export_e098_sage_ncu.py
echo "COMPLETE: source counters and API timeline saved; $(date -Is)"
