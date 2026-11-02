#!/usr/bin/env bash
set -euo pipefail

cd /home/wjq/workspace/svdquant-exp

sample=/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p1_s00.pt
calibration=/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p11_s00_calib.pt
vllm_python=/data1/models/svdquant-wjq/research/envs/vllm-linear-bench-20261014/bin/python
flashinfer_python=/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python
cutlass_site=/data1/models/svdquant-wjq/research/deps/cutlass-dsl-4.6.2-cu13
result_dir=results/research/E097
log_dir=results/logs
mkdir -p "$result_dir" "$log_dir"

check_gpu() {
  local used
  used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  printf '%s GPU0 memory.used=%s MiB\n' "$(date --iso-8601=seconds)" "$used"
  if (( used > 1024 )); then
    printf 'GPU0 is occupied; stopping before the next benchmark.\n' >&2
    exit 2
  fi
}

run_stage() {
  local name=$1
  shift
  check_gpu
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$name"
  timeout 2700 "$@" 2>&1 | tee "$log_dir/e097_${name}_gpu0.log"
  printf '%s END %s\n' "$(date --iso-8601=seconds)" "$name"
}

export PATH=/home/wjq/.conda/envs/convrot-wan/bin:$PATH

run_stage vllm_int8_int4 "$vllm_python" scripts/research/bench_h3_vllm_int8_int4.py \
  --sample "$sample" --output "$result_dir/vllm_int8_int4_gpu0.json" \
  --gpu 0 --warmup 50 --batch-repeats 30 --batches 11

run_stage nvfp4_w4a4_cutlass "$flashinfer_python" scripts/research/bench_h3_vllm_style_nvfp4.py \
  --sample "$sample" --calibration "$calibration" \
  --output "$result_dir/nvfp4_w4a4_cutlass_gpu0.json" \
  --gpu 0 --backend cutlass --warmup 50 --batch-repeats 30 --batches 11

run_stage nvfp4_w4a16_cute "$flashinfer_python" scripts/research/bench_h3_flashinfer_nvfp4_w4a16.py \
  --sample "$sample" --output "$result_dir/nvfp4_w4a16_cute_gpu0.json" \
  --gpu 0 --backend cute-dsl --cutlass-site "$cutlass_site" \
  --warmup 50 --batch-repeats 30 --batches 11
