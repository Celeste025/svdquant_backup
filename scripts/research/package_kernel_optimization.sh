#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
mkdir -p results/research/E194
out=results/research/E194/kernel_optimization.patch
: > "$out"
emit_diff() {
 local before="$1" current="$2" relative="$3"
 if diff -u --label "a/$relative" --label "b/$relative" "$before" "$current" >> "$out"; then :; else
  local code=$?; if [ "$code" != 1 ]; then exit "$code"; fi
 fi
}
prefix=third_party/anemoi-code
emit_diff results/local/E170/source_before/q128_microscaling_preparation.cu "$prefix/csrc/attention/cuda/sm120/q128_microscaling_preparation.cu" csrc/attention/cuda/sm120/q128_microscaling_preparation.cu
emit_diff results/local/E170/source_before/q64_attention.cuh "$prefix/csrc/attention/cuda/sm120/q64_attention.cuh" csrc/attention/cuda/sm120/q64_attention.cuh
emit_diff results/local/E173/source_before/q64_attention_host.cu "$prefix/csrc/attention/cuda/sm120/q64_attention_host.cu" csrc/attention/cuda/sm120/q64_attention_host.cu
emit_diff results/local/E192/source_before/inst_q128_k64_d128_nv_int8_fp16.cu "$prefix/csrc/attention/cuda/sm120/instantiations/inst_q128_k64_d128_nv_int8_fp16.cu" csrc/attention/cuda/sm120/instantiations/inst_q128_k64_d128_nv_int8_fp16.cu
sha256sum "$prefix/anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so" "$out" > results/research/E194/artifact_hashes.txt
wc -l "$out"
