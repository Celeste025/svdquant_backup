#!/usr/bin/env bash
set -euo pipefail
cd /home/wjq/workspace/svdquant-exp
export CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098 TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=2
export PATH="$CUDA_HOME/bin:/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
for d in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do export CPATH="$CPATH:$d"; done
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
target=third_party/anemoi-code/csrc/attention/cuda/sm120
trap 'cp /tmp/e206_candidate.cuh "$target/q64_attention.cuh"; cp /tmp/e206_candidate.inl "$target/q64_attention_phase_composer.inl"' EXIT
for variant in A1 B A2; do
 if [[ "$variant" == B ]]; then src=e206_candidate; else src=e206_base; fi
 cp "/tmp/${src}.cuh" "$target/q64_attention.cuh"
 cp "/tmp/${src}.inl" "$target/q64_attention_phase_composer.inl"
 /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/bench_e206_aba.py "results/research/E206/aba_${variant}.json"
done
