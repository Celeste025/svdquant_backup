#!/usr/bin/env bash
set -Eeuo pipefail
root=/home/wjq/workspace/svdquant-exp
cuda_dir=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098
build_dir=/data1/models/svdquant-wjq/research/20261031/E139_draftattention2_build
export CUDA_VISIBLE_DEVICES=0
export MPA_PYTHON=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
export MPA_CUDA_HOME="$cuda_dir" CUDA_HOME="$cuda_dir"
export PATH="$cuda_dir/bin:/home/wjq/.conda/envs/convrot-wan/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$cuda_dir/targets/x86_64-linux/include${CPATH:+:$CPATH}"
for include_dir in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do
    CPATH="$CPATH:$include_dir"
done
export CPATH
export MPA_BUILD_ROOT="$build_dir" MPA_BUILD_COMPONENTS=sm120 MPA_MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=12.0a
mkdir -p "$build_dir"
bash "$root/third_party/anemoi-code/scripts/build_attention_cuda.sh"
