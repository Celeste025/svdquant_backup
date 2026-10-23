#!/usr/bin/env bash
set -euo pipefail

source_dir=/home/wjq/workspace/svdquant-exp/third_party/sage3_native_mixed
cuda_dir=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
export E114_CUTLASS_ROOT=/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/cutlass

test -x "$cuda_dir/bin/nvcc"
test -f "$E114_CUTLASS_ROOT/include/cute/tensor.hpp"
mkdir -p "$source_dir/tmp"
export CUDA_HOME="$cuda_dir"
export PATH="$cuda_dir/bin:/home/wjq/.conda/envs/convrot-wan/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$cuda_dir/targets/x86_64-linux/include${CPATH:+:$CPATH}"
for include_dir in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do
    CPATH="$CPATH:$include_dir"
done
export CPATH
export TMPDIR="$source_dir/tmp"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
cd "$source_dir"
"$python_bin" setup_core.py build_ext --inplace --force
"$python_bin" -c 'import torch, fp4attn_mixed_cuda; print(torch.__version__, fp4attn_mixed_cuda.__file__)'
