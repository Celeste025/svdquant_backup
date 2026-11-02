#!/usr/bin/env bash
set -euo pipefail

probe_dir=/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe
cuda_dir=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python

test -f "$probe_dir/sageattn3/blackwell/launch.h"
test -x "$cuda_dir/bin/nvcc"
mkdir -p "$probe_dir/tmp"
export CUDA_HOME="$cuda_dir"
export PATH="$cuda_dir/bin:/home/wjq/.conda/envs/convrot-wan/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$cuda_dir/targets/x86_64-linux/include${CPATH:+:$CPATH}"
for include_dir in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do
  CPATH="$CPATH:$include_dir"
done
export CPATH
export TMPDIR="$probe_dir/tmp"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2

cd "$probe_dir"
"$python_bin" setup.py build_ext --inplace --force
"$python_bin" -c 'import torch, fp4attn_cuda; print(torch.__version__, fp4attn_cuda.__file__)'
