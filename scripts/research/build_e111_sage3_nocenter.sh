#!/usr/bin/env bash
set -euo pipefail

probe_dir=${1:?pass an isolated Sage3 source directory}
cuda_dir=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098
python_bin=/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python
nvidia_lib=/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/lib/python3.12/site-packages/nvidia

test -f "$probe_dir/sageattn3_prescale/blackwell/mainloop_tma_ws.h"
test -x "$cuda_dir/bin/nvcc"
export CUDA_HOME="$cuda_dir"
export PATH="$cuda_dir/bin:$PATH"
export LD_LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$cuda_dir/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$cuda_dir/targets/x86_64-linux/include${CPATH:+:$CPATH}"
for include_dir in "$nvidia_lib"/*/include; do
  CPATH="$CPATH:$include_dir"
done
export CPATH
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2

cd "$probe_dir"
"$python_bin" setup.py build_ext --inplace --force
"$python_bin" -c 'import torch, fp4attn_cuda; print(torch.__version__, fp4attn_cuda.__file__)'
