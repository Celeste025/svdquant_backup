#!/usr/bin/env bash
set -euo pipefail

source_dir=/data1/models/svdquant-wjq/third_party/SageAttention-official-20261004
target_dir=/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python

mkdir -p "$target_dir"
export TORCH_CUDA_ARCH_LIST=12.0
export EXT_PARALLEL=2
export MAX_JOBS=4
export CUDA_HOME=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098
export PATH=$CUDA_HOME/bin:/home/wjq/.conda/envs/convrot-wan/bin:$PATH
torch_site=/home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
for include_dir in "$torch_site"/nvidia/*/include; do
  CPATH=$CPATH:$include_dir
done
export CPATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

"$python_bin" -m pip install --no-build-isolation --no-deps --target "$target_dir" "$source_dir"
PYTHONPATH="$target_dir" "$python_bin" -c 'import sageattention; from sageattention import sageattn; print(sageattention.__file__, sageattn.__module__)'
