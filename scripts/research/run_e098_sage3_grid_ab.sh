#!/usr/bin/env bash
set -euo pipefail

original=/data1/models/svdquant-wjq/third_party/SageAttention-official-20261004/sageattention3_blackwell
probe=/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe
python_bin=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
script=scripts/research/bench_h3_sage_two_scope.py

test -f "$original/fp4attn_cuda.cpython-312-x86_64-linux-gnu.so"
test -f "$probe/fp4attn_cuda.cpython-312-x86_64-linux-gnu.so"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

"$python_bin" "$script" --sage3-dir "$original" \
  --output results/research/E098/sage3_grid_ab_original_1_gpu0.json
"$python_bin" "$script" --sage3-dir "$probe" \
  --output results/research/E098/sage3_grid_ab_sm110_gpu0.json
"$python_bin" "$script" --sage3-dir "$original" \
  --output results/research/E098/sage3_grid_ab_original_2_gpu0.json
