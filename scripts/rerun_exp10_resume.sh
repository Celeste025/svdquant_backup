#!/usr/bin/env bash
# Resume from Exp10 after weight-script env mismatch.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WANPY=/data/home/jinqiwen/miniconda3/envs/wan-qvdit/bin/python
FLUXPY=/data/home/jinqiwen/miniconda3/envs/svdquant/bin/python
TORCHLIB=/data/home/jinqiwen/miniconda3/envs/svdquant/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="${TORCHLIB}:${LD_LIBRARY_PATH:-}"

mkdir -p logs/exp7_10_rerun

run() {
  local tag="$1"; shift
  echo "===== $(date -Is) START: $tag =====" | tee -a "logs/exp7_10_rerun/master.log"
  "$@" 2>&1 | tee "logs/exp7_10_rerun/${tag}.log"
  local rc=${PIPESTATUS[0]}
  echo "===== $(date -Is) DONE: $tag rc=$rc =====" | tee -a "logs/exp7_10_rerun/master.log"
  return $rc
}

run exp10_weight_wan "$WANPY" scripts/analyze_selected_weight_reconstruction.py --model wan
run exp10_weight_flux "$FLUXPY" scripts/analyze_selected_weight_reconstruction.py --model flux
run exp10_wan_block_bf16 "$WANPY" scripts/collect_wan_firststep_block_error.py --mode bf16
run exp10_wan_block_quant "$WANPY" scripts/collect_wan_firststep_block_error.py --mode quant
run exp10_wan_dense_bf16 "$WANPY" scripts/collect_wan_firststep_dense_outputhead.py --mode bf16
run exp10_wan_dense_quant "$WANPY" scripts/collect_wan_firststep_dense_outputhead.py --mode quant
run exp10_flux_block_bf16 "$FLUXPY" scripts/collect_flux_firststep_block_error.py --mode bf16
run exp10_flux_block_quant "$FLUXPY" scripts/collect_flux_firststep_block_error.py --mode quant
run exp10_plot "$WANPY" scripts/plot_quant_error_diagnosis.py

echo "ALL DONE $(date -Is)" | tee -a logs/exp7_10_rerun/master.log
