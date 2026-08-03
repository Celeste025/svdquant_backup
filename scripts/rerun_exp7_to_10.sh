#!/usr/bin/env bash
# Re-run experiments 7–10 with fixed collectors and dual MSE/NMSE metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

# --- Exp 7 round 1 ---
run exp7_wan_r1 "$WANPY" scripts/analyze_scaled_activation_quant_error.py --mode wan \
  --output outputs/activation_error_comparison/wan_metrics.json
run exp7_flux_r1 "$FLUXPY" scripts/collect_flux_dev_scaled_activation_metrics.py
run exp7_plot_r1 "$WANPY" scripts/plot_activation_error_comparison.py

# --- Exp 7 round 2 ---
run exp7_flux_r2 "$FLUXPY" scripts/collect_selected_postsmooth_activation.py --model flux
run exp7_wan_r2 "$WANPY" scripts/collect_selected_postsmooth_activation.py --model wan

# --- Exp 8 ---
run exp8_flux_single "$FLUXPY" scripts/collect_timestep_activation_outliers.py --model flux
run exp8_wan_single "$WANPY" scripts/collect_timestep_activation_outliers.py --model wan
run exp8_plot_single "$WANPY" scripts/plot_timestep_activation_outliers.py
run exp8_flux_multi "$FLUXPY" scripts/collect_multilayer_timestep_outliers.py --model flux
run exp8_wan_multi "$WANPY" scripts/collect_multilayer_timestep_outliers.py --model wan
run exp8_plot_multi "$WANPY" scripts/plot_multilayer_timestep_outliers.py
if [[ -f scripts/plot_multilayer_fig1_style.py ]]; then
  run exp8_plot_fig1 "$WANPY" scripts/plot_multilayer_fig1_style.py || true
fi

# --- Exp 9 ---
run exp9_wan_bf16 "$WANPY" scripts/collect_rollout_latent_error.py --model wan --mode bf16
run exp9_wan_quant "$WANPY" scripts/collect_rollout_latent_error.py --model wan --mode quant
run exp9_wan_forced "$WANPY" scripts/collect_rollout_latent_error.py --model wan --mode forced
run exp9_flux_bf16 "$FLUXPY" scripts/collect_rollout_latent_error.py --model flux --mode bf16
run exp9_flux_quant "$FLUXPY" scripts/collect_rollout_latent_error.py --model flux --mode quant
run exp9_flux_forced "$FLUXPY" scripts/collect_rollout_latent_error.py --model flux --mode forced
run exp9_plot "$WANPY" scripts/plot_rollout_latent_error.py

# --- Exp 10 ---
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
