#!/usr/bin/env bash
set -Eeuo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
plot_py=/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python
gpu=${1:-5}
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if (( free_mib < 45000 )); then
    echo "GPU $gpu has only $free_mib MiB free; refusing shared-GPU replay" >&2
    exit 2
fi
cd "$root"
export CUDA_VISIBLE_DEVICES="$gpu"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH="$root/scripts:$root/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio"

echo "E104 start $(date -Is), GPU $gpu, free $free_mib MiB"
"$py" -u scripts/research/capture_h3_paired_output_patterns.py \
  --calls /data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases/clap0/bf16_teacher_calls.pt \
  --state results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt \
  --e101-report results/research/E101/clap0/svdquant.json \
  --raw-dir /data1/models/svdquant-wjq/research/20261016/E104_h3_clap0_output_patterns/raw \
  --manifest results/research/E104/manifest.json \
  --expected-prompt-id 161 --expected-seed 38001610
"$py" -u scripts/research/analyze_h3_output_error_patterns.py \
  --manifest results/research/E104/manifest.json \
  --summary results/research/E104/summary.json \
  --arrays results/research/E104/derived_arrays.npz
"$plot_py" -u scripts/research/render_h3_output_error_patterns.py \
  --summary results/research/E104/summary.json \
  --arrays results/research/E104/derived_arrays.npz \
  --output-dir results/research/E104/figures
echo "E104 complete $(date -Is)"
