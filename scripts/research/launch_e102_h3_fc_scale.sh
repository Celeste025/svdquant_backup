#!/usr/bin/env bash
set -euo pipefail

root=/home/wjq/workspace/svdquant-exp
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
gpu=${1:-4}
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
if (( free_mib < 45000 )); then
    echo "GPU $gpu has only $free_mib MiB free; refusing shared-GPU run" >&2
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

echo "E102 start $(date -Is), physical GPU $gpu, free ${free_mib} MiB"
"$py" scripts/research/e102_h3_local_global_scale.py \
    --calls /data1/models/svdquant-wjq/research/20261015/E100_h3_svd_fixed1_p1/bf16_teacher_calls.pt \
    --state results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt \
    --layer blocks.0.mlp.fc1 \
    --capture /data1/models/svdquant-wjq/research/20261015/E102_h3_fc_scale/p1_step0_b0_fc1_input_weight.pt \
    --output results/research/E102/p1_step0_b0_fc1.json
echo "E102 complete $(date -Is)"
