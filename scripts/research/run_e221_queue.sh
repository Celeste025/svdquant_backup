#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
gpu="$1"; shift
export CUDA_VISIBLE_DEVICES="$gpu" DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export CUDA_HOME=/data1/models/svdquant-wjq/research/envs/cuda-toolkit-12p8-e098 TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=2
export PATH="$CUDA_HOME/bin:/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin:/home/wjq/.conda/envs/convrot-wan/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
for d in /home/wjq/.conda/envs/convrot-wan/lib/python3.12/site-packages/nvidia/*/include; do export CPATH="$CPATH:$d"; done
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
py=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
for job in "$@"; do
 policy="${job%:*}"; pid="${job#*:}"; log="results/logs/E221_${policy}_p${pid}.log"
 if (date -Is && "$py" -u scripts/research/run_e221_four_video.py --prompt-id "$pid" --policy "$policy" --phase denoise && "$py" -u scripts/research/run_e221_four_video.py --prompt-id "$pid" --policy "$policy" --phase decode && "$py" -u scripts/research/evaluate_e221.py --prompt-id "$pid" --policy "$policy" && date -Is) >"$log" 2>&1; then :; else printf 'FAILED %s gpu %s\n' "$job" "$gpu" >> "$log"; fi
done
