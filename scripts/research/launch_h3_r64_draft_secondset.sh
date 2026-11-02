#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/wjq/workspace/svdquant-exp
gpu=${1:?gpu}; variant=${2:?variant}; pid=${3:?pid}; phase=${4:?phase}
export CUDA_VISIBLE_DEVICES="$gpu" DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SVDQUANT_DATA_ROOT=/data1/models/svdquant-wjq DIFFSYNTH_ROOT=/home/wjq/workspace/DiffSynth-Studio
export MINIMAX_H3_MODEL_ROOT=/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3
export MINIMAX_H3_DIT_PATH=/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
export MINIMAX_H3_PROMPT_FILE=/home/wjq/workspace/178866172854036
export PYTHONPATH=/home/wjq/workspace/svdquant-exp/scripts:/home/wjq/workspace/svdquant-exp/scripts/research:/home/wjq/workspace/svdquant-exp/third_party/deepcompressor:/home/wjq/workspace/DiffSynth-Studio:/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages:/data1/models/svdquant-wjq/research/deps/cutlass-dsl-4.6.2-cu13:/data1/models/svdquant-wjq/research/deps/cutlass-dsl-4.6.2-cu13/nvidia_cutlass_dsl/dsl_packages
export PATH=/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=4
export FLASHINFER_WORKSPACE_BASE=/tmp/h3_fused_flashinfer
exec /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python -u scripts/research/run_h3_r64_draft_secondset.py --variant "$variant" --pid "$pid" --phase "$phase"
