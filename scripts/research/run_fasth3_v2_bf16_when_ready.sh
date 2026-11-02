#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT=/data1/models/svdquant-wjq/models/fastvideo-v2-triad
MODEL_FILE="$ASSET_ROOT/bf16/diffusion_models/fastvideo_fasth3_8step_v2_pruned_bf16.safetensors"
MANIFEST_FILE="$ASSET_ROOT/bf16/download_manifest.json"
RUNNER=/home/wjq/workspace/svdquant-exp/scripts/research/run_fasth3_v2_comfy_case.py
PROMPT_SOURCE=/home/wjq/workspace/178866172854036
COMFY_ROOT=/data1/models/svdquant-wjq/comfyui
REPORT_ROOT=/home/wjq/workspace/svdquant-exp/results/research/fasth3_v2_triad
PYTHON=/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python

for attempt in $(seq 1 360); do
    if test -s "$MANIFEST_FILE" && test -s "$MODEL_FILE"; then
        break
    fi
    if test "$attempt" -eq 360; then
        echo "BF16 checkpoint not verified after 3 hours" >&2
        exit 1
    fi
    sleep 30
done

for prompt_id in 16 11 51; do
    padded_id=$(printf '%03d' "$prompt_id")
    "$PYTHON" -u "$RUNNER" \
        --arm bf16 \
        --prompt-source "$PROMPT_SOURCE" \
        --prompt-id "$prompt_id" \
        --asset-root "$ASSET_ROOT" \
        --comfy-root "$COMFY_ROOT" \
        --report "$REPORT_ROOT/p${padded_id}_bf16_aligned.json"
done
