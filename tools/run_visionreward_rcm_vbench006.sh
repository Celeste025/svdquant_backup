#!/usr/bin/env bash
# Durable GPU-2 VisionReward evaluation for the rCM vbench_006 samples.
set -Eeuo pipefail

repo_dir=/home/wjq/workspace/svdquant-exp
model_dir=/data1/models/svdquant-wjq/models/VisionReward-Video
vr_dir=/data1/models/svdquant-wjq/third_party/VisionReward
py=/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python
out_dir="$repo_dir/results/visionreward/rcm_vbench006"
runner="$repo_dir/tools/evaluate_visionreward_video.py"
target_gpu="${TARGET_GPU:?set TARGET_GPU to the physical GPU index before launch}"
mkdir -p "$out_dir"

# The downloader writes files atomically; wait for all six model shards.
while [ "$(find "$model_dir" -maxdepth 1 -name 'model-*-of-00006.safetensors' -type f | wc -l)" -ne 6 ]; do
  date -Is | xargs -I{} echo "{} waiting for VisionReward model shards"
  sleep 30
done

common=(
  --model-path "$model_dir"
  --questions "$vr_dir/VisionReward_Video/VisionReward_video_qa_select.txt"
  --weights "$vr_dir/VisionReward_Video/weight.json"
)
bf16="$repo_dir/results/samples/rcm_vbench251_seed0_480p77f_4step/cases/vbench_006/bf16.json"

CUDA_VISIBLE_DEVICES="$target_gpu" "$py" "$runner" "${common[@]}" \
  --case-json "$bf16" --max-questions 1 --output "$out_dir/smoke.json"

CUDA_VISIBLE_DEVICES="$target_gpu" "$py" "$runner" "${common[@]}" \
  --case-json "$bf16" \
  --case-json "$repo_dir/results/samples/rcm_vbench251_seed0_480p77f_4step/cases/vbench_006/nvfp4.json" \
  --case-json "$repo_dir/results/samples/rcm_vbench251_seed0_480p77f_4step/cases/vbench_006/nvfp4_svdquant.json" \
  --case-json "$repo_dir/results/samples/rcm_int4_svdquant_vbench51_seed0_480p77f_4step/cases/vbench_006/int4_plain.json" \
  --case-json "$repo_dir/results/samples/rcm_int4_svdquant_vbench51_seed0_480p77f_4step/cases/vbench_006/int4_svdquant.json" \
  --case-json "$repo_dir/results/samples/rcm_real_nvfp4_svdquant_g20_r32_vbench51_seed0_480p77f_4step/cases/vbench_006/nvfp4_svdquant.json" \
  --case-json "$repo_dir/results/samples/rcm_real_nvfp4_svdquant_g10_r64_vbench51_seed0_480p77f_4step/cases/vbench_006/nvfp4_svdquant.json" \
  --output "$out_dir/results.json"
