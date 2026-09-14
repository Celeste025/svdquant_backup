#!/usr/bin/env bash
set -euo pipefail

selection=${1:?usage: run_rcm_signed_restore_sweep.sh signed-top|signed-bottom}
if [[ "$selection" != "signed-top" && "$selection" != "signed-bottom" ]]; then
  echo "selection must be signed-top or signed-bottom" >&2
  exit 2
fi

env_path=/data/models/svdquant-wjq/conda-envs/svdquant-ptq
output_root=results/samples/rcm_restore_video_comparison_all
latent_root=results/reports/rcm_signed_restore_top_bottom/latents
mkdir -p "$output_root" "$latent_root"

names=(airplane person fastmotion watch)
seeds=(303 307 308 309)
prompts=(
  "An airplane soaring through a clear blue sky above white clouds."
  "A woman in a bright yellow coat walking toward the camera through a rainy city street at night, cinematic close-up."
  "A professional skateboarder performing a high-speed kickflip down a concrete stair set, dynamic tracking shot."
  "Macro cinematic close-up of a luxury mechanical wristwatch, intricate engraved gears and moving second hand, sharp highlights."
)

label=${selection//-/_}
for index in "${!names[@]}"; do
  case_name=${names[$index]}_seed${seeds[$index]}
  conda run --no-capture-output -p "$env_path" python scripts/exp_rcm_restore_top_fraction_bf16.py \
    --selection "$selection" \
    --top-fraction 0.10 \
    --prompt "${prompts[$index]}" \
    --seed "${seeds[$index]}" \
    --output "$output_root/${case_name}__${label}.mp4" \
    --latent-output "$latent_root/${case_name}__${label}.pt"
done
