# E159 — first-step exact-route reuse versus per-step DraftMap

Status: complete

Existing E150 BF16 final latents are reused. Prompt, seed, resolution, frames, scheduler settings and initial video/audio latent hashes are checked for each arm. Checkpoint hashes are verified. No BF16 regeneration.

Both arms: BF16 linears, Q128, no skipped tiles, NVFP4 90% / INT8 10%; step 0 dense INT8 except layer 0 BF16. Layer 0 remains BF16 for all steps. Steps 1–19: proxy rebuilds DraftMap; reuse uses per-layer exact QK tile-max routes captured at step 0. Text prefix uses dense INT8; refiner and padding remain BF16.

768 x 1344, 20 steps, CFG 1, flow shift 12, audio flow shift 3, CPU RNG. E150 frame alignment is preserved. GPUs 2 and 3 each process two cases, two arms per case.

| Case | Frames | Seed | Proxy NMSE | Reuse NMSE | Reuse change |
|---|---:|---:|---:|---:|---:|
| Cats playing Mahjong (5s) | 124 | 2141 | 0.26680205 | 0.31144487 | +16.73% |
| Kayaking (5s) | 124 | 9939 | 0.11023886 | 0.10417085 | -5.50% |
| Swimming (10s) | 243 | 17080 | 0.09607190 | 0.14559078 | +51.54% |
| Astronaut on Mars (10s) | 243 | 59417 | 0.23930129 | 0.25086102 | +4.83% |

NMSE = sum((final video latent - BF16 latent)^2) / sum(BF16 latent^2). Lower is better. This measures final latent agreement, not perceptual video quality. Four cases and one seed each do not establish population-wide statistical superiority.

Pending arms: 0

Artifacts: comparison.json, comparison.csv, aligned_v2/{proxy,reuse}/pXXX/bf16_denoise.json. The inherited report filename is retained; mode and attention_router identify the actual arm. Checkpoints are local under results/local/E159/aligned_v2.
