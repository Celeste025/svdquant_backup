# E221: three additional four-video experiments

Same four prompts, seed, initial noise, BF16 linears, 768x1344, 124/243 frames, 20 steps and E150 BF16 references as E202. One-based steps. Refiner BF16. Metrics are all-frame RGB PSNR and LPIPS Alex.

- refresh64: Q64xK64, main block 0 BF16 as E202; blocks 1–48 dense INT8 on steps 1, 2 and 10, probability mean capture at steps 2 and 10 using fused INT8 kernel, output dense INT8 on capture steps; steps 3–9 and 11–20 reuse prior capture, 10% INT8 and 90% NVFP4, zero skip. Block 49 dense INT8 steps 1–2, thereafter original Q64 DraftAttention2 20% INT8, 80% skip, no capture.
- sparse128: Q128 original baseline all 50 blocks, steps 1–2 dense INT8 with no skip; steps 3–20 75% skip, 10% INT8 and 15% NVFP4 overall (retained ratios 40%/60%).
- sparse_refresh64: same as refresh64, except reuse steps on blocks 1–48 use 75% skip, 10% INT8 and 15% NVFP4 overall. Dense INT8 captures at steps 2 and 10 remain dense. Last block remains 20% INT8 and 80% skip.

Per-step timings are single runs. Captures use the fused Q64 probability-mean experimental path, not the E202 two-pass capture. All outputs and metrics are stored under E221, separate from earlier incomplete attempts.
