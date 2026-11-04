# E219: four-video protected baseline and probability refresh

E150/E202 matched prompts 25, 48, 44, 7; 768x1344, 124/243 frames, 20 steps, BF16 linears, same seeds and initial audio/video noise, all-frame RGB PSNR and LPIPS Alex. One-based step numbers. Refiner BF16 throughout.

- protect128: E162 all 50 blocks baseline, Q128 original DraftMap, except first two steps dense INT8.
- protect64: same with Q64 original DraftMap.
- refresh64: layer 0 BF16 as E202; other layers dense INT8 steps 1-2, capture fused INT8 probability mean Q64xK64 at steps 2,8,14 for layers 1-48. Steps 3-20 use 10% INT8/90% NVFP4 with captured routes. At steps 8,14 run mixed attention using the previously cached route, then capture the current map for use starting at the next step. The current implementation is a sequential correctness prototype, not the requested optimized kernel; do not run the third video group until performance is evaluated. Layer 49: dense INT8 steps 1-2, original Q64 DraftAttention2 20% INT8 and 80% skipped thereafter; never captures or reuses.

Capture API produces dense INT8 output; refreshing and then producing mixed output requires two attention calls. Timing includes both. GPU logs and per-step reports retained. All comparisons are four selected videos; timing is not a controlled multi-run benchmark.
