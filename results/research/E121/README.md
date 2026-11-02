# E121 — split mixed Sage3 in real MiniMax-H3 generation

> **Baseline correction (E122 follow-up):** The arm named `bf16` below used
> DiffSynth's default `sage_attention` backend, which internally quantizes Q/K
> to INT8 and V to FP16. Its model weights and helper inputs/outputs were
> BF16, but its attention arithmetic was **not BF16 SDPA**. Therefore the
> speedups and NMSE values below are relative to that default SageAttention
> path, not a true BF16-attention baseline. The original records are retained
> for audit; E122 re-profiles against forced PyTorch BF16 SDPA.

## Result

The corrected E120 split P8×V8 operator works in the 50 main attention calls
of a full 20-step MiniMax-H3 generation. On one RTX PRO 5000 72GB, the
observed **warm-cache** denoise-plus-decode wall time was **202.21 s** for the
split operator, **199.86 s** for corrected pure Sage3, and **217.77 s** for
BF16 attention. Thus split was **7.69% faster than warm BF16** (1.0769×
speedup) and **1.18% slower than corrected Sage3**, well below the proposed
15% slowdown threshold for this case. This is one generation per arm, not a
cross-prompt or multi-run confidence interval; it does not guarantee the
same bound on other shapes or hardware.

| Attention arm | Denoise wall | 20 DiT calls | Decode wall | Denoise + decode | Median DiT step 1–19 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16, cold-cache reference | 330.78 s | 192.63 s | 36.14 s | 366.92 s | 9.542 s |
| BF16, warm-cache control | 205.76 s | 197.76 s | 12.01 s | 217.77 s | 9.644 s |
| Corrected Sage3 FP4 | 188.49 s | 180.23 s | 11.38 s | 199.86 s | 8.940 s |
| Split mixed FP4/P8×V8 | 190.86 s | 182.79 s | 11.36 s | 202.21 s | 9.057 s |

The initial BF16 arm loaded model files and the VAEs from a cold OS cache;
using its 366.92 s total as the speed denominator would substantially
overstate the attention gain. The repeated BF16 run produced **bitwise
identical terminal latents**. Warm BF16 had a few 10–11 s DiT steps; its
median is therefore a useful companion to the wall-clock result. Median
stable-step split/Sage3 ratio was 1.0131×.

## Workload and correctness

The four runs use race prompt ID 16 and seed 20026, 576×1024, 124 frames,
20 denoising steps, CFG 1, flow shift 12, audio flow shift 3, and tiled VAE
decode. The same BF16 H3 weights and all 200 BF16 target linears are used;
only the 50 main attention helpers change. Refiner and independent padding
segments remain BF16. Each main attention has 22,677 valid tokens and 43
padding tokens. GPU 4 was idle before launch. All arms have identical initial
video/audio latent hashes, all 20 steps and 124 decoded frames, finite
outputs, and the same target shape. See `race/summary.json` for exact values
and media paths.

On the cached teacher-forced first DiT call, corrected Sage3 video/audio
output NMSE relative to BF16 was 1.4369%/0.3848%; split mixed was
1.4064%/0.3957%. These local output metrics are **not** a claim about visual
quality. The final video latent NMSE versus BF16 was 18.92% for Sage3 and
15.33% for split; split versus Sage3 final video latent NMSE was 11.40%.
Small attention perturbations accumulated across recurrent denoising, so the
generated videos are not numerically interchangeable. No VBench, LPIPS, or
human quality assessment was run here.

## Long-sequence routing bug found and fixed

The previous E120 split wrapper reused an E119 route-compaction kernel that
only scanned the first **128** V tiles. This H3 workload has **178** tiles.
The V4-zeroing kernel removed selected tiles in all 178 positions, but tiles
128–177 were omitted from the FP8 correction. As a result, the first complete
DiT preflight had split video NMSE **11.37%**, versus Sage3 **1.44%**. The
new `_compact_route_all` kernel scans `next_power_of_2(actual_tiles)` with
masking. A last-tile-only boundary test passed, a real 22,677-token
block-25 QKV check passed, and full-DiT split video NMSE fell to **1.406%**.
Both pre- and post-fix reports are retained; only the post-fix operator was
used in the four video generations.

## Reproduction and limits

Run `bash scripts/research/launch_e121_h3_split_generation.sh race 4 all`
from this checkout on an idle GPU 4, then
`bash scripts/research/launch_e121_h3_split_generation.sh race 4 warmbf16`
and `/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/summarize_e121_h3_split_generation.py --case race`.
The launcher refuses a GPU with less than 55 GiB free and preserves existing
arm outputs. Logs are `results/logs/e121_h3_generation_gpu4.log` and
`results/logs/e121_h3_bf16_warm_gpu4.log`; media and 4 MiB terminal latent
checkpoints are under
`/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race`.
The GPU was NVIDIA RTX PRO 5000 72GB,
driver 580.126.20. The split wrapper SHA-256 is
`29805659c2d1458de5f9eceb1cd16582e094d2a59ba300dd4c9a70acf3b4de43`;
the corrected FP4 and LSE extension hashes are recorded in E119/E120 and
the per-arm router manifests. The LSE extension used here has SHA-256
`09c3a8c1078aff8c01f8ecd1730bb3fdaebaac81fb654bbee3fc44b21bc8bbcf`.

The timing includes H3 model loading, 20-step denoising, separate VAE model
loading/decoding, and media writing, with the same two-phase procedure for
every arm. It does not include the preparatory teacher-forced preflight.
Because model load/decode are repeated in separate processes, this is a
matched complete-generation workflow timing, not the latency of a resident
long-running inference server. Median DiT steps isolate the steady-state
compute effect. No CPU/GPU concurrent load control beyond selecting an idle
GPU was imposed, and only one prompt/seed was tested.
