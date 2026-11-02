# E093 — H3 FC2 native NVFP4 operand-error distribution

Completed 2026-10-14. This is **plain W4A4 NVFP4**, not SVDQuant or smoothing. Full real `blocks.25.mlp.fc2` activation inputs from p1 and p11 at step 0, plus the original BF16 weight, are packed with native FlashInfer E2M1 and K16 E4M3 scales. Activation FP32 global scale is computed from each current input, as in the standard H3 dynamic-activation Q/DQ; weight FP32 global scale is fixed from its weight tensor. Actual codes/scales are decoded in 128-row chunks and compared elementwise with BF16 operands. Whole-operand NMSE matches E090 to less than `2e-10` absolute.

All numbers below describe **operand Q/DQ error**, not complete-linear output NMSE. `NMSE = sum((decoded-original)^2)/sum(original^2)`. Modality ranges are original packed positions, including p1's 16 padding rows in the global total.

| Operand | Global NMSE | Top 1% rows: error / signal energy | Top 1% input channels: error / signal energy | Top 1% K16 groups: error / signal energy |
|---|---:|---:|---:|---:|
| p1 activation | 0.486% | 4.94% / 6.56% | 6.65% / 23.15% | 3.68% / 6.43% |
| p11 activation | 0.485% | 4.43% / 5.53% | 7.05% / 25.61% | 4.01% / 6.50% |
| weight | 0.887% | 1.24% / 1.23% | 1.43% / 2.22% | 1.21% / 1.22% |

Each top-1% share is independently ranked by error energy or BF16 signal energy; they need not select identical positions. If error were equally distributed over positions, its top 1% would hold 1%. Thus activation **absolute error** varies by row and channel, while weight error is nearly flat across output rows and K16 groups. But activation signal energy is even more concentrated than its error, especially by input channel. The highest-energy activation channels are not simply the worst-relative-error channels.

| Call | Text rows / NMSE | Audio rows / NMSE | Video rows / NMSE | Padding rows / NMSE |
|---|---:|---:|---:|---:|
| p1 | 658 / 0.365% | 414 / 0.372% | 21,312 / 0.491% | 16 / 0.528% |
| p11 | 738 / 0.373% | 414 / 0.409% | 21,312 / 0.491% | none |

Video accounts for approximately 95%–96% of signal energy and 96%–97% of activation Q/DQ error energy. Its normalized error is about 1.3x the text region's. The activation per-row NMSE median/p99 is 0.533%/0.647% on p1 and 0.530%/0.644% on p11; the weight per-output-row median/p99 is 0.887%/0.917%. Across individual input channels, p99 local NMSE is 12.1% on p1 and 13.8% on p11, but those high relative errors often occur where signal energy is small; they do not imply those channels dominate the total error.

Conclusion: weight errors are approximately uniform at useful row/K16-group resolution. Activation errors are less uniform in absolute size, but not concentrated beyond the underlying signal energy; normalized error varies moderately by modality and row, much more by individual input channel. This does not establish which operand matters more for **linear-output** or full-model quality, because weight/activation error can interact and cancel. Only one layer, two prompts, and one timestep were measured.

Raw reductions and top indices: `h3_fc2_nvfp4_operand_error_distribution.json`. Reproducer: `scripts/research/analyze_h3_nvfp4_operand_errors.py`. Log: `results/logs/e093_h3_nvfp4_operand_error_distribution.log`.

## Raw 3000×3000 error heatmaps

Repository note: the full-resolution `*_3000x3000.png` images remain local and
are excluded from Git because each is roughly 17–24 MiB. Git retains matching
`*_preview_1000.png` images, downsampled with nearest-neighbor sampling for
inspection; unlike the originals, these previews are not element-per-pixel
scientific data. The numerical metadata and reproducer are retained.

Six lossless PNG heatmaps in `figures/` show p1 activation, p11 activation, and weight, each as signed `QDQ(x)-x` and absolute `|QDQ(x)-x|` error. Each PNG is exactly 3000×3000 pixels: **one original tensor element maps to one pixel**, with no binning, averaging, spatial resizing, or interpolation in the file. Y is tensor rows `[0,3000)`; X is input channels `[4500,7500)`. For activation, p1 text/audio/video boundaries are rows 658/1072; p11 boundaries are 738/1152. For weight, Y is output-channel index.

The signed maps use blue for negative error, off-white for zero, orange for positive error. Absolute maps use off-white for zero and darker gold/red for greater error. Color values are linear. P1 and p11 activation share an absolute display limit of `2.7704` for direct visual comparison; weight has its own display limit of `0.0540365` because its units have a different scale. Roughly 0.45%–0.49% of pixels exceed these display limits and are shown at the saturated end color; their actual error values are **not** modified in quantization or the analytical reductions. The exact crop, limits, raw min/max and saturation counts are in `figures/heatmap_metadata.json`.

At screen size, the app may shrink a 3000×3000 image to fit; open the PNG at full size to inspect individual cells. These are direct per-element operand errors, not linear-output errors.

## Native W4A4 linear-output error heatmaps

The four additional `*_output_{absolute,signed}_3000x3000.png` images use the same p1 and p11 step-0 calls but show `blocks.25.mlp.fc2` **output** error: native FlashInfer W4A4 NVFP4 GEMM minus BF16 GEMM, each rounded to BF16. This linear has no bias. Every image maps one tensor element to one pixel without pooling or averaging: rows 0–2999 are tokens; columns 0–2999 are output channels. The activation global scale is computed from the **full** cached input, not merely the displayed rows. Both prompts share the linear absolute/signed color limit of 72 output units; fewer than 0.35% of pixels saturate.

For the first 3000 rows and **all 5376 output channels**, the regional comparisons are:

| Prompt | Region | BF16 output RMS | W4A4 error RMS | Output NMSE |
|---|---|---:|---:|---:|
| p1 | text, rows 0–657 | 220.24 | 20.54 | 0.870% |
| p1 | audio, rows 658–1071 | 48.84 | 4.74 | 0.941% |
| p1 | video, rows 1072–2999 | 203.61 | 19.42 | 0.910% |
| p11 | text, rows 0–737 | 224.24 | 21.12 | 0.887% |
| p11 | audio, rows 738–1151 | 42.64 | 4.23 | 0.985% |
| p11 | video, rows 1152–2999 | 179.64 | 17.39 | 0.937% |

The audio band is lighter in a **shared absolute-error color scale** because its BF16 FC2 output is about 4–5× smaller in RMS; its absolute W4A4 error is likewise about 4–5× smaller. After dividing squared error by BF16 output energy, audio is **not** more accurate: its NMSE is slightly higher than text and video in both examples. Thus the visual band primarily reflects output scale, not an audio-specific quantization advantage. This is one layer, two prompts, one timestep, and a row crop, so it is not a general statement about all H3 audio tokens.

Exact full-width regional statistics, color limits and source paths: `figures/output_heatmap_metadata.json`. Reproducer: `scripts/research/render_h3_nvfp4_output_error_heatmaps.py`. Log: `results/logs/e093_h3_nvfp4_output_heatmaps_ninja.log`.
