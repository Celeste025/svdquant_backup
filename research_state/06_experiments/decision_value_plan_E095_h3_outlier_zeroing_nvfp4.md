# E095 — H3 FC2 outlier zeroing before native NVFP4 W4A4

- **Mode:** exploration.
- **Problem addressed:** Are a few extreme H3 activation values the main cause of FC2 W4A4 output error or p99-smoothing transfer failure?
- **Hypothesis:** Zeroing rare large activation entries shrinks the activation global scale and thereby reduces quantization-only error; whether it improves the *original* BF16 target depends on the information lost by zeroing.
- **Decision informed:** Whether outlier removal itself is viable, or whether any benefit requires a sparse high-precision bypass / separate outlier representation.
- **Minimum valid setup:** Real block-25 FC2 BF16 weight and p1/p11 step-0 cached activations. Preserve the original BF16 `XW.T` as the primary target. Compare no zeroing with predeclared absolute thresholds `2512,1024,512,256` under native FlashInfer NVFP4 W4A4 with both current-input and fixed p11-original activation global scales. Measure (i) quantization error against the zeroed-input BF16 output, (ii) signal-removal error against original BF16 output, and (iii) final W4A4 error against original BF16 output, plus changed-element counts by modality. Include a targeted p1-video p99-smoothing alpha-0.5 static-scale check if the raw test leaves the E094 mechanism unresolved.
- **Invalid cheap proxy to avoid:** Report lower error only relative to the modified BF16 target, call zeroing exact outlier preservation, or attribute a one-layer/two-prompt result to all H3 layers and timesteps.
- **Expected outcomes:** Quantization-only gain but final error deterioration implies outliers are costly yet informative; final error gain implies a viable zeroing tradeoff in this local setting; no quantization-only gain argues extreme values are not the main bottleneck.
- **Decision for each outcome:** Continue only if final original-target NMSE materially improves on both prompts or a sparse bypass control shows a substantial quantization-only gain worth engineering. Otherwise stop literal zeroing and retain outlier handling only as a separate-precision hypothesis.
- **Stop condition:** One FC2 layer, two calls, four fixed thresholds; no threshold tuning on the observed result.
- **Evidence class:** Necessary for answering the user’s outlier-zeroing hypothesis; model-wide evidence is future work.
