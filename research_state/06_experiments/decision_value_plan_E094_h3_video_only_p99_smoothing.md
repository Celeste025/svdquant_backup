# E094 — H3 FC2 video-only p99 smoothing

- **Mode:** exploration.
- **Problem addressed:** Is mixed text/audio/video calibration responsible for weak p99 SmoothQuant transfer on H3 FC2 native NVFP4 W4A4?
- **Hypothesis:** Per-channel p99 statistics from video tokens alone yield a channel scale that lowers independently evaluated video-output NMSE more than the same statistic from the mixed sequence.
- **Decision informed:** Whether modality-specific smoothing merits extension to more H3 layers or should be deprioritized against activation-scale policy/outlier handling.
- **Minimum valid setup:** Fixed real block-25 FC2 BF16 weight; p11 step-0 calibration and disjoint p1 step-0 evaluation. Both measured outputs use exactly the video-token rows and all 5376 output channels. Compare unsmoothed, mixed-calibration p99 and video-only-calibration p99 with alpha grid `{0,.25,.5,.75,1}`. Select alpha on p11 video output only. Native FlashInfer K16 E2M1/E4M3 W4A4 GEMM, BF16 target, and both static-calibration and current-input global activation scales. Record mixed/video p99-vector similarity and unsmoothed controls to separate smoothing-statistic changes from removing non-video rows from per-tensor scale.
- **Invalid cheap proxy to avoid:** Evaluate different token populations between arms; choose alpha on p1; compare only activation Q/DQ instead of complete FC2 output; treat video-only FC2 as a whole-model video-quality result.
- **Expected outcomes:** Meaningful held-out gain from video-only p99 supports testing across layers/prompts; near-identical scales/errors argues mixed-modality p99 is not the main cause; in-sample-only gain indicates overfit; gains confined to dynamic scale point to global-scale mismatch.
- **Decision for each outcome:** Extend only if calibration-selected video-only p99 improves p1 video NMSE by at least 5% relative versus mixed p99 and at least 2% versus unsmoothed, without merely changing the runtime global-scale policy. Otherwise stop this one-layer probe.
- **Stop condition:** One p11/p1 pair and one FC2 layer; no additional alpha/prompt search based on p1. Abort conclusions if native unsmoothed/video-only output or modality boundaries fail validation.
- **Evidence class:** Necessary for the current modality-mixing hypothesis; broader model evidence is supporting.
