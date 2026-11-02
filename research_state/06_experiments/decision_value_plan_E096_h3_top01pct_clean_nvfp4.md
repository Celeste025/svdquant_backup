# E096 — Clear top 0.1% H3 FC2 video activations before W4A4

- **Mode:** exploration.
- **Problem addressed:** Could the broader activation tail, rather than only a few extreme outliers, explain weak plain NVFP4 and SmoothQuant output accuracy?
- **Hypothesis:** Removing the exact top 0.1% raw absolute activation entries substantially lowers W4A4 output NMSE against the corresponding cleaned BF16 multiplication; p99 SmoothQuant may then become more useful.
- **Decision informed:** Whether tail isolation/sparse preservation merits further method development, or whether the remaining error is mostly distributed across ordinary entries.
- **Minimum valid setup:** One real H3 block-25 FC2, all 21,312 video rows on p11 step 0 calibration and independent p1 step 0 evaluation. Define each call's mask as the exact top 0.1% of raw `|X|` entries over its video tensor, deterministic flat-index tie break. Both arms use the identical cleaned tensor and cleaned BF16 `X_clean W.T` target. Compare no-clean controls against plain native K16 NVFP4 W4A4 and alpha-0.5 p99 smoothing recalibrated on cleaned p11; report static p11 activation-global scale and dynamic per-call scale. Quantify how much the cleaned BF16 output differs from the original separately.
- **Invalid cheap proxy to avoid:** Measure only activation Q/DQ, use different masks or denominators between arms, reselect alpha on p1, or claim cleaned-target NMSE is improvement on the original H3 computation.
- **Expected outcomes:** Strong paired output-NMSE reduction would implicate the broader tail; small/no change means the remaining quantization error is broadly distributed; improvement only under static scale points to calibration range mismatch.
- **Decision for each outcome:** Pursue a preserving outlier side path only if cleaned-target p1 NMSE falls materially (at least 10% relative) while identifying the signal-removal cost; otherwise stop this coarse tail-removal hypothesis for this layer.
- **Stop condition:** One predeclared fraction (0.1%), one layer, two calls, no fraction/alpha search on p1.
- **Evidence class:** Necessary for the user's top-0.1% question; model-wide evidence is future work.
