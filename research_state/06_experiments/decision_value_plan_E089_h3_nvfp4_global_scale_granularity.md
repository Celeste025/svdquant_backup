# E089 — H3 NVFP4 second-level scale granularity

- **Mode:** exploration.
- **Problem addressed:** Does the per-tensor FP32 scale in two-level NVFP4 materially contribute to H3 W4A4 linear-output error?
- **Hypothesis:** Replacing one per-tensor scale with one per 512, 128, 32, or 1 matrix rows reduces output NMSE when row-wise dynamic ranges differ, while retaining the FP8 scale per 16 contiguous K elements.
- **Decision informed:** Whether a row/group-scale dense NVFP4 kernel merits implementation. A negligible gain means stop at the present per-tensor contract.
- **Minimum valid setup:** Real H3 block-25 FC2 BF16 weight and exported cached inputs; A=[tokens,K], B=[output channels,K], both packed over contiguous K blocks of 16. Compare calibrated per-tensor A + per-tensor B, evaluation-tensor A + per-tensor B, grouped A only, grouped B only, and both grouped. Use native FlashInfer FP4 packing, dequantize its actual codes and E4M3 scales, and compare complete linear outputs with the same BF16 reference. Gate the dequantized baseline against native NVFP4 GEMM accuracy.
- **Invalid cheap proxy to avoid:** Quantizing across rows instead of K; assuming an independent BF16-QDQ rounding error is the true NVFP4 result without a native baseline gate; optimizing/evaluating only elementwise quantization MSE while claiming linear-output benefit.
- **Expected outcomes:** (1) Consistent sizeable NMSE reduction, especially from activation groups: proceed to a compatible GEMM/epilogue design. (2) Only in-sample or small benefit: do not implement a new kernel yet. (3) No benefit: retain per-tensor scale.
- **Stop condition:** Stop at one layer and two cached calls if all grouped variants change NMSE by less than 2% relative to a **current-input per-tensor activation scale** (separating row granularity from cross-input scale mismatch); otherwise extend to a second layer or timestep before making a general claim. Do not report a numerical conclusion if native pack/dequant gate fails.
- **Evidence class:** Necessary for deciding whether to pursue this scale-granularity idea; multi-layer generalization is supporting evidence.
- **Scope:** Error only. No latency or throughput measurement.

