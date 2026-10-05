# E016 profiler attribution

All three completed reports and Chrome trace hashes were checked. No GPU was launched for this analysis.

| Attention arm | Steady host median [min, max], ms, including checks | Speedup | Measured peak allocated / reserved, GiB |
|---|---:|---:|---:|
| svd_bf16 | 6791.286 [6780.626, 6803.870] | 1.0000× | 16.80573 / 30.23242 |
| svd_block_mean | 5124.274 [5123.026, 5131.842] | 1.3253× | 17.71330 / 32.55273 |
| svd_global_mean | 4968.463 [4966.731, 4979.043] | 1.3669× | 16.80603 / 31.73438 |

The next table is **exclusive GPU kernel duration from one separate diagnostic profile**, not steady execution time. Finite checks are absent in the timed repeats.

| Profile category, ms | BF16 | Block mean | Global mean |
|---|---:|---:|---:|
| bf16_flash_attention | 3339.679 | 0.585 | 0.587 |
| native_fp4_attention | 0.000 | 1047.236 | 1031.110 |
| official_mean_center_and_pad | 0.000 | 248.269 | 246.791 |
| official_correction_fp32_casts | 0.000 | 45.841 | 45.275 |
| official_correction_fp32_matmul | 0.000 | 174.224 | 35.901 |
| official_qkv_fp4_pack | 0.000 | 63.307 | 61.647 |
| native_fp4_linear | 1008.035 | 994.409 | 992.241 |
| h3_linear_activation_pack | 199.968 | 196.413 | 196.227 |
| diagnostic_finite_checks_only | 111.552 | 682.690 | 441.376 |
| other_model_and_routing | 2235.746 | 2370.089 | 2365.434 |
| Total kernel sum | 6894.981 | 5823.062 | 5416.588 |

Each low-precision profile contains exactly 50 native attention kernels, 150 official Q/K/V pack kernels, 50 correction matmul kernels, 200 native linear GEMMs, and 600 linear activation-pack kernels. BF16 attention counts are 52 for the low-precision arms versus 102 for the BF16 arm.

Correction uses `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` for block mean and `gemv2T_kernel_val` for global mean. Native attention is `nvfp4_attention::attention_kernel_ws<…float_e2m1…>`; Q/K packing uses `scaled_fp4_quant_kernel<…, false/true,…>` and V uses `scaled_fp4_quant_trans_kernel`.

- Only cat=kernel, ph=X durations enter the mutually exclusive kernel sum; no CPU/GPU annotation double count.
- Each independent profile includes diagnostics (52/252 flags), unlike all steady repeats. The profile sum is not steady latency, critical path, or predicted speedup.
- Generic preprocessing ops are attributed only after all 50 contiguous 13-kernel sequences match the official source via External id CPU links. Other layout/copy/model/LR work remains other.
- FP32 correction GEMM/GEMV and casts are separated from mean/center/pad and official Q/K/V packing. Both modes center K and Q; they are complete official recipes, not a one-factor causal comparison.
- Memory is actual post-warmup three-repeat allocated/reserved peak and measured resident storage. No full-model memory estimate from single correction tensors.
- CUDA runtime CPU duration may include host synchronization waits; GPU transfers are separate, not an explanation by pure bandwidth. CPU runtime sums are not added to GPU sums.

Source-linked per-category kernel names/counts and actual memory/timing records are in `E016_profile_attribution.json`. No quality or new-method claim follows from this profile.
