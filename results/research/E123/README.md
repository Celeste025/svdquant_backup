# E123 — MiniMax-H3 GPU-resident DiT timing

Two real cached race-case DiT inputs (denoising steps 0 and 10, 22,677 valid
main-attention tokens, 576×1024, 124 frames) were profiled on an otherwise
idle NVIDIA RTX PRO 5000 72GB. Only the BF16 DiT was loaded; VAE and text
encoder were not loaded. Model load and one warm-up per arm/step were outside
timing. All **264** linear modules had CUDA-resident weights after warm-up;
none of the 200 target linears used disk offload. Peak CUDA allocation was
44.31 GiB. The direct column is synchronized uninstrumented wall time; the
component columns use CUDA events on a separate forward. Each is the mean of
steps 0 and 10, not a 20-step generation estimate.

The final matched rerun uses the same resident DiT for BF16, Sage2++ (the
local Blackwell SageAttention2++ fork, INT8 QK and FP8 PV), corrected Sage3,
and corrected split mixed. Sage2++ includes the required `K.clone()` inside
the attention-helper timing because its preprocessing centers K in place.

| Attention arm | Direct DiT (s) | Main attention helper (s) | All linear modules (s) | Other (s) | Speedup vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| True BF16 SDPA | 8.338 | 3.510 | 3.669 | 1.180 | 1.000× |
| Sage2++ INT8/FP8 | 6.891 | 2.093 | 3.627 | 1.180 | 1.210× |
| Corrected Sage3 FP4 | 6.717 | 1.940 | 3.593 | 1.181 | 1.241× |
| Corrected split FP4/P8×V8 | 6.911 | 2.148 | 3.581 | 1.181 | 1.206× |

On the same cached DiT inputs, output NMSE relative to true BF16 SDPA was
**0.0896% video / 0.0490% audio** for Sage2++, **1.515% / 0.920%** for
Sage3, and **1.598% / 0.866%** for split mixed. These are teacher-forced
two-step output errors, not final video-quality scores.

Using the instrumented BF16 total of 8.360 s, the narrow attention-helper
share is **42.0%**. The linear GEMM subtotal is 3.669 s; linear-module time
minus GEMM time is about **0.00051 s**. The BF16 QKV projection takes about 1.10 s
and attention output projection 0.368 s, both counted as linear modules. If
the whole attention sublayer is defined as attention helper + QKV projection
+ output projection, its measured time is about **4.98 s**, or **59.5%** of the
instrumented DiT forward. These buckets are exclusive at the measured scope;
the broad sublayer is a regrouping, not extra runtime.

For context, the earlier E122 disk-offloaded BF16 profile measured 10.344 s
direct, 3.388 s main attention, 5.654 s linear modules, and 1.211 s other.
Its ~33% attention share reflected the offloaded execution path, including
2.191 s of linear-module time outside GEMM. The E122 and E123 timings were
separate GPU sessions, so their latency difference is descriptive rather than
a perfectly paired estimate of the offload penalty.

The final matched raw timing and output-NMSE record is
`h3_forward_profile_resident_sage2pp.json`; the earlier resident four-arm
record including DiffSynth's older default `sage_attention` is
`h3_forward_profile_resident.json`. No full-trajectory generation or quality
evaluation was run here.
