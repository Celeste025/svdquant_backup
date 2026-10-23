# E122 — H3 attention attribution and long-sequence mixed-route validation

## Bottom line

The earlier E121 arm labeled `bf16` did **not** use BF16 attention arithmetic:
DiffSynth selected its default `sage_attention` implementation, which quantizes
Q/K internally. E122 explicitly forces PyTorch BF16 SDPA as the reference and
excludes model loading. On two real cached MiniMax-H3 DiT calls (denoising steps
0 and 10), corrected Sage3 reduces mean direct-forward latency from **10.344 s
to 8.673 s** (1.193× speedup); the corrected split FP4/P8×V8 operator takes
**8.870 s** (1.166× versus BF16, 2.26% slower than Sage3). This is a
teacher-forced, resident-DiT timing proxy, **not** a new 20-step generation
wall-clock measurement or a video-quality result.

## Attribution: where the time goes

All four arms use the same H3 BF16 weights and 200 checked BF16 target linears.
There are 50 main attention helpers and two refiner helpers per DiT call. The
refiner and main-sequence padding use the selected unmodified DiffSynth backend;
for the corrected Sage3 and split arms it is PyTorch SDPA. GPU: NVIDIA RTX PRO
5000 72GB Blackwell. Model construction/loading, lazy compilation, and one
warm-up forward per arm/step occur outside the measured calls. Means below are
over steps 0 and 10; `direct` is uninstrumented synchronized wall time, while
the remaining columns come from CUDA-event instrumentation on a separate call.

| Arm | Direct DiT (s) | Main attention (s) | All linear modules (s) | Other (s) | Direct speedup vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| True BF16 SDPA | 10.344 | 3.388 | 5.654 | 1.211 | 1.000× |
| DiffSynth default `sage_attention` (old E121 reference) | 9.690 | 2.848 | 5.619 | 1.214 | 1.067× |
| Corrected Sage3 FP4 | 8.673 | 1.873 | 5.552 | 1.203 | 1.193× |
| Corrected split FP4/P8×V8 | 8.870 | 2.087 | 5.573 | 1.332 | 1.166× |

In the BF16 profile, main attention accounts for **33.0%**, all linear modules
**55.1%**, and other work **11.8%** of the instrumented 10.253 s forward.
Within linear modules, GEMMs take 3.463 s and non-GEMM module work an estimated
2.191 s. All 200 target linears are disk-offloaded, so the non-GEMM bucket
includes weight materialization/transfers and should not be interpreted as
pure disk-I/O time. The measured attention-helper speedups are **1.809×** for
Sage3 and **1.623×** for split, but the much larger unchanged linear share
limits the whole-DiT gain. Holding the BF16 non-attention work fixed gives
Amdahl predictions of **1.173×** and **1.145×**, close to the direct observed
**1.193×** and **1.166×**. The small residual reflects independent instrumented
and direct calls and changes/noise in non-attention time; this decomposition
does not establish a causal explanation for every millisecond.

The original E121 warm-generation speedup was against the default quantized
SageAttention backend and included denoising/decode overhead, so it is not
directly comparable with E122's pure-BF16, single-DiT-forward denominator.
E122's default-backend arm explicitly reproduces that baseline distinction.

## Length sweep and correctness

Q/K/V are captured from real H3 block 25 at 56 heads, head dimension 128,
noncausal attention, and 22,677 valid tokens. Shorter tests take a prefix of
this same tensor; they stress shape boundaries, not independent H3 examples.
Times are the mean of **three independent processes**, each with four warmups
and four batches of eight CUDA-event calls per arm/length (96 timed calls per
cell overall). The full Sage3 call includes the `K.clone()` required by this
wrapper, matching the routed H3 path; the benchmark excludes model loading.

| Tokens | BF16 SDPA (ms) | Corrected Sage3 (ms) | Split mixed (ms) | Split / Sage3 |
| ---: | ---: | ---: | ---: | ---: |
| 127 | 0.018 | 0.140 | 0.250 | 1.79× |
| 128 | 0.020 | 0.111 | 0.179 | 1.61× |
| 129 | 0.026 | 0.136 | 0.231 | 1.69× |
| 255 | 0.028 | 0.150 | 0.231 | 1.54× |
| 256 | 0.029 | 0.115 | 0.184 | 1.59× |
| 1,024 | 0.189 | 0.198 | 0.318 | 1.60× |
| 4,096 | 2.284 | 1.504 | 2.029 | 1.35× |
| 8,192 | 8.834 | 4.946 | 5.828 | 1.18× |
| 16,384 | 35.741 | 17.181 | 19.011 | 1.11× |
| 16,385 | 36.312 | 19.829 | 21.680 | 1.09× |
| 22,677 | 69.419 | 35.195 | 38.078 | 1.08× |

At the actual 22,677-token shape, the split kernel is **8.2% slower** than
corrected Sage3 attention but **1.823× faster** than BF16 SDPA. Small lengths
are slower than BF16 because launch/preprocessing overhead dominates; no
universal 15% performance bound is claimed. The boundary at 16,384/16,385
tokens shows a scheduler/padding discontinuity rather than smooth scaling.

The corrected route-compaction kernel scans all `ceil(tokens/128)` V tiles,
including tiles beyond index 127. Unit checks for tile counts 1, 2, 127, 128,
129, 178, 256 and 257 pass. The full H3 case selected 117/9,968 V tiles,
including **three** after tile 127 that the old compactor omitted. Across all
11 lengths and all three independent runs, split-vs-native mixed output NMSE
is at most **1.753e-6**. At 22,677 tokens, the output NMSE versus BF16 SDPA
is **0.704%** for corrected Sage3 and **0.703%** for split; the tiny difference
is not a claim of downstream generation-quality improvement. Every tested
output is finite and shape/dtype-correct.

## Reproduction and artifacts

The branch versions `scripts/research/h3_mixed_sage3_api.py` with the
all-tile compactor, plus self-contained `e122_*` scripts. Re-run on an idle
GPU using `scripts/research/launch_e122_profile.sh` and
`scripts/research/launch_e122_sweep.sh`, then
`python scripts/research/summarize_e122_h3_profile.py`. The launchers set the
runtime environment and check free GPU memory; the mixed-attention wrapper
prepends the corrected Sage3 source path. See their usage lines for arguments.
Source inputs remain local and unversioned under `/data1`:
`E101_h3_r32_fixed1_three_cases/race/bf16_teacher_calls.pt` for the two DiT
calls and `E121_h3_split_generation/race/block25_qkv_h56.pt` for the sweep.

Reviewable numerical outputs are `h3_forward_profile_refactored.json`, the
three `mixed_length_sweep*.json` files, and `summary.json` in this directory.
Logs stay local in `results/logs/`. These two cached calls and one captured
block cannot establish cross-prompt or full-trajectory latency/quality
distributions. The old E121 full-generation record remains useful only with
its corrected baseline label.
