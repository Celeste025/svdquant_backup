# E120 — split rare P8×V8 work from the Sage3 NVFP4 core

## Decision

Experimental candidate only; **not integrated into H3 generation**. On the
captured H3 block-25 attention call, it reduced the original native mixed
online latency from 18.45 to 8.81 ms, or 2.09× faster. It remained 1.15×
the corrected pure-FP4 online latency (7.67 ms), slightly above the hoped-for
1.10×. It retained 94.4% of the native mixed path's BF16-NMSE improvement,
but it is **not numerically equivalent** to native mixed.

The input is `/data1/models/svdquant-wjq/research/20261014/E098/
h3_p1_s0_b25_480x832_73f_qkv.pt`: `[1,56,9482,128]`, noncausal,
RTX PRO 5000 72GB. The same corrected Sage3 tail mask and threshold 0.10
are used throughout; 106 of 4,200 V tiles route to FP8. BF16 is the full
SDPA reference on the captured Q/K/V. No H3 video is generated here.

## Method

The existing sparse router packs selected V tiles as E4M3 FP8. Selected
tiles in a private copy of packed V4 are set to zero, and the spill-free
pure-FP4 Sage3 core computes all QK scores, softmax normalization and
unselected P4×V4 contributions. Its LSE-export build is the one validated
in E119. A small Triton pass unpacks the already-quantized Q4 to FP16 and
only selected K4 tiles to FP16, preserving their E2M1 values and E4M3
scales. The sparse correction kernel recomputes selected quantized QK
scores, applies the same delta-S and tile-relative P8 E4M3 conversion,
performs a real Tensor-Core **P8×V8** dot, normalizes using the core's LSE,
and adds the result to the main output. This is not E119's BF16
`P×(V8−V4)` approximation.

The unpacked Q and K buffers reserve approximately 138 MB each for this
shape (K writes only routed tiles); these are additional runtime buffers.
The V4 zeroing, route compaction and unpacking are included in the online
candidate timing. The benchmark-only K restoration needed because the
Sage3 preprocessing mutates K is outside all timed operators.

## Numerical checks

| Case | Route bits | Split vs native NMSE | Pure FP4 vs BF16 NMSE | Native vs BF16 NMSE | Split vs BF16 NMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 tokens, forced all-FP8, 4 heads | 4/4 | 1.01e-9 | 0.85429% | 0.32452% | 0.32452% |
| 256 tokens, natural routing, 56 heads | 0/112 | 0 | 1.01843% | 1.01843% | 1.01843% |
| 1,024 tokens, natural routing, 56 heads | 94/448 | 2.97e-6 | 1.00023% | 0.98489% | 0.98513% |
| 4,096 tokens, natural routing, 56 heads | 101/1,792 | 2.21e-6 | 0.85288% | 0.84670% | 0.84686% |
| 9,482 tokens, natural routing, 56 heads | 106/4,200 | 2.22e-6 | 0.74843% | 0.74524% | 0.74542% |

At the full H3 shape, the native mixed improvement over pure FP4 is
0.003197 percentage points; the split version's is 0.003019 percentage
points, a 94.4% retention. The full split/native maximum absolute output
difference is 0.5. The forced-all-FP8 short control matching to 1e-9
supports the P8×V8 math; the remaining sparse/full-sequence difference is
consistent with writing the unselected main output to BF16 before adding
the selected FP8 contribution. This is an inference, not a proven complete
error decomposition.

## Performance

Each row below is the mean of three **independent processes**; each process
used 20 warmups then 8 batches × 12 CUDA-event calls. All online paths
include Sage3 preprocessing, dynamic Q/K/V quantization and output crop;
mixed paths additionally include V8 routing. The split path additionally
includes V4 zeroing, route compaction, Q/K unpacking and sparse P8×V8.

| Full online path | Mean time | Relative to pure FP4 |
| --- | ---: | ---: |
| Corrected pure FP4 | 7.673 ms | 1.000× |
| Original native mixed | 18.450 ms | 2.404× |
| E120 split P8×V8 | 8.813 ms | 1.149× |

The three split-process means were 8.81094, 8.81512 and 8.81314 ms;
the matching pure-FP4 means were 7.67044, 7.67320 and 7.67545 ms.
At the best `BM=64`, 8 warps, 2 pipeline stages, component CUDA-event
times were approximately: V4 zero 0.014 ms, route compaction 0.013 ms,
Q4 unpack 0.176 ms, selected K4 unpack 0.019 ms, FP4+LSE core 4.683 ms,
and sparse P8×V8 0.726 ms. Component timings are measured separately and
should not be added to infer Python/API overhead. The sparse kernel reports
190 registers per thread and zero spills. Two warps spilled 194 values and
slowed the correction to 1.58 ms; 16 warps were also slower. `BM=128`,
8 warps exceeded the device's shared-memory limit.

## Reproduction and limits

Run `scripts/research/e120_split_native_mixed.py` with `--gpu 4 --tokens
9482 --heads 56 --bm 64 --warps 8 --stages 2 --predecode --full-reference`
for component timing and numerical checks, then run
`scripts/research/e120_full_online_benchmark.py` with `--gpu 4 --bm 64
--warps 8 --stages 2` for online timing. Raw per-process results are
`full_online_gpu4_w8_s2_run{1,2,3}.json`; length and tuning controls are
also in this directory.

An isolated FP32-intermediate-output build was attempted to test the
remaining rounding gap. Compilation failed because the existing Sage3
epilogue shared-memory copy layout is specialized for 16-bit output. That
exploratory build was removed; no verified binary was replaced. Closing
the residual numerical gap would require an explicit FP32 accumulator
export/redesigned epilogue, likely with a speed/memory cost. This result
is limited to one captured H3 call, one GPU and a fixed route threshold;
there is no generation-quality conclusion or evidence that 1.15× holds
for every block or step.
