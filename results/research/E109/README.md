# E109: SageAttention3 prescale on the captured H3 attention input

Date: 2026-10-20. Fork commit: `673a3db` plus the local device-aware grid patch in
`third_party/SageAttention3_prescale/sageattention3_blackwell/sageattn3_pvhad/blackwell/launch.h`.
The patch replaces the fork's hard-coded `110` in `Scheduler::get_grid_dim`
with `at::cuda::getCurrentDeviceProperties()->multiProcessorCount`. This is
110 on the tested NVIDIA RTX PRO 5000. The fork's `pre_scale_o` remains enabled;
`p_hadamard=False` in every fork call.

## Input and timing

- Real BF16 H3 prompt 1, step 0, block 25 attention capture from E098,
  480×832, 73 frames. Main segment is `[B,H,N,D]=[1,56,9482,128]`,
  noncontiguous HND Q/K/V, non-causal. The 54-token tail is excluded.
- GPU 0, one build per process, 25 warmups, 12 batches × 12 CUDA-event calls.
  Values below are means of batch means. `online` includes the fork/official
  QKV preprocessing and FP4 packing but excludes the benchmark-only K reset.
  `core` excludes all QKV preprocessing and output crop/copy.
- Useful TOPS uses `4·B·H·N²·D` operations, MAC = 2 operations. NMSE is relative
  to PyTorch BF16 SDPA on the same Q/K/V.
- Script: `scripts/research/bench_e109_sage3_prescale.py`.

| Build | Online (ms) | Online TOPS | Core (ms) | Core TOPS | Output NMSE |
|---|---:|---:|---:|---:|---:|
| Official SA3, historical 170 grid | 8.716 | 295.8 | 5.535 | 465.7 | 0.750226% |
| Official SA3, device-sized 110 grid, mean of 2 runs | 7.632 | 337.8 | 4.568 | 564.4 | 0.750226% |
| Prescale fork, isolated 170 grid, mean of 2 runs | 8.931 | 288.7 | 5.746 | 448.7 | 0.750234% |
| Prescale fork, hard-coded 110 grid, mean of 3 runs | 7.706 | 334.5 | 4.608 | 559.4 | 0.750234% |
| Prescale fork with Hadamard disabled, device-aware grid | 7.708 | 334.4 | 4.602 | 560.2 | 0.750234% |
| **Prescale-only build, Hadamard removed, device-aware grid, mean of 2 runs** | **7.706** | **334.5** | **4.605** | **559.8** | **0.750234%** |

The prescale-only build is about **11.6% faster online** and **16.8% faster core-only**
than the historical official 170-grid build. That gain is largely the grid
correction, **not evidence of a prescale gain**. Against the matched official
110-grid baseline, the prescale-only build is **1.0% slower online** and **0.9% slower
core-only** on this H3 shape. Three hard-coded-110 fork runs gave virtually
the same online time (7.703–7.709 ms) and core time (4.608–4.609 ms).
Output NMSE changed by only `7.94e-8` in absolute NMSE units. Core outputs
matched their own online API exactly (NMSE 0).

The author's `PRESCALE_OPTIMIZATION.md` reports 5.7–8.2% improvements on
different shapes, including 32K tokens and 14 heads. This result does not
disprove those measurements; it shows that their claimed speedup does not
transfer to the measured H3 9.5K-token, 56-head call. The fork also differs
from official code in its optional Hadamard infrastructure and build flags,
although Hadamard was disabled. Hence the small difference cannot be
attributed solely to `pre_scale_o` without a further controlled kernel-only
ablation. No full H3 generation latency was measured here.

Changing the pre-cleanup fork itself to a 170-block grid raises online latency from
7.708 to 8.931 ms and core latency from 4.602 to 5.746 ms. This isolated
170-grid diagnostic is also 2.5% slower online and 3.8% slower core-only
than the historical official 170-grid build. The fork's BF16-relative NMSE
remains 0.750234%, showing this grid change affects scheduling/speed, not
the measured output error. The diagnostic source and binary live only in an
isolated temporary build; the checkout and its saved extension remain
device-aware.

## Raw records and build

- Official 170 grid: `sage3_official_sm170_gpu0.json`.
- Official 110 grid: `sage3_official_sm110_a_gpu0.json`,
  `sage3_official_sm110_b_gpu0.json`.
- Fork hard-coded 110: `sage3_prescale_sm110_gpu0.json`,
  `sage3_prescale_sm110_b_gpu0.json`,
  `sage3_prescale_sm110_checkout_gpu0.json`.
- Fork isolated 170: `sage3_prescale_sm170_gpu0.json`,
  `sage3_prescale_sm170_b_gpu0.json`.
- Final fork device-aware grid: `sage3_prescale_dynamic_grid_gpu0.json`.
- Prescale-only build with Hadamard code and API removed:
  `sage3_prescale_only_dynamic_grid_gpu0.json`,
  `sage3_prescale_only_checkout_gpu0.json`.

The final compiled CUDA extensions are local, unversioned files in the fork's
`sageattention3_blackwell/` directory. The prescale-only `fp4attn_cuda`
SHA-256 is
`a6b31e3fc9a0da178f79b092280850acf228ba2e8df59e5643a421fb3e297643`.
The official 110-grid attention extension SHA-256 is
`a326923ecd2fc3aa45b6c780c240c7405a8f06448333722406c3d7094ea7f98a`.
