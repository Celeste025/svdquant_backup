# E098 SageAttention profiling archive

This archive is for subsequent kernel tuning, not just the final timing table.
It uses the same real H3 p1/step-0/block-25 non-causal Q/K/V capture
(`B=1, H=56, N=9482, D=128`) for both native cores. The 54-token H3 tail is
excluded from core profiling. Sage3 uses the official-derived SM-count-grid
build, not the 170-CTA upstream default. The two native cores reproduce their
respective public-API outputs exactly (NMSE 0).

## Retained profiles and validation

- CUDA-event benchmark: `sage_h3_sm110_two_scope_gpu0.json`, 25 warmups,
  12 batches x 12 calls. Core and full public-API latency, useful TOPS,
  accuracy, dispersion, and original Q/K/V geometry are recorded.
- Both native-core Nsight Compute reports on `/data1`:
  `h3_sage2pp_core_breakdown.ncu-rep` and
  `h3_sage3_sm110_core_breakdown.ncu-rep`. Each includes LaunchStats,
  SpeedOfLight, Occupancy, SchedulerStats, WarpStateStats,
  ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, and InstructionStats.
  These are native FP8/INT8 and FP4 kernels, not wrappers or fallbacks.
- Source/PC-sampling reports for both cores:
  `h3_sage2pp_core_source.ncu-rep` and
  `h3_sage3_sm110_core_source.ncu-rep`, including SourceCounters and
  `smsp__pcsamp_sample_count` (833,376 and 579,441 sampled PCs, respectively).
- `sage_ncu_metric_bundle.json`: all 2,344 retained raw NCU metrics,
  their units, report SHA-256 hashes, the Q/K/V capture SHA-256, checked
  launch geometry, CUDA-event timing, and a compact side-by-side summary.
  Regenerate with `python scripts/research/export_e098_sage_ncu.py`.
- Nsight Systems API timeline:
  `h3_sage2pp_sage3_official_api_timeline.nsys-rep`, its exported `.sqlite`,
  and four CSVs (`_nvtx_gpu_proj_sum`, `_cuda_gpu_kern_sum`,
  `_cuda_gpu_trace`, `_cuda_api_sum`). It contains three measured calls per
  arm after two warmups, marked by per-call NVTX ranges. Sage2++ used seven
  GPU ops/call; Sage3 used 18 GPU ops/call, including the harness's fresh-K
  restoration before the public API (see timing-scope caveat below).
- Existing preprocessing timings:
  `sage3_sm110_preprocess_stages_gpu0.json` and
  `sage2pp_preprocess_stages_gpu0.json`; interpretation in `README.md`.

The binary `.ncu-rep` files and large Q/K/V tensor live at
`/data1/models/svdquant-wjq/research/20261014/E098/` and are intentionally
not committed. Open the `.ncu-rep` files with Nsight Compute GUI or use
`ncu -i <report>.ncu-rep --page details --print-details all`; the source
report can be explored with `--page source --print-source cuda,sass`.
Nsight replay changes clocks and duration, so **do not** compare profiler
duration as a latency benchmark; use CUDA events in the benchmark JSON.
Pipeline percentages are not additive per-stage times.

## Compact comparison

| Metric | Sage2++ | Sage3 FP4, grid 110 |
| --- | ---: | ---: |
| Native core CUDA-event mean | 6.430 ms | 4.670 ms |
| Native core useful speed | 400.9 TOPS | 552.0 TOPS |
| Public API CUDA-event mean, original scope | 7.307 ms | 7.628 ms |
| NCU grid / block | 4,200 / 128 | 110 / 384 |
| NCU registers / thread | 255 | 168 |
| NCU dynamic shared memory / CTA | 32.768 KB | 100.352 KB |
| NCU tensor-pipe active, elapsed-clock basis | 73.62% | 50.34% |
| NCU DRAM throughput / peak | 2.80% | 4.84% |
| NCU L2 sector hit rate | 96.73% | 93.30% |

The Nsys timeline recorded mean projected GPU ranges of 7.124 ms for Sage2++
and 7.785 ms for Sage3; the native core kernels averaged 6.067 and 4.376 ms
in that trace. The Nsys run is a separate GPU-1 measurement and **not** the
25-warmup/144-call CUDA-event benchmark. Its Sage3 NVTX range also includes a
fresh K copy (about 0.24 ms device-to-device per call) because the official
API centers K in place while the captured H3 Q/K/V must remain reusable.
Therefore compare that range's composition, not its absolute number, against
the original no-copy online CUDA-event timing. The raw per-call and per-kernel
times remain in the CSV/SQLite for exact attribution.

## Capture circumstances and limits

At launch all eight GPUs had another user's compute process. A named
`research-e098-sage-profile-capture` tmux session waited for a *process-free*
card (<1 GiB allocated and <5% utilization twice 10 seconds apart), then
captured the missing reports on GPU 1. The card was subsequently reclaimed by
the other user's training process. The capture log is
`results/logs/e098_sage_profile_when_idle.log`; no occupied GPU was profiled.

The current Sage2++ binary was built without CUDA `-lineinfo`, so its
SourceCounters report identifies hot SASS PCs/opcodes and stalls, but may not
map every hot instruction to a C++ line. Rebuilding with line information is
a separate, unperformed diagnostic step. Sage3's report includes source
correlation. Neither report should be interpreted as proof of a particular
high-level stage's exact elapsed milliseconds inside the fused kernel.
