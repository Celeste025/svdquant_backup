# E111: Recompiled Sage3 FP4 kernel without Q/K mean correction

Date: 2026-10-20. Same H3 Q/K/V capture and timing boundary as E110:
prompt 1, step 0, block 25, `[1,56,9482,128]`, non-causal, RTX PRO 5000 GPU 0.
The user-requested all-off arm leaves Q and K uncentered and does not compute
`Qmean × Kᵀ`. BF16 PyTorch SDPA on the original input is the accuracy reference.

Unlike E110, this experiment recompiles the FP4 attention core in an isolated
`/tmp/e111_sage3_nocenter_7DxMl2` copy. The patch removes compensation TMA
copies, subtracts their bytes from the K-pipeline completion count, and
initializes each QK accumulator to zero instead of loading compensation from
shared memory. The original extension was not modified. The patch is saved in
`configs/research/e111_sage3_no_qk_mean_core.patch`; the CUDA build environment
is in `scripts/research/build_e111_sage3_nocenter.sh`.

| Kernel used for all-off arm | Full API, ms | Prepacked core, ms | Output NMSE |
|---|---:|---:|---:|
| Original, zero compensation buffer | 6.575 | 4.584 | 1.0578% |
| Recompiled, compensation transfer/use removed | 6.548 | 4.486 | 1.0578% |

Values are means of two process-level runs in A–B–B–A order. Each process
measured two rounds of 25 warmups plus 12 batches × 12 CUDA-event calls.
The revised kernel is **2.18% faster at the core** but only **0.41% faster
end-to-end** than the zero-buffer version. Relative to the original fully
centered E110 baseline of 7.701 ms, the new full-call speedup is **1.176×**,
not 1.3×. The needed time for 1.3× is about 5.924 ms, a further 0.624 ms
below this measured recompiled version.

All four runs gave exactly the same BF16-relative NMSE
`0.010577645014255566`. This compares aggregate error, not elementwise output
identity. The modified source still declares an unused delta_s descriptor and
small shared-memory slot; only the runtime TMA loads and QK compensation use
are removed. A more invasive ABI/shared-memory rewrite may differ slightly,
but this controlled comparison shows no large benefit from the runtime
compensation transfer/use alone on this H3 shape.

Original extension SHA-256:
`a6b31e3fc9a0da178f79b092280850acf228ba2e8df59e5643a421fb3e297643`.
Recompiled extension SHA-256:
`d2a8d861c1581453f0ad9fc3b27ff51fc4e8db04af810d164da086f06faa5257`.

Raw results: `original_no_qk_gpu0.json`, `original_no_qk_gpu0_b.json`,
`recompiled_no_qk_gpu0.json`, `recompiled_no_qk_gpu0_b.json`.
Benchmark: `scripts/research/bench_e110_sage3_mean_ablation.py --only-no-qk`.
