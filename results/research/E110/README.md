# E110: SageAttention3 FP4 mean-removal ablation

Captured real MiniMax-H3 Q/K/V from prompt 1, denoising step 0, DiT block 25,
480×832/73 frames. The tested non-causal main segment has shape
`[B,H,N,D]=[1,56,9482,128]`; the 54-token tail is excluded. RTX PRO 5000 GPU 0,
current `SageAttention3_prescale` build, dynamic 110-SM persistent grid.

The three requested ablations are independent changes from the standard
baseline, not cumulative except for the last row. “Remove Q mean” also omits
its Q-mean × Kᵀ correction. “Remove both” omits Q/K centering and the
correction. There is no separate QK-score mean subtraction in this code.

| Arm | Full API time | Speed vs baseline | BF16-relative output NMSE | NMSE vs baseline |
|---|---:|---:|---:|---:|
| Original Q/K centering + correction | 7.701 ms | 1.000× | 0.7502% | — |
| No Q centering or correction | 6.940 ms | 1.109× | 1.0530% | +40.35% |
| No K centering; retain Q centering/correction | 7.325 ms | 1.051× | 0.7379% | −1.65% |
| No Q/K centering or correction | 6.571 ms | 1.172× | 1.0578% | +40.99% |

Timing uses CUDA events, 25 warmups and 12 batches × 12 calls per arm per
round, with two reversed-order rounds. Means are across rounds. Disposable K
reset is excluded from the timed region, matching E109's online API boundary.
Custom baseline output exactly matched the public Sage3 API output (NMSE=0).
Reference is BF16 PyTorch SDPA on the same original Q/K/V.

**Important implementation boundary:** no-correction arms reuse a preallocated
all-zero `delta_s` tensor. They skip constructing `Qmean × Kᵀ`, but the
unchanged native attention kernel still reads and applies zeros. These are
measured end-to-end preprocessing ablations, not measurements of a specialized
kernel with correction handling compiled out. Such a kernel could change both
speed and numerical output slightly. The accuracy result is only one prompt,
step and block and should not be generalized to all H3 calls without replay.

Script: `scripts/research/bench_e110_sage3_mean_ablation.py`.
Raw results: `sage3_mean_ablation_h3_p1_s0_b25_gpu0.json`.
