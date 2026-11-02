# E154 — exact first-step route capture and reuse

## Outcome

The requested execution mode is implemented for SM120 Q64 and Q128:

1. `first_step=True` skips the rough DraftMap, prepares only INT8 operands,
   returns dense INT8 attention, and reduces the actual quantized QK values
   into a per-tile maximum importance map inside the main kernel. The final
   CTA-wide reduction uses shared memory and writes one FP16 value per tile;
   it does not issue a global atomic for every contributing warp.
2. The existing global budget router converts this exact map into the selected
   NVFP4/INT8 tile lists. The selected route is cached per layer.
3. `first_step=False` skips pooling, DraftMap scoring, and route sorting, then
   directly materializes and executes the cached route.
4. The cache uses a ragged int16 ID stream and native CUDA materializer rather
   than retaining the full square int32 route matrix.

## Correctness evidence

Real MiniMax-H3 block-0 QKV captures were used (56 heads, head dimension 128,
non-causal). The policy was 75% skipped tiles and, among retained tiles, 60%
NVFP4 plus 40% INT8.

| Q tile | Tokens | First-step vs ordinary dense INT8 NMSE | Reuse vs legacy proxy-route NMSE | Reuse vs dense INT8 NMSE |
|---:|---:|---:|---:|---:|
| 64 | 38,661 | 0 | 0.008038 | 0.017385 |
| 128 | 38,661 | 0 | 0.009887 | 0.019045 |
| 64 | 110,013 | 0 | 0.003725 | 0.007313 |

The reuse call was also executed with both DraftMap scoring and route selection
patched to raise. It completed, proving that the later-step path does not
silently recompute either operation.

The “reuse vs legacy proxy-route” number measures the difference between two
routing policies; it is not an error-to-BF16 quality result.

## Cache memory

| Q tile | Tokens | Full-width int32 prototype | Final ragged cache | Reduction | Approx. 50 layers |
|---:|---:|---:|---:|---:|---:|
| 64 | 38,661 | 78.90 MB | 10.34 MB | 86.9% | 0.52 GB |
| 128 | 38,661 | — | 2.72 MB | — | 0.14 GB |
| 64 | 110,013 | about 0.66 GB | 83.60 MB | about 87% | 4.18 GB |

## Clean-card paired timing

The following full-API measurements used an otherwise idle RTX PRO 5000 and
CUDA events. They include operand preparation and routing work in each path,
but exclude model loading.

| Q tile | Tokens | Cached exact-route reuse | Legacy DraftMap every step | Later-step reduction |
|---:|---:|---:|---:|---:|
| 64 | 38,661 | 42.287 ms | 44.914 ms | 5.85% |
| 128 | 38,661 | 40.797 ms | 42.441 ms | 3.87% |
| 64 | 110,013 | 271.612 ms | 290.550 ms | 6.52% |

The cached path gives a real but modest 3.9--6.5% later-step saving. The former
shared-GPU timing snapshot overstated the gain and is intentionally superseded
by this table.

## Importance-capture overhead

An isolated prepacked dense-INT8 kernel comparison shows that exact in-kernel
capture is still the main cost. The stable shared-reduction implementation
preserves the attention output exactly (capture vs plain NMSE = 0):

| Q tile | Tokens | Plain dense INT8 | Dense + capture | Capture overhead |
|---:|---:|---:|---:|---:|
| 64 | 38,661 | 116.89 ms | 171.33 ms | 46.57% |
| 128 | 38,661 | 101.36 ms | 122.31 ms | 20.67% |

Q128 is therefore the practical default for this execution mode: it has lower
capture overhead and a substantially smaller route cache. Removing the final
CTA synchronization was tested and rejected because it did not improve either
Q64 or Q128 timing.

A final deliberately small Q128 full-API smoke run (one warmup, two measured
calls) reported 131.49 ms for first-step dense capture, 40.87 ms for cached
reuse, and 42.48 ms for rebuilding the legacy proxy route. It also verified
that later reuse completes while DraftMap and route-selection functions are
mocked to raise.

## Artifacts

- `race_768p_124f_step0_block0_q64.json`
- `race_768p_124f_step0_block0_q128.json`
- `race_768p_362f_step0_block0.json` (Q64 long-sequence run)
- `scripts/research/validate_e154_exact_route_reuse.py`
- `scripts/research/profile_e154_capture_overhead.py` (`--quick` for the two
  representative 38k-token cases)
- `scripts/research/build_e154_exact_route_reuse.sh`
