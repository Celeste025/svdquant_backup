# Experimental mixed-precision Sage3 operators

This branch contains two noncausal SM120 attention paths for BF16 Q/K/V of
shape `[1, H, L, 128]`:

- `native_mixed_attention`: corrected Sage3 native mixed FP4/FP8 kernel.
- `split_mixed_attention`: spill-free FP4 kernel plus sparse FP8 P×V kernel.

Both use the same dynamic, V-error-based routing threshold (`0.10` by
default), Q/K/V preprocessing and corrected K-tail mask. The split path is
experimental: it is not yet integrated into H3 generation and does not
produce bitwise-identical output to the native mixed kernel. See
`results/research/E120/README.md` for the numerical and timing comparison.
The noncausal tail-mask finding and repair are documented separately in
`docs/sage3_noncausal_tail_mask_bug.md`. The fixed vendored Sage3 fork is
used here; an external upstream Sage3 installation is not changed.

## Build on the current server

The build scripts use this machine's CUDA 12.8, CUTLASS and Python paths.
They compile into their respective `third_party/` directories; binary
extensions are intentionally not committed.

```bash
bash scripts/research/build_e116_sage3_maskfix.sh
bash scripts/research/build_e117_sage3_router_sparse.sh
bash scripts/research/build_e119_sage3_lse.sh
```

The first command builds the corrected pure FP4 core, FP4 Q/K/V quantizer,
and corrected native mixed core. The second builds the sparse V8 router;
the third builds the pure-FP4 core variant that additionally exports LSE.
Use the H3 SVDQuant Python environment listed in `AGENTS.md` (on this host,
`/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python`).

## Call from Python

Run from the repository root, with `q`, `k` and `v` already on the same CUDA
device as BF16 tensors:

```python
from scripts.research.h3_mixed_sage3_api import (
    native_mixed_attention,
    split_mixed_attention,
)

# q.shape == k.shape == v.shape == (1, heads, tokens, 128)
out_native = native_mixed_attention(q, k, v, threshold=0.10)
out_split = split_mixed_attention(q, k, v, threshold=0.10)
# Both outputs: BF16 [1, heads, tokens, 128].
```

These functions allocate and quantize Q/K/V on every call; the input K is
cloned because Sage3 preprocessing mutates it. The split path additionally
reserves unpacked Q/K scratch buffers. They currently support batch size 1,
head dimension 128, noncausal attention, and per-128-token Q centering.
They are research wrappers, not a production/autograd API.

For the original measured H3 capture, run
`scripts/research/e120_split_native_mixed.py` for component/numerical checks
or `scripts/research/e120_full_online_benchmark.py` for 20-warmup,
96-CUDA-event-call online benchmarks. These benchmark scripts refer to a
local, unversioned QKV capture and are not generic input loaders.
