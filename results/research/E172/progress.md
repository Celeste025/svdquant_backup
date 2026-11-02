# Optimization progress: E170–E172

Goal remains active: deeply optimize first capture, route reuse, QKV preparation,
route handling and NVFP4 main attention. Current work is a verified initial
improvement, not completion of the goal.

Current package-local SM120 binary SHA256:
7290bdf6fd56736145eef6e530e2b2edfe940f96648e9d23b24a1d3a1296f97d.
Original binary/source backup: results/local/E170/source_before.
E170 binary backup: results/local/E172/source_before.
Current optimized binary backup: results/local/E172/optimized_sm120.so.

## Accepted experimental source changes

1. q128_microscaling_preparation.cu: skip FP16 packed Q/K stores when both
   has_pool and has_fp16 are false. Tensor shapes/allocations are retained for
   the existing interface. This applies to capture/reuse paths with prepared
   low-precision operands; proxy and FP16 paths retain the writes.
2. q64_attention.cuh: exact-importance reduction computes the maximum before
   multiplying by the shared positive scale. Per-element products become one
   scaled tile maximum. Preserve the original finite lower bound for fully
   masked tiles after scaling. Attention arithmetic is unchanged.

Direct original-versus-optimized tests on real 480p and 768p 124-frame inputs
produce identical SHA256 hashes for all four attention outputs and all captured
route tensors. Raw hashes: *_original_check_hashes.json and
*_after_check_hashes.json in E172. Original binary was temporarily selected
under an EXIT restoration trap; current package binary was verified restored.

## Timings (GPU1, Q128, skip0, 90/10, 5 warmups + 12 calls)

| Input | Original capture | E170 capture | E172 capture | E172 reuse |
|---|---:|---:|---:|---:|
| 480p/124f | 36.809 ms | 36.564 ms | 35.914 ms | 29.024 ms |
| 768p/124f | 136.198 ms | 136.577 ms | 134.171 ms | 99.155 ms |

Original 768p timing was on GPU2, so do not treat the absolute cross-build
delta as a rigorous paired speedup. E170 and E172 were on GPU1. Repeat/order
controls remain necessary before claiming a stable small performance gain.

## Rejected as a large speedup: physical-route memoization

E171 caches the materialized int32 physical route for the same fixed cached
input plan. It materializes once; output is bitwise equal. 30-call ABA results:
480p baseline 29.048 / 29.042 ms, cached 28.936 ms;
768p baseline 99.507 / 99.897 ms, cached 99.572 ms.
This does not recover the apparent 3/16 ms in-chain materialization event span.
The prototype stays in benchmark code, not the production cache API.

## Next concrete investigations

- Capture importance currently introduces two CTA barriers per tile. Test
  per-warp partial maxima with a separate reduction against the exact current
  map, retaining the same route and dense INT8 attention output.
- NVFP4 main loop currently performs several CTA synchronizations per K64
  tile, FP4 probability conversion, and one K/V shared-memory buffer. Profile
  instruction/occupancy and test pipeline changes with identical routes.
- Analyze per-row precision counts to determine whether static routes can
  eliminate preparation for unused precisions. Dense INT8 prefix still needs
  global INT8 K/V, so do not assume all 90%-FP4 tiles avoid INT8 preparation.
- Expand correctness to Q64/long inputs and repeated/reversed timings before
  finalizing any implementation.
