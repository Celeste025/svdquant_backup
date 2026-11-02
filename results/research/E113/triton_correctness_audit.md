# E113 mixed P/V Triton correctness audit

Scope: the local `third_party/MixAttention/triton_adaptive_pv.py` implementation,
used by `scripts/research/e113_mixed_attention.py`. This is a numerical
BF16-matmul emulation of FP4/FP8 P/V quantization, not a native packed
FP4/FP8 Tensor Core kernel or deployable speed measurement.

The original kernel computed the online-softmax running maximum separately
for each 16-key quantization group. Its PyTorch reference computes the maximum
once per 128-key KV tile before quantizing the eight P groups. With quantized
P scales these are not strictly equivalent. The local Triton copy was fixed to
find a 128-key tile maximum first, then use it for every 16-key P group; the
softmax denominator remains computed from unquantized P.

Validation uses the saved real H3 block-25 Q/K/V capture
`/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt`.
`scripts/research/validate_e113_mixed_triton.py` compares the corrected Triton
output with an independent PyTorch implementation that uses the same official
Sage3 Q/K preprocessing and per-head V-tile decision. It asserts finite output
and Triton-vs-reference NMSE below 1e-6 for each tested case. Thresholds
0.001, 0.10, and 0.50 exercise all-FP8, actual mixed, and all-FP4 routing.
The mixed test for head 4 selects 5 of 8 full KV tiles for FP4. It also tests
a 54-token padded sequence and a full 9,482-token main sequence.

Observed post-fix output NMSE versus the PyTorch implementation is
4.90e-9/5.15e-9 for the two mixed 1,024-token heads and 8.35e-8 for the
full 9,482-token head. See `triton_correctness.json` for all 13 cases.
Remaining differences are floating-point/GEMM ordering and BF16 output
rounding. These checks establish numerical consistency for the covered H3
shapes, not general correctness for arbitrary shapes or the speed of a future
native mixed-bit kernel.
