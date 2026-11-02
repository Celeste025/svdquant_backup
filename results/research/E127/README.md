# E127 — H3 plain NVFP4 W4A4 linears × attention

Requested generation: repeat the 12 E124 prompts and five E126 VBench prompts
with two new arms: plain NVFP4 W4A4 for all 200 main DiT target linears plus
BF16 SDPA attention, and the same W4A4 linears plus corrected Sage3 attention.
The refiner and other non-target linears stay BF16. No SVDQuant smoothing or
low-rank branch is used. The existing H3 fake-quant Q/DQ implementation uses
dynamic tensor-wise scales, E4M3 block-16 scales and E2M1 values. It then
performs BF16 GEMM; generation time is **not** native NVFP4 kernel latency.

Settings and prompt/seed sources are unchanged from E124 (576×1024, 124 frames,
20 steps) and E126 (768×1344, 124 frames, 50 steps). Input trajectories are
paired within each suite by seed and checked against prior initial latent hashes.
All output media and terminal latent checkpoints are stored outside Git at
`/data1/models/svdquant-wjq/research/20261024/E127_h3_w4a4_attention`.

Status: preflight pending.
