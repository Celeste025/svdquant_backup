# E009b: existing FlashInfer fused-up feasibility

2026-10-02, preregistered bounded engineering check. This is not a new method.

Use a new isolated environment inheriting the frozen FlashInfer 0.7.0.post1 environment. Candidate CuTe 4.7.0a0 was unavailable; pin release 4.7.0 (satisfies >=4.7.0a0), retain torch2.11/cu128 and cuda-python12.9.7. Save resolution/wheels/SHA. No package upgrade in prior environments.

1. Synthetic M128/N128/K256 and M257/N256/K512, rank32. Nonuniform E4M3 SF, non-power-of-two FP32 globals, separate main-only/up-only/combined. Compare fused output with independent FP32 same-packet + BF16-rescaled-up reference; finite and NMSE<=1e-4. Compare official unfused and existing torch native path, report numerical differences rather than demand bit equality.
2. Once E009 exporter has passed its independent parity gates, real block0 qkv/fc2, same weights and legacy activation packets, M512 and M22400. Preserve BF16 division smoothing and original BF16 down. Compute alpha=gA*gW and BF16(B/alpha) every call. Validate finiteness and isolate main, up, and combined numerical changes. Same-code main safety threshold NMSE<=1e-4; full combined drift reported, not labeled quality.
3. For these four shapes only: same-code prepared-packet component timings (1 warmup + 5 repeats), with and without down/dynamic-up rescaling cost. Packing and smoothing excluded and explicitly labeled; no whole-linear/DiT acceleration claim. Profile actual fused kernel.

JIT/support budget <=60min. Stop if dependencies/SM120 compilation cannot be resolved within budget; no custom fused-GEMM reimplementation. Full-model integration requires same-environment A replay and separate endpoint propagation check. No high-level svdquant_linear import that changes the smoothing/rounding contract silently.

Reference: [FlashInfer mm_nvfp4_svdquant official API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_nvfp4_svdquant.html), local installed source SHA captured by runner.
