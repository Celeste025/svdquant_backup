# E128 — native NVFP4 W4A4 linears × BF16/Sage3 attention

This supersedes the E127 fake-quant generation. E127's incomplete logs are
retained separately. E128 uses the already exported E014 original-W plain
NVFP4 checkpoint for all 200 main DiT linears, with the validated fast
dynamic activation packer and real `torch.nn.functional.scaled_mm` FP4 GEMM.
There is no smoothing or low-rank branch. The eight refiner linears and all
other layers stay BF16. Attention is either BF16 SDPA or corrected Sage3.

The 12 E124 prompts and five E126 VBench prompts keep their original
prompt/seed/resolution/frame/step configurations. E014's packed weight files
were deleted in a prior disk cleanup. E128 rehydrates them under
`/data1/models/svdquant-wjq/research/20261024/E128_plain_export_rehydrated`,
requiring every source-weight and NVFP4 code/scale hash to match the retained
E014 manifest;
the original BF16 source checkpoint is checked by path, size and mtime before
installation. Each native DiT forward validates 200 activation-pack calls.
Media and final latents go to
`/data1/models/svdquant-wjq/research/20261024/E128_h3_native_w4a4_attention`.

Status: **complete**. The final audit passed **34/34 videos** with no missing
arms; each MP4 hash, 124-frame dimensions, full denoising-step count and
initial-latent pairing against the original BF16 arm were checked. See
`final_audit.json`. Both native preflights passed. E124-size first DiT 6.10 s vs fake-QDQ
20.99 s; E126-size first DiT with fixed Sage3 10.05 s vs fake-QDQ 36.34 s.
All 200 activation packs per forward passed numerical-domain validation;
peak allocated memory was 37.60 GiB on a 72 GiB GPU.
