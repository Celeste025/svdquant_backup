# E145 — retained-pair precision comparison on text/detail prompt

Reuse E144 prompt 25, seed 2141, BF16 reference and generation contract.  Both
new arms use 70% skip, first two steps fully dense, and block 0 dense on every
later step. Compare the remaining 30% video-region interactions as either all
INT8 or all NVFP4. Linears, refiner and independent padding remain BF16.

The all-NVFP4 video phase cannot use the native prefix-query INT8 kernel because
there is no active video INT8 phase. Its prefix K/V remain INT8 while prefix
queries use dense BF16. This difference is explicit in reports and must be
considered when interpreting generation quality and time.

Verify identical initial latent hashes against E124/E144, 882 sparse main calls,
finite outputs and complete 124-frame videos. Report decoded RGB NMSE/LPIPS to
BF16 and render BF16 / mixed 20+10 / pure INT8 / pure NVFP4 comparison media.
