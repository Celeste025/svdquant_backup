# E146 — p025 retained 20% precision comparison

Reuse the E124/E144/E145 p025 contract and initial noise. Steps 0–1 are fully
dense and block 0 remains dense on steps 2–19. For blocks 1–49 retain only 20%
of video-region interactions (80% skip), comparing all retained work in INT8
against an equal split of 10% total INT8 and 10% total NVFP4. Prefix Q/K/V stay
on the same official INT8 path for both arms; linears, refiners and independent
padding remain BF16. Expected sparse main calls per arm: 882.

Verify prompt/seed/settings and initial latent hashes, finite 20-step output,
complete 124-frame video, decoded RGB NMSE/LPIPS versus BF16, and paired media.
