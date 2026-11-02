# E144 — text/detail DraftAttention2 case

Use prompt 25 (seed 2141) from the immutable E124 twelve-prompt manifest: four
cats at a mahjong table, visible tile details, and requested final subtitle
“胡了！”. Reuse the completed E124 BF16 reference at 576x1024, 124 frames,
20 denoising steps, CFG 1, video/audio flow shifts 12/3 and CPU RNG.

Generate one paired DraftAttention2 arm with BF16 linears and main attention:

- 70% video-block interactions skipped;
- 20% of all interactions NVFP4 and 10% INT8 (retained ratio 2/3 vs 1/3);
- first two denoising steps fully BF16-dense;
- block 0 remains BF16-dense on every later step;
- blocks 1–49 use DraftAttention2 on steps 2–19;
- refiner and independent padding segments remain BF16.

Expected sparse main calls: 18 * 49 = 882. Verify prompt/seed, initial latent
hash against E124 BF16, all step outputs finite, and decode exactly 124 frames
with one audio stream. Compare the paired videos and report video NMSE/LPIPS.
