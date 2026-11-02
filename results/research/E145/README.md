# E145 — p025 retained 30% INT8 versus retained 30% NVFP4

Same prompt, seed, initial video/audio latents and generation contract as E144.
Steps 0–1 are fully dense; block 0 remains dense on steps 2–19; the other 49
blocks retain 30% of video-region interactions.  Compare all retained video
interactions in INT8 versus all in NVFP4.  E144's 20% NVFP4 + 10% INT8 arm is
included as a reference.

| arm | RGB NMSE vs BF16 | LPIPS-Alex mean | summed DiT time | vs BF16 time |
|---|---:|---:|---:|---:|
| 20% NVFP4 + 10% INT8 | 21.4466% | 0.3333 | 185.34 s | -12.2% |
| 30% INT8 | **19.8716%** | **0.3216** | **176.32 s** | **-16.5%** |
| 30% NVFP4 | 22.4835% | 0.3681 | 177.86 s | -15.8% |

All variants preserve readable tile markings and the final “胡了！” subtitle.
For this one closed-loop sample, all-INT8 retained attention is both the closest
to BF16 and marginally fastest.  Mixed precision is slowest because it executes
two precision phases.  Pure NVFP4 is slightly slower than pure INT8 here because
the official native API cannot run prefix-query INT8 without an active video
INT8 phase: its prefix K/V remain INT8 but prefix queries use dense BF16.  This
prefix difference means the INT8/NVFP4 timing comparison is representative of
these deployable API contracts, not a perfectly isolated video-kernel ablation.

The four-arm contact sheet and video show that metric differences mostly reflect
camera/pose/tile-layout trajectory drift rather than obvious text corruption.
One prompt is insufficient to establish a general precision ranking.
