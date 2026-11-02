# E144 — text/detail case, BF16 versus DraftAttention2 mixed precision

Selected E124 prompt 25 (seed 2141, category `文字`): four cats playing mahjong,
close-ups of tile faces, and requested final subtitle “胡了！”.  The completed
E124 BF16 reference was reused; only the mixed arm was regenerated.

Both arms use 576x1024, 124 frames, 20 steps, CFG 1, video/audio flow shifts
12/3, BF16 linears and identical initial video/audio latent hashes.  Mixed main
attention skips 70% of video-region block interactions, computes 20% in NVFP4
and 10% in INT8.  Steps 0–1 are fully BF16-dense; block 0 remains dense on every
later step; blocks 1–49 are sparse on steps 2–19 (882 sparse main calls). Refiner
and independent padding segments remain BF16.

## Results

| metric versus matched BF16 | mixed result |
|---|---:|
| decoded RGB NMSE | 21.4466% |
| LPIPS-Alex mean | 0.3333 |
| LPIPS-Alex median | 0.3424 |
| LPIPS-Alex maximum | 0.8049 |
| summed DiT time | 185.34 s |
| historical matched BF16 summed DiT time | 211.13 s |

The mixed video preserves the requested content surprisingly well despite the
large pixel/LPIPS trajectory divergence: both render four cats, readable
mahjong-tile markings, and a large legible final “胡了！” subtitle. Fine fur,
whiskers and tile edges remain visually sharp in the sampled frames. Differences
are mainly camera framing, cat pose, tile arrangement and motion trajectory, so
full-frame NMSE should not be interpreted as text-recognition failure.

Artifacts:

- `p025/evaluation.json`: full metrics and per-frame LPIPS;
- `p025/five_timepoints.jpg`: five paired timepoints;
- paired comparison MP4 under the E144 data directory (no audio);
- both individual originals retain their audio streams.
