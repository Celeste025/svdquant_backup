# E105 — H3 output-error interventions (completed 2026-10-16)

Two fixed 576×1024, 124-frame, 20-step MiniMax-H3 cases use the E101 rank-32 dynamic-scale SVDQuant checkpoint (`e5ccf2d809f0b03e6a0eaef400e4f610ccfc37478eb3715d75040cf07446652d`). Race is prompt 16/seed 20026; clap is prompt 161/seed 38001610. Each intervention recomputes BF16 and SVDQuant full-DiT outputs on *its own current input* at every step. The existing BF16 and original SVDQuant videos are the visual controls. All six intervention runs have 20 finite video/audio steps and validated 124-frame MP4s with audio. The first paired step matches E101.

| Case | Arm | Mean video output NMSE on own trajectory | Final decoded RGB NMSE vs BF16 control | Change vs original SVDQuant RGB NMSE |
|---|---|---:|---:|---:|
| Race | Original SVDQuant | 0.07417 | 0.51910 | — |
| Race | Restore `>4σ` error elements | 0.06748 realized (0.07279 target) | 0.44543 | −14.2% |
| Race | Equal-magnitude random-sign error | 0.07399 realized (0.07406 target) | 0.10188 | −80.4% |
| Race | Equal-energy Gaussian error | 0.07624 realized (0.07624 target) | 0.07994 | −84.6% |
| Clap | Original SVDQuant | 0.02263 | 0.57200 | — |
| Clap | Restore `>4σ` error elements | 0.02125 realized (0.02348 target) | 0.56206 | −1.7% |
| Clap | Equal-magnitude random-sign error | 0.02427 realized (0.02428 target) | 0.28419 | −50.3% |
| Clap | Equal-energy Gaussian error | 0.02335 realized (0.02334 target) | 0.29364 | −48.7% |

The tail intervention restores a mean 0.308%/0.305% of video output elements in race/clap, removing 7.87%/7.48% of squared video output error on the intervention trajectories. Its final video effect is much smaller and case dependent. The random-sign and Gaussian arms match the locally measured quantized video/audio MSE to within 0.5% at every step after BF16 rounding; their average video output NMSE is similar to, or slightly worse than, the original SVDQuant run, yet their final RGB divergence from BF16 is much lower in both cases. Four sampled frames per video show that the random arms retain the BF16 scene layout and camera trajectory more closely than SVDQuant, especially for the clap case; the tail arm resembles SVDQuant. This is a qualitative inspection, not a blind perceptual score.

This supports a **two-case exploratory observation**: the *form* of the complete-DiT output error matters to closed-loop generation beyond its aggregate NMSE. It does **not** isolate whether spatial/temporal correlation, channel bias, heavy tails, systematic direction, or cross-modal error is responsible. The matched target is recomputed on each arm's divergent trajectory, not fixed to the original BF16 trajectory. Decoded H.264 RGB distance is a trajectory-divergence proxy, not video quality; the videos include audio but no listening-based audio-quality conclusion is made. Replicate prompts/seeds and isolate one error statistic at a time before making a general causal claim.

Artifacts: [race poster](race/five_arm_poster.jpg), [clap poster](clap0/five_arm_poster.jpg), [race summary](race/five_arm_summary.json), [clap summary](clap0/five_arm_summary.json). The two five-arm, 124-frame silent comparison montages (original individual videos contain audio) are at `/data1/models/svdquant-wjq/research/20261016/E105_h3_output_interventions/{race,clap0}/{case}_five_arm_montage.mp4`. Per-arm denoise and video JSON reports are in the corresponding case directories. Large terminal latents and individual intervention MP4s are on `/data1`. Source: `scripts/research/run_e105_h3_output_interventions.py`, `scripts/research/launch_e105_h3_output_interventions.sh`, `scripts/research/summarize_e105_h3_output_interventions.py`.

Runtime: first denoising arms started 22:15; the last of six intervention MP4s finished 23:01 (Asia/Shanghai). Four GPUs were used without terminating other jobs. Earlier failed preflight/first-attempt logs are retained; they were corrected before these final complete runs and are not used in the table.
