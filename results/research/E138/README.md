# E138 — activation stochastic rounding feasibility

Matched H3 rank-64 SVDQuant, dynamic NVFP4 tensor scales, group-16 E4M3
local scales, BF16 attention. Weight quantization, smoothing and low-rank
branches retain the existing recipe. Only activation E2M1 rounding changes.
For adjacent representable values l/u, select u with probability
`clamp((z-l)/(u-l),0,1)`, otherwise l. Saturation remains at the endpoints.
FP8 scale rounding is unchanged, so this is not a claim of overall unbiased
NVFP4 quantization. A dedicated CUDA RNG separates rounding randomness from
initial video/audio noise generation.

Four rollouts: race and clap0, each nearest versus activation stochastic
rounding, 20 steps, 576x1024, 124 frames, matched prompt/seed/checkpoint.
This uses fake quantization to screen quality; timing is not deployment speed.

Local diagnostic uses the saved prompt-1 step-0 block-0 FC1 input, shape
15552x5376. Tensor scale uses the entire input; output metrics cover first
2048 rows. This historical local payload is separate from the rank-64 rollout
checkpoint and is only a numerical diagnostic, not a rank-64 quality proxy.

Local result: nearest output NMSE 0.8526%, stochastic 1.3415–1.3442% over
three seeds. Centered adjacent-row output-error cosine 0.3618 versus
0.2439–0.2469. This index adjacency does not establish spatial/video-time
adjacency. It establishes a variance/correlation tradeoff that full rollout
must evaluate.

Success criterion: lower decoded LPIPS/RGB divergence with visibly preserved
content on both cases despite increased local MSE. Mixed results are
inconclusive. Consistent worsening is a reason to stop plain activation
stochastic rounding. A positive result still needs more prompts/seeds and
hardware-compatible implementation before becoming a research contribution.

Source: `scripts/research/run_e138_stochastic_rounding.py`.
Media/tensors: `/data1/models/svdquant-wjq/research/20261031/E138_stochastic_rounding`.
Launch logs: `results/logs/e138_{race,clap}_{nearest,sr}.log`.

## Completed rollout result

All four arms completed 20 steps and 124-frame videos with audio. Initial
video/audio latent SHA256 matches within each pair and with its BF16 reference;
the nearest/SR checkpoint hashes also match. Full-resolution LPIPS-Alex and
RGB NMSE use all 124 frames against the E121/E125 BF16 videos.

| Case | Rounding | RGB NMSE | LPIPS-Alex |
|---|---|---:|---:|
| Race | Nearest | 0.239484 | 0.397907 |
| Race | Activation SR | 0.246978 | 0.422399 |
| Clap | Nearest | 0.573803 | 0.500562 |
| Clap | Activation SR | 0.622723 | 0.578404 |

Decision: stop plain activation stochastic rounding as a candidate method.
Both measured BF16-reference distances worsen in both cases. This rejects
the cheap proposed implementation, not every possible error-structure method.
One rounding seed per rollout and two prompts cannot establish a broad
perceptual-quality ranking. Local diagnostics and rollout use different
checkpoint payloads, so their metric changes are not a quantitative mediation
analysis. The new nearest race baseline is rank-64 and must not be confused
with E105/E137's older rank-32 baseline (RGB NMSE about 0.519).

Numerical records: `race/video_metrics.json`, `clap0/video_metrics.json`,
`local_fc_check.json`, `rounding_validation.json`.
