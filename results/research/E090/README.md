# E090 — SmoothQuant-style smoothing + native NVFP4 W4A4 on H3 FC2

Completed 2026-10-14. Error-only exploratory experiment on the same real
`blocks.25.mlp.fc2` FC2 weight and two cached calls as E089. No low-rank
branch, timestep sweep, whole-block output, speed, or generation is involved.

## Contract

Input `A` is `[tokens, K]`, weight `W` is `[output channels, K]`, and the
linear output is `A @ W.T`. Prompt 11 step 0 (`M=22464`) is the **calibration
call**; prompt 1 step 0 (`M=22400`) is the **cross-input evaluation call**.
The fixed SmoothQuant-style scale is

`s_j = max_calibration |A[:,j]|^alpha / max |W[:,j]|^(1-alpha)`

with both bases and the final scale clamped to at least `1e-5`. For every
alpha, native FlashInfer NVFP4 packs `(A/s).to(BF16)` and `(W*s).to(BF16)`
over **16 contiguous K values**, with E4M3 block scales and a second-level
FP32 global scale. The primary deployment-style arm derives the activation
global scale from the smoothed **calibration** call and uses it unchanged on
the evaluation call. Weight global scale comes from the smoothed weight.
The mathematical BF16 target is always the original `A @ W.T`.

The unsmoothed control reproduces E089 exactly: calibration-call NMSE
`0.009607384`, evaluation-call static-scale NMSE `0.010424481`. The extra
evaluation-current-global-scale control reproduces E089's `0.009643157`.
The alpha grid was fixed before the run; alpha is selected only on the
calibration call. The current-input global-scale control below is
**diagnostic**, not the primary static-scale recipe and not used to select
alpha.

## Complete-linear output NMSE

Values are NMSE × 100, i.e. percent of the original BF16 output energy.

| SmoothQuant alpha | p11 calibration, static | p1 evaluation, static | p1 evaluation, current-input global scale | p1 smoothed amax / p11 smoothed amax |
|---:|---:|---:|---:|---:|
| None (no smoothing) | 0.96074% | **1.04245%** | **0.96432%** | 2.05× |
| 0 | 1.09465% | 1.13473% | 1.08298% | 1.89× |
| 0.25 | 0.85745% | 1.19887% | **0.93104%** | 4.34× |
| 0.5 | **0.80352%** | **1.61375%** | **0.93261%** | 10.0× |
| 0.75 | 0.93640% | 2.93625% | 1.15102% | 30.13× |
| 1 | 1.41527% | 6.10482% | 1.84872% | 100× |

Alpha `0.5` is the calibration-selected variant. It reduces p11 in-sample
NMSE by **16.36% relative** but *increases* p1 static-scale NMSE by
**54.80% relative**. Alpha `0.25` also improves p11 (10.75% relative) but
worsens p1 static-scale NMSE (15.01% relative). Hence smoothing with the
same static activation-global-scale calibration contract is **not an
improvement on this cross-input test**.

The dynamic-global diagnostic changes only p1's activation-global scale,
keeping the p11-derived channel `s` and the smoothed weight fixed. Under
that different scale policy, calibration-selected alpha `0.5` improves p1
NMSE from `0.96432%` to `0.93261%` (**3.29% relative**). Alpha `0.25`
would reach `0.93104%` (3.45% relative), but this is a post-hoc observation
on p1, not an independently chosen alpha. A runtime reduction to get the
current-input global scale is outside this error-only experiment.

## Why the static path fails

With alpha `0.5`, smoothing cuts p11's activation amax from `2512` to `48`,
but p1's smoothed amax is `480`, or **10×** the calibration value. A static
NVFP4 scale derived from p11 can represent only the calibrated maximum with
the E4M3×E2M1 range, so this is a severe range mismatch. p1 activation
quantize-dequantize NMSE is `0.01446` with the static global scale but
`0.00538` with a current-input global scale. The corresponding unsmoothed
figures are `0.00564` and `0.00486`. This numerical evidence supports scale
mismatch as the immediate mechanism; it does not claim that all smoothing
variants will fail on broader calibration data.

BF16 roundoff from the reversible `A/s, W*s` transformation is only about
`8.5e-6` NMSE on p1 for nontrivial alpha values, far below the observed FP4
output errors. The difference is therefore dominated by quantization, not
the BF16 smoothing cast.

## Decision and limitations

Do not adopt this simple, single-call calibrated SmoothQuant recipe for this
H3 FC2. The more promising conditional result is **SmoothQuant plus a
current-input activation-global scale**, but its 3.29% relative NMSE gain is
from one linear and one cross-input call and needs more prompts/timesteps
before any H3-wide claim. An existing SVDQuant smoothing scale and low-rank
branch are different recipes; neither is included here.

Authoritative raw file:
`h3_fc2_p11cal_p1eval_smoothquant_nvfp4_diagnostic.json`.
The earlier `h3_fc2_p11cal_p1eval_smoothquant_nvfp4.json` contains the same
primary static-scale rows without the additional dynamic diagnostic.
Reproducer: `scripts/research/eval_h3_smoothquant_nvfp4_fc2.py`. Local logs
are under `results/logs/e090_h3_smoothquant_fc2*.log`.

Formula reference: [SmoothQuant's official implementation](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py).

## Activation shape check (p1 versus p11)

The complete raw FC2 inputs are `[22400, 14336]` for p1 and `[22464, 14336]`
for p11. Both cached calls are step 0 and use the same model/block/linear, but
their prompt IDs and seeds differ (22529 versus 1571); their prompt-embedding
lengths are 658 versus 738. Thus this is **not** a prompt-text-only controlled
comparison. Raw activation RMS is 8.262 versus 7.841, a modest difference
beside the maxima of 5152 versus 2512. Out of approximately 321–322 million
entries, p1 has only 9 values with absolute magnitude above 2512 (3 above
4000), while p11 has none above 2512. This supports a rare-outlier account
of the extreme-scale difference, without attributing it to prompt text alone.
The outlier channels are also different: at input channel 4880, the largest
absolute value over p1's tokens is 5152 but only 97.5 over p11's; at channel
8643 the order reverses (119 for p1, 2512 for p11). This directly explains
why p11-derived per-channel smoothing can underprepare a channel that becomes
an outlier in p1, although prompt text versus seed effects remain confounded.

3D figures in `figures/`: `h3_fc2_p1_p11_activation_3d_overview.png` is a
common-linear-z overview of exact per-bin maximum absolute values across all
token rows and channels; `h3_fc2_p1_p11_activation_3d_peaks.png` shows raw
**signed**, unpooled values in each prompt's separate peak window, using
individually labeled linear z-ranges. The overview bins are 350×224 tensor
cells for p1 and 351×224 for p11. Peak coordinates and full source paths are
recorded in `h3_fc2_p1_p11_activation_3d_metadata.json`.
