# Experiment log

- E214: full four-prompt head/layer/step completeness experiment, plan in decision_value_plan_E214.md; execution and findings recorded in results/research/E214.
- E214 complete: four prompts × 20 steps × 50 layers × 56 heads; strict reversal 13,932/224,000 head observations, layer-aggregate reversal 75/4,000; see results/research/E214/README.md for definitions, effect sizes and limitations.

## E142 — completed 2026-10-31

Three real BF16-trajectory QKV captures,race s0b0/race s10b25/clap s10b49;
19arms each,full output and video-only errors,144CUDA-event measurements per
arm.50skip/25FP4/25INT8 is faster and lower video NMSE than optimized Sage3
on all3calls; tail49prefers pureINT8 within Sage3 latency.70skip/30INT8 is
~1ms slower but lower error than70/20/10 on all3. Sage3 tail videoNMSE13.43%
confirmed identical with non-optimized path; inputs unchanged. Plot/CSV/raw
reports complete; no video-quality generalization. See E142/README.md.

## E141 — completed 2026-10-31

First2steps dense,no layer exemption. Four paired race/clap rollouts completed,
900 sparse/100dense main calls per arm,BF16 linears,70% skip,same precision
ratios. Mixed RGBNMSE/LPIPS race .17291/.31564,clap .12653/.19564: closer
than E140 all-sparse,but farther than E13910steps+2layers dense. INT8
.17342/.30652 and .40823/.35198; precision ranking varies by case. All124-frame
metrics,initial pairing and comparison media validated. See E141/README.md.
All jobs ended; no new quality or stable timing ranking claim.

## E136 — completed 2026-10-28

Four H3 precision-window comparisons on the matched race `p016` and swimming
`vbench_000` cases. The requested first-only and last-only BF16/SVDQuant arms
are complete E134 results and will be reused after integrity checks. New arms
are symmetric W8A8 group-128 Q/DQ at both ends with rank-64 SVDQuant W4A4 in
the middle, and full-BF16 ends with rank-64 SVDQuant plus corrected Sage3 in
the middle. GPU 4/5 run race, GPU 6/7 run swimming. This is an exploratory
closed-loop generation comparison, not a deployment-speed benchmark for INT8.
All four new race/swimming video generations finished with matched initial
latents, finite outputs and the intended 20/50-step precision schedules.
Corrected Sage3 ran exactly 50 main FP4 and 2 BF16 refiner attention calls
per middle step. Reused E134 arms passed the same pairing check. First-only
BF16 was the dominant error reduction in these two cases; last-only BF16 had
negligible RGB NMSE effect. INT8 K128 endpoints nearly matched BF16 endpoints
in swimming but not race. Sage3 middle increased RGB NMSE versus the matching
BF16-attention middle in both cases. See the plan and
`results/research/E136/README.md` for the complete table and caveats. Stop
after these two exploratory cases; broaden prompts only for a subsequent
population-level claim.

## E098 — completed 2026-10-14

Captured real H3 prompt-1 step-0 block-25 post-norm/RoPE QKV at
480×832, 73 frames (~3.04 s), then benchmarked official SageAttention2
FP32 path, SageAttention2++ and SageAttention3 Blackwell on GPU 0 against
BF16 SDPA. The main 9,482-token segment took 12.269 ms BF16, 7.362 ms
Sage2 FP32 path, 7.490 ms Sage2++, and 8.897 ms Sage3. Including H3's
54-token tail and output assembly: 12.643, 8.028, 8.156, and 9.192 ms
respectively. Main-output NMSEs were 0.0212% Sage2 FP32 path, 0.0210%
Sage2++, and 0.7502% Sage3 versus BF16. The Sage2 FP32 path is not the
paper's per-thread INT4 variant. Independent profiling isolates fused-attention,
quantization and preprocessing kernels; see `results/research/E098/README.md`
and the raw JSON. Scope is one actual-input attention call, not a DiT block
or end-to-end inference. The older 22,384-token diagnostic is not 3 s.
Follow-up paper-TOPS audit: official charts exclude quantization and
smoothing; direct-core timing on the real H3 call yielded 412.4 TOPS
Sage2 FP32, 402.5 TOPS Sage2++, and 465.1 TOPS Sage3, versus the
previous full-API 350.1/344.2/289.8 TOPS. Direct-core outputs replayed
the public APIs exactly. Synthetic B4/H32/N8192 changed core rates only
slightly. The remaining gap to RTX-5090 paper peaks is not yet isolated.

## E097 — completed 2026-10-14

Isolated GPU-0 full-shape H3 FC2 native-operator benchmark, following
`decision_value_plan_E097_h3_deployment_linear_gpu0.md`. All successful
arms passed the 3% timing-CV gate with 50 warmups and 11×30 timed calls.
Paired BF16 was ~14.1 ms. vLLM W8A8 Triton INT8 full path was 8.998 ms
(NMSE 0.2319%); FlashInfer NVFP4 W4A4 CUTLASS full path was 4.971 ms
(NMSE 1.0424%). vLLM Marlin group-128 INT4 W4A16 was 15.983 ms
(NMSE 1.0438%), and FlashInfer CuTe NVFP4 W4A16 was 55.149 ms
(NMSE 0.6391%). The latter two are native but do not accelerate this
large-token shape. W4A4 auto backend was no faster than explicit CUTLASS;
W4A16 cuDNN was unavailable. See `results/research/E097/README.md` and
the raw JSON/logs. Stop this one-linear baseline; do not extrapolate to
full-DiT runtime.

## E096 — completed 2026-10-14

H3 block-25 FC2 top-0.1% absolute raw-activation clearing on the p11/p1
21,312-row video calls: exactly 305,529 entries per call, deterministic
tie handling. Both native plain and p99-smoothed alpha-0.5 W4A4 use the
same cleaned BF16 target; smoothing is recalibrated on cleaned p11.
On held-out p1, current-scale plain NMSE rises 0.96824%→1.14462% and
smoothed 0.97224%→1.11975%. Cleaned-data smoothing gains only 2.17%
relative over plain. Cleanup changes the underlying BF16 output by
35.16% NMSE, so it is not a valid deployment improvement without
preserving removed values. See
`decision_value_plan_E096_h3_top01pct_clean_nvfp4.md` and
`results/research/E096/README.md`. Stop this literal-removal screen;
do not infer other layers/timesteps.

## E095 — completed 2026-10-14

Literal activation-outlier zeroing before native H3 FC2 NVFP4 W4A4.
At p1, zeroing only nine values above 2512 reduces amax 5152→2272
but raises final original-BF16-target NMSE 0.9643%→1.5242% (current
global scale); removal alone causes 0.5632% NMSE. At E094's p99
alpha-0.5 p1 video, zeroing 91 smoothed entries above p11-video amax
388 reduces quantization-only error 1.0863%→0.9540% but raises final
error 1.0863%→1.7027%; 0.7491% comes from signal removal. Stop literal
zeroing; an outlier-preserving side path remains a distinct untested
option. A same-mask/same-clean-BF16-target paired p1-video check shows
plain 0.97260% versus p99-smoothed 0.96313% NMSE with the 91-entry
smoothed-outlier mask and current-input scales: only 0.97% relative
smoothing gain. With the 8-entry raw-amax mask, smoothing is worse.
See `decision_value_plan_E095_h3_outlier_zeroing_nvfp4.md` and
`results/research/E095/README.md`.

## E094 — completed 2026-10-14

H3 block-25 FC2 video-only versus mixed-modality p99 SmoothQuant on native
NVFP4 W4A4. Both arms evaluate the same p11/p1 21,312 video rows; p1's
16 padding rows are excluded. Alpha 0.5 selected on p11 video for each.
Video-only calibration slightly lowers in-sample video NMSE (0.73075%
to 0.72761%) but raises p1 static-scale video NMSE (0.96001% to
1.08629%); with a current-input scale, 0.95984% to 0.97224%. Its p11
smoothed amax is 388 versus mixed 844, but p1 video amax remains ~900,
creating a 2.32x versus 1.09x scale mismatch. Stop this one-call
video-only p99 recipe; see `decision_value_plan_E094_h3_video_only_p99_smoothing.md`
and `results/research/E094/README.md`.

## E093 — completed 2026-10-14

Plain NVFP4 W4A4 operand-error distribution on real H3 block-25 FC2 p1/p11
step-0 calls and BF16 weight, using current-input activation global scales.
Quantify per-modality/row/channel/K16-group and weight-row concentration;
see `decision_value_plan_E093_h3_nvfp4_operand_error_distribution.md`.
Full p1/p11 activation and weight native-code Q/DQ reductions passed E090
operand NMSE gates (<2e-10 absolute). Weight error is nearly uniform across
output rows/K16 groups; activation absolute error varies by row/channel,
but its signal energy is more concentrated still. Video activation NMSE is
~0.491% versus text 0.365%-0.373%. See `results/research/E093/README.md`
and raw JSON. Stop this one-layer distribution probe; extend only if a
targeted method requires multi-layer evidence.

## E092 — completed 2026-10-14

H3 block-25 FC2 error-only comparison of plain signed INT4 W4A4 K-group-128
against absmax- and abs-p99-derived static channel smoothing. Same p11/p1
cached calls as E090/E091; dynamic per-row K128 activation quantization and
offline per-output-row K128 weight quantization. A matched signed-INT4 K16
error-only control isolates group granularity. See
`decision_value_plan_E092_h3_int4_g128_smoothing.md`. Native INT4 kernel speed
is out of scope.
Full-shape p11 calibration / p1 held-out runs and integer-code gates passed.
Plain p1 NMSE was 0.861% at K16 and 2.604% at K128. At K128, p99
smoothing selected on p11 reduced p11 NMSE 2.585% to 2.171%, but increased
p1 NMSE 2.604% to 2.680%; absmax only reduced p1 to 2.598%. K16 p99
also selected an overfitting alpha and raised p1 to 1.114%. Stop this
single-call p99 recipe; see `results/research/E092/README.md` and raw JSON.

## E090 — completed 2026-10-14

SmoothQuant-style channel smoothing (`A/s`, `W*s`) + native NVFP4 W4A4 on
the same real H3 block-25 FC2 inputs as E089. Prompt 11 step 0 supplies
calibration channel maxima and alpha selection; prompt 1 step 0 is cross-input
evaluation. See `decision_value_plan_E090_h3_smoothquant_nvfp4_fc2.md`.
Alpha 0.5 was best on the calibration call (16.36% relative NMSE reduction)
but worsened cross-input static-scale NMSE by 54.80%. A current-input
global-scale diagnostic converted this to a 3.29% relative gain, isolating
the static scale mismatch. See `results/research/E090/README.md` and its
raw JSON. Stop this single-call static-calibration variant; broader dynamic-
scale evidence remains future work.

## E089 — completed 2026-10-14

MiniMax-H3 NVFP4 W4A4 second-level scale granularity: per-tensor versus
per-512/128/32/1 rows for activation and weight. Error-only exploration on
real block-25 FC2 tensors. See
`decision_value_plan_E089_h3_nvfp4_global_scale_granularity.md`.
Two cross-calibrated real calls passed the native-NVFP4-versus-dequantized-
GEMM numerical gate. Per-row activation scaling improved output NMSE by
0.735% and 0.794% relative to a current-input per-tensor activation scale;
32/128/512-row groups improved less. Full results and limitations are at
`results/research/E089/README.md`. Stop at this exploratory two-call FC2
result: improvement is below the predeclared 2% relative decision threshold.

## E085 — started 2026-10-06

MiniMax-H3 SVDQuant timestep NMSE split validation.  The calibration-side
teacher-forced measurement already exists at
`results/reports/minimax_h3_svdquant_standard_8p64s/calib_teacher_forced_step_error.json`.
The held-out side reuses standard sample prompt IDs 2, 16, 26, 42, and 51.
Prompt 16 has an existing full teacher-forced output; prompts 2, 26, 42, and
51 are to be run on GPU 1.  See `decision_value_plan_E085_h3_svdquant_timestep_generalization.md`.

## E102 — completed 2026-10-15

H3 `blocks.0.mlp.fc1` prompt 1/step 0 real activation, paired dynamic versus
fixed-one FP32 activation global scale, same rank-32 SVDQuant checkpoint and
BF16 `Wx` reference. Measured dynamic scale 0.0007178897. Output NMSE 1.2368%
versus 1.2600% (+1.88% relative for fixed-one); activation NMSE 0.8431%
versus 0.8770%. Full input/weight capture retained on `/data1`. See
`decision_value_plan_E102_h3_fc_global_scale.md` and
`results/research/E102/README.md`. Stop at the predeclared one-layer decision
point; do not generalize to the whole model.

## E103 — completed 2026-10-16

Captured all 20 paired complete-DiT BF16/rank-32 SVDQuant outputs for the
race prompt 16/seed 20026. Raw signed FP32 errors and both BF16-valued arms
are stored as 20 hash-checked files on `/data1`; all per-step NMSE values
match E101. Initial analysis found spatial/temporal error clustering and
moderate, stable channel differences, with pooled video NMSE 8.19%.
See `decision_value_plan_E103_h3_output_error_patterns.md` and
`results/research/E103/README.md`. Quality impact at equal NMSE remains an
unanswered follow-up, not a conclusion of this capture.

## E105 — completed 2026-10-16

Three closed-loop complete-DiT output-error interventions on the fixed race
and clap cases: restore elements beyond 4σ, equal-energy random-sign flat
error, and equal-energy Gaussian error. Each step recomputes BF16 and rank-32
SVDQuant outputs on the current arm's own input; video and audio are handled
separately. See `decision_value_plan_E105_h3_output_error_interventions.md`.
All six runs completed 20 steps with finite video/audio metrics and validated
124-frame videos. At matched per-step output MSE, random-sign and Gaussian
errors produced videos substantially closer to the BF16 controls than the
structured SVDQuant error in both fixed cases; restoring only >4σ output
errors helped much less. This is two-case exploratory evidence that output
error distribution/pattern matters beyond aggregate NMSE, not identification
of a specific causal pattern or a general perceptual-quality claim. See
`results/research/E105/README.md` and per-arm JSON reports.

## E106 — completed 2026-10-17

Closed-loop full-element permutation of genuine signed SVDQuant complete-DiT
output error, independently for video and audio, on the same fixed race and
clap cases as E105. This preserves the FP32 marginal error values exactly
before the mandatory BF16 cast while breaking their tensor-coordinate
placement. See `decision_value_plan_E106_h3_error_permutation.md`.
Both cases completed 20 steps, passed video/audio finite and permutation
energy/rounding gates, and yielded hash-checked 124-frame videos. Despite
practically unchanged complete-DiT output NMSE, decoded RGB distance to the
BF16 controls fell from 0.5191 to 0.03737 (race) and from 0.5720 to 0.2972
(clap). This is two-case exploratory evidence that the location/structure of
the real quantization error matters, not a general perceptual-quality claim.
See `results/research/E106/README.md` and two six-arm summaries.

## E107 — completed 2026-10-17

Read-only oracle diagnostic on the 20 cached E103/E104 BF16-trajectory paired
DiT outputs: compare next-step error `e_t` against `e_t - k e_{t-1}` for
fixed `k=0.5,0.8`, separately for video and audio. See
`decision_value_plan_E107_h3_previous_error_subtraction.md`.
Both fixed coefficients reduce pooled video NMSE on the 19 eligible steps:
race 35.7%/31.8% and clap 29.6%/23.0% for k=0.5/0.8. Audio also improves
on the pooled metric, but individual steps can worsen, especially with 0.8.
This uses the true preceding BF16-vs-quantized error and is an oracle
teacher-forced diagnostic, not a deployable correction or closed-loop video
result. See `results/research/E107/README.md`, JSON, and per-step CSV.
The pooled k-sweep curve from 0 to 1.5 (0.01 increments) and analytic pooled
minima are saved in the same E107 results directory; k=0/0.5/0.8 match the
primary calculation exactly.

## E108 — completed 2026-10-17

Closed-loop oracle generation for race and clap: step 0 uses original
SVDQuant output; each subsequent step returns `Q_t - 0.6 e_(t-1)`, using
the previous raw paired BF16-vs-SVDQuant output error on this arm's own
trajectory, with video and audio corrected separately. See
`decision_value_plan_E108_h3_oracle_temporal_correction.md`.
Both cases completed 20 paired steps and hash-checked 124-frame media. On
the corrected arm's own inputs, pooled steps 1–19 video output NMSE fell
from 9.252% to 5.483% for race and from 2.990% to 2.216% for clap; audio
NMSE also fell. Decoded RGB distance to BF16 improved 0.5191→0.3125 for
race and 0.5720→0.5256 for clap. Qualitatively, race is closer in scene
progression but clap still has a different viewpoint from BF16. This is a
two-case oracle result, not a deployable correction or general perceptual
quality finding. See `results/research/E108/README.md`, per-case JSON,
three-arm videos, and the per-step figure.

## E112 — completed 2026-10-20

Real H3 block-25, prompt-1, step-0 QKV mixed P/V quantization reference.
After a 160-query preflight, recomputed the complete 9,482-query main attention
output and the complete 54-query short attention output. Threshold 0.10 routes
106/4,200 main `(head, KV tile)` pairs to FP8 (2.5238% of tile GEMMs). Main
output NMSE vs BF16: all-FP4 0.75804%, all-FP8 0.53578%, threshold-mixed
0.75504%, strict top-1%-tile mixed 0.75795%. The threshold-mixed improvement
is only 0.397% relative to all-FP4 NMSE, capturing 1.35% of the possible
all-FP8 improvement. See `decision_value_plan_E112_h3_mixed_pv_tile.md` and
`results/research/E112_full/README.md`, `summary.json`, per-tile CSV, and PNG.
This is a single captured call and a PyTorch numeric reference, not native
kernel timing or video quality.

## E113 — completed 2026-10-20

MiniMax-H3 race and clap free-generation experiment with all 200 target
linears BF16 and three attention arms: official Sage3, native Triton all-FP4,
and native Triton V-error-routed FP4/FP8. Both Triton and PyTorch reference
use dynamic E4M3 FP8 scales per 128 reduction elements; native NVFP4 uses
E2M1 plus E4M3 scales per 16. Thirteen real-input numerical comparisons
passed with maximum output NMSE 3.218e-6, including a full 9,482-token head.
Each new arm completed 20 steps and a validated 124-frame video with audio;
within each case, initial video/audio latent hashes match. The cross-case
`verify_e113_native.py --require-analysis` audit passed. Relative to the
matched native all-FP4 arm, the mixed arm was farther from BF16 in both cases:
RGB NMSE 0.16597→0.17030 (race), 0.66988→0.67982 (clap); LPIPS-Alex
0.31497→0.31819 and 0.47934→0.50536. Thus the current threshold-0.10 V
router does not show a consistent full-generation benefit in these two fixed
cases. These decoded-video distances are not teacher-forced DiT NMSE and do
not establish a broad perceptual-quality ranking. See `results/research/E113/README.md`
and per-case four-arm video summaries, LPIPS reports, and montages.

## E137 — in progress 2026-10-30

Decision-value test for whether H3 SVDQuant rollout degradation depends on
where an unchanged per-step error field is placed.  The intervention retains
the whole FP32 96-channel error vector at every video token and preserves its
per-step error energy, while either sharing one random spatial HxW permutation
over all video frames or permuting whole temporal frames; audio is unmodified.
The valid race/spatial run completed all 20 steps.  Its maximum per-step
energy mismatch after BF16 casting was 0.0227% (gate: <=0.5%), while decoded
RGB divergence to the matched BF16 rollout fell from 0.51910 to 0.19442.
This is preliminary one-arm, one-case evidence only: the temporal arm and a
second case remain queued for a genuinely empty GPU.  It is an oracle paired
BF16-vs-quantized-output intervention, not a deployable quantizer.  See
`decision_value_plan_E137_h3_error_axis_permutation.md` and
`results/research/E137/attempt2/race/axis_permutation_summary.json`.

## E138 — completed 2026-10-31

Deployable-rule feasibility screen: activation E2M1 stochastic rounding with
original nearest weight quantization, dynamic NVFP4 scales, smoothing, rank-64
low-rank branches and BF16 attention. Four paired 20-step 576x1024/124-frame
race/clap generations completed; initial latent and checkpoint hashes matched.
Local FC diagnostic: output NMSE 0.8526% -> about 1.342%, centered adjacent-row
error cosine 0.3618 -> about 0.245 (three local SR seeds). Closed-loop video
distances worsened: race RGB NMSE 0.23948 -> 0.24698, LPIPS 0.39791 -> 0.42240;
clap RGB NMSE 0.57380 -> 0.62272, LPIPS 0.50056 -> 0.57840. Decision: stop
plain activation SR; the oracle spatial-permutation observation has not yet
yielded a useful method. See `results/research/E138/README.md`.

## E139 — completed 2026-10-31

User requested official DraftAttention2 generation before routing research.
Pinned anemoi97a1c8b; two build compatibility changes, no arithmetic changes.
BF16 linears, Q64/.2 MeanMax, official H3 scales and INT8 prefix; official
dense-first10steps/2layers preserved. Race/clap each use70% skipped +30% INT8
versus70% skipped +20% NVFP4 +10% INT8, same initial noise. Four arms completed;
new clap BF16 reference run due historical first-step hash mismatch. Local
dense FP16 smoke NMSE5.72e-6; full124-frame metrics and labeled triptychs done.
Race RGBNMSE/LPIPS .01624/.08622 -> .01971/.10633; clap .01771/.06031 ->
.02148/.06799. Mixed and INT8 sparse-step DiT times similar (~8.4–8.5s),
single-run/offload-inclusive. No optimal-allocation or new quality claim.
See `results/research/E139/README.md`. No running experiment remains.

## E140 — completed 2026-10-31

User removed both dense-first exemptions. All20steps x50 main blocks use
DraftAttention2, BF16 linears; same70% skip and either30% INT8 or20% FP4+10%
INT8. Four matched race/clap videos and two124-frame four-column comparisons
completed. Mixed RGBNMSE/LPIPS race .28049/.41235, clap .54897/.44332;
INT8 race .20970/.35256, clap .55786/.45047. Distances much larger than E139
warmup, with visible viewpoint/background/pose shifts in sampled frames;
not a standalone quality verdict. Mixed total DiT175.39/174.33seconds versus
187.84/188.16 warmup. All-step sparse-call assertions, initial noise pairing,
finite outputs and media counts passed. See `results/research/E140/README.md`.

- 2026-11-02：启动 DraftAttention2 单 prompt 真实块命中率探索，设计见 `decision_value_plan_draftmap_hit_20261102.md`；固定 20 steps × 5 layers × 8 heads，比较真实 softmax 块质量和真实 QK 最大值两种参考。
- 2026-11-02：DraftAttention2 命中率实验完成。赛车单条 BF16 轨迹固定 20×5×8；真实 softmax 块质量 top 10/20/30/50 命中率 75.70/79.63/81.92/86.68%，QK tile max 口径 51.18/60.79/66.55/76.35%。中层较低、非单调；时间排序统计仅限单轨迹且交换性假设未证实。详见 `results/research/draftmap_hit_20261102/README.md`。
- 2026-11-02：补充 DraftMap 对块内最大 softmax 概率的 top-k 命中率；固定原实验数据和配置，设计见 `decision_value_plan_draftmap_maxprob_20261102.md`。
- 2026-11-02：补充块内最大 softmax 概率 oracle；top 10/20/30/50 命中率为 60.68/68.19/72.61/80.15%，沿用赛车 20×5×8 轨迹。原两种 oracle 6400 条命中率与旧运行逐项一致；详见 `results/research/draftmap_maxprob_20261102/README.md`。
