# E142 — real QKV precision/sparsity Pareto screen

Three BF16-trajectory captures, BF16 model: race/step0/block0,
race/step10/block25,clap/step10/block49. Complete56head QKV after norms/RoPE;
exclude independent padding segment, retain packed text/audio prefix and all
37x18x32 video tokens. Exact source calls and prefix saved in packets.

19 arms per input: BF16 SDPA,Sage2++,optimized Sage3,16 official DraftAttention2
configurations. Fixed Q64,MeanMax0.2,INT8 prefix and official layer calibration.
Drop fractions0,.3,.5,.7,.8 x retained INT8 fractions.25,.5,1 (15 configurations),
plus70%drop/20%NVFP4/10%INT8. No FP16 retained video phase. No per-layer/step
dense exemption: each measurement is one native attention call.

Complete-output and video-only NMSE/MSE vs same-input BF16; not sampled query
rows. Full API CUDA-event timing includes native preprocessing,quantization,
score/route/pack,attention and output. Native contiguous layouts prebuilt outside
timing, no cloned-input cost.25warmups+72calls each forward/reverse order,
144measured calls per arm. Report group CV,do not silently trim outliers.

Compare per-input frontiers before aggregates. Results are operator tradeoffs,
not generation-quality validation. Fresh native sparse output shapes/finite
checks on every arm. No new video generation/training. Source checkpoints and
calibration remain unchanged. All new tensors underdata1.
