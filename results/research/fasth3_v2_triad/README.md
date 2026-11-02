# FastH3 V2: BF16 / PTQ / QAD local run-through

Run date: 2026-10-29 to 2026-10-30 (Asia/Shanghai). Prompt source:
`/home/wjq/workspace/178866172854036`, recovered by `prompt_id`, with original
seeds. Prompts 16 (racing/rainbow), 11 (starlit boat), and 51 (skier) were run.

Each arm generated 1024×576, 124-frame, 24 FPS video with audio. BF16 and PTQ
use the same ComfyUI 0.36 graph: eight `res_multistep` denoising steps, simple
schedule, video/audio shifts 10/3, and VSA retaining 20% of video blocks from
the first step. QAD uses the FastVideo 8-forward DMD schedule and VSA sparsity
0.8 (also retaining 20%); its optional FP4 attention was disabled because the
installed `attn_qat_infer` extension lacked a required function. QAD's NVFP4
linear weights remain intact.

The locally existing Comfy NVFP4-AWQ text encoder and Comfy video/audio VAEs
were reused directly for BF16/PTQ. The text encoder was converted locally with
FastVideo's official converter for QAD, without downloading another copy.
QAD's released VAE uses a different Diffusers key layout, so its 4.23 GB video
VAE and 0.61 GB audio VAE were fetched from the same QAD release rather than
pretending the Comfy VAE was directly loadable.

BF16 and PTQ use the official pruned Comfy checkpoints. For QAD, only 238
non-AdaLN dense tensors were range-downloaded and repacked into six valid
safetensors shards. The 100 per-block AdaLN weight/bias tensors were omitted;
FastVideo instead uses the release's `adaln_tables.pt`. This reduced the dense
shard transfer from 27,710,855,896 to 1,689,927,184 bytes (saving
26,020,928,712 bytes, about 26.0 GB). The QAD FP4 packed weights were
downloaded separately. All full-file downloads used ModelScope without proxy
and passed size/SHA-256 verification.

`summary.json` records the nine exact video paths, prompt hashes, seeds,
decoded-frame counts, durations, audio-stream presence, and report paths.
The validator decoded all 124 frames of every video; all nine passed. The
middle frame of every video had nonzero pixel variance.

Interpretation limit: identical prompt text, seed, geometry and nominal
schedule do not prove identical initial noise or numerical pairing across
ComfyUI and FastVideo. BF16/PTQ are directly paired within ComfyUI; QAD is a
separate-runtime visual comparison, not evidence by itself that QAD improves
quality over PTQ. QAD also comes from the V2.1 QAD release, while BF16/PTQ
come from the V2 checkpoint family.
