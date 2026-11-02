# E124 — 12-prompt H3 BF16 / Sage3 / split-mixed generation

Status: **complete**. Four GPUs (4–7) each processed three prompts. Every
prompt has three 20-step, 576×1024, 124-frame videos with audio: true BF16
SDPA, corrected Sage3 FP4, and the corrected split mixed FP4/P8×V8 attention.
All other H3 weights and inference settings were held fixed. The source prompt
is the full `prompt` field from `/home/wjq/workspace/178866172854036`, with
the row's seed. The source JSONL hash and assignment are in `manifest.json`.
Its original duration/frame metadata was deliberately overridden by the
common 124-frame test contract, including for the 15-second race prompt.

| GPU | Prompt IDs (source seeds) |
| --- | --- |
| 4 | 1 (22529), 11 (1571), 16 (20026; race/rainbow) |
| 5 | 20 (13378), 25 (2141), 42 (52386) |
| 6 | 51 (41698), 68 (38691), 76 (52544) |
| 7 | 93 (13169), 103 (38792), 116 (35796) |

The final `summary.json` verifies **12/12 complete prompt triplets, 36/36
videos, zero missing arms**. Each arm has 20 finite DiT steps, 50 main and two
refiner attention calls per step; each decoded MP4 has 124 frames, 1024×576
pixels and an audio stream. The three arms of each prompt share the exact same
initial video/audio latent hashes. Terminal latent NMSE is recorded only as a
numerical diagnostic, not a perceptual quality score.

Mean **sum of 20 DiT-forward times per prompt** was 205.35 s for BF16,
180.36 s for Sage3, and 183.58 s for split mixed. The mean of within-prompt
BF16/quantized ratios is **1.139×** for Sage3 and **1.119×** for split mixed.
This timing includes the selected attention helper and DiT linear/offload
behavior, but excludes text encoding, VAE decode, and model startup; it was
not designed to compare GPU-resident weights. Four GPUs ran concurrently, so
small cross-GPU differences should not be overinterpreted.

The race trio is under `p016/{bf16,sage3_fixed,split_mixed}` in the media
directory. The 36 MP4s are packaged as
`/data1/models/svdquant-wjq/research/20261023/E124_h3_attention_12prompt/E124_12prompts_3arms_36videos.tar`
(266 MB, 36 entries). Generated videos and final latent checkpoints remain
local on `/data1`; only the small numerical report and prompt manifest are in
this repository.

One valid boundary was uncovered during the run: prompt 11 has zero padding
tokens. The older H3 routing adapter incorrectly required a nonempty padding
suffix and stopped before its first DiT step. The adapter was updated to
accept empty padding and count only segments actually computed; the failed
report and empty output directory were renamed and retained as
`failed_zero_padding_20261023`. Prompt 11 was then rerun successfully through
all three arms. No completed output was overwritten.

Reproduction: `bash scripts/research/launch_e124_h3_attention_gpu.sh GPU_ID`
for GPU IDs 4–7; completed arms are skipped and incomplete reports require
inspection. After all groups finish, run
`/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/summarize_e124_h3_attention_prompts.py`.
