# E126 — MixAttention VBench5 prompts with BF16, corrected Sage3 and split mixed attention

Status: **complete** (`finalize_status.json`). The five prompt strings and
seed 42 come from the original MixAttention `run_bf16_vbench5.py` and
`run_sageattn2pp_vbench5.py`. We preserved their default 1344×768,
124-frame, 50-inference-step and 24-fps settings. All 15 videos were produced
with the same local BF16 MiniMax-H3 model used in E124/E125, through the
DiffSynth pipeline. The three attention arms are true BF16 SDPA, corrected
noncausal Sage3 FP4, and the split mixed FP4/FP8 operator at threshold 0.10.
Only main attention changes; linears, refiner attention and model weights stay
BF16. Each prompt's three arms have identical initial video/audio latent
SHA-256 hashes and 50 finite DiT calls. Video/audio MP4 streams, dimensions,
frame counts and hashes were checked. Main attention has 37,715–37,721 valid
tokens depending on prompt.

The original scripts use a Diffusers `torch.Generator`; E126 uses DiffSynth's
CPU-seeded H3 pipeline. The same numeric seed does not imply bitwise-identical
initial noise across these two frameworks. E126's own three arms are exactly
paired by initial-latent hash. This is the same model and comparison path as
our previous H3 attention videos, not an execution of those original scripts.

## Full-generation video distances

Reference is the same-prompt BF16 MP4. LPIPS is TorchMetrics pretrained
AlexNet averaged over all 124 full-resolution RGB8 frames; RGB NMSE is total
squared pixel error divided by BF16 pixel energy. Audio is excluded. Lower
means closer to BF16, **not necessarily better perceptual quality**. These
are open-loop generation distances, not same-input local attention errors.

| Prompt | Sage3 LPIPS | Split LPIPS | Sage3 RGB NMSE | Split RGB NMSE |
| --- | ---: | ---: | ---: | ---: |
| Swimming in ocean | 0.2218 | 0.2250 | 5.24% | 5.14% |
| Presentation | 0.3859 | 0.3485 | 21.83% | 17.22% |
| Washing dishes | 0.2505 | 0.2188 | 18.01% | 15.38% |
| Eating a burger | 0.3011 | 0.6184 | 11.78% | 51.13% |
| Walking in snowstorm | 0.2303 | 0.2517 | 5.98% | 5.47% |
| **Mean of five prompts** | **0.2779** | **0.3325** | **12.57%** | **18.87%** |

The burger prompt is an important outlier: split mixed is much farther from
BF16 after 50 recurrent steps. On the **same first DiT input** for that
prompt, split mixed video-output NMSE versus BF16 was **0.808%**, below
corrected Sage3's **0.982%**; split versus Sage3 was **0.194%**. Thus the
large final-video gap is not already present as a gross error on the first
matched DiT call. This single-step control does not rule out later-step
distribution shift or numerical instability in the recurrent trajectory.
See `vbench_003/paired_first_dit.json`.

Mean sum of 50 DiT-forward timings was 1015.0 s BF16, 772.9 s Sage3,
794.1 s split. Arms ran concurrently on different GPUs (5–7) with disk
offload, so these are descriptive timings, not a controlled kernel benchmark.
No official VBench score was computed; the name “VBench5” refers to the five
prompt strings in the source scripts.

`summary.json` contains per-case scores and terminal latent NMSE. The five
hash-verified, video-only three-column previews are under
`/data1/models/svdquant-wjq/research/20261023/E126_h3_vbench5_attention/previews/`;
full MP4s retain their audio under each prompt/arm directory. Source and
recovery entry points are `scripts/research/run_e126_h3_vbench5_attention.py`,
`launch_e126_h3_vbench5_gpu.sh`, `finalize_e126_h3_vbench5.py`,
`summarize_e126_h3_vbench5_attention.py`, and
`render_e126_triplet_previews.py`.
