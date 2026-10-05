---
base_model: MiniMaxAI/MiniMax-H3
library_name: diffusers
license: other
tags:
- nunchaku-lite
- svdquant
- quantized
- nvfp4
- video
---

# MiniMax-H3 transformer — Nunchaku Lite NVFP4

NVFP4 SVDQuant quantizations of the 30.8B MiniMax-H3 video+audio transformer
(the `transformer/` partition, i.e. the `t2va` / `fl2va` workflows), packed in
the Nunchaku Lite kernel layout. This is the **transformer component only** —
pair it with the other components from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

This repo holds **two** checkpoints that differ in calibration and packaging.
Pick one:

| | **Data-free** (repo root) | **Calibrated 8×20** |
| --- | --- | --- |
| diffusers load | `from_pretrained(repo)` | `from_pretrained(repo, subfolder="calibrated-8x20")` |
| Files | root `config.json` + `diffusion_pytorch_model-*.safetensors` + index | `calibrated-8x20/` `config.json` + `diffusion_pytorch_model.safetensors`; also the raw `svdq-nvfp4_r32-minimax-h3-t2va.safetensors` + `.config.yaml` |
| Calibration | none (weight-span smoothing only) | 8 prompts × 20 steps, default canvas |
| Common | NVFP4 SVDQuant, group 16, rank-32 low-rank branch, 362 quantized linears (SVDQ fp4 + AdaLN AWQ W4A16 int4) | same |

Both are ~19 GB (vs ~66 GB bf16) and both now load with a one-line
`from_pretrained` (the calibrated one via `subfolder=`).

## 1. Data-free checkpoint (default `from_pretrained` path)

Data-free SVDQuant: weight-span smoothing + rank-32 SVD low-rank branch + NVFP4
group quantization (group size 16), no calibration data. Loads through
diffusers' pre-quantized fast path — the extra single-file checkpoint in this
repo is ignored, since `from_pretrained` reads only the shards named in
`diffusion_pytorch_model.safetensors.index.json`.

```python
import torch
from diffusers import MiniMaxH3Transformer3DModel

transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    "rootonchair/MiniMax-H3-nunchaku-lite-nvfp4", dtype=torch.bfloat16, device_map="cuda",
)
```

## 2. Calibrated 8×20 checkpoint

Calibrated SVDQuant produced with
[diffuse-compressor](https://github.com/rootonchair/diffuse-compressor)
(`examples/text_to_video/quantize_minimax_h3.py`); calibration is 8 prompts ×
20 denoising steps at the default 768-short-edge canvas. Same quantization is
provided two ways.

**Recommended — diffusers-native (`subfolder=`):** repackaged into the same
pre-quantized format as the data-free build, loads in one line.

```python
import torch
from diffusers import MiniMaxH3Transformer3DModel

transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    "rootonchair/MiniMax-H3-nunchaku-lite-nvfp4", subfolder="calibrated-8x20",
    dtype=torch.bfloat16, device_map="cuda",
)
```

**Alternative — raw single-file export.** The original
`svdq-nvfp4_r32-minimax-h3-t2va.safetensors` (embedded
`nunchaku_lite.runtime_manifest` v1, precision `mixed`: fp4 SVDQ + int4 AWQ, plus
a `.config.yaml` sidecar) is also kept. It is not diffusers-native, so it is
applied by patching a meta-constructed transformer:

```python
import torch
from huggingface_hub import hf_hub_download
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3RotaryPosEmbed, MiniMaxH3Transformer3DModel)
from nunchaku_lite.core import _patch_component

ckpt = hf_hub_download("rootonchair/MiniMax-H3-nunchaku-lite-nvfp4",
                       "svdq-nvfp4_r32-minimax-h3-t2va.safetensors")

config = MiniMaxH3Transformer3DModel.load_config("MiniMaxAI/MiniMax-H3", subfolder="transformer")
with torch.device("meta"):
    transformer = MiniMaxH3Transformer3DModel.from_config(config)

# rope.inv_freq is a non-persistent buffer (absent from the checkpoint), so the
# meta-assign load would leave it on the meta device — rebuild it off-meta first.
transformer.rope.inv_freq = MiniMaxH3RotaryPosEmbed(
    rope_freq_dim=transformer.config.rope_freq_dim,
    rope_theta=transformer.config.rope_theta).inv_freq

_patch_component(transformer, ckpt, target="manifest", precision="fp4",
                 torch_dtype=torch.bfloat16, device="cuda", strict=True,
                 adapter_options=None, assign=True)
```

## Requirements (both checkpoints)

Requires the `kernels` package, `DIFFUSERS_TRUST_REMOTE_KERNELS=true`, and a
**Blackwell-or-newer** NVIDIA GPU — NVFP4 kernels need sm_120, PyTorch >= 2.7
with CUDA >= 12.8. Keep the transformer fully resident on the GPU (~19 GB)
rather than offloading, for best throughput. (For non-Blackwell GPUs, use the
int4 build at
[rootonchair/MiniMax-H3-nunchaku-lite-int4](https://huggingface.co/rootonchair/MiniMax-H3-nunchaku-lite-int4).)
