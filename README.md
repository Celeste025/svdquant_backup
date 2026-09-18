# Nunchaku

> This repository is also the experiment workspace for the rCM-Wan, MiniMax-H3,
> and FLUX SVDQuant work. The upstream Nunchaku documentation starts below;
> this section is the operational handoff guide for a new server.

## Experiment workspace quick start

### What is in Git vs. ModelScope

- Git holds code, fixed third-party source revisions, PTQ recipes, launchers,
  manifests, and lightweight metric records. It deliberately does **not** hold
  weights, calibration caches, generated MP4s, or run directories.
- ModelScope holds the final quantized states and the formal video results.
  The rCM and FLUX repositories are public; MiniMax-H3 repositories are private
  because of the upstream H3 license.
- Upstream BF16 base models must be downloaded separately. The quantized package
  replaces only the transformer/quantization state; it is not a standalone
  text encoder, VAE, tokenizer, or scheduler.

Clone with the exact external source revisions:

```bash
git clone --recurse-submodules https://github.com/Celeste025/svdquant_backup.git svdquant-exp
cd svdquant-exp
git submodule update --init --recursive
```

The two experiment-critical submodules are `third_party/DiffSynth-Studio` for
H3 inference and `third_party/ViDiT-Q` for the VBench implementation. Do not
replace them with a moving `main` checkout unless intentionally updating the
experiment environment.

### Directory convention

Use one writable data disk. Paths are configurable, but the examples use the
following layout:

```text
SVDQUANT_DATA_ROOT=/data/svdquant-wjq
${SVDQUANT_DATA_ROOT}/models/       # upstream BF16 base checkpoints
${SVDQUANT_DATA_ROOT}/artifacts/    # downloaded ModelScope quantized packages
${SVDQUANT_DATA_ROOT}/datasets/     # calibration cache only, never required for inference
${SVDQUANT_DATA_ROOT}/runs/         # PTQ outputs and logs
```

```bash
export SVDQUANT_DATA_ROOT=/data/svdquant-wjq
export DIFFSYNTH_ROOT="$PWD/third_party/DiffSynth-Studio"
export VBENCH_ROOT="$PWD/third_party/ViDiT-Q/eval/video/Vbench"
mkdir -p "$SVDQUANT_DATA_ROOT"/{models,artifacts,datasets,runs}
source scripts/env_svdquant_ptq.sh
```

`scripts/env_svdquant_ptq.sh` sets cache paths and activates the intended PTQ
Python when it already exists. For a fresh environment, install a CUDA-matched
PyTorch first, then install the local libraries and evaluation dependencies:

```bash
python -m pip install -e third_party/deepcompressor
python -m pip install -e 'third_party/DiffSynth-Studio[audio]'
python -m pip install -r third_party/ViDiT-Q/eval/video/requirements.txt
python -m pip install modelscope diffusers accelerate safetensors imageio[ffmpeg]
```

The exact split, tested versions, and commands for the two core and two optional
environments are in [docs/environment_matrix.md](docs/environment_matrix.md).

VBench downloads several metric backbones on its first run. Keep its cache on
the data disk (for example, set `HF_HOME=$SVDQUANT_DATA_ROOT/hf`) and run one
small evaluation before scheduling a full benchmark.

### Upstream base-model prerequisites

Download these from their original publishers and put them below
`$SVDQUANT_DATA_ROOT/models`. Observe each publisher's license/gating terms.

| Family | Required upstream files | Expected local path / use |
|---|---|---|
| rCM-Wan | `Wan-AI/Wan2.1-T2V-1.3B` Diffusers pipeline plus the rCM transformer | `models/Wan2.1-T2V-1.3B-Diffusers/` and `models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer/`; pass them as `--model` and `--transformer` |
| MiniMax-H3 | `Comfy-Org/MiniMax-H3` pruned DiT and `MiniMax/MiniMax-H3` FL2VA components | DiffSynth-Studio resolves these through its Hugging Face cache; authenticate to Hugging Face if required |
| FLUX.1-dev / schnell | `black-forest-labs/FLUX.1-dev` or `FLUX.1-schnell` pipeline | set `FLUX_MODEL_PATH=$SVDQUANT_DATA_ROOT/models/FLUX.1-dev` (or schnell); dev is license-gated/non-commercial |

The rCM transformer is an additional required input: ModelScope packages contain
the quantized state but do not duplicate the rCM base Transformer. If moving
servers, copy or obtain that approved upstream rCM Transformer before inference.

### Download published quantized states and video results

Install `modelscope`, then use the registry downloader. It downloads and checks
only the requested variant, including `SHA256SUMS` verification:

```bash
# Public rCM example
python tools/fetch_published_artifact.py rcm-wan real-nvfp4-g10-r64 \
  --namespace Celeste025 --output "$SVDQUANT_DATA_ROOT/artifacts"

# Private H3: set a read-capable token in the shell, never in source code.
export MODELSCOPE_API_TOKEN='...'
python tools/fetch_published_artifact.py minimax-h3 nvfp4-g10-r64 \
  --namespace Celeste025 --output "$SVDQUANT_DATA_ROOT/artifacts"

# Formal video/metadata datasets (the H3 dataset requires the same read token).
python tools/fetch_published_video_dataset.py rcm-wan \
  --namespace Celeste025 --output "$SVDQUANT_DATA_ROOT/videos"
```

Published repositories:

- `Celeste025/svdquant-rcm-wan2.1-1.3b` — public rCM variants.
- `Celeste025/svdquant-flux1` — public FLUX quantized states.
- `Celeste025/svdquant-videoeval-rcm-wan` — public rCM formal videos.
- `Celeste025/svdquant-minimax-h3` and `Celeste025/svdquant-videoeval-minimax-h3` — private H3 states/videos; require the owner's read token.

See `docs/modelscope_release.md` for package layout and publishing details.
For a concise, copy-ready task brief for an agent on another server, see
[`docs/new_server_handoff_prompt.md`](docs/new_server_handoff_prompt.md).

### Main entry points

| Task | Script | Notes |
|---|---|---|
| rCM quantized smoke inference | `scripts/infer_rcm_wan_4step.py` | Load the Wan base + rCM Transformer + a downloaded rCM artifact; use 480x832, 77 frames, four steps for the established smoke contract. |
| H3 BF16 smoke inference | `scripts/minimax_h3_bf16_smoke.py` | Verifies DiffSynth-Studio and H3 base downloads before quantized generation. |
| H3 VBench-51 generation | `scripts/run_minimax_h3_vbench51.py` | Pass `--state .../quant_state.pt --variants svdquant`; supports r32 and r64. |
| VBench evaluation | `scripts/eval_rcm_vbench_subset.py` | Evaluates paired generated video directories using the vendored VBench implementation. |
| FLUX quantized image smoke | `scripts/infer_bf16_vs_w4a4_one.py` | Set `FLUX_MODEL_PATH`; pass `--ckpt` to a downloaded FLUX artifact. |
| rCM PTQ | `scripts/run_rcm_wan_real_nvfp4_svdquant.sh`, `scripts/run_rcm_wan_int4_s16_g10.sh` | Requires the calibration cache and an idle GPU. |

Example rCM inference:

```bash
python scripts/infer_rcm_wan_4step.py \
  --model "$SVDQUANT_DATA_ROOT/models/Wan2.1-T2V-1.3B-Diffusers" \
  --transformer "$SVDQUANT_DATA_ROOT/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer" \
  --quant-ckpt "$SVDQUANT_DATA_ROOT/artifacts/variants/real-nvfp4-g10-r64" \
  --prompt 'A short deterministic test prompt.' --steps 4 --height 480 --width 832 --frames 77 \
  --output rcm_smoke.mp4
```

### Reproducibility boundary

Inference and evaluation can be reconstructed from Git + upstream base model +
the ModelScope artifact/video package. Re-running PTQ **exactly** also needs the
original calibration cache (64 samples for the formal recipes), CUDA/PyTorch
versions, and the recorded recipe. Recollecting calibration inputs is supported,
but will not guarantee bitwise-identical checkpoints. Generated videos, smoke
outputs, run caches, and ModelScope credentials must not be committed to Git.

### Terminology

- **Loader** reconstructs quantized layers, smoothing, activation hooks, and
  low-rank branches from a published artifact on top of an upstream base model.
- **Recipe/config** records the model-specific quantization contract: format,
  group size, rank, grid, calibration settings, and related PTQ parameters.
- **Worker** is a resumable per-GPU generation process that writes per-case
  video/status files; a runner/launcher prepares manifests and invokes workers.

Nunchaku is an inference engine designed for 4-bit diffusion models, as demonstrated in our paper [SVDQuant](http://arxiv.org/abs/2411.05007). Please check [DeepCompressor](https://github.com/mit-han-lab/deepcompressor) for the quantization library.

### [Paper](http://arxiv.org/abs/2411.05007) | [Project](https://hanlab.mit.edu/projects/svdquant) | [Blog](https://hanlab.mit.edu/blog/svdquant) | [Demo](https://svdquant.mit.edu)

- **[2025-01-23]** 🚀 **4-bit [SANA](https://nvlabs.github.io/Sana/) support is here!** Experience a 2-3× speedup compared to the 16-bit model. Check out the [usage example](./examples/sana_1600m_pag.py) and the [deployment guide](app/sana/t2i) for more details. Explore our live demo at [svdquant.mit.edu](https://svdquant.mit.edu)!
- **[2025-01-22]** 🎉 [**SVDQuant**](http://arxiv.org/abs/2411.05007) has been accepted to **ICLR 2025**!
- **[2024-12-08]** Support [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Please check [comfyui/README.md](comfyui/README.md) for the usage.
- **[2024-11-07]** 🔥 Our latest **W4A4** Diffusion model quantization work [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) is publicly released! Check [**DeepCompressor**](https://github.com/mit-han-lab/deepcompressor) for the quantization library.

![teaser](./assets/teaser.jpg)
SVDQuant is a post-training quantization technique for 4-bit weights and activations that well maintains visual fidelity. On 12B FLUX.1-dev, it achieves 3.6× memory reduction compared to the BF16 model. By eliminating CPU offloading, it offers 8.7× speedup over the 16-bit model when on a 16GB laptop 4090 GPU, 3× faster than the NF4 W4A16 baseline. On PixArt-∑, it demonstrates significantly superior visual quality over other W4A4 or even W4A8 baselines. "E2E" means the end-to-end latency including the text encoder and VAE decoder.

**SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**<br>
[Muyang Li](https://lmxyy.me)\*, [Yujun Lin](https://yujunlin.com)\*, [Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang)\*, [Tianle Cai](https://www.tianle.website/#/), [Xiuyu Li](https://xiuyuli.com), [Junxian Guo](https://github.com/JerryGJX), [Enze Xie](https://xieenze.github.io), [Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), and [Song Han](https://hanlab.mit.edu/songhan) <br>
*MIT, NVIDIA, CMU, Princeton, UC Berkeley, SJTU, and Pika Labs* <br>

<p align="center">
  <img src="assets/demo.gif" width="100%"/>
</p>

## Method

#### Quantization Method -- SVDQuant

![intuition](./assets/intuition.gif)Overview of SVDQuant. Stage1: Originally, both the activation $\boldsymbol{X}$ and weights $\boldsymbol{W}$ contain outliers, making 4-bit quantization challenging.  Stage 2: We migrate the outliers from activations to weights, resulting in the updated activation $\hat{\boldsymbol{X}}$ and weights $\hat{\boldsymbol{W}}$. While $\hat{\boldsymbol{X}}$ becomes easier to quantize, $\hat{\boldsymbol{W}}$ now becomes more difficult. Stage 3: SVDQuant further decomposes $\hat{\boldsymbol{W}}$ into a low-rank component $\boldsymbol{L}_1\boldsymbol{L}_2$ and a residual $\hat{\boldsymbol{W}}-\boldsymbol{L}_1\boldsymbol{L}_2$ with SVD. Thus, the quantization difficulty is alleviated by the low-rank branch, which runs at 16-bit precision. 

#### Nunchaku Engine Design

![engine](./assets/engine.jpg) (a) Naïvely running low-rank branch with rank 32 will introduce 57% latency overhead due to extra read of 16-bit inputs in *Down Projection* and extra write of 16-bit outputs in *Up Projection*. Nunchaku optimizes this overhead with kernel fusion. (b) *Down Projection* and *Quantize* kernels use the same input, while *Up Projection* and *4-Bit Compute* kernels share the same output. To reduce data movement overhead, we fuse the first two and the latter two kernels together.


## Performance

![efficiency](./assets/efficiency.jpg)SVDQuant reduces the model size of the 12B FLUX.1 by 3.6×. Additionally, Nunchaku, further cuts memory usage of the 16-bit model by 3.5× and delivers 3.0× speedups over the NF4 W4A16 baseline on both the desktop and laptop NVIDIA RTX 4090 GPUs. Remarkably, on laptop 4090, it achieves in total 10.1× speedup by eliminating CPU offloading.

## Installation

**Note**:

*  Ensure your CUDA version is **≥ 12.2 on Linux** and **≥ 12.6 on Windows**.

*  For Windows user, please refer to [this issue](https://github.com/mit-han-lab/nunchaku/issues/6) for the instruction. Please upgrade your MSVC compiler to the latest version.

*  We currently support only NVIDIA GPUs with architectures sm_86 (Ampere: RTX 3090, A6000), sm_89 (Ada: RTX 4090), and sm_80 (A100). See [this issue](https://github.com/mit-han-lab/nunchaku/issues/1) for more details.


1. Install dependencies:
	```shell
	conda create -n nunchaku python=3.11
	conda activate nunchaku
	pip install torch torchvision torchaudio
	pip install diffusers ninja wheel transformers accelerate sentencepiece protobuf
	pip install huggingface_hub peft opencv-python einops gradio spaces GPUtil
	```
	
2. Install `nunchaku` package:
    Make sure you have `gcc/g++>=11`. If you don't, you can install it via Conda:
  
	```shell
	conda install -c conda-forge gxx=11 gcc=11
	```
	
	Then build the package from source:
	```shell
	git clone https://github.com/mit-han-lab/nunchaku.git
	cd nunchaku
	git submodule init
	git submodule update
	pip install -e .
	```

## Usage Example

In [examples](examples), we provide minimal scripts for running INT4 [FLUX.1](https://github.com/black-forest-labs/flux) and [Sana](https://github.com/NVlabs/Sana) models with Nunchaku. For example, the [script](examples/flux.1-dev.py) for [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) is as follows:

```python
import torch
from diffusers import FluxPipeline

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save("flux.1-dev.png")
```

Specifically, `nunchaku` shares the same APIs as [diffusers](https://github.com/huggingface/diffusers) and can be used in a similar way.

## ComfyUI

Please refer to [comfyui/README.md](comfyui/README.md) for the usage in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Gradio Demos

### FLUX.1 Models

#### Text-to-Image

```shell
cd app/flux.1/t2i
python run_gradio.py
```

* The demo also defaults to the FLUX.1-schnell model. To switch to the FLUX.1-dev model, use `-m dev`.
* By default, the Gemma-2B model is loaded as a safety checker. To disable this feature and save GPU memory, use `--no-safety-checker`.
* To further reduce GPU memory usage, you can enable the W4A16 text encoder by specifying `--use-qencoder`.
* By default, only the INT4 DiT is loaded. Use `-p int4 bf16` to add a BF16 DiT for side-by-side comparison, or `-p bf16` to load only the BF16 model.

#### Sketch-to-Image

```shell
cd app/flux.1/i2i
python run_gradio.py
```

* Similarly, the demo loads the Gemma-2B model as a safety checker by default. To disable this feature, use `--no-safety-checker`.
* To further reduce GPU memory usage, you can enable the W4A16 text encoder by specifying `--use-qencoder`.
* By default, we use our INT4 model. Use  `-p bf16` to switch to the BF16 model.

### Sana

#### Text-to-Image

```shell
cd app/sana/t2i
python run_gradio.py
```

## Benchmark

Please refer to [app/flux/t2i/README.md](app/flux/t2i/README.md) for instructions on reproducing our paper's quality results and benchmarking inference latency on FLUX.1 models.

## Roadmap

- [ ] Easy installation
- [x] Comfy UI node
- [ ] Customized LoRA conversion instructions
- [ ] Customized model quantization instructions
- [ ] FLUX.1 tools support
- [ ] Modularization
- [ ] IP-Adapter integration
- [ ] Video Model support
- [ ] Metal backend

## Citation

If you find `nunchaku` useful or relevant to your research, please cite our paper:

```bibtex
@inproceedings{
  li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Related Projects

* [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048), NeurIPS 2022 & T-PAMI 2023
* [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023
* [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304), ICCV 2023
* [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024
* [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR 2024
* [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532), ArXiv 2024
* [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629), ICLR 2025

## Acknowledgments

We thank MIT-IBM Watson AI Lab, MIT and Amazon Science Hub, MIT AI Hardware Program, National Science Foundation, Packard Foundation, Dell, LG, Hyundai, and Samsung for supporting this research. We thank NVIDIA for donating the DGX server.

We use [img2img-turbo](https://github.com/GaParmar/img2img-turbo) to train the sketch-to-image LoRA. Our text-to-image and sketch-to-image UI is built upon [playground-v.25](https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py) and [img2img-turbo](https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py), respectively. Our safety checker is borrowed from [hart](https://github.com/mit-han-lab/hart).

Nunchaku is also inspired by many open-source libraries, including (but not limited to) [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), [QServe](https://github.com/mit-han-lab/qserve), [AWQ](https://github.com/mit-han-lab/llm-awq), [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), and [Atom](https://github.com/efeslab/Atom).
