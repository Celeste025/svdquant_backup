# Environment matrix

This workspace deliberately uses separate environments where dependency stacks
conflict. A new server needs **two core environments**; MJVideo and ConvRot are
optional evaluation/ablation environments. All examples assume a CUDA-capable
Linux host and a writable `$SVDQUANT_DATA_ROOT`.

## 1. Core PTQ + rCM/FLUX + VBench

Purpose: rCM-Wan inference/PTQ, FLUX inference/PTQ, and the established VBench
evaluator. This is the default environment selected by
`scripts/env_svdquant_ptq.sh`.

Tested local baseline:

| Component | Tested version |
|---|---|
| Python | 3.12.13 |
| PyTorch / CUDA wheel | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| diffusers | 0.33.1 |
| transformers | 4.49.0 |
| ModelScope | 1.39.1 |
| NumPy | 2.5.2 |

Bootstrap outline (select the PyTorch CUDA wheel compatible with the target
driver; do not blindly install a CPU wheel):

```bash
export SVDQUANT_DATA_ROOT=/data/svdquant-wjq
python3.12 -m venv "$SVDQUANT_DATA_ROOT/conda-envs/svdquant-ptq"
source "$SVDQUANT_DATA_ROOT/conda-envs/svdquant-ptq/bin/activate"
pip install --upgrade pip
# Install the target CUDA build of torch/torchvision first.
pip install -e third_party/deepcompressor
pip install modelscope==1.39.1 diffusers==0.33.1 transformers==4.49.0 \
  accelerate safetensors numpy opencv-python imageio[ffmpeg] einops
pip install -r third_party/ViDiT-Q/eval/video/requirements.txt
```

VBench is sourced directly from the pinned submodule, not from a separately
cloned VBench checkout:

```bash
export VBENCH_ROOT="$PWD/third_party/ViDiT-Q/eval/video/Vbench"
export PYTHONPATH="$VBENCH_ROOT:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
python scripts/eval_rcm_vbench_subset.py --help
```

The evaluator carries a local compatibility shim for NumPy >=2 and the trusted
legacy VBench checkpoint loader. Its first real evaluation downloads metric
backbones; use a smoke subset first and retain `HF_HOME` on the data disk.

## 2. Core MiniMax-H3 generation

Purpose: BF16 and SVDQuant H3 generation. Keep it separate because the tested
DiffSynth stack uses newer diffusers/transformers than the PTQ stack.

Tested local baseline:

| Component | Tested version |
|---|---|
| Python | 3.12.13 |
| PyTorch / CUDA wheel | 2.11.0+cu128 |
| diffusers | 0.40.0 |
| transformers | 5.16.1 |
| ModelScope | 1.39.1 |
| NumPy | 2.5.2 |
| PyAV | 18.1.0 |

```bash
python3.12 -m venv .venvs/minimax-h3
source .venvs/minimax-h3/bin/activate
pip install --upgrade pip
# Install the target CUDA build of torch/torchvision first.
pip install -e third_party/deepcompressor
pip install -e 'third_party/DiffSynth-Studio[audio]'
pip install modelscope==1.39.1 diffusers==0.40.0 transformers==5.16.1 \
  safetensors imageio[ffmpeg] av accelerate
export DIFFSYNTH_ROOT="$PWD/third_party/DiffSynth-Studio"
export PYTHONPATH="$PWD/scripts:$DIFFSYNTH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Before applying a quantized H3 state, verify upstream H3 downloads and audio
video writing with:

```bash
python scripts/minimax_h3_bf16_smoke.py --output h3_bf16_smoke.mp4
```

The script downloads/resolves `Comfy-Org/MiniMax-H3` and `MiniMax/MiniMax-H3`
through the Hugging Face cache. Authenticate to Hugging Face in this shell if
the base weights are gated. Then pass the ModelScope-downloaded
`quant_state.pt` using `scripts/run_minimax_h3_vbench51.py --state ...`.

## 3. Optional MJVideo environment

Purpose: MJVideo metric evaluation only. It is not required for model loading,
PTQ, VBench, or video generation.

Tested baseline: Python 3.12.13, PyTorch 2.11.0+cu128, diffusers 0.33.1,
transformers 4.49.0, `decord==0.6.0`, and `av==18.1.0`. It also requires the
separately acquired MJ-VIDEO-2B checkpoint. The pinned upstream source is
`third_party/MJ-Video`; the local wrapper is `scripts/eval_mjvideo_rcm_vbench51.py`.

```bash
python3.12 -m venv .venvs/mjvideo
source .venvs/mjvideo/bin/activate
# Install the target CUDA build of torch/torchvision first.
pip install -r requirements/optional-mjvideo-py312.txt
export MJVIDEO_REPO="$PWD/third_party/MJ-Video"
export MJVIDEO_MODEL="$SVDQUANT_DATA_ROOT/models/MJ-VIDEO-2B"
```

Run the wrapper with explicit `--mjvideo-repo` and checkpoint/model arguments
shown by `python scripts/eval_mjvideo_rcm_vbench51.py --help`. Do not install it
into the H3 environment unless resolving dependency differences deliberately.

## 4. Optional ConvRot environment

Purpose: H3 ConvRot ablation generation only. It is not necessary for the
formal SVDQuant r32/r64 pipeline.

Tested baseline: Python 3.12.13, PyTorch 2.11.0+cu128, diffusers 0.40.0,
transformers 5.16.1. It additionally requires the separate ConvRot repository
and its checkpoint, neither of which is needed for the released SVDQuant
artifacts. The original experiment used the following isolated environment:

```bash
git clone https://github.com/feice-huang/ConvRot.git "$SVDQUANT_DATA_ROOT/third_party/ConvRot"
cd "$SVDQUANT_DATA_ROOT/third_party/ConvRot"
conda env create -f "$OLDPWD/requirements/optional-convrot-wan.yml"
conda activate convrot-wan
export CONVROT_ROOT="$PWD"
```

Important: the local ConvRot checkout used for the Wan/H3 experiment contains
uncommitted Wan support and experiment configs on top of upstream commit
`c80281c`. It is intentionally **not** treated as a clean upstream submodule
yet. The formal released rCM/H3 SVDQuant path does not need it; do not expect a
fresh upstream ConvRot clone to reproduce that optional ablation until its patch
set is reviewed and published separately.

## Verification order on a new server

1. `git submodule update --init --recursive` and activate environment 1.
2. Download one public rCM artifact and run `infer_rcm_wan_4step.py`.
3. Add VBench `PYTHONPATH` and run a one-case evaluator smoke test.
4. Activate environment 2, authenticate/download H3 base weights, and run
`minimax_h3_bf16_smoke.py`.
5. Download the private H3 artifact with an authorized ModelScope token, then
   run one SVDQuant H3 case.

For the optional tools, create their environment only when needed: MJVideo also
needs the independently obtained MJ-VIDEO-2B reward checkpoint; ConvRot needs
the separately maintained local patch set as well as its environment. Neither
is a prerequisite for loading a published SVDQuant checkpoint.

The original 64-sample calibration caches are intentionally outside this
matrix: they are required for exact PTQ reproduction, not for loading or using
published quantized artifacts.
