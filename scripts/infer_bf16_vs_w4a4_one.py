#!/usr/bin/env python3
"""Generate one BF16 and one PTQ W4A4 image with the same prompt/seed."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import torch  # noqa: E402
import pyarrow  # noqa: E402,F401


PROMPT = "A cat holding a sign that says hello world"
SEED = 42
STEPS = int(os.environ.get("FLUX_NUM_STEPS", "50"))
GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "3.5"))
HEIGHT = 1024
WIDTH = 1024
MODEL_PATH = Path(os.environ.get("FLUX_MODEL_PATH", str(DATA_ROOT / "models" / "FLUX.1-dev")))
MODEL_CONFIG = os.environ.get("FLUX_MODEL_CONFIG", "configs/model/flux.1-dev.yaml")
CALIB_PATH = os.environ.get("FLUX_CALIB_PATH", str(DATA_ROOT / "datasets/torch.bfloat16/flux.1-dev/fmeuler50-g3.5/qdiff/s64"))


def _parse_cfg(argv: list[str]):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    sys.argv = ["infer_compare_one.py", *argv]
    config, *_rest = DiffusionPtqRunConfig.get_parser().parse_known_args()
    return config


def load_quant_pipeline(ckpt_dir: Path):
    """Load Flux via official pipeline.build + ptq(load_from=ckpt).

    Must go through DiffusionPipelineConfig.build() so FluxSingleTransformerBlock.proj_out
    is converted to ConcatLinear before DiffusionModelStruct.construct / weight load.
    """
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    # Keep full pipeline on GPU for image generation (not PTQ-only mode).
    os.environ["DEEPCOMPRESSOR_TRANSFORMER_ONLY"] = "0"

    repo = Path(__file__).resolve().parents[1]
    diffusion = repo / "third_party" / "deepcompressor" / "examples" / "diffusion"
    os.chdir(diffusion)
    out_root = DATA_ROOT / "compare" / "bf16_vs_w4a4_one" / "ptq_load_scratch"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _parse_cfg(
        [
            MODEL_CONFIG,
            "configs/svdquant/int4.yaml",
            "configs/svdquant/fast.yaml",
            "configs/svdquant/lowmem.yaml",
            f"--pipeline-path={MODEL_PATH}",
            f"--calib-path={CALIB_PATH}",
            "--calib-num-samples=64",
            f"--load-from={ckpt_dir}",
            f"--output-root={out_root}",
            f"--cache-root={out_root}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )

    model_path = ckpt_dir / "model.pt"
    branch_path = ckpt_dir / "branch.pt"
    smooth_path = ckpt_dir / "smooth.pt"
    assert model_path.is_file(), model_path
    assert branch_path.exists(), branch_path
    assert smooth_path.exists(), smooth_path

    print("[quant] building Flux pipeline (with ConcatLinear patch)...", flush=True)
    pipe = cfg.pipeline.build()
    model = DiffusionModelStruct.construct(pipe)

    print(f"[quant] loading PTQ ckpt from {ckpt_dir} (smooth + model + branch + act hooks)...", flush=True)
    ptq(
        model,
        cfg.quant,
        cache=None,
        load_dirpath=str(ckpt_dir),
        save_dirpath="",
        copy_on_save=False,
        save_model=False,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return pipe


def load_bf16_pipeline(dtype=torch.bfloat16):
    from diffusers import FluxPipeline

    print("[bf16] building FluxPipeline...", flush=True)
    return FluxPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
    ).to("cuda")


@torch.inference_mode()
def generate(pipe, prompt: str, seed: int, out_path: Path) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    image = pipe(
        prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    info = {
        "path": str(out_path),
        "seconds": elapsed,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    print(f"[gen] saved {out_path} in {elapsed:.1f}s peak_alloc={info['peak_alloc_gib']:.2f}GiB", flush=True)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DATA_ROOT / "ckpts" / "flux.1-dev-int4-fast-svd-lowrank",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_ROOT / "compare" / "bf16_vs_w4a4_one",
    )
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--only", choices=["both", "bf16", "quant"], default="both")
    args = parser.parse_args()

    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "height": HEIGHT,
        "width": WIDTH,
        "ckpt": str(args.ckpt),
    }

    if args.only in ("both", "bf16"):
        pipe = load_bf16_pipeline()
        report["bf16"] = generate(pipe, args.prompt, args.seed, args.out_dir / f"bf16_seed{args.seed}.png")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    if args.only in ("both", "quant"):
        pipe = load_quant_pipeline(args.ckpt)
        report["w4a4"] = generate(pipe, args.prompt, args.seed, args.out_dir / f"w4a4_svd_lowrank_seed{args.seed}.png")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
