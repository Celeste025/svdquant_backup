#!/usr/bin/env python3
"""Generate one BF16 and one PTQ W4A4 Wan2.1 video with the same prompt/seed."""
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
os.environ.setdefault("DEEPCOMPRESSOR_TRANSFORMER_ONLY", "0")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_FLOW_SHIFT", "3.0")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import torch  # noqa: E402
import pyarrow  # noqa: E402,F401

# Held-out vs Random(0) smoke-8 calib set (see vbench_t2v_simple.yaml).
PROMPT = "A person is riding a bike"
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)
SEED = 42
STEPS = 50
GUIDANCE = 6.0
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 33


def _parse_cfg(argv: list[str]):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    sys.argv = ["infer_wan_compare.py", *argv]
    config, *_rest = DiffusionPtqRunConfig.get_parser().parse_known_args()
    return config


def _save_video(frames, path: Path) -> None:
    from PIL import Image
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    def _one(frame):
        if hasattr(frame, "save"):
            return frame
        arr = np.asarray(frame)
        if np.issubdtype(arr.dtype, np.floating):
            max_v = float(arr.max()) if arr.size else 1.0
            if max_v <= 1.0 + 1e-3:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr)

    pil_frames = [_one(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0,
    )


def load_bf16_pipeline():
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    repo = Path(__file__).resolve().parents[1]
    diffusion = repo / "third_party" / "deepcompressor" / "examples" / "diffusion"
    os.chdir(diffusion)
    cfg = _parse_cfg(
        [
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/real_nvfp4.yaml",
            "configs/svdquant/wan_s16.yaml",
            f"--pipeline-path={DATA_ROOT}/models/Wan2.1-T2V-1.3B-Diffusers",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    print("[bf16] building Wan pipeline...", flush=True)
    return cfg.pipeline.build()


def load_quant_pipeline(ckpt_dir: Path):
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    repo = Path(__file__).resolve().parents[1]
    diffusion = repo / "third_party" / "deepcompressor" / "examples" / "diffusion"
    os.chdir(diffusion)
    out_root = DATA_ROOT / "compare" / "wan_bf16_vs_w4a4_one" / "ptq_load_scratch"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _parse_cfg(
        [
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/real_nvfp4.yaml",
            "configs/svdquant/wan_s16.yaml",
            f"--pipeline-path={DATA_ROOT}/models/Wan2.1-T2V-1.3B-Diffusers",
            f"--load-from={ckpt_dir}",
            f"--output-root={out_root}",
            f"--cache-root={out_root}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    print("[quant] building Wan pipeline + loading PTQ ckpt...", flush=True)
    pipe = cfg.pipeline.build()
    model = DiffusionModelStruct.construct(pipe)
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


@torch.inference_mode()
def generate(pipe, prompt: str, seed: int, out_path: Path) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe(
        prompt,
        negative_prompt=NEGATIVE,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    )
    frames = out.frames[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    _save_video(frames, out_path)
    info = {
        "path": str(out_path),
        "seconds": elapsed,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "n_frames": len(frames),
    }
    print(f"[gen] saved {out_path} in {elapsed:.1f}s peak={info['peak_alloc_gib']:.2f}GiB", flush=True)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-smoke")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "compare" / "wan_bf16_vs_w4a4_one")
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--only", choices=["both", "bf16", "quant"], default="both")
    parser.add_argument("--tag", type=str, default="smoke", help="suffix for w4a4 gif filename")
    args = parser.parse_args()

    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "ckpt": str(args.ckpt),
        "tag": args.tag,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("both", "bf16"):
        pipe = load_bf16_pipeline()
        report["bf16"] = generate(pipe, args.prompt, args.seed, args.out_dir / f"bf16_seed{args.seed}.gif")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    if args.only in ("both", "quant"):
        pipe = load_quant_pipeline(args.ckpt)
        report["w4a4"] = generate(
            pipe, args.prompt, args.seed, args.out_dir / f"w4a4_{args.tag}_seed{args.seed}.gif"
        )
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
