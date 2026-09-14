#!/usr/bin/env python3
"""BF16 vs ungated-s16 vs gated-s16 Wan2.1 GIF compare (same prompt/seed)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
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

PROMPT_HELDOUT = "A person is riding a bike"
PROMPT_CALIB = "an airplane soaring through a clear blue sky"
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

CKPT_UNGATED = DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16"
CKPT_GATED = DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16-gated"
PRIOR_S16 = DATA_ROOT / "compare" / "wan_bf16_vs_w4a4_s16"


def _parse_cfg(argv: list[str]):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    sys.argv = ["infer_wan_threeway.py", *argv]
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
    repo = Path(__file__).resolve().parents[1]
    diffusion = repo / "third_party" / "deepcompressor" / "examples" / "diffusion"
    os.chdir(diffusion)
    cfg = _parse_cfg(
        [
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/wan_smoke.yaml",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    print("[bf16] building Wan pipeline...", flush=True)
    return cfg.pipeline.build()


def load_quant_pipeline(ckpt_dir: Path, scratch_name: str):
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    repo = Path(__file__).resolve().parents[1]
    diffusion = repo / "third_party" / "deepcompressor" / "examples" / "diffusion"
    os.chdir(diffusion)
    out_root = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "ptq_load_scratch" / scratch_name
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _parse_cfg(
        [
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/wan_smoke.yaml",
            f"--load-from={ckpt_dir}",
            f"--output-root={out_root}",
            f"--cache-root={out_root}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    print(f"[quant] building Wan pipeline + loading {ckpt_dir}...", flush=True)
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


def make_side_by_side(paths: list[Path], labels: list[str], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    seqs = [Image.open(p) for p in paths]
    n = min(getattr(im, "n_frames", 1) for im in seqs)
    frames_out = []
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(n):
        crops = []
        for im in seqs:
            im.seek(i)
            crops.append(im.convert("RGB").copy())
        h = max(c.height for c in crops)
        w = sum(c.width for c in crops)
        canvas = Image.new("RGB", (w, h + 22), (20, 20, 20))
        x = 0
        for c, lab in zip(crops, labels):
            canvas.paste(c, (x, 22))
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 6, 4), lab, fill=(240, 240, 240), font=font)
            x += c.width
        frames_out.append(canvas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_out[0].save(
        out_path,
        save_all=True,
        append_images=frames_out[1:],
        duration=100,
        loop=0,
    )
    print(f"[montage] saved {out_path}", flush=True)
    for im in seqs:
        im.close()


def maybe_reuse(src: Path, dst: Path) -> dict | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    return {"path": str(dst), "reused_from": str(src)}


def run_case(
    out_dir: Path,
    prompt: str,
    seed: int,
    variants: set[str],
    reuse_prior: bool,
    prior_subdir: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "prompt": prompt,
        "seed": seed,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "ckpt_ungated": str(CKPT_UNGATED),
        "ckpt_gated": str(CKPT_GATED),
    }

    bf16_path = out_dir / f"bf16_seed{seed}.gif"
    ungated_path = out_dir / f"ungated_s16_seed{seed}.gif"
    gated_path = out_dir / f"gated_s16_seed{seed}.gif"

    if "bf16" in variants:
        reused = None
        if reuse_prior:
            reused = maybe_reuse(PRIOR_S16 / prior_subdir / f"bf16_seed{seed}.gif", bf16_path)
        if reused:
            report["bf16"] = reused
            print(f"[bf16] reused {reused['reused_from']}", flush=True)
        else:
            pipe = load_bf16_pipeline()
            report["bf16"] = generate(pipe, prompt, seed, bf16_path)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

    if "ungated" in variants:
        reused = None
        if reuse_prior:
            reused = maybe_reuse(PRIOR_S16 / prior_subdir / f"w4a4_s16_seed{seed}.gif", ungated_path)
        if reused:
            report["ungated_s16"] = reused
            print(f"[ungated] reused {reused['reused_from']}", flush=True)
        else:
            pipe = load_quant_pipeline(CKPT_UNGATED, "ungated")
            report["ungated_s16"] = generate(pipe, prompt, seed, ungated_path)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

    if "gated" in variants:
        pipe = load_quant_pipeline(CKPT_GATED, "gated")
        report["gated_s16"] = generate(pipe, prompt, seed, gated_path)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    trio = [
        (bf16_path, "BF16"),
        (ungated_path, "ungated-s16"),
        (gated_path, "gated-s16"),
    ]
    if all(p.is_file() for p, _ in trio):
        montage = out_dir / f"side_by_side_seed{seed}.gif"
        make_side_by_side([p for p, _ in trio], [lab for _, lab in trio], montage)
        report["side_by_side"] = str(montage)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["heldout_bike", "calib_airplane", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["bf16", "ungated", "gated", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--reuse-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse prior BF16/ungated GIFs from wan_bf16_vs_w4a4_s16 when present",
    )
    args = parser.parse_args()

    cases = set(args.cases)
    if "all" in cases:
        cases = {"heldout_bike", "calib_airplane"}
    variants = set(args.variants)
    if "all" in variants:
        variants = {"bf16", "ungated", "gated"}

    case_cfgs = {
        "heldout_bike": (PROMPT_HELDOUT, "heldout_bike"),
        "calib_airplane": (PROMPT_CALIB, "calib_airplane"),
    }

    summary = {}
    for name in sorted(cases):
        prompt, prior = case_cfgs[name]
        print(f"\n=== case {name} ===", flush=True)
        summary[name] = run_case(
            args.out_root / name,
            prompt,
            args.seed,
            variants,
            args.reuse_prior,
            prior,
        )

    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nALL DONE → {args.out_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
