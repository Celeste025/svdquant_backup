#!/usr/bin/env python3
"""Wan BF16 vs W4A4 short-frame smoke: num_frames in {1,5} (4n+1).

Reuses the same prompt/seed protocol as infer_wan_bf16_vs_w4a4_one.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
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

PROMPT = "A person is riding a bike"
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)
STEPS = 50
GUIDANCE = 6.0
HEIGHT = 480
WIDTH = 832


def _parse_cfg(argv: list[str]):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    sys.argv = ["infer_wan_short_frames.py", *argv]
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
        append_images=pil_frames[1:] if len(pil_frames) > 1 else [],
        duration=100,
        loop=0,
    )


def _side_by_side(bf16_path: Path, quant_path: Path, out_path: Path) -> None:
    from PIL import Image
    import numpy as np

    bf = Image.open(bf16_path)
    q = Image.open(quant_path)
    bf_frames, q_frames = [], []
    try:
        while True:
            bf_frames.append(bf.copy().convert("RGB"))
            bf.seek(bf.tell() + 1)
    except EOFError:
        pass
    try:
        while True:
            q_frames.append(q.copy().convert("RGB"))
            q.seek(q.tell() + 1)
    except EOFError:
        pass
    n = max(len(bf_frames), len(q_frames), 1)
    while len(bf_frames) < n:
        bf_frames.append(bf_frames[-1] if bf_frames else Image.new("RGB", (WIDTH, HEIGHT)))
    while len(q_frames) < n:
        q_frames.append(q_frames[-1] if q_frames else Image.new("RGB", (WIDTH, HEIGHT)))
    merged = []
    for a, b in zip(bf_frames, q_frames):
        w = a.width + b.width
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        merged.append(canvas)
    merged[0].save(out_path, save_all=True, append_images=merged[1:], duration=100, loop=0)


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


def load_quant_pipeline(ckpt_dir: Path, quant_yaml: str):
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    repo = Path(__file__).resolve().parents[1]
    diffusion = repo / "third_party" / "deepcompressor" / "examples" / "diffusion"
    os.chdir(diffusion)
    out_root = DATA_ROOT / "compare" / "wan_short_frames" / "ptq_load_scratch"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _parse_cfg(
        [
            "configs/model/wan2.1-1.3b.yaml",
            quant_yaml,
            "configs/svdquant/wan_smoke.yaml",
            f"--load-from={ckpt_dir}",
            f"--output-root={out_root}",
            f"--cache-root={out_root}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    print(f"[quant] building Wan + load {ckpt_dir} ({quant_yaml})...", flush=True)
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
def generate(pipe, prompt: str, seed: int, num_frames: int, out_path: Path) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe(
        prompt,
        negative_prompt=NEGATIVE,
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
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
        "num_frames_req": num_frames,
    }
    print(
        f"[gen] {out_path.name}: req={num_frames} got={len(frames)} "
        f"{elapsed:.1f}s peak={info['peak_alloc_gib']:.2f}GiB",
        flush=True,
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16",
        help="PTQ ckpt (default: int4 s16 W4A4)",
    )
    parser.add_argument(
        "--quant-yaml",
        type=str,
        default="configs/svdquant/int4.yaml",
        help="Must match ckpt recipe (int4.yaml or real_nvfp4.yaml)",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "compare" / "wan_short_frames")
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--only", choices=["both", "bf16", "quant"], default="both")
    parser.add_argument("--tag", type=str, default="int4_s16")
    args = parser.parse_args()

    for f in args.frames:
        if (f - 1) % 4 != 0:
            raise SystemExit(f"num_frames={f} invalid: need 4n+1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "height": HEIGHT,
        "width": WIDTH,
        "frames": args.frames,
        "ckpt": str(args.ckpt),
        "quant_yaml": args.quant_yaml,
        "tag": args.tag,
        "runs": {},
    }

    bf16_pipe = quant_pipe = None
    if args.only in ("both", "bf16"):
        bf16_pipe = load_bf16_pipeline()
    if args.only in ("both", "quant"):
        # free bf16 first if both: generate all bf16 then all quant to save VRAM
        pass

    if args.only in ("both", "bf16"):
        for nf in args.frames:
            key = f"f{nf}"
            report["runs"].setdefault(key, {})
            report["runs"][key]["bf16"] = generate(
                bf16_pipe,
                args.prompt,
                args.seed,
                nf,
                args.out_dir / f"bf16_f{nf}_seed{args.seed}.gif",
            )
        del bf16_pipe
        gc.collect()
        torch.cuda.empty_cache()

    if args.only in ("both", "quant"):
        quant_pipe = load_quant_pipeline(args.ckpt, args.quant_yaml)
        for nf in args.frames:
            key = f"f{nf}"
            report["runs"].setdefault(key, {})
            report["runs"][key]["quant"] = generate(
                quant_pipe,
                args.prompt,
                args.seed,
                nf,
                args.out_dir / f"quant_{args.tag}_f{nf}_seed{args.seed}.gif",
            )
            bf = args.out_dir / f"bf16_f{nf}_seed{args.seed}.gif"
            q = args.out_dir / f"quant_{args.tag}_f{nf}_seed{args.seed}.gif"
            if bf.is_file() and q.is_file():
                sbs = args.out_dir / f"side_by_side_bf16_vs_{args.tag}_f{nf}_seed{args.seed}.gif"
                _side_by_side(bf, q, sbs)
                report["runs"][key]["side_by_side"] = str(sbs)
                print(f"[sbs] {sbs}", flush=True)
        del quant_pipe
        gc.collect()
        torch.cuda.empty_cache()

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
