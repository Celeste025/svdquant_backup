#!/usr/bin/env python3
"""Low-invasiveness ablation: restore Wan blocks[18:25] to BF16 on top of ungated-s16.

Does not modify deepcompressor. Loads existing W4A4 ckpt, swaps mid blocks with
deepcopied BF16 modules, then (1) writes a GIF and (2) first-step layer NMSE curve.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
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

# Reuse loaders / hooks from existing scripts without touching deepcompressor.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_firststep_block_nmse import (  # noqa: E402
    collect_wan_acts,
    compare_stores,
    load_wan_bf16,
    load_wan_quant,
)
from infer_wan_bf16_vs_w4a4_one import NEGATIVE, _save_video  # noqa: E402
from plot_firststep_block_nmse import plot_one  # noqa: E402

PROMPT = "A person is riding a bike"
SEED = 42
KEEP_BLOCKS = list(range(18, 25))  # 18..24 inclusive
CKPT = DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16"
OUT = DATA_ROOT / "compare" / "wan_keep_bf16_b18_24"


def restore_blocks_from_bf16(quant_pipe, bf16_blocks_cpu: dict[int, torch.nn.Module]) -> None:
    device = next(quant_pipe.transformer.parameters()).device
    for i, blk_cpu in bf16_blocks_cpu.items():
        quant_pipe.transformer.blocks[i] = copy.deepcopy(blk_cpu).to(
            device=device, dtype=torch.bfloat16
        )
        print(f"[restore] blocks[{i}] <- BF16 (hooks/lowrank cleared by module replace)", flush=True)


@torch.inference_mode()
def generate_gif(pipe, prompt: str, seed: int, out_path: Path) -> dict:
    import time

    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe(
        prompt,
        negative_prompt=NEGATIVE,
        height=480,
        width=832,
        num_frames=33,
        num_inference_steps=50,
        guidance_scale=6.0,
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
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    frames_out = []
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
            ImageDraw.Draw(canvas).text((x + 6, 4), lab, fill=(240, 240, 240), font=font)
            x += c.width
        frames_out.append(canvas)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_out[0].save(
        out_path, save_all=True, append_images=frames_out[1:], duration=100, loop=0
    )
    print(f"[montage] {out_path}", flush=True)
    for im in seqs:
        im.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--keep-blocks", type=int, nargs="+", default=KEEP_BLOCKS)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--skip-nmse", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keep = sorted(args.keep_blocks)
    print(f"=== keep BF16 blocks {keep} on top of {args.ckpt} ===", flush=True)

    # 1) Snapshot BF16 mid-blocks on CPU, then free BF16 pipe.
    print("[1/4] load BF16 and snapshot mid-blocks...", flush=True)
    bf16_pipe = load_wan_bf16()
    bf16_blocks_cpu = {
        i: copy.deepcopy(bf16_pipe.transformer.blocks[i]).cpu() for i in keep
    }
    # Also capture BF16 first-step acts now (before deleting) if NMSE needed.
    bf16_acts = None
    if not args.skip_nmse:
        print("[1b] collect BF16 first-step acts...", flush=True)
        bf16_acts = collect_wan_acts(bf16_pipe, args.prompt, args.seed)
        print(f"  captured {len(bf16_acts)} tensors", flush=True)
    del bf16_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 2) Load quant and restore mid-blocks.
    print("[2/4] load W4A4 and restore mid-blocks to BF16...", flush=True)
    quant_pipe = load_wan_quant(args.ckpt)
    restore_blocks_from_bf16(quant_pipe, bf16_blocks_cpu)
    del bf16_blocks_cpu
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "ckpt": str(args.ckpt),
        "keep_bf16_blocks": keep,
        "method": "replace transformer.blocks[i] with deepcopy(BF16 block)",
    }

    # 3) GIF
    if not args.skip_gif:
        print("[3/4] generate hybrid GIF...", flush=True)
        hybrid_gif = args.out_dir / f"hybrid_keep_b{keep[0]}-{keep[-1]}_seed{args.seed}.gif"
        report["hybrid_gif"] = generate_gif(quant_pipe, args.prompt, args.seed, hybrid_gif)
        # Side-by-side with prior BF16 / ungated if present.
        prior = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "heldout_bike"
        trio = [
            (prior / f"bf16_seed{args.seed}.gif", "BF16"),
            (prior / f"ungated_s16_seed{args.seed}.gif", "ungated-s16"),
            (hybrid_gif, f"keep-b{keep[0]}-{keep[-1]}"),
        ]
        if all(p.is_file() for p, _ in trio):
            montage = args.out_dir / f"side_by_side_seed{args.seed}.gif"
            make_side_by_side([p for p, _ in trio], [lab for _, lab in trio], montage)
            report["side_by_side"] = str(montage)

    # 4) First-step NMSE
    if not args.skip_nmse:
        print("[4/4] collect hybrid first-step acts + NMSE...", flush=True)
        assert bf16_acts is not None
        hybrid_acts = collect_wan_acts(quant_pipe, args.prompt, args.seed)
        rows = compare_stores(bf16_acts, hybrid_acts)
        meta = {
            "model": "wan",
            "prompt": args.prompt,
            "seed": args.seed,
            "ckpt": str(args.ckpt),
            "keep_bf16_blocks": keep,
            "n_layers": len(rows),
            "rows": rows,
        }
        out_json = args.out_dir / "wan_keep_bf16_b18_24_firststep_block_nmse.json"
        out_json.write_text(json.dumps(meta, indent=2))
        peak = max(rows, key=lambda r: r["nmse"])
        print(
            f"[nmse] peak={100 * peak['nmse']:.3f}% @ {peak['name']}; "
            f"final={100 * rows[-1]['nmse']:.3f}% @ {rows[-1]['name']}",
            flush=True,
        )
        out_png = args.out_dir / "wan_keep_bf16_b18_24_firststep_block_nmse.png"
        summary = plot_one(
            out_json,
            out_png,
            f"Wan first-step NMSE (W4A4 + BF16 blocks {keep[0]}-{keep[-1]})",
        )
        report["nmse"] = summary
        # Compare peak vs baseline full W4A4 curve if available.
        base = DATA_ROOT / "compare" / "firststep_block_nmse" / "wan_firststep_block_nmse.json"
        if base.is_file():
            base_rows = json.loads(base.read_text())["rows"]
            base_peak = max(base_rows, key=lambda r: r["nmse"])
            report["baseline_full_w4a4"] = {
                "peak_name": base_peak["name"],
                "peak_nmse_pct": 100 * base_peak["nmse"],
                "final_nmse_pct": 100 * base_rows[-1]["nmse"],
            }
            report["delta_peak_nmse_pct"] = (
                report["nmse"]["peak_nmse_pct"] - report["baseline_full_w4a4"]["peak_nmse_pct"]
            )

    del quant_pipe
    gc.collect()
    torch.cuda.empty_cache()
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"ALL DONE → {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
