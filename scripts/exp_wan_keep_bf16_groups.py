#!/usr/bin/env python3
"""Keep contiguous Wan block groups in BF16 for the FULL denoising trajectory.

Loads ungated-s16, permanently replaces selected transformer.blocks[i] with BF16
copies, then runs the full 50-step pipeline (every timestep uses those BF16
blocks). Records GIF + per-step latent NMSE vs BF16. Does not modify deepcompressor.
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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pyarrow  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_firststep_block_nmse import load_wan_bf16, load_wan_quant  # noqa: E402
from infer_wan_bf16_vs_w4a4_one import NEGATIVE, _save_video  # noqa: E402

PROMPT = "A person is riding a bike"
SEED = 42
CKPT = DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16"
OUT = DATA_ROOT / "compare" / "wan_keep_bf16_groups"

# Wan 1.3B has blocks[0..29]. User "b25-30" → blocks 25..29 (no block 30).
CONFIGS: list[tuple[str, list[int]]] = [
    ("b0-6", list(range(0, 7))),
    ("b7-12", list(range(7, 13))),
    ("b13-18", list(range(13, 19))),
    ("b19-24", list(range(19, 25))),
    ("b25-30", list(range(25, 30))),  # 25..29
]


def snapshot_blocks(transformer, indices: list[int]) -> dict[int, torch.nn.Module]:
    return {i: copy.deepcopy(transformer.blocks[i]).cpu() for i in indices}


def apply_blocks(transformer, blocks_cpu: dict[int, torch.nn.Module], indices: list[int]) -> None:
    device = next(transformer.parameters()).device
    for i in indices:
        # Keep dtype of snapshot; only move device. Avoid stripping hook branch state.
        blk = copy.deepcopy(blocks_cpu[i]).to(device=device)
        transformer.blocks[i] = blk
        print(f"  [apply] blocks[{i}] <- {type(blk).__name__}", flush=True)


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).pow(2).mean() / a.pow(2).mean().clamp_min(1e-12))


@torch.inference_mode()
def run_full_with_latents(pipe, prompt: str, seed: int, save_gif: Path | None) -> tuple[list[torch.Tensor], dict | None]:
    """Full 50-step generate; capture latents after every step. Optionally save GIF."""
    import time

    latents: list[torch.Tensor] = []

    def cb(_pipe, index, _timestep, kwargs):
        lat = kwargs["latents"]
        latents.append(lat.detach().to("cpu", torch.bfloat16).contiguous().clone())
        return kwargs

    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    kwargs = dict(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        height=480,
        width=832,
        num_frames=33,
        num_inference_steps=50,
        guidance_scale=6.0,
        generator=gen,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    if save_gif is None:
        kwargs["output_type"] = "latent"
        pipe(**kwargs)
        info = None
    else:
        out = pipe(**kwargs)
        frames = out.frames[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        _save_video(frames, save_gif)
        info = {
            "path": str(save_gif),
            "seconds": elapsed,
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
            "n_frames": len(frames),
        }
        print(f"[gen] {save_gif.name} {elapsed:.1f}s peak={info['peak_alloc_gib']:.2f}GiB steps={len(latents)}", flush=True)
    assert len(latents) == 50, f"expected 50 latent steps, got {len(latents)}"
    return latents, info


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
    for im in seqs:
        im.close()
    print(f"[montage] {out_path}", flush=True)


def gif_pixel_mae_vs_ref(ref_gif: Path, hyp_gif: Path) -> dict:
    from PIL import Image

    ref, hyp = Image.open(ref_gif), Image.open(hyp_gif)
    n = min(getattr(ref, "n_frames", 1), getattr(hyp, "n_frames", 1))
    maes = []
    for i in range(n):
        ref.seek(i)
        hyp.seek(i)
        a = np.asarray(ref.convert("RGB"), dtype=np.float32)
        b = np.asarray(hyp.convert("RGB"), dtype=np.float32)
        maes.append(float(np.mean(np.abs(a - b))))
    ref.close()
    hyp.close()
    return {"mean_mae": float(np.mean(maes)), "max_mae": float(np.max(maes)), "n_frames": n}


def plot_multistep(curves: dict[str, list[float]], out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    for lab, ys in curves.items():
        ax.plot(np.arange(len(ys)), [100.0 * y for y in ys], label=lab, linewidth=1.6)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("latent NMSE vs BF16 (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-ungated-gif", action="store_true",
                        help="Reuse heldout ungated GIF if present")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_idx = sorted({i for _, blks in CONFIGS for i in blks})
    n_blocks = None
    print(f"=== full-trajectory keep-BF16 groups on {args.ckpt} ===", flush=True)
    print(f"configs: {[(t, blks) for t, blks in CONFIGS]}", flush=True)

    # --- BF16 reference: snapshot blocks + full traj + GIF ---
    print("[1] BF16: snapshot blocks + full generate...", flush=True)
    bf16_pipe = load_wan_bf16()
    n_blocks = len(bf16_pipe.transformer.blocks)
    print(f"  transformer has {n_blocks} blocks", flush=True)
    for tag, blks in CONFIGS:
        bad = [i for i in blks if i < 0 or i >= n_blocks]
        if bad:
            raise RuntimeError(f"{tag} has invalid block indices {bad} (n_blocks={n_blocks})")
    bf16_cpu = snapshot_blocks(bf16_pipe.transformer, all_idx)
    bf16_gif = args.out_dir / f"bf16_seed{args.seed}.gif"
    bf16_lats, bf16_info = run_full_with_latents(bf16_pipe, args.prompt, args.seed, bf16_gif)
    del bf16_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- W4A4 load ---
    print("[2] load W4A4 + snapshot quant blocks...", flush=True)
    quant_pipe = load_wan_quant(args.ckpt)
    quant_cpu = snapshot_blocks(quant_pipe.transformer, all_idx)

    curves: dict[str, list[float]] = {}
    summaries: list[dict] = []

    # ungated full traj + gif
    print("[3] ungated full generate...", flush=True)
    prior_u = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "heldout_bike" / f"ungated_s16_seed{args.seed}.gif"
    ungated_gif = args.out_dir / f"ungated_seed{args.seed}.gif"
    if args.skip_ungated_gif and prior_u.is_file():
        import shutil
        shutil.copy2(prior_u, ungated_gif)
        u_lats, _ = run_full_with_latents(quant_pipe, args.prompt, args.seed, save_gif=None)
        u_info = {"path": str(ungated_gif), "copied_from": str(prior_u)}
    else:
        u_lats, u_info = run_full_with_latents(quant_pipe, args.prompt, args.seed, ungated_gif)
    curves["ungated"] = [nmse(bf16_lats[i], u_lats[i]) for i in range(50)]
    u_mae = gif_pixel_mae_vs_ref(bf16_gif, ungated_gif)
    summaries.append(
        {
            "tag": "ungated",
            "keep_bf16_blocks": [],
            "final_lat_nmse_pct": 100.0 * curves["ungated"][-1],
            "mean_lat_nmse_pct": 100.0 * float(np.mean(curves["ungated"])),
            "gif_mean_mae_vs_bf16": u_mae["mean_mae"],
            "gif": u_info,
        }
    )
    print(
        f"[ungated] final_lat_nmse={summaries[-1]['final_lat_nmse_pct']:.3f}% "
        f"gif_mae={u_mae['mean_mae']:.2f}",
        flush=True,
    )

    prior_bike = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "heldout_bike"

    for tag, keep in CONFIGS:
        keep = sorted(keep)
        cfg_dir = args.out_dir / tag
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [{tag}] keep BF16 {keep} for ALL denoising steps ===", flush=True)
        apply_blocks(quant_pipe.transformer, bf16_cpu, keep)
        gif_path = cfg_dir / f"hybrid_{tag}_seed{args.seed}.gif"
        h_lats, h_info = run_full_with_latents(quant_pipe, args.prompt, args.seed, gif_path)
        # restore quant for next config
        apply_blocks(quant_pipe.transformer, quant_cpu, keep)

        ys = [nmse(bf16_lats[i], h_lats[i]) for i in range(50)]
        curves[tag] = ys
        mae = gif_pixel_mae_vs_ref(bf16_gif, gif_path)
        entry = {
            "tag": tag,
            "keep_bf16_blocks": keep,
            "final_lat_nmse_pct": 100.0 * ys[-1],
            "mean_lat_nmse_pct": 100.0 * float(np.mean(ys)),
            "gif_mean_mae_vs_bf16": mae["mean_mae"],
            "gif_max_mae_vs_bf16": mae["max_mae"],
            "gif": h_info,
            "delta_final_lat_vs_ungated": 100.0 * (ys[-1] - curves["ungated"][-1]),
            "delta_gif_mae_vs_ungated": mae["mean_mae"] - u_mae["mean_mae"],
        }
        # side-by-side
        trio = [
            (bf16_gif, "BF16"),
            (ungated_gif, "ungated"),
            (gif_path, tag),
        ]
        if all(p.is_file() for p, _ in trio):
            montage = cfg_dir / f"side_by_side_{tag}_seed{args.seed}.gif"
            make_side_by_side([p for p, _ in trio], [lab for _, lab in trio], montage)
            entry["side_by_side"] = str(montage)
        (cfg_dir / "metrics.json").write_text(json.dumps({**entry, "lat_nmse": ys}, indent=2))
        summaries.append(entry)
        print(
            f"[{tag}] final_lat={entry['final_lat_nmse_pct']:.3f}% "
            f"(Δungated {entry['delta_final_lat_vs_ungated']:+.3f}) "
            f"gif_mae={mae['mean_mae']:.2f} (Δ {entry['delta_gif_mae_vs_ungated']:+.2f})",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    plot_multistep(
        curves,
        args.out_dir / "multistep_latent_nmse.png",
        "Wan latent NMSE vs BF16 (keep-BF16 groups, full trajectory)",
    )

    # big montage: bf16 | ungated | all groups (may be wide — use mid frame strip too)
    group_gifs = [(args.out_dir / t / f"hybrid_{t}_seed{args.seed}.gif", t) for t, _ in CONFIGS]
    all_paths = [bf16_gif, ungated_gif] + [p for p, _ in group_gifs]
    all_labs = ["BF16", "ungated"] + [t for t, _ in CONFIGS]
    if all(p.is_file() for p in all_paths):
        make_side_by_side(all_paths, all_labs, args.out_dir / f"side_by_side_all_seed{args.seed}.gif")

    lines = ["tag\tkeep\tfinal_lat%\tmean_lat%\tgif_mae\tdelta_final_lat\tdelta_mae"]
    for e in summaries:
        lines.append(
            f"{e['tag']}\t{e['keep_bf16_blocks']}\t{e['final_lat_nmse_pct']:.3f}\t"
            f"{e['mean_lat_nmse_pct']:.3f}\t{e['gif_mean_mae_vs_bf16']:.2f}\t"
            f"{e.get('delta_final_lat_vs_ungated', 0):+.3f}\t"
            f"{e.get('delta_gif_mae_vs_ungated', 0):+.2f}"
        )
    table = "\n".join(lines) + "\n"
    (args.out_dir / "summary_table.tsv").write_text(table)
    print("\n" + table, flush=True)

    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "ckpt": str(args.ckpt),
        "n_blocks": n_blocks,
        "note": "Selected blocks stay BF16 for every denoising timestep of the full 50-step generate.",
        "note_b25_30": "Tag b25-30 maps to blocks[25:30] i.e. indices 25..29 (model has 30 blocks).",
        "configs": summaries,
        "bf16_gif": bf16_info,
        "overlay": str(args.out_dir / "multistep_latent_nmse.png"),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    del quant_pipe, bf16_cpu, quant_cpu, bf16_lats
    gc.collect()
    torch.cuda.empty_cache()
    print(f"ALL DONE → {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
