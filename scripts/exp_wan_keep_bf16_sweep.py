#!/usr/bin/env python3
"""Sweep: keep different Wan block sets in BF16 on top of ungated-s16.

Low-invasiveness: load BF16+W4A4 once, snapshot both block sets on CPU, then
for each config swap→measure first-step NMSE→swap back. Optional GIFs for a
subset. Does not modify deepcompressor.
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
from collect_firststep_block_nmse import (  # noqa: E402
    collect_wan_acts,
    compare_stores,
    load_wan_bf16,
    load_wan_quant,
)
from exp_wan_keep_bf16_blocks_18_24 import generate_gif, make_side_by_side  # noqa: E402
from plot_firststep_block_nmse import plot_one  # noqa: E402

PROMPT = "A person is riding a bike"
SEED = 42
CKPT = DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16"
OUT = DATA_ROOT / "compare" / "wan_keep_bf16_sweep"

# name -> block indices (inclusive ranges expanded below)
CONFIGS: list[tuple[str, list[int]]] = [
    ("b17", [17]),
    ("b21", [21]),
    ("b21-22", list(range(21, 23))),
    ("b14-17", list(range(14, 18))),
    ("b15-17", list(range(15, 18))),
    ("b17-21", list(range(17, 22))),
    ("b12-17", list(range(12, 18))),
    ("b14-24", list(range(14, 25))),
    ("b0-16", list(range(0, 17))),
    ("b17_21", [17, 21]),
]

# GIFs are slow (~2min); only for these after NMSE (can override via CLI).
DEFAULT_GIF_TAGS = ["b14-17", "b17-21", "b14-24", "b0-16", "b17_21"]


def tag_slug(blocks: list[int]) -> str:
    return "b" + "-".join(str(i) for i in blocks) if len(blocks) <= 4 else f"b{blocks[0]}-{blocks[-1]}"


def snapshot_blocks(transformer, indices: list[int]) -> dict[int, torch.nn.Module]:
    return {i: copy.deepcopy(transformer.blocks[i]).cpu() for i in indices}


def apply_blocks(transformer, blocks_cpu: dict[int, torch.nn.Module], indices: list[int]) -> None:
    device = next(transformer.parameters()).device
    for i in indices:
        transformer.blocks[i] = copy.deepcopy(blocks_cpu[i]).to(device=device, dtype=torch.bfloat16)


def plot_overlay(curves: dict[str, list[dict]], out_png: Path, title: str) -> None:
    """curves: label -> rows from compare_stores / JSON."""
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=160)
    cmap = plt.get_cmap("tab10")
    baseline = curves.get("full_w4a4")
    others = [(k, v) for k, v in curves.items() if k != "full_w4a4"]

    def _draw(label: str, rows: list[dict], color, lw, alpha, zorder):
        xs = np.arange(len(rows))
        ys = np.asarray([100.0 * r["nmse"] for r in rows])
        ax.plot(xs, ys, label=label, color=color, linewidth=lw, alpha=alpha, zorder=zorder, marker="o", markersize=2.5)

    if baseline is not None:
        _draw("full_w4a4", baseline, "#444444", 2.2, 0.9, 3)
    for i, (lab, rows) in enumerate(others):
        _draw(lab, rows, cmap(i % 10), 1.4, 0.85, 2)

    ref_rows = baseline or (others[0][1] if others else None)
    if ref_rows:
        labels = [r["name"].split("_", 1)[-1] for r in ref_rows]
        step = max(1, len(labels) // 16)
        tick_idx = list(range(0, len(labels), step))
        if tick_idx[-1] != len(labels) - 1:
            tick_idx.append(len(labels) - 1)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("NMSE (%)")
    ax.set_xlabel("Layer (model input → output)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[overlay] {out_png}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--gif-tags",
        type=str,
        nargs="*",
        default=DEFAULT_GIF_TAGS,
        help="Config tags to also generate GIFs for (empty = none)",
    )
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = CONFIGS
    all_idx = sorted({i for _, blks in configs for i in blks})
    gif_tags = set() if args.skip_gif else set(args.gif_tags)

    print(f"=== sweep keep-BF16 blocks on {args.ckpt} ===", flush=True)
    print(f"configs: {[t for t, _ in configs]}", flush=True)
    print(f"union blocks: {all_idx}", flush=True)
    print(f"gif tags: {sorted(gif_tags)}", flush=True)

    # --- BF16 snapshot + acts ---
    print("[1] load BF16, snapshot blocks, collect first-step acts...", flush=True)
    bf16_pipe = load_wan_bf16()
    bf16_cpu = snapshot_blocks(bf16_pipe.transformer, all_idx)
    bf16_acts = collect_wan_acts(bf16_pipe, args.prompt, args.seed)
    print(f"  captured {len(bf16_acts)} BF16 tensors; snapped {len(bf16_cpu)} blocks", flush=True)
    del bf16_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- W4A4 load + quant snapshot ---
    print("[2] load W4A4, snapshot quant blocks...", flush=True)
    quant_pipe = load_wan_quant(args.ckpt)
    quant_cpu = snapshot_blocks(quant_pipe.transformer, all_idx)
    print(f"  snapped {len(quant_cpu)} quant blocks", flush=True)

    baseline_path = DATA_ROOT / "compare" / "firststep_block_nmse" / "wan_firststep_block_nmse.json"
    curves: dict[str, list[dict]] = {}
    if baseline_path.is_file():
        curves["full_w4a4"] = json.loads(baseline_path.read_text())["rows"]

    # Prior b18-24 result if present (not re-run).
    prior = DATA_ROOT / "compare" / "wan_keep_bf16_b18_24" / "wan_keep_bf16_b18_24_firststep_block_nmse.json"
    if prior.is_file():
        curves["b18-24"] = json.loads(prior.read_text())["rows"]

    summaries: list[dict] = []
    prior_bike = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "heldout_bike"

    for tag, keep in configs:
        keep = sorted(keep)
        cfg_dir = args.out_dir / tag
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [{tag}] keep BF16 {keep} ===", flush=True)

        apply_blocks(quant_pipe.transformer, bf16_cpu, keep)
        hybrid_acts = collect_wan_acts(quant_pipe, args.prompt, args.seed)
        rows = compare_stores(bf16_acts, hybrid_acts)
        # Restore quant blocks for next config.
        apply_blocks(quant_pipe.transformer, quant_cpu, keep)

        peak = max(rows, key=lambda r: r["nmse"])
        final = rows[-1]
        meta = {
            "model": "wan",
            "prompt": args.prompt,
            "seed": args.seed,
            "ckpt": str(args.ckpt),
            "tag": tag,
            "keep_bf16_blocks": keep,
            "n_layers": len(rows),
            "rows": rows,
        }
        out_json = cfg_dir / f"{tag}_firststep_block_nmse.json"
        out_json.write_text(json.dumps(meta, indent=2))
        out_png = cfg_dir / f"{tag}_firststep_block_nmse.png"
        plot_one(out_json, out_png, f"Wan first-step NMSE (keep BF16: {tag})")

        entry = {
            "tag": tag,
            "keep_bf16_blocks": keep,
            "peak_name": peak["name"],
            "peak_nmse_pct": 100.0 * peak["nmse"],
            "final_nmse_pct": 100.0 * final["nmse"],
            "json": str(out_json),
            "png": str(out_png),
        }
        if "full_w4a4" in curves:
            bp = max(curves["full_w4a4"], key=lambda r: r["nmse"])
            entry["delta_peak_vs_full_w4a4"] = entry["peak_nmse_pct"] - 100.0 * bp["nmse"]
            entry["delta_final_vs_full_w4a4"] = entry["final_nmse_pct"] - 100.0 * curves["full_w4a4"][-1]["nmse"]
        summaries.append(entry)
        curves[tag] = rows
        print(
            f"[{tag}] peak={entry['peak_nmse_pct']:.3f}% @{peak['name']}  "
            f"final={entry['final_nmse_pct']:.3f}%",
            flush=True,
        )

        if tag in gif_tags:
            print(f"[{tag}] generating GIF...", flush=True)
            apply_blocks(quant_pipe.transformer, bf16_cpu, keep)
            gif_path = cfg_dir / f"hybrid_{tag}_seed{args.seed}.gif"
            gif_info = generate_gif(quant_pipe, args.prompt, args.seed, gif_path)
            entry["hybrid_gif"] = gif_info
            trio = [
                (prior_bike / f"bf16_seed{args.seed}.gif", "BF16"),
                (prior_bike / f"ungated_s16_seed{args.seed}.gif", "ungated-s16"),
                (gif_path, tag),
            ]
            if all(p.is_file() for p, _ in trio):
                montage = cfg_dir / f"side_by_side_{tag}_seed{args.seed}.gif"
                make_side_by_side([p for p, _ in trio], [lab for _, lab in trio], montage)
                entry["side_by_side"] = str(montage)
            apply_blocks(quant_pipe.transformer, quant_cpu, keep)
            gc.collect()
            torch.cuda.empty_cache()

    # Overlay + table
    overlay_png = args.out_dir / "overlay_firststep_block_nmse.png"
    plot_overlay(curves, overlay_png, "Wan first-step NMSE: keep-BF16 block sweeps")

    # Compact CSV-ish table
    table_lines = [
        "tag\tkeep\tpeak%\tpeak_layer\tfinal%\tdelta_peak\tdelta_final",
    ]
    for e in summaries:
        table_lines.append(
            f"{e['tag']}\t{e['keep_bf16_blocks']}\t{e['peak_nmse_pct']:.3f}\t"
            f"{e['peak_name']}\t{e['final_nmse_pct']:.3f}\t"
            f"{e.get('delta_peak_vs_full_w4a4', float('nan')):+.3f}\t"
            f"{e.get('delta_final_vs_full_w4a4', float('nan')):+.3f}"
        )
    table_txt = "\n".join(table_lines) + "\n"
    (args.out_dir / "summary_table.tsv").write_text(table_txt)
    print("\n" + table_txt, flush=True)

    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "ckpt": str(args.ckpt),
        "configs": summaries,
        "overlay": str(overlay_png),
        "prior_b18_24_included_in_overlay": prior.is_file(),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    del quant_pipe, bf16_cpu, quant_cpu, bf16_acts
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
