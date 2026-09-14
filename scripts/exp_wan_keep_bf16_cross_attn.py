#!/usr/bin/env python3
"""Ablation: restore ALL Wan cross-attn (blocks.*.attn2) to BF16 on ungated W4A4.

- Generate one heldout video (GIF)
- First-timestep per-block NMSE vs BF16 (line chart, one point per layer)
- Overlay vs full W4A4 baseline curve if available

Does not modify deepcompressor: module-swap only.
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
os.environ.setdefault("DEEPCOMPRESSOR_WAN_GATED", "0")

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
OUT = DATA_ROOT / "compare" / "wan_keep_bf16_cross_attn"
BASELINE_JSON = DATA_ROOT / "compare" / "firststep_block_nmse" / "wan_firststep_block_nmse.json"
PRIOR_GIF = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "heldout_bike"


def snapshot_cross_attns(transformer) -> dict[int, torch.nn.Module]:
    out: dict[int, torch.nn.Module] = {}
    for i, block in enumerate(transformer.blocks):
        if not hasattr(block, "attn2"):
            raise AttributeError(f"blocks[{i}] has no attn2")
        out[i] = copy.deepcopy(block.attn2).cpu()
    return out


def restore_cross_attns(transformer, attn2_cpu: dict[int, torch.nn.Module]) -> None:
    device = next(transformer.parameters()).device
    for i, mod_cpu in attn2_cpu.items():
        transformer.blocks[i].attn2 = copy.deepcopy(mod_cpu).to(device=device, dtype=torch.bfloat16)
        print(f"[restore] blocks[{i}].attn2 <- BF16", flush=True)


def plot_overlay(hybrid_json: Path, baseline_json: Path, out_png: Path) -> dict:
    hy = json.loads(hybrid_json.read_text())["rows"]
    base = json.loads(baseline_json.read_text())["rows"]
    # Align by name
    base_by = {r["name"]: r["nmse"] for r in base}
    names = [r["name"] for r in hy]
    ys_h = [100.0 * r["nmse"] for r in hy]
    ys_b = [100.0 * base_by.get(n, float("nan")) for n in names]
    xs = np.arange(len(names))
    labels = [n.split("_", 1)[-1] for n in names]

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    ax.plot(xs, ys_b, marker="o", markersize=3.0, linewidth=1.5, color="#888888", label="full W4A4")
    ax.plot(xs, ys_h, marker="o", markersize=3.5, linewidth=1.7, color="#1f4e79", label="W4A4 + BF16 cross-attn")
    ax.set_ylabel("first-step NMSE vs BF16 (%)")
    ax.set_xlabel("Layer (input → output)")
    ax.set_title("Wan first-step error propagation — keep all attn2 in BF16")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    step = max(1, len(labels) // 16)
    tick_idx = list(range(0, len(labels), step))
    if tick_idx[-1] != len(labels) - 1:
        tick_idx.append(len(labels) - 1)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)
    return {
        "baseline_peak_pct": float(np.nanmax(ys_b)),
        "hybrid_peak_pct": float(np.nanmax(ys_h)),
        "baseline_final_pct": float(ys_b[-1]),
        "hybrid_final_pct": float(ys_h[-1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--prompt", type=str, default=PROMPT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-gif", action="store_true")
    ap.add_argument("--skip-nmse", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== keep ALL cross-attn (attn2) BF16 on {args.ckpt} ===", flush=True)

    # 1) BF16: snapshot attn2 + first-step acts
    print("[1/4] load BF16, snapshot attn2...", flush=True)
    bf16_pipe = load_wan_bf16()
    n_blocks = len(bf16_pipe.transformer.blocks)
    attn2_cpu = snapshot_cross_attns(bf16_pipe.transformer)
    print(f"  snapshotted {len(attn2_cpu)}/{n_blocks} attn2 modules", flush=True)

    bf16_acts = None
    if not args.skip_nmse:
        print("[1b] collect BF16 first-step acts...", flush=True)
        bf16_acts = collect_wan_acts(bf16_pipe, args.prompt, args.seed)
        print(f"  captured {len(bf16_acts)} tensors", flush=True)

    del bf16_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 2) Quant + restore attn2
    print("[2/4] load W4A4 and restore all attn2 to BF16...", flush=True)
    quant_pipe = load_wan_quant(args.ckpt)
    restore_cross_attns(quant_pipe.transformer, attn2_cpu)
    del attn2_cpu
    gc.collect()
    torch.cuda.empty_cache()

    report: dict = {
        "prompt": args.prompt,
        "seed": args.seed,
        "ckpt": str(args.ckpt),
        "keep_bf16": "all blocks.*.attn2 (cross-attn)",
        "n_blocks": n_blocks,
        "method": "replace transformer.blocks[i].attn2 with deepcopy(BF16 attn2)",
    }

    # 3) GIF
    if not args.skip_gif:
        print("[3/4] generate hybrid GIF...", flush=True)
        hybrid_gif = args.out_dir / f"hybrid_bf16_crossattn_seed{args.seed}.gif"
        report["hybrid_gif"] = generate_gif(quant_pipe, args.prompt, args.seed, hybrid_gif)
        trio = [
            (PRIOR_GIF / f"bf16_seed{args.seed}.gif", "BF16"),
            (PRIOR_GIF / f"ungated_s16_seed{args.seed}.gif", "ungated-s16"),
            (hybrid_gif, "BF16-cross-attn"),
        ]
        if all(p.is_file() for p, _ in trio):
            montage = args.out_dir / f"side_by_side_seed{args.seed}.gif"
            make_side_by_side([p for p, _ in trio], [lab for _, lab in trio], montage)
            report["side_by_side"] = str(montage)

    # 4) First-step NMSE vs BF16
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
            "keep_bf16": "all attn2",
            "n_layers": len(rows),
            "rows": rows,
        }
        out_json = args.out_dir / "wan_bf16_crossattn_firststep_block_nmse.json"
        out_json.write_text(json.dumps(meta, indent=2))
        peak = max(rows, key=lambda r: r["nmse"])
        print(
            f"[nmse] peak={100 * peak['nmse']:.3f}% @ {peak['name']}; "
            f"final={100 * rows[-1]['nmse']:.3f}% @ {rows[-1]['name']}",
            flush=True,
        )
        out_png = args.out_dir / "wan_bf16_crossattn_firststep_block_nmse.png"
        summary = plot_one(
            out_json,
            out_png,
            "Wan first-step NMSE (W4A4 + all cross-attn BF16)",
        )
        report["nmse"] = summary

        if BASELINE_JSON.is_file():
            overlay_png = args.out_dir / "overlay_vs_full_w4a4_firststep_block_nmse.png"
            report["overlay_vs_full_w4a4"] = plot_overlay(out_json, BASELINE_JSON, overlay_png)
            base_rows = json.loads(BASELINE_JSON.read_text())["rows"]
            base_peak = max(base_rows, key=lambda r: r["nmse"])
            report["baseline_full_w4a4"] = {
                "peak_name": base_peak["name"],
                "peak_nmse_pct": 100 * base_peak["nmse"],
                "final_nmse_pct": 100 * base_rows[-1]["nmse"],
            }
            report["delta_peak_nmse_pct"] = (
                report["nmse"]["peak_nmse_pct"] - report["baseline_full_w4a4"]["peak_nmse_pct"]
            )
            report["delta_final_nmse_pct"] = (
                report["nmse"]["final_nmse_pct"] - report["baseline_full_w4a4"]["final_nmse_pct"]
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
