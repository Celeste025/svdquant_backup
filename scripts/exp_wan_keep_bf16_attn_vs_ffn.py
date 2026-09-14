#!/usr/bin/env python3
"""Ablation: W4A4 Wan with either all Attention or all FFN restored to BF16.

Configs:
  A) bf16_attn_quant_ffn  — blocks.*.attn1 + attn2 = BF16; FFN stays W4A4
  B) quant_attn_bf16_ffn  — blocks.*.ffn = BF16; Attention stays W4A4

For each: heldout GIF + first-step per-block NMSE vs BF16; overlay vs full W4A4.
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
OUT = DATA_ROOT / "compare" / "wan_keep_bf16_attn_vs_ffn"
BASELINE_JSON = DATA_ROOT / "compare" / "firststep_block_nmse" / "wan_firststep_block_nmse.json"
PRIOR_GIF = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "heldout_bike"


def snapshot_part(transformer, part: str) -> dict[int, dict[str, torch.nn.Module]]:
    """part in {'attn','ffn'} → per-block submodule deepcopies on CPU."""
    out: dict[int, dict[str, torch.nn.Module]] = {}
    for i, block in enumerate(transformer.blocks):
        if part == "attn":
            if not hasattr(block, "attn1") or not hasattr(block, "attn2"):
                raise AttributeError(f"blocks[{i}] missing attn1/attn2")
            out[i] = {
                "attn1": copy.deepcopy(block.attn1).cpu(),
                "attn2": copy.deepcopy(block.attn2).cpu(),
            }
        elif part == "ffn":
            if not hasattr(block, "ffn"):
                raise AttributeError(f"blocks[{i}] missing ffn")
            out[i] = {"ffn": copy.deepcopy(block.ffn).cpu()}
        else:
            raise ValueError(part)
    return out


def apply_part(
    transformer,
    snaps: dict[int, dict[str, torch.nn.Module]],
    tag: str,
    *,
    force_bf16: bool = True,
) -> None:
    device = next(transformer.parameters()).device
    for i, mods in snaps.items():
        for name, mod_cpu in mods.items():
            mod = copy.deepcopy(mod_cpu)
            if force_bf16:
                mod = mod.to(device=device, dtype=torch.bfloat16)
            else:
                mod = mod.to(device=device)
            # LowRank / act-quant hooks may hold separate Parameter modules
            for m in mod.modules():
                for hook_dict in (getattr(m, "_forward_hooks", {}), getattr(m, "_forward_pre_hooks", {})):
                    for h in hook_dict.values():
                        br = getattr(h, "branch", None)
                        if isinstance(br, torch.nn.Module):
                            br.to(device=device)
                        proc = getattr(h, "processor", None)
                        if isinstance(proc, torch.nn.Module):
                            proc.to(device=device)
            setattr(transformer.blocks[i], name, mod)
        print(f"[apply {tag}] blocks[{i}] <- {{{', '.join(mods)}}} force_bf16={force_bf16}", flush=True)


def plot_multi_overlay(
    curves: dict[str, Path],
    baseline_json: Path | None,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    colors = {
        "full_w4a4": "#888888",
        "bf16_attn_quant_ffn": "#1f4e79",
        "quant_attn_bf16_ffn": "#b85c38",
    }
    ref_names = None
    if baseline_json is not None and baseline_json.is_file():
        base = json.loads(baseline_json.read_text())["rows"]
        ref_names = [r["name"] for r in base]
        ys = [100.0 * r["nmse"] for r in base]
        ax.plot(
            np.arange(len(ys)),
            ys,
            marker="o",
            markersize=2.8,
            linewidth=1.4,
            color=colors["full_w4a4"],
            label="full W4A4",
        )
    for tag, jp in curves.items():
        rows = json.loads(jp.read_text())["rows"]
        if ref_names is None:
            ref_names = [r["name"] for r in rows]
            xs = np.arange(len(rows))
            ys = [100.0 * r["nmse"] for r in rows]
        else:
            by = {r["name"]: r["nmse"] for r in rows}
            xs = np.arange(len(ref_names))
            ys = [100.0 * by.get(n, float("nan")) for n in ref_names]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            color=colors.get(tag, None),
            label=tag,
        )
    assert ref_names is not None
    labels = [n.split("_", 1)[-1] for n in ref_names]
    ax.set_ylabel("first-step NMSE vs BF16 (%)")
    ax.set_xlabel("Layer (input → output)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
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


def run_one_config(
    *,
    tag: str,
    quant_pipe,
    bf16_snaps: dict[int, dict[str, torch.nn.Module]],
    quant_snaps_backup: dict[int, dict[str, torch.nn.Module]],
    bf16_acts: dict,
    out_dir: Path,
    prompt: str,
    seed: int,
    skip_gif: bool,
    skip_nmse: bool,
) -> dict:
    """Apply BF16 snaps for this part; after run, restore quant snaps."""
    print(f"\n=== config {tag} ===", flush=True)
    apply_part(quant_pipe.transformer, bf16_snaps, tag)

    report: dict = {"tag": tag, "prompt": prompt, "seed": seed}
    cfg_dir = out_dir / tag
    cfg_dir.mkdir(parents=True, exist_ok=True)

    if not skip_gif:
        gif = cfg_dir / f"{tag}_seed{seed}.gif"
        report["gif"] = generate_gif(quant_pipe, prompt, seed, gif)
        trio = [
            (PRIOR_GIF / f"bf16_seed{seed}.gif", "BF16"),
            (PRIOR_GIF / f"ungated_s16_seed{seed}.gif", "ungated-s16"),
            (gif, tag),
        ]
        if all(p.is_file() for p, _ in trio):
            montage = cfg_dir / f"side_by_side_seed{seed}.gif"
            make_side_by_side([p for p, _ in trio], [lab for _, lab in trio], montage)
            report["side_by_side"] = str(montage)
        elif PRIOR_GIF.joinpath(f"bf16_seed{SEED}.gif").is_file() and gif.is_file():
            # fallback montage vs heldout seed42 BF16 if this seed has no prior
            montage = cfg_dir / f"side_by_side_vs_bf16seed{SEED}_seed{seed}.gif"
            make_side_by_side(
                [PRIOR_GIF / f"bf16_seed{SEED}.gif", gif],
                [f"BF16-s{SEED}", f"{tag}-s{seed}"],
                montage,
            )
            report["side_by_side"] = str(montage)

    if not skip_nmse:
        assert bf16_acts is not None
        print(f"[{tag}] first-step acts + NMSE...", flush=True)
        hybrid_acts = collect_wan_acts(quant_pipe, prompt, seed)
        rows = compare_stores(bf16_acts, hybrid_acts)
        meta = {
            "model": "wan",
            "prompt": prompt,
            "seed": seed,
            "tag": tag,
            "n_layers": len(rows),
            "rows": rows,
        }
        out_json = cfg_dir / f"{tag}_firststep_block_nmse.json"
        out_json.write_text(json.dumps(meta, indent=2))
        peak = max(rows, key=lambda r: r["nmse"])
        print(
            f"[{tag}] peak={100*peak['nmse']:.3f}% @ {peak['name']}; "
            f"final={100*rows[-1]['nmse']:.3f}% @ {rows[-1]['name']}",
            flush=True,
        )
        out_png = cfg_dir / f"{tag}_firststep_block_nmse.png"
        summary = plot_one(out_json, out_png, f"Wan first-step NMSE ({tag})")
        report["nmse"] = summary
        report["nmse_json"] = str(out_json)

        if BASELINE_JSON.is_file():
            base_rows = json.loads(BASELINE_JSON.read_text())["rows"]
            base_peak = max(base_rows, key=lambda r: r["nmse"])
            report["baseline_full_w4a4"] = {
                "peak_nmse_pct": 100 * base_peak["nmse"],
                "final_nmse_pct": 100 * base_rows[-1]["nmse"],
            }
            report["delta_peak_nmse_pct"] = summary["peak_nmse_pct"] - report["baseline_full_w4a4"]["peak_nmse_pct"]
            report["delta_final_nmse_pct"] = summary["final_nmse_pct"] - report["baseline_full_w4a4"]["final_nmse_pct"]
    else:
        print(f"[{tag}] skip NMSE", flush=True)

    # restore quant modules for this part so next config starts clean
    print(f"[{tag}] restore quant modules...", flush=True)
    apply_part(quant_pipe.transformer, quant_snaps_backup, f"restore_quant:{tag}", force_bf16=False)
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--prompt", type=str, default=PROMPT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-gif", action="store_true")
    ap.add_argument("--skip-nmse", action="store_true")
    ap.add_argument(
        "--only",
        choices=["both", "bf16_attn_quant_ffn", "quant_attn_bf16_ffn"],
        default="both",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== attn vs ffn BF16 ablation on {args.ckpt} ===", flush=True)

    print("[1] load BF16; snapshot attn1/attn2 + ffn; collect first-step acts...", flush=True)
    bf16_pipe = load_wan_bf16()
    n_blocks = len(bf16_pipe.transformer.blocks)
    bf16_attn = snapshot_part(bf16_pipe.transformer, "attn")
    bf16_ffn = snapshot_part(bf16_pipe.transformer, "ffn")
    print(f"  snapshotted attn+ffn for {n_blocks} blocks", flush=True)
    bf16_acts = None
    if not args.skip_nmse:
        bf16_acts = collect_wan_acts(bf16_pipe, args.prompt, args.seed)
        print(f"  captured {len(bf16_acts)} tensors", flush=True)
    else:
        print("  skip BF16 first-step acts (--skip-nmse)", flush=True)
    del bf16_pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("[2] load W4A4; snapshot quant attn/ffn for restore...", flush=True)
    quant_pipe = load_wan_quant(args.ckpt)
    quant_attn = snapshot_part(quant_pipe.transformer, "attn")
    quant_ffn = snapshot_part(quant_pipe.transformer, "ffn")

    configs: list[tuple[str, dict, dict]] = []
    if args.only in ("both", "bf16_attn_quant_ffn"):
        configs.append(("bf16_attn_quant_ffn", bf16_attn, quant_attn))
    if args.only in ("both", "quant_attn_bf16_ffn"):
        configs.append(("quant_attn_bf16_ffn", bf16_ffn, quant_ffn))

    all_reports: dict[str, dict] = {}
    curve_paths: dict[str, Path] = {}
    for tag, bf16_snaps, quant_snaps in configs:
        rep = run_one_config(
            tag=tag,
            quant_pipe=quant_pipe,
            bf16_snaps=bf16_snaps,
            quant_snaps_backup=quant_snaps,
            bf16_acts=bf16_acts,
            out_dir=args.out_dir,
            prompt=args.prompt,
            seed=args.seed,
            skip_gif=args.skip_gif,
            skip_nmse=args.skip_nmse,
        )
        all_reports[tag] = rep
        if rep.get("nmse_json"):
            curve_paths[tag] = Path(rep["nmse_json"])

    overlay = None
    if curve_paths:
        overlay = args.out_dir / "overlay_attn_vs_ffn_vs_full_w4a4.png"
        plot_multi_overlay(
            curve_paths,
            BASELINE_JSON if BASELINE_JSON.is_file() else None,
            overlay,
            "Wan first-step NMSE: BF16-attn vs BF16-ffn vs full W4A4",
        )

    summary = {
        "ckpt": str(args.ckpt),
        "prompt": args.prompt,
        "seed": args.seed,
        "n_blocks": n_blocks,
        "configs": all_reports,
        "overlay": str(overlay) if overlay else None,
        "note": (
            "bf16_attn_quant_ffn = all attn1+attn2 BF16, FFN quantized; "
            "quant_attn_bf16_ffn = all FFN BF16, attn quantized."
        ),
    }
    report_name = f"report_seed{args.seed}.json" if args.seed != SEED else "report.json"
    (args.out_dir / report_name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(
        {
            k: {
                "peak": v.get("nmse", {}).get("peak_nmse_pct"),
                "final": v.get("nmse", {}).get("final_nmse_pct"),
                "delta_peak": v.get("delta_peak_nmse_pct"),
                "delta_final": v.get("delta_final_nmse_pct"),
            }
            for k, v in all_reports.items()
        },
        indent=2,
    ), flush=True)

    del quant_pipe
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
