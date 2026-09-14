#!/usr/bin/env python3
"""Replot saved Wan/Flux calib local NMSE with shared y-axes (Wan top / Flux bottom).

Sources (already collected):
  - compare/calib_local_block_nmse/{wan,flux}/*_calib_local_nmse.json
      → per-block modules: attn1/attn2/attn, ffn_up, ffn_down
  - compare/calib_qkvo_ffn_nmse/{wan,flux}/*_qkvo_ffn_nmse.json
      → q/k/v/o projs + ffn_compose (up+act+down as one)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path("/ssd/2/wenjinqi.wjq")
LOCAL_DIR = DATA_ROOT / "compare" / "calib_local_block_nmse"
QKVO_DIR = DATA_ROOT / "compare" / "calib_qkvo_ffn_nmse"
OUT_DEFAULT = DATA_ROOT / "compare" / "calib_nmse_shared_ylim"

COLORS = {
    "attn1": "#1f4e79",
    "attn2": "#2a9d8f",
    "attn": "#5c4d7a",
    "ffn_down": "#b85c38",
    "ffn_up": "#e9a46c",
    "ffn_compose": "#8b3a2a",
    "ffn_compose_ctx": "#c47a5a",
    "q": "#1f4e79",
    "k": "#2a9d8f",
    "v": "#b85c38",
    "o": "#5c4d7a",
}
LABELS = {
    "attn1": "attn1 (Wan self / Flux dual)",
    "attn2": "attn2 (Wan cross)",
    "attn": "attn (Flux single)",
    "ffn_down": "ffn_down",
    "ffn_up": "ffn_up",
    "ffn_compose": "ffn_compose (up+act+down)",
    "ffn_compose_ctx": "ffn_compose_ctx (Flux dual)",
    "q": "q",
    "k": "k",
    "v": "v",
    "o": "o",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def blocks_from_local(res: dict) -> list[str]:
    # preserve discovery order from modules
    seen: list[str] = []
    for m in res["modules"]:
        bid = m["block_id"]
        if bid not in seen:
            seen.append(bid)
    return seen


def kind_map_local(res: dict) -> dict[str, dict[str, float]]:
    """block_id -> kind -> nmse."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for m in res["modules"]:
        out[m["block_id"]][m["kind"]] = float(m["nmse"])
    return out


def kind_series_qkvo(res: dict, kind: str) -> list[tuple[str, float]]:
    rows = [m for m in res["modules"] if m.get("kind_canon", m["kind"]) == kind]
    # keep file order
    return [(m["block_id"] + "|" + m["name"], float(m["nmse"])) for m in rows]


def attn_compose_per_block(local: dict, qkvo: dict) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Merge whole-attn (local) + ffn_compose (qkvo) keyed by block_id."""
    blocks = blocks_from_local(local)
    loc = kind_map_local(local)
    ffn_img: dict[str, float] = {}
    ffn_ctx: dict[str, float] = {}
    for m in qkvo["modules"]:
        if m.get("kind_canon", m["kind"]) != "ffn_compose":
            continue
        bid = m["block_id"]
        name = m["name"]
        if "ff_context" in name or name.endswith(".ff_context"):
            ffn_ctx[bid] = float(m["nmse"])
        else:
            ffn_img[bid] = float(m["nmse"])
    merged: dict[str, dict[str, float]] = {}
    for bid in blocks:
        d = dict(loc.get(bid, {}))
        # drop per-proj FFN kinds so bars focus on composed modules
        d.pop("ffn_up", None)
        d.pop("ffn_down", None)
        if bid in ffn_img:
            d["ffn_compose"] = ffn_img[bid]
        if bid in ffn_ctx:
            d["ffn_compose_ctx"] = ffn_ctx[bid]
        merged[bid] = d
    return blocks, merged


def shared_ylim(*vals: float, pad: float = 0.08) -> tuple[float, float]:
    mx = max(vals) if vals else 1.0
    return 0.0, mx * (1.0 + pad) * 100.0  # return in percent space if vals are fractions


def _bar_group(ax, blocks: list[str], data: dict[str, dict[str, float]], kinds: list[str], ylim: tuple[float, float]):
    n = len(blocks)
    k = len(kinds)
    x = np.arange(n, dtype=float)
    width = min(0.8 / max(k, 1), 0.22)
    offsets = (np.arange(k) - (k - 1) / 2.0) * width
    for oi, kind in enumerate(kinds):
        ys = []
        for bid in blocks:
            v = data.get(bid, {}).get(kind)
            ys.append(100.0 * v if v is not None else np.nan)
        ax.bar(
            x + offsets[oi],
            ys,
            width=width * 0.95,
            label=LABELS.get(kind, kind),
            color=COLORS.get(kind, "#666"),
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("NMSE (%)")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)


def plot_all_modules_shared(wan: dict, flux: dict, out_png: Path) -> None:
    wan_blocks = blocks_from_local(wan)
    flux_blocks = blocks_from_local(flux)
    wan_data = kind_map_local(wan)
    flux_data = kind_map_local(flux)
    ymax = 100.0 * max(
        max(m["nmse"] for m in wan["modules"]),
        max(m["nmse"] for m in flux["modules"]),
    )
    ylim = (0.0, ymax * 1.08)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), dpi=150, sharey=True)
    _bar_group(axes[0], wan_blocks, wan_data, ["attn1", "attn2", "ffn_down", "ffn_up"], ylim)
    axes[0].set_title(f"Wan ungated s16 — per-module calib local NMSE (n={wan['n_modules']})")

    # Flux: attn1 (dual) + attn (single) + ffn
    _bar_group(axes[1], flux_blocks, flux_data, ["attn1", "attn", "ffn_down", "ffn_up"], ylim)
    axes[1].set_title(
        f"Flux s16 — per-module calib local NMSE (n={flux['n_modules']}; dual attn1 + single attn)"
    )
    axes[1].set_xlabel("block")

    fig.suptitle(
        "Teacher-forced calib local NMSE — individual modules (shared y-axis)\n"
        "BF16 inputs → BF16 vs W4A4+LoRA · Wan 64/32 · Flux 16/-1",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def plot_attn_ffn_compose_shared(wan_l: dict, flux_l: dict, wan_q: dict, flux_q: dict, out_png: Path) -> None:
    wan_blocks, wan_data = attn_compose_per_block(wan_l, wan_q)
    flux_blocks, flux_data = attn_compose_per_block(flux_l, flux_q)

    vals = []
    for data in (wan_data, flux_data):
        for d in data.values():
            for kind in ("attn1", "attn2", "attn", "ffn_compose", "ffn_compose_ctx"):
                if kind in d:
                    vals.append(d[kind])
    ylim = (0.0, 100.0 * max(vals) * 1.08)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), dpi=150, sharey=True)
    _bar_group(axes[0], wan_blocks, wan_data, ["attn1", "attn2", "ffn_compose"], ylim)
    axes[0].set_title("Wan — attention (whole) + FFN compose (up+act+down)")

    _bar_group(
        axes[1],
        flux_blocks,
        flux_data,
        ["attn1", "attn", "ffn_compose", "ffn_compose_ctx"],
        ylim,
    )
    axes[1].set_title("Flux — attention (whole) + FFN compose (img + dual ctx)")
    axes[1].set_xlabel("block")

    fig.suptitle(
        "Teacher-forced calib local NMSE — composed modules (shared y-axis)\n"
        "attn from local-block hooks · ffn_compose from QKVO/FFN probe",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def plot_qkvo_shared(wan_q: dict, flux_q: dict, out_png: Path) -> None:
    """Q/K/V/O + ffn_compose as Wan/Flux stacked rows with one shared ylim."""
    kinds = ["q", "k", "v", "o", "ffn_compose"]
    series = {}
    all_vals = []
    for tag, res in (("wan", wan_q), ("flux", flux_q)):
        series[tag] = {}
        for kind in kinds:
            ys = [100.0 * v for _, v in kind_series_qkvo(res, kind)]
            series[tag][kind] = ys
            all_vals.extend(ys)
    # Flux also has add_* / ffn_compose_ctx — skip for main compare
    ylim = (0.0, max(all_vals) * 1.08 if all_vals else 1.0)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.8), dpi=150, sharey=True)
    for col, kind in enumerate(kinds):
        for row, tag, color in ((0, "wan", "#1f4e79"), (1, "flux", "#b85c38")):
            ax = axes[row, col]
            ys = series[tag][kind]
            ax.plot(range(len(ys)), ys, "-o", ms=2.5, lw=1.0, color=color)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.35)
            if row == 0:
                ax.set_title(f"{kind}\n(n={len(ys)})", fontsize=10)
            else:
                ax.set_xlabel("module index")
            if col == 0:
                ax.set_ylabel(f"{tag.upper()}  NMSE (%)")
            else:
                ax.set_ylabel("")
            ax.text(
                0.02,
                0.95,
                f"mean={np.mean(ys):.2f}%" if ys else "n=0",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                color=color,
            )
    fig.suptitle(
        "Calib local NMSE — Q/K/V/O + FFN compose (Wan top / Flux bottom, shared y)\n"
        "teacher-forced · W4A4 · in-Attention hooks for QKVO",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def plot_qkvo_overlay_shared(wan_q: dict, flux_q: dict, out_png: Path) -> None:
    """Same 2×3 overlay style as before, but all panels share one y-limit."""
    kinds = ["q", "k", "v", "o", "ffn_compose"]
    all_vals = []
    data = {}
    for kind in kinds:
        w = [100.0 * v for _, v in kind_series_qkvo(wan_q, kind)]
        f = [100.0 * v for _, v in kind_series_qkvo(flux_q, kind)]
        data[kind] = (w, f)
        all_vals.extend(w)
        all_vals.extend(f)
    ylim = (0.0, max(all_vals) * 1.08)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=150, sharey=True)
    axes_flat = list(axes.ravel())
    for i, kind in enumerate(kinds):
        ax = axes_flat[i]
        w, f = data[kind]
        ax.plot(range(len(w)), w, "-o", ms=2.5, lw=1.0, color="#1f4e79", label=f"Wan n={len(w)}")
        ax.plot(range(len(f)), f, "-o", ms=2.5, lw=1.0, color="#b85c38", label=f"Flux n={len(f)}")
        ax.set_title(kind)
        ax.set_xlabel("module index (model order)")
        ax.set_ylabel("NMSE (%)")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7)

    # mean bar
    ax = axes_flat[5]
    means_w = [np.mean(data[k][0]) for k in kinds]
    means_f = [np.mean(data[k][1]) for k in kinds]
    x = np.arange(len(kinds))
    ax.bar(x - 0.18, means_w, 0.36, color="#1f4e79", label="Wan")
    ax.bar(x + 0.18, means_f, 0.36, color="#b85c38", label="Flux")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("mean NMSE (%)")
    ax.set_title("mean NMSE by kind")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=7)

    fig.suptitle(
        "Calib local NMSE — Q/K/V/O & composed FFN (shared y across panels)\n"
        "teacher-forced, W4A4",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def _collect_vals(modules: list[dict], pred) -> list[float]:
    return [float(m["nmse"]) for m in modules if pred(m)]


def mean_nmse_table(wan_l: dict, flux_l: dict, wan_q: dict, flux_q: dict) -> dict:
    """Per-kind mean NMSE for single + composed modules (comparable categories)."""

    def local_kind(res: dict, kind: str) -> list[float]:
        return _collect_vals(res["modules"], lambda m: m["kind"] == kind)

    def local_attn(res: dict) -> list[float]:
        return _collect_vals(res["modules"], lambda m: m["kind"] in ("attn1", "attn2", "attn"))

    def qkvo_kind(res: dict, kind: str) -> list[float]:
        # use kind_canon; for ffn_compose include both img and dual-ctx (same kind)
        return _collect_vals(res["modules"], lambda m: m.get("kind_canon", m["kind"]) == kind)

    cats = [
        ("q", "single", "qkvo", "q"),
        ("k", "single", "qkvo", "k"),
        ("v", "single", "qkvo", "v"),
        ("o", "single", "qkvo", "o"),
        ("ffn_up", "single", "local", "ffn_up"),
        ("ffn_down", "single", "local", "ffn_down"),
        ("attn", "whole", "local_attn", None),
        ("ffn_compose", "whole", "qkvo", "ffn_compose"),
    ]
    out: dict = {"categories": [], "wan": {}, "flux": {}}
    for name, group, src, key in cats:
        if src == "local":
            wv, fv = local_kind(wan_l, key), local_kind(flux_l, key)
        elif src == "local_attn":
            wv, fv = local_attn(wan_l), local_attn(flux_l)
        else:
            wv, fv = qkvo_kind(wan_q, key), qkvo_kind(flux_q, key)
        out["categories"].append({"name": name, "group": group})
        out["wan"][name] = {
            "n": len(wv),
            "nmse_mean": float(np.mean(wv)) if wv else float("nan"),
            "nmse_mean_pct": float(100.0 * np.mean(wv)) if wv else float("nan"),
        }
        out["flux"][name] = {
            "n": len(fv),
            "nmse_mean": float(np.mean(fv)) if fv else float("nan"),
            "nmse_mean_pct": float(100.0 * np.mean(fv)) if fv else float("nan"),
        }
    return out


def plot_mean_by_kind(wan_l: dict, flux_l: dict, wan_q: dict, flux_q: dict, out_png: Path, out_json: Path) -> None:
    tab = mean_nmse_table(wan_l, flux_l, wan_q, flux_q)
    out_json.write_text(json.dumps(tab, indent=2))
    print(f"[json] {out_json}", flush=True)

    names = [c["name"] for c in tab["categories"]]
    groups = [c["group"] for c in tab["categories"]]
    wan_y = [tab["wan"][n]["nmse_mean_pct"] for n in names]
    flux_y = [tab["flux"][n]["nmse_mean_pct"] for n in names]
    wan_n = [tab["wan"][n]["n"] for n in names]
    flux_n = [tab["flux"][n]["n"] for n in names]

    x = np.arange(len(names), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=160)
    b1 = ax.bar(x - width / 2, wan_y, width, color="#1f4e79", label="Wan", edgecolor="none")
    b2 = ax.bar(x + width / 2, flux_y, width, color="#b85c38", label="Flux", edgecolor="none")

    # group separator between single | whole
    split_at = next(i for i, g in enumerate(groups) if g == "whole")
    ax.axvline(split_at - 0.5, color="#888", ls="--", lw=1.0, alpha=0.7)
    ymax = max(wan_y + flux_y) * 1.22
    ax.set_ylim(0.0, ymax)
    ax.text((split_at - 1) / 2.0, ymax * 0.97, "single modules", ha="center", va="top", fontsize=9, color="#444")
    ax.text(
        (split_at + len(names) - 1) / 2.0,
        ymax * 0.97,
        "whole modules",
        ha="center",
        va="top",
        fontsize=9,
        color="#444",
    )

    def _annotate(bars, ns):
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            if not np.isfinite(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.04,
                f"{h:.2f}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333",
            )

    _annotate(b1, wan_n)
    _annotate(b2, flux_n)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "q",
            "k",
            "v",
            "o",
            "ffn_up",
            "ffn_down",
            "attn\n(whole)",
            "ffn_compose\n(whole)",
        ]
    )
    ax.set_ylabel("mean NMSE (%)")
    ax.set_title(
        "Wan vs Flux — mean calib local NMSE by module kind\n"
        "single: q/k/v/o (QKVO hooks) + ffn_up/down · whole: attn + ffn_compose"
    )
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)

    # console summary
    print("\n=== mean NMSE (%) ===", flush=True)
    print(f"{'kind':14s} {'Wan':>10s} {'Flux':>10s} {'Δ(W-F)':>10s}", flush=True)
    for n in names:
        w, f = tab["wan"][n]["nmse_mean_pct"], tab["flux"][n]["nmse_mean_pct"]
        print(f"{n:14s} {w:10.3f} {f:10.3f} {w-f:+10.3f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--means-only", action="store_true", help="only regenerate mean bar chart")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wan_l = load_json(LOCAL_DIR / "wan" / "wan_calib_local_nmse.json")
    flux_l = load_json(LOCAL_DIR / "flux" / "flux_calib_local_nmse.json")
    wan_q = load_json(QKVO_DIR / "wan" / "wan_qkvo_ffn_nmse.json")
    flux_q = load_json(QKVO_DIR / "flux" / "flux_qkvo_ffn_nmse.json")

    plot_mean_by_kind(
        wan_l,
        flux_l,
        wan_q,
        flux_q,
        args.out_dir / "mean_nmse_by_kind_wan_vs_flux.png",
        args.out_dir / "mean_nmse_by_kind_wan_vs_flux.json",
    )

    if args.means_only:
        print(f"[done] {args.out_dir}", flush=True)
        return 0

    plot_all_modules_shared(wan_l, flux_l, args.out_dir / "all_modules_shared_ylim.png")
    plot_attn_ffn_compose_shared(
        wan_l, flux_l, wan_q, flux_q, args.out_dir / "attn_and_ffn_compose_shared_ylim.png"
    )
    plot_qkvo_shared(wan_q, flux_q, args.out_dir / "qkvo_ffn_wan_flux_rows_shared_ylim.png")
    plot_qkvo_overlay_shared(wan_q, flux_q, args.out_dir / "qkvo_ffn_overlay_shared_ylim.png")

    # also refresh into original folders for convenience
    plot_all_modules_shared(wan_l, flux_l, LOCAL_DIR / "all_modules_calib_local_nmse_shared_ylim.png")
    plot_qkvo_overlay_shared(wan_q, flux_q, QKVO_DIR / "wan_vs_flux_qkvo_ffn_nmse_shared_ylim.png")
    print(f"[done] {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
