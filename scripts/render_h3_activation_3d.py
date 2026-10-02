#!/usr/bin/env python3
"""Render MiniMax-H3 DiT activation outliers as 3D surfaces.

Reads the ``.npz`` payloads written by ``survey_h3_activation_outliers.py`` and
draws the activation value as height over a (token x channel) base plane, so a
handful of oversized channels stand out as spikes.  A ``symlog`` variant
(height = sign(v) * log10(1 + |v|)) keeps the spike structure readable when the
linear view is dominated by a single column.

Run with a numpy+matplotlib interpreter (no torch needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SUFFIXES = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
TRANSFORMS = {
    "linear": (lambda values: values, "value"),
    "symlog": (lambda values: np.sign(values) * np.log10(1.0 + np.abs(values)), "sign(v)*log10(1+|v|)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 24, 49])
    parser.add_argument("--overview-steps", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--chanpeak-layers", nargs="+",
                        default=["blocks.0.attn.qkv_proj", "blocks.49.mlp.fc2"])
    parser.add_argument("--scales", nargs="+", choices=sorted(TRANSFORMS), default=["linear", "symlog"])
    parser.add_argument("--pool", type=int, default=2, help="peak-preserving downsample factor for surfaces")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


LAYOUT_COLORS = {"text": "#ffd0a3", "audio": "#bcd8ff", "video": "#d6ecd2", "pad": "#e8e8e8"}


def load_case(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.keys()}
    steps = sorted({int(key.split("/")[0][1:]) for key in arrays if key.startswith("s")})
    layout = json.loads(str(arrays["layout"])) if "layout" in arrays else None
    return {
        "case": str(arrays["case"]),
        "prompt": str(arrays["prompt"]),
        "steps": steps,
        "targets": [str(t) for t in arrays["targets"]],
        "layout": layout,
        "arrays": arrays,
    }


def modality_of(layout: dict | None, index: int) -> str:
    if not layout:
        return "?"
    for name, (start, stop) in layout.items():
        if start <= index < stop:
            return {"text": "T", "audio": "A", "video": "V", "pad": "P"}[name]
    return "?"


def shade_layout(ax, layout: dict | None) -> None:
    if not layout:
        return
    for name, (start, stop) in layout.items():
        ax.axvspan(start, stop, color=LAYOUT_COLORS[name], alpha=0.5, linewidth=0)


def peak_pool(grid: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by keeping the signed value with the largest magnitude per cell."""
    if factor <= 1:
        return grid
    rows = grid.shape[0] - grid.shape[0] % factor
    cols = grid.shape[1] - grid.shape[1] % factor
    reshaped = grid[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    flat = reshaped.transpose(0, 2, 1, 3).reshape(rows // factor, cols // factor, factor * factor)
    picks = np.take_along_axis(flat, np.abs(flat).argmax(axis=2, keepdims=True), axis=2)
    return picks[..., 0]


def draw_surface(ax, grid: np.ndarray, *, title: str, variant: str, zlim: float, pool: int,
                 bar_height: float | None = None) -> None:
    transform, _ = TRANSFORMS[variant]
    data = transform(peak_pool(grid, pool))
    rows, cols = data.shape
    y, x = np.mgrid[0:rows, 0:cols]
    norm = matplotlib.colors.Normalize(vmin=-zlim, vmax=zlim)
    ax.plot_surface(x, y, data, cmap="coolwarm", norm=norm, rcount=rows, ccount=cols,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_zlim(-zlim, zlim)
    if bar_height is not None:
        ax.set_box_aspect((1, 1, bar_height))
    ax.set_xlabel("channel", fontsize=6, labelpad=-6)
    ax.set_ylabel("token", fontsize=6, labelpad=-6)
    ax.tick_params(labelsize=5, pad=-2)
    ax.set_title(title, fontsize=7.5, pad=1)
    ax.view_init(elev=28, azim=-60)


def overview_figure(case: dict, step: int, blocks: list[int], pool: int, dpi: int, variant: str,
                    output: Path) -> None:
    transform, axis_label = TRANSFORMS[variant]
    fig = plt.figure(figsize=(4 * 4.2, 3 * 3.4), dpi=dpi)
    arrays = case["arrays"]
    for row, block in enumerate(blocks):
        for col, suffix in enumerate(SUFFIXES):
            layer = f"blocks.{block}.{suffix}"
            grid = arrays[f"s{step:02d}/{layer}/grid"]
            stats = arrays[f"s{step:02d}/{layer}/stats"]
            absmax = float(stats[0])
            ax = fig.add_subplot(len(blocks), len(SUFFIXES), row * len(SUFFIXES) + col + 1, projection="3d")
            draw_surface(ax, grid, variant=variant, zlim=float(transform(np.array([absmax]))[0]), pool=pool,
                         bar_height=0.45,
                         title=f"{layer}\nabsmax={absmax:.3g} peak/mean={float(stats[2]):.0f}x")
    fig.suptitle(f"{case['case']} - {case['prompt'][:70]}\ndenoise step {step}: linear-layer output, height = {axis_label} (bf16)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def chanpeak_figure(case: dict, layer: str, pool: int, dpi: int, variant: str, output: Path) -> None:
    steps = case["steps"]
    transform, axis_label = TRANSFORMS[variant]
    fig = plt.figure(figsize=(2.6 * len(steps), 3.6), dpi=dpi)
    arrays = case["arrays"]
    for col, step in enumerate(steps):
        grid = arrays[f"s{step:02d}/{layer}/grid_sorted"]
        stats = arrays[f"s{step:02d}/{layer}/stats"]
        top = arrays[f"s{step:02d}/{layer}/top_channels"]
        absmax = float(stats[0])
        ax = fig.add_subplot(1, len(steps), col + 1, projection="3d")
        draw_surface(ax, grid, variant=variant, zlim=float(transform(np.array([absmax]))[0]), pool=pool,
                     bar_height=0.5, title=f"step {step}\nabsmax={absmax:.3g}")
        if col == 0:
            ax.set_zlabel(axis_label, fontsize=6, labelpad=-4)
        if top.size:
            print(f"  {layer} step {step}: top peak channels {top[:3].tolist()}", flush=True)
    fig.suptitle(f"{case['case']} - {layer}\nchannels sorted by per-channel peak (desc), top {arrays[f's{steps[0]:02d}/{layer}/grid_sorted'].shape[1]}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _grid_of_axes(case: dict, blocks: list[int], figsize, dpi: int):
    layers = [f"blocks.{b}.{s}" for b in blocks for s in SUFFIXES]
    fig, axes = plt.subplots(len(blocks), len(SUFFIXES), figsize=figsize, dpi=dpi)
    return layers, fig, axes


def colpeak_figure(case: dict, step: int, blocks: list[int], dpi: int, output: Path) -> None:
    layers, fig, axes = _grid_of_axes(case, blocks, (4.4 * len(SUFFIXES), 2.4 * len(blocks)), dpi)
    arrays = case["arrays"]
    for index, layer in enumerate(layers):
        row, col = divmod(index, len(SUFFIXES))
        peak = arrays[f"s{step:02d}/{layer}/col_peak"]
        ax = axes[row][col]
        ax.plot(np.arange(peak.size), peak, linewidth=0.6, color="#1f4e79")
        ax.set_yscale("log")
        ax.set_title(f"{layer}\nmax={peak.max():.3g} median={np.median(peak):.3g}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25, linewidth=0.4)
        if row == len(blocks) - 1:
            ax.set_xlabel("channel index", fontsize=7)
        if col == 0:
            ax.set_ylabel("per-channel |peak| (log)", fontsize=7)
    fig.suptitle(f"{case['case']} - per-channel activation peak over all tokens, step {step}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def rowpeak_figure(case: dict, step: int, blocks: list[int], dpi: int, output: Path) -> None:
    layers, fig, axes = _grid_of_axes(case, blocks, (4.4 * len(SUFFIXES), 2.4 * len(blocks)), dpi)
    arrays = case["arrays"]
    for index, layer in enumerate(layers):
        row, col = divmod(index, len(SUFFIXES))
        peak = arrays[f"s{step:02d}/{layer}/row_peak"]
        stats = arrays[f"s{step:02d}/{layer}/stats"]
        ax = axes[row][col]
        shade_layout(ax, case.get("layout"))
        ax.plot(np.arange(peak.size), peak, linewidth=0.6, color="#7f3f00")
        ax.axhline(float(stats[0]), color="red", linewidth=0.7, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"{layer}\nabsmax={float(stats[0]):.3g} median row={np.median(peak):.3g}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25, linewidth=0.4)
        if row == len(blocks) - 1:
            ax.set_xlabel("packed token index", fontsize=7)
        if col == 0:
            ax.set_ylabel("per-token |peak| (log)", fontsize=7)
    fig.suptitle(f"{case['case']} - per-token activation peak over all channels, step {step} (dashed = tensor absmax)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def histogram_figure(case: dict, step: int, blocks: list[int], dpi: int, output: Path) -> None:
    layers, fig, axes = _grid_of_axes(case, blocks, (4.4 * len(SUFFIXES), 2.3 * len(blocks)), dpi)
    arrays = case["arrays"]
    edges = np.linspace(-8.0, 8.0, 49)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for index, layer in enumerate(layers):
        row, col = divmod(index, len(SUFFIXES))
        hist = arrays[f"s{step:02d}/{layer}/abs_hist"]
        stats = arrays[f"s{step:02d}/{layer}/stats"]
        ax = axes[row][col]
        ax.bar(centers, hist, width=0.3, color="#26456e")
        ax.set_yscale("log")
        ax.set_title(f"{layer}\nabsmax={float(stats[0]):.3g} mean|v|={float(stats[1]):.3g}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25, linewidth=0.4)
        if row == len(blocks) - 1:
            ax.set_xlabel("log10 |activation|", fontsize=7)
        if col == 0:
            ax.set_ylabel("count (log)", fontsize=7)
    fig.suptitle(f"{case['case']} - distribution of |activation|, step {step}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def spread_figure(case: dict, step: int, blocks: list[int], dpi: int, output: Path) -> None:
    """How many channels the worst tokens exceed the per-channel median in.

    Token-localized outliers that hit *many* channels cannot be removed by
    channel-wise smoothing; a handful of channels can.
    """
    layers = [f"blocks.{b}.{s}" for b in blocks for s in SUFFIXES]
    arrays = case["arrays"]
    layout = case.get("layout")
    fig, axes = plt.subplots(len(blocks), len(SUFFIXES), figsize=(4.4 * len(SUFFIXES), 2.5 * len(blocks)), dpi=dpi)
    for index, layer in enumerate(layers):
        row, col = divmod(index, len(SUFFIXES))
        stats = arrays[f"s{step:02d}/{layer}/spike_stats"]
        tokens = arrays[f"s{step:02d}/{layer}/spike_tokens"]
        n_channels = int(arrays[f"s{step:02d}/{layer}/stats"][5])
        ax = axes[row][col]
        colors = [{"T": "#e07b39", "A": "#2f6fd0", "V": "#3f8f4f", "P": "#888888"}[modality_of(layout, int(t))]
                  for t in tokens]
        ax.bar(np.arange(len(tokens)), stats[:, 3], color=colors, width=0.7)
        ax.bar(np.arange(len(tokens)), stats[:, 4], color="#111111", width=0.35)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(tokens)),
                      [f"{int(t)}\n{modality_of(layout, int(t))}" for t in tokens], fontsize=5)
        ax.set_title(f"{layer}\nmax ratio={stats[:, 2].max():.0f}x of {n_channels} channels", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25, linewidth=0.4)
        if row == len(blocks) - 1:
            ax.set_xlabel("spike token index (T=text A=audio V=video)", fontsize=7)
        if col == 0:
            ax.set_ylabel("channels > 10x / 100x\nper-channel median", fontsize=7)
    fig.suptitle(f"{case['case']} - channel spread of the top-8 spike tokens, step {step}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def spectrum_figure(case: dict, step: int, blocks: list[int], dpi: int, output: Path,
                    top_k: int = 64) -> None:
    """3D bars of the top channel peaks: exact, full-tensor statistics (no token sampling)."""
    layers = [f"blocks.{b}.{s}" for b in blocks for s in SUFFIXES]
    arrays = case["arrays"]
    fig = plt.figure(figsize=(4 * 4.4, 3 * 3.6), dpi=dpi)
    for index, layer in enumerate(layers):
        row, col = divmod(index, len(SUFFIXES))
        peak = np.sort(arrays[f"s{step:02d}/{layer}/col_peak"])[::-1][:top_k]
        ax = fig.add_subplot(len(blocks), len(SUFFIXES), index + 1, projection="3d")
        x = np.arange(peak.size)
        ax.bar3d(x, np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float),
                 np.full(x.size, 0.8), np.full(x.size, 0.8), peak,
                 color="#b6492b", shade=True, edgecolor="none")
        ax.set_zlim(0, float(peak[0]) * 1.05)
        ax.set_xlabel("channel rank", fontsize=6, labelpad=-6)
        ax.set_ylabel("", fontsize=6, labelpad=-8)
        ax.set_yticks([])
        ax.tick_params(labelsize=5, pad=-2)
        median = float(np.median(arrays[f"s{step:02d}/{layer}/col_peak"]))
        ax.set_title(f"{layer}\ntop peak={peak[0]:.3g} median channel={median:.3g} ({peak[0] / median:.0f}x)",
                     fontsize=7.5, pad=1)
        ax.view_init(elev=24, azim=-62)
    fig.suptitle(f"{case['case']} - per-channel activation peak, top {top_k} of all channels, step {step}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def stats_heatmap(case: dict, blocks: list[int], dpi: int, output: Path) -> None:
    layers = [f"blocks.{b}.{s}" for b in blocks for s in SUFFIXES]
    steps = case["steps"]
    arrays = case["arrays"]
    ratio = np.array([[float(arrays[f"s{step:02d}/{layer}/stats"][2]) for step in steps] for layer in layers])
    fig, ax = plt.subplots(figsize=(0.75 * len(steps) + 5.5, 0.42 * len(layers) + 2.2), dpi=dpi)
    image = ax.imshow(np.log10(ratio), cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(steps)), [f"{s}" for s in steps])
    ax.set_yticks(np.arange(len(layers)), layers, fontsize=7)
    ax.set_xlabel("denoise step")
    ax.set_title(f"{case['case']} - absmax / mean|activation| (annotation = ratio)", fontsize=10)
    for i in range(len(layers)):
        for j in range(len(steps)):
            ax.text(j, i, f"{ratio[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if np.log10(ratio[i, j]) < np.log10(ratio).max() - 1.2 else "black")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="log10 ratio")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("*.npz"))
    if args.cases:
        wanted = set(args.cases)
        paths = [path for path in paths if path.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no .npz payloads in {args.input_dir}")
    print(f"cases: {[path.stem for path in paths]}", flush=True)

    for path in paths:
        case = load_case(path)
        first_step = case["steps"][0]
        for variant in args.scales:
            for step in args.overview_steps:
                if step not in case["steps"]:
                    continue
                target = args.output_dir / f"overview_{variant}_{case['case']}_s{step:02d}.png"
                overview_figure(case, step, args.blocks, args.pool, args.dpi, variant, target)
                print(f"wrote {target.name}", flush=True)
            for layer in args.chanpeak_layers:
                target = args.output_dir / f"chanpeak_{variant}_{case['case']}_{layer}.png"
                chanpeak_figure(case, layer, args.pool, args.dpi, variant, target)
                print(f"wrote {target.name}", flush=True)
        for name, function in (
            ("spectrum", spectrum_figure),
            ("spread", spread_figure),
            ("colpeak", colpeak_figure),
            ("rowpeak", rowpeak_figure),
            ("hist", histogram_figure),
        ):
            target = args.output_dir / f"{name}_{case['case']}_s{first_step:02d}.png"
            function(case, first_step, args.blocks, args.dpi, target)
            print(f"wrote {target.name}", flush=True)
        target = args.output_dir / f"stats_{case['case']}.png"
        stats_heatmap(case, args.blocks, args.dpi, target)
        print(f"wrote {target.name}", flush=True)
    print("ALL FIGURES DONE", flush=True)


if __name__ == "__main__":
    main()
