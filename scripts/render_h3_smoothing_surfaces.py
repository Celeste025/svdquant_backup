#!/usr/bin/env python3
"""Render matched absolute-activation 3D surfaces from compact H3 arrays."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/data1/models/svdquant-wjq/results/E087_h3_patterns")

for source in sorted(OUT.glob("h3_linear_pattern_surface_*.npz")):
    data = np.load(source, allow_pickle=True)
    x, xs = np.abs(data["x"]), np.abs(data["xs"])
    # Render a readable 65×75 mesh; both panels retain exactly the same z-axis.
    x, xs = x[::4, ::4], xs[::4, ::4]
    yy, xx = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), indexing="ij")
    zmax = np.percentile(x, 99)
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    for idx, (surface, label) in enumerate(((x, "Activation |x| (original)"), (xs, "Activation |x/s| (smoothed)")), 1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.plot_surface(xx, yy, surface, cmap="coolwarm", linewidth=0, antialiased=False, rcount=65, ccount=75)
        ax.set_title(label)
        ax.set_xlabel("Input channel (downsampled)")
        ax.set_ylabel("Token (downsampled)")
        ax.set_zlabel("Absolute value")
        ax.set_zlim(0, zmax)
        ax.view_init(elev=28, azim=-62)
    fig.suptitle(f"MiniMax-H3 | prompt {int(data['prompt'])}, step {int(data['step'])}, block {int(data['block'])}, {str(data['suffix'])}\nshared z-axis: 0–{zmax:.3g} (original |x| p99)")
    fig.savefig(OUT / source.name.replace("_surface", "_activation_surface").replace(".npz", ".png"), dpi=170)
    plt.close(fig)
    # Shape-only view: each surface receives its own p99 z-range.
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    for idx, (surface, label) in enumerate(((x, "Activation |x| (original; own z-axis)"), (xs, "Activation |x/s| (smoothed; own z-axis)")), 1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        local_max = np.percentile(surface, 99)
        ax.plot_surface(xx, yy, surface, cmap="coolwarm", linewidth=0, antialiased=False, rcount=65, ccount=75)
        ax.set_title(label)
        ax.set_xlabel("Input channel (downsampled)")
        ax.set_ylabel("Token (downsampled)")
        ax.set_zlabel("Absolute value")
        ax.set_zlim(0, local_max)
        ax.view_init(elev=28, azim=-62)
    fig.suptitle(f"MiniMax-H3 | prompt {int(data['prompt'])}, step {int(data['step'])}, block {int(data['block'])}, {str(data['suffix'])}\neach panel uses its own |activation| p99 z-axis")
    fig.savefig(OUT / source.name.replace("_surface", "_activation_surface_autoscale").replace(".npz", ".png"), dpi=170)
    plt.close(fig)
    # A scale-free tail comparison makes shape changes inspectable.
    fig, ax = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    for values, label, color in ((x, "|x| / RMS(x)", "tab:red"), (xs, "|x/s| / RMS(x/s)", "tab:blue")):
        z = values.ravel() / np.sqrt(np.mean(values**2))
        grid = np.geomspace(1e-3, max(1.0, np.percentile(z, 99.9)), 250)
        ax.semilogy(grid, [(z > q).mean() for q in grid], label=label, color=color)
    ax.set_title("Normalized absolute-activation tail shape")
    ax.set_xlabel("absolute activation / its own RMS")
    ax.set_ylabel("fraction greater than threshold")
    ax.grid(True, alpha=.25); ax.legend()
    fig.savefig(OUT / source.name.replace("_surface", "_activation_shape").replace(".npz", ".png"), dpi=170)
    plt.close(fig)
