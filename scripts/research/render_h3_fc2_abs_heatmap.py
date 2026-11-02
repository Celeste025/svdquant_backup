#!/usr/bin/env python3
"""2D absolute-activation counterpart of the E090 FC2 3D overview."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch


SAMPLES = {
    1: Path("/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p1_s00.pt"),
    11: Path("/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p11_s00_calib.pt"),
}
OUTPUT = Path("results/research/E090/figures")


def load_pool(prompt_id: int, path: Path) -> dict:
    sample = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if sample.get("format") != "h3-native-bench-sample-v1" or sample.get("target_block") != 25:
        raise ValueError(f"Unexpected sample format: {path}")
    x = sample["linear_inputs"]["mlp.fc2"]
    if x.ndim != 2 or x.shape[1] != 14336 or x.shape[0] % 64:
        raise ValueError(f"Unexpected FC2 input shape: {tuple(x.shape)}")
    # Exactly the same 64 x 64 max-absolute pooling as the 3D surface.
    pooled = x.abs().reshape(64, x.shape[0] // 64, 64, x.shape[1] // 64)
    pooled = pooled.amax(dim=(1, 3)).float().numpy()
    peak = float(pooled.max())
    if prompt_id == 1 and peak != 5152 or prompt_id == 11 and peak != 2512:
        raise ValueError(f"Peak differs from the original 3D chart: p{prompt_id} {peak}")
    return {"prompt_id": prompt_id, "path": str(path), "shape": list(x.shape),
            "pooled": pooled, "peak": peak, "bin_values": [x.shape[0] // 64, x.shape[1] // 64]}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = [load_pool(pid, path) for pid, path in SAMPLES.items()]
    upper = float(max(np.quantile(item["pooled"], 0.99) for item in data))
    full_max = max(item["peak"] for item in data)
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.6), dpi=160)
    fig.suptitle("H3 block 25 FC2 input | step 0 | absolute activation",
                 fontsize=17, weight="semibold")
    for row, (vmax, title) in enumerate(((full_max, "Full range"),
                                          (upper, "Detail · top 1% saturated"))):
        im = None
        for col, item in enumerate(data):
            ax = axes[row, col]
            im = ax.imshow(item["pooled"], origin="upper", interpolation="nearest",
                           extent=(0, item["shape"][1], item["shape"][0], 0),
                           aspect="auto", cmap="viridis", norm=Normalize(0, vmax))
            ax.set_title(f"p{item['prompt_id']} · peak {item['peak']:,.0f}", fontsize=12)
            ax.set_xlabel("Input channel (K)")
            ax.set_ylabel("Token row")
            ax.tick_params(labelsize=9)
        cax = fig.add_axes((0.87, 0.57 if row == 0 else 0.13, 0.014, 0.30))
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Max |activation| per bin")
        fig.text(0.87, 0.88 if row == 0 else 0.44, title, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.075, right=0.83, bottom=0.105, top=0.92,
                        hspace=0.34, wspace=0.22)
    fig.text(0.5, 0.025,
             "Each pixel is the exact max |x| over one token-row × channel bin: "
             "p1 350×224 values, p11 351×224.\nBoth prompts share the same linear "
             "color range in each row; the lower row clips values above the shared p99.",
             ha="center", va="bottom", fontsize=9)
    output = OUTPUT / "h3_fc2_p1_p11_activation_abs_heatmap.png"
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    meta = {"format": "h3-fc2-activation-abs-heatmap-v1", "output": str(output),
            "pooling": "exact max(abs(x)) into 64x64 bins, same as 3D overview",
            "full_scale_max": full_max, "shared_detail_cap_p99": upper,
            "panels": [{k: v for k, v in item.items() if k != "pooled"} for item in data]}
    (OUTPUT / "h3_fc2_p1_p11_activation_abs_heatmap.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
