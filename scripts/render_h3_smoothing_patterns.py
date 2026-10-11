#!/usr/bin/env python3
"""Render smoothing factors and matched x versus x/s activation patterns."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
OUT = Path("/data1/models/svdquant-wjq/results/E087_h3_patterns")


def heatmap(x: torch.Tensor, lim: float) -> Image.Image:
    z = (x.float().clamp(-lim, lim) / lim).numpy()
    positive, negative = z.clip(min=0), (-z).clip(min=0)
    rgb = torch.stack((torch.from_numpy(1-negative), torch.from_numpy(1-positive-negative*.35),
                       torch.from_numpy(1-positive)), dim=-1).clamp(0, 1).mul(255).byte().numpy()
    return Image.fromarray(rgb, mode="RGB")


def qlimit(x: torch.Tensor) -> float:
    return float(torch.quantile(x.float().abs().flatten(), .99).clamp_min(1e-12))


def factor_line(draw: ImageDraw.ImageDraw, smooth: torch.Tensor, left: int, top: int, width: int, height: int, font) -> None:
    s = smooth.float().cpu()
    bins = min(width, s.numel())
    edges = torch.linspace(0, s.numel(), bins + 1).round().long()
    values = torch.stack([s[edges[i]:edges[i+1]].median() for i in range(bins)])
    lo, hi = 0.0, float(s.max())
    hi *= 1.05
    draw.rectangle((left, top, left+width, top+height), outline="black", width=1)
    y_one = top + height * (hi - 1.0) / (hi - lo)
    draw.line((left, y_one, left+width, y_one), fill=(190, 40, 40), width=1)
    points=[]
    for i, v in enumerate(values.tolist()):
        x = left + i * width / max(1, bins-1)
        y = top + height * (hi - v) / (hi - lo)
        points.append((x,y))
    draw.line(points, fill=(30, 90, 180), width=2)
    for val in (s.min().item(), s.median().item(), s.max().item()):
        y = top + height * (hi - val) / (hi - lo)
        draw.text((left+4, y-12), f"{val:.3g}", fill="black", font=font)
    draw.text((left, top+height+8), "input-feature coordinate; blue = binned median s, red = s=1; linear y-axis", fill="black", font=font)


def color(value: float, limit: float) -> tuple[int, int, int]:
    z = max(-1.0, min(1.0, value / limit))
    if z >= 0:
        return (255, int(255 * (1-z)), int(255 * (1-z)))
    return (int(255 * (1+z)), int(255 * (1+z*.35)), 255)


def scatter3d(draw: ImageDraw.ImageDraw, x: torch.Tensor, left: int, top: int, limit: float, font) -> None:
    """Simple fixed-view 3D projection: token × feature × activation value."""
    rows, cols = 55, 70
    rr = torch.linspace(0, x.shape[0]-1, rows).round().long()
    cc = torch.linspace(0, x.shape[1]-1, cols).round().long()
    z = x.index_select(0,rr).index_select(1,cc)
    ox, oy = left+75, top+275
    # Ground-plane axes, followed by a point cloud with the same z scale in both panels.
    draw.line((ox,oy,ox+270,oy),fill="black",width=1)
    draw.line((ox,oy,ox+65,oy-65),fill="black",width=1)
    draw.line((ox,oy,ox,oy-165),fill="black",width=1)
    for r in range(rows):
        for c in range(cols):
            v=float(z[r,c]); px=ox+c*270/(cols-1)+r*65/(rows-1); py=oy-r*65/(rows-1)-v/limit*150
            draw.ellipse((px-1,py-1,px+1,py+1),fill=color(v,limit))
    draw.text((left,top+292),"feature → ; token ↗ ; activation ↑ (same z-axis scale)",fill="black",font=font)


def main() -> None:
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    for path in sorted(OUT.glob("h3_linear_pattern_data_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        prompt, step, block, suffix = data["spec"]
        s = state["layers"][f"blocks.{block}.{suffix}"]["smooth"].float()
        x, xs = data["panels"]["input"], data["panels"]["smoothed_input"]
        figure = Image.new("RGB", (1420, 490), "white"); draw = ImageDraw.Draw(figure)
        draw.text((20,16), f"MiniMax-H3 | prompt {prompt}, step {step}, block {block}, {suffix}", fill="black", font=title_font)
        draw.text((20,46), f"smoothing s: min {s.min():.3g}, median {s.median():.3g}, p99 {torch.quantile(s,.99):.3g}, max {s.max():.3g}", fill="black", font=font)
        draw.text((20,80), "Per-channel smoothing factor s (linear scale)", fill="black", font=font)
        factor_line(draw, s, 20, 110, 420, 270, small)
        common_lim = max(qlimit(x), qlimit(xs))
        for tensor, label, left in ((x, "BF16 input x (area-averaged)", 500), (xs, "smoothed input x/s (area-averaged; shared scale)", 960)):
            lim = common_lim; draw.text((left,80), label, fill="black", font=font)
            figure.paste(heatmap(tensor,lim).resize((420,300),Image.Resampling.NEAREST),(left,110))
            draw.text((left,420), "token × input-feature coordinate", fill="black", font=small)
            draw.text((left,438), f"symmetric range ±{lim:.3g} (p99)", fill="black", font=small)
        figure.save(OUT / path.name.replace("_data", "_smoothing").replace(".pt", ".png"))
        cloud = Image.new("RGB", (980, 420), "white"); cd = ImageDraw.Draw(cloud)
        cd.text((20,16), f"3D activation point cloud | prompt {prompt}, step {step}, block {block}, {suffix}", fill="black", font=title_font)
        cd.text((20,44), f"Shared z-axis: ±{common_lim:.3g} (input p99); each dot is an area-averaged token × feature cell", fill="black", font=font)
        cd.text((40,82), "before smoothing: x", fill="black", font=font)
        cd.text((510,82), "after smoothing: x/s", fill="black", font=font)
        scatter3d(cd,x,20,105,common_lim,small); scatter3d(cd,xs,490,105,common_lim,small)
        cloud.save(OUT / path.name.replace("_data", "_activation_3d").replace(".pt", ".png"))


if __name__ == "__main__":
    main()
