#!/usr/bin/env python3
"""Render captured E087 H3 linear tensors as compact scientific heatmaps."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/data1/models/svdquant-wjq/results/E087_h3_patterns")


def downsample(x: torch.Tensor, rows: int = 260, cols: int = 300) -> torch.Tensor:
    x = x.float()
    return F.interpolate(x[None, None], size=(min(rows, x.shape[0]), min(cols, x.shape[1])), mode="area")[0, 0]


def limit(x: torch.Tensor) -> float:
    return float(torch.quantile(x.abs().flatten(), 0.99).clamp_min(1e-12))


def heatmap(x: torch.Tensor, lim: float) -> Image.Image:
    """A diverging blue-white-red map without relying on matplotlib."""
    z = (x.clamp(-lim, lim) / lim).numpy()
    positive, negative = z.clip(min=0), (-z).clip(min=0)
    rgb = torch.stack((
        torch.from_numpy(1.0 - negative),
        torch.from_numpy(1.0 - positive - negative * 0.35),
        torch.from_numpy(1.0 - positive),
    ), dim=-1).clamp(0, 1).mul(255).byte().numpy()
    return Image.fromarray(rgb, mode="RGB")


def render(path: Path) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    prompt_id, step, block, suffix = saved["spec"]
    p, m = saved["panels"], saved["metrics"]
    panels = [
        (p["weight"], "BF16 weight\n(output × input; averaged)"),
        (p["weight_residual"], "effective weight residual\n(SVDQuant − BF16; averaged)"),
        (p["input"], "BF16 input x\n(token × input; averaged)"),
        (p["smoothed_input"], "smoothed input x/s\n(token × input; averaged)"),
        (p["bf16_output"], "BF16 output y = Wx\n(token × output; averaged)"),
        (p["main_error"], "main-only error\nQ(Ws−BA)Q(x/s) − y\n(raw values, sampled)"),
        (p["low_rank"], "low-rank term\nBA(x/s)\n(raw values, sampled)"),
        (p["full_error"], "full error\nQ(Ws−BA)Q(x/s)+BA(x/s)−y\n(raw values, sampled)"),
    ]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    width, height, top, panel_w, panel_h = 1400, 830, 135, 330, 260
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    draw.text((20, 16), f"MiniMax-H3 | prompt {prompt_id}, cached denoising step {step}, block {block}, {suffix}", fill="black", font=title_font)
    draw.text((20, 48), f"main-only NMSE {m['main_nmse']:.3%} (rRMS {m['main_rrmse']:.2%}) → full NMSE {m['full_nmse']:.3%} (rRMS {m['full_rrmse']:.2%})", fill="black", font=font)
    draw.text((20, 72), f"low-rank RMS/output RMS {m['low_rank_rms_over_output_rms']:.2%}; error-energy reduction {m['error_energy_reduction_fraction']:.2%}; BF16 check {m['bf16_check_nmse']:.1e}", fill="black", font=font)
    for i, (tensor, title) in enumerate(panels):
        image = downsample(tensor)
        lim = limit(image)
        col, row = i % 4, i // 4
        left, y0 = 20 + col * 344, top + row * 350
        for line_no, line in enumerate(title.split("\n")):
            draw.text((left, y0 - 42 + 14 * line_no), line, fill="black", font=small)
        img = heatmap(image, lim).resize((panel_w, panel_h), Image.Resampling.NEAREST)
        figure.paste(img, (left, y0))
        draw.text((left, y0 + panel_h + 4), "feature coordinate (downsampled)", fill="black", font=small)
        draw.text((left, y0 + panel_h + 20), f"symmetric range ±{lim:.3g} (99th percentile)", fill="black", font=small)
    figure.save(OUT / path.name.replace("_data", "").replace(".pt", ".png"))


def main() -> None:
    for path in sorted(OUT.glob("h3_linear_pattern_data_*.pt")):
        render(path)


if __name__ == "__main__":
    main()
