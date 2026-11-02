#!/usr/bin/env python3
"""Overlay paired H3 DiT video-token errors on approximate BF16 video frames.

This is a spatial-location guide, not a VAE-decoded pixel-error attribution.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path(__file__).resolve().parents[2]
CASES = {"race": "E103", "clap0": "E104"}
STEPS = (0, 4, 10, 19)
LATENT_TIMES = (8, 18, 28)
GRID = (37, 18, 32)
SIZE = (512, 288)
PALETTE = ((16, 19, 37), (54, 47, 111), (133, 48, 111), (213, 89, 67), (248, 190, 83), (253, 246, 189))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representative_frame(latent_t: int) -> int:
    """Approximate display-time center; VAE mixing makes this non-exact."""
    if latent_t < 2:
        return round((latent_t + 0.5) * 5 / 2 - 0.5)
    chunk, offset = divmod(latent_t - 2, 5)
    return round(5 + 17 * chunk + (offset + 0.5) * 17 / 5 - 0.5)


def selected_video_frames(path: Path, requested: tuple[int, ...]) -> dict[int, Image.Image]:
    selected: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height, float(stream.average_rate)) != (1024, 576, 24.0):
            raise RuntimeError(f"unexpected BF16 video format: {path}")
        for index, frame in enumerate(container.decode(stream)):
            if index in requested:
                selected[index] = Image.fromarray(frame.to_ndarray(format="rgb24"))
    if set(selected) != set(requested) or index != 123:
        raise RuntimeError(f"BF16 video lacks required 124 frames: {path}")
    return selected


def palette_rgb(values: np.ndarray) -> np.ndarray:
    u = np.clip(values, 0, 1) * (len(PALETTE) - 1)
    lo = np.floor(u).astype(np.int32)
    hi = np.minimum(lo + 1, len(PALETTE) - 1)
    frac = (u - lo)[..., None]
    colors = np.asarray(PALETTE, dtype=np.float32)
    return colors[lo] * (1 - frac) + colors[hi] * frac


def overlay(frame: Image.Image, values: np.ndarray, vmax: float) -> Image.Image:
    if values.shape != GRID[1:]:
        raise RuntimeError(f"unexpected spatial map shape: {values.shape}")
    background = np.asarray(frame.resize(SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    normalized = np.clip(values / vmax, 0, 1)
    color = palette_rgb(normalized)
    heat = np.repeat(np.repeat(color, SIZE[1] // GRID[1], axis=0), SIZE[0] // GRID[2], axis=1)
    opacity = np.repeat(np.repeat(0.82 * np.sqrt(normalized), SIZE[1] // GRID[1], axis=0),
                        SIZE[0] // GRID[2], axis=1)[..., None]
    return Image.fromarray(np.uint8(np.clip(background * (1 - opacity) + heat * opacity, 0, 255)))


def load_error_maps(case: str) -> tuple[dict, dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, str]]:
    directory = ROOT / "results/research" / CASES[case]
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["format"] != "h3-e103-paired-output-manifest-v1" or
            tuple(manifest["video_grid_thw"]) != GRID or manifest["steps"] != 20 or
            len(manifest["records"]) != 20):
        raise RuntimeError(f"unexpected capture manifest: {case}")
    maps: dict[int, dict[int, np.ndarray]] = {}
    step_nmse: dict[int, float] = {}
    source_hashes: dict[int, str] = {}
    for step in STEPS:
        record = manifest["records"][step]
        path = Path(record["path"])
        if record["step"] != step or sha256(path) != record["sha256"]:
            raise RuntimeError(f"capture hash or order mismatch: {case}/step {step}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["step"] != step or tuple(payload["video_grid_thw"]) != GRID:
            raise RuntimeError(f"capture metadata mismatch: {case}/step {step}")
        reference = payload["video_bf16"].float().numpy().reshape(*GRID, 96)
        error = payload["video_error_fp32"].numpy().reshape(*GRID, 96)
        quantized = payload["video_svdquant"].float().numpy().reshape(*GRID, 96)
        if not np.array_equal(error, quantized - reference):
            raise RuntimeError(f"signed error failed exact reconciliation: {case}/step {step}")
        if not np.isfinite(error).all() or not np.isfinite(reference).all():
            raise RuntimeError(f"nonfinite output: {case}/step {step}")
        signal_rms = math.sqrt(float(np.mean(np.square(reference, dtype=np.float64))))
        nmse = float(np.square(error, dtype=np.float64).sum() / np.square(reference, dtype=np.float64).sum())
        if signal_rms <= 0 or abs(nmse - payload["metrics"]["video"]["nmse"]) > 1e-7:
            raise RuntimeError(f"step NMSE mismatch: {case}/step {step}")
        # Each cell is the 96-channel error RMS at a token, divided by the
        # whole-step BF16 output RMS. It is not per-token NMSE or RGB error.
        per_token = np.sqrt(np.mean(np.square(error, dtype=np.float64), axis=-1)) / signal_rms * 100
        maps[step] = {latent_t: per_token[latent_t].astype(np.float32) for latent_t in LATENT_TIMES}
        step_nmse[step] = nmse
        source_hashes[step] = record["sha256"]
    return manifest, maps, step_nmse, source_hashes


def draw_case(case: str) -> None:
    manifest, maps, step_nmse, source_hashes = load_error_maps(case)
    six_arm = json.loads((ROOT / "results/research/E106" / case / "six_arm_summary.json").read_text())
    video_path = Path(six_arm["sources"]["bf16"])
    if sha256(video_path) != six_arm["source_sha256"]["bf16"]:
        raise RuntimeError(f"BF16 video hash mismatch: {case}")
    video_indices = tuple(representative_frame(t) for t in LATENT_TIMES)
    frames = selected_video_frames(video_path, video_indices)
    all_values = np.concatenate([maps[step][t].reshape(-1) for step in STEPS for t in LATENT_TIMES])
    vmax = float(np.quantile(all_values, 0.99))
    if not math.isfinite(vmax) or vmax <= 0:
        raise RuntimeError("invalid heatmap scale")

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 27)
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    margin_x, margin_top, gap_x, gap_y = 220, 132, 12, 40
    width = margin_x + 3 * SIZE[0] + 2 * gap_x + 40
    height = margin_top + 5 * SIZE[1] + 4 * gap_y + 134
    canvas = Image.new("RGB", (width, height), "#fafbfc")
    draw = ImageDraw.Draw(canvas)
    prompt = manifest["prompt_id"]
    draw.text((24, 16), f"{case.upper()}  |  prompt {prompt}  |  paired DiT video-output error on BF16 video", font=title_font, fill="#1c2833")
    draw.text((24, 58), "Token grid 37 x 18 x 32; selected latent times 8, 18, 28; denoise steps 0, 4, 10, 19", font=font, fill="#34495e")
    draw.text((24, 90), "Heat color = token error RMS / whole-step BF16 output RMS (%); one linear scale across this figure", font=small, fill="#34495e")

    labels = ("BF16 frame\n(spatial guide)",) + tuple(f"Step {step:02d}\nNMSE {100 * step_nmse[step]:.2f}%" for step in STEPS)
    for row, label in enumerate(labels):
        y = margin_top + row * (SIZE[1] + gap_y)
        draw.multiline_text((24, y + 12), label, font=font, fill="#1c2833", spacing=7)
        for col, (latent_t, frame_index) in enumerate(zip(LATENT_TIMES, video_indices, strict=True)):
            x = margin_x + col * (SIZE[0] + gap_x)
            frame = frames[frame_index]
            tile = frame.resize(SIZE, Image.Resampling.BILINEAR) if row == 0 else overlay(frame, maps[STEPS[row - 1]][latent_t], vmax)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + SIZE[0] - 1, y + SIZE[1] - 1), outline="#2e4053", width=1)
            if row == 0:
                draw.text((x + 8, y + 8), f"latent t={latent_t}  ~decoded frame {frame_index}", font=small,
                          fill="white", stroke_width=2, stroke_fill="black")

    footer_y = margin_top + 5 * SIZE[1] + 4 * gap_y + 17
    bar_x, bar_y, bar_w, bar_h = margin_x, footer_y, 600, 20
    gradient = np.linspace(0, 1, bar_w, dtype=np.float32)[None, :]
    gradient_rgb = np.uint8(np.repeat(palette_rgb(gradient), bar_h, axis=0))
    canvas.paste(Image.fromarray(gradient_rgb), (bar_x, bar_y))
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline="#2e4053", width=1)
    for frac in (0, 0.25, 0.5, 0.75, 1):
        tx = bar_x + round(frac * bar_w)
        draw.text((tx - (8 if frac == 0 else 27), bar_y + 24), f"{frac * vmax:.1f}%", font=small, fill="#1c2833")
    clipped = int(np.count_nonzero(all_values > vmax))
    draw.text((24, footer_y + 60), f"Shared scale clips top {clipped}/{all_values.size} cells (~1%); video is a location guide, not decoded pixel error.",
              font=small, fill="#34495e")

    directory = ROOT / "results/research" / CASES[case]
    figure = directory / "figures/video_frame_aligned_token_error.png"
    report = directory / "video_frame_aligned_token_error.json"
    if figure.exists() or report.exists():
        raise FileExistsError(f"refusing to overwrite existing overlay: {case}")
    figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure, optimize=True)
    payload = {
        "format": "h3-paired-dit-token-error-video-overlay-v1", "case": case,
        "definition": "100 * sqrt(mean_channel((SVDQuant-BF16)^2)) / sqrt(mean_all_video_output(BF16^2)); full-step BF16 input paired DiT output",
        "caveat": "Token positions are spatial guides over final BF16 decoded video; VAE mixes latent tokens across space/time, so the heatmap is not decoded-pixel-error attribution; temporal frame centers are approximate.",
        "selected_steps": list(STEPS), "latent_times": list(LATENT_TIMES),
        "approximate_decoded_frames": list(video_indices), "video_grid_thw": list(GRID),
        "step_video_nmse": {str(s): step_nmse[s] for s in STEPS},
        "case_shared_colorbar_max_percent_p99": vmax,
        "clipped_cells": clipped, "total_cells": int(all_values.size),
        "source_capture_manifest": str((directory / "manifest.json").resolve()),
        "source_capture_sha256": source_hashes,
        "bf16_video": str(video_path), "bf16_video_sha256": six_arm["source_sha256"]["bf16"],
        "figure": str(figure.resolve()), "figure_sha256": sha256(figure),
    }
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{case}: {figure} colorbar_max={vmax:.3f}% frames={video_indices}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    for selected_case in CASES:
        draw_case(selected_case)
