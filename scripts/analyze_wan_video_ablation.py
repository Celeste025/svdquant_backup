#!/usr/bin/env python3
"""Measure spatial and temporal residuals against matched BF16 Wan videos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def read_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    try:
        return np.stack([frame.astype(np.float32) for frame in reader])
    finally:
        reader.close()


def metrics(reference: np.ndarray, quantized: np.ndarray) -> dict[str, float]:
    if reference.shape != quantized.shape:
        raise ValueError(f"shape mismatch: {reference.shape} versus {quantized.shape}")
    error = quantized - reference
    mse = float(np.mean(error**2))
    temporal_error = np.diff(error, axis=0)

    # Evaluate temporal residual specifically around high-gradient BF16 regions.
    gray = reference.mean(axis=-1)
    gx = np.abs(np.diff(gray, axis=2, prepend=gray[:, :, :1]))
    gy = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1, :]))
    gradient = gx + gy
    thresholds = np.percentile(gradient.reshape(gradient.shape[0], -1), 80, axis=1)
    edge_mask = gradient[1:] >= thresholds[1:, None, None]

    # A simple high-pass residual reports quantization-correlated noise texture.
    spatial_average = (
        error
        + np.roll(error, 1, axis=1)
        + np.roll(error, -1, axis=1)
        + np.roll(error, 1, axis=2)
        + np.roll(error, -1, axis=2)
    ) / 5
    high_pass = error - spatial_average

    edge_values = np.abs(temporal_error)[edge_mask[..., None].repeat(3, axis=-1)]
    return {
        "rgb_mae": float(np.mean(np.abs(error))),
        "rgb_psnr_db": float(10 * math.log10(255**2 / max(mse, 1e-12))),
        "temporal_residual_mae": float(np.mean(np.abs(temporal_error))),
        "edge_temporal_residual_mae": float(np.mean(edge_values)),
        "high_frequency_residual_mae": float(np.mean(np.abs(high_pass))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "BF16", "QUANT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for label, bf16_path, quant_path in args.pair:
        print(f"reading {label}", flush=True)
        report[label] = metrics(read_video(Path(bf16_path)), read_video(Path(quant_path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
