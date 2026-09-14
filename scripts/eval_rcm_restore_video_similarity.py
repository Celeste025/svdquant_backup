#!/usr/bin/env python3
"""Compare rCM-Wan restore-sweep videos against their matched BF16 videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]


def cases() -> dict[str, dict[str, Path]]:
    samples = ROOT / "results" / "samples"
    airplane = samples / "rcm-wan_bf16_vs_nvfp4"
    restore = samples / "rcm_restore_smoothaware"
    random20 = samples / "rcm_restore_random20"
    sweep = samples / "rcm_restore_sweep"
    result = {
        "airplane_seed303": {
            "bf16": airplane / "bf16_airplane_seed303.mp4",
            "top0": airplane / "w4a4_real_nvfp4_airplane_seed303.mp4",
            "top5": restore / "top5pct_airplane_seed303.mp4",
            "top10": restore / "top10pct_airplane_seed303.mp4",
            "top20": restore / "top20pct_airplane_seed303.mp4",
            "random20": random20 / "airplane_seed303_random20.mp4",
        }
    }
    for case in ("fox_seed304", "bus_seed305", "ocean_seed306"):
        base = sweep / case
        result[case] = {
            "bf16": base / "bf16.mp4",
            "top0": base / "top0_nvfp4.mp4",
            "top5": base / "top05pct_bf16.mp4",
            "top10": base / "top10pct_bf16.mp4",
            "top20": base / "top20pct_bf16.mp4",
            "random20": random20 / f"{case}_random20.mp4",
        }
    prompt_sweep = samples / "rcm_restore_prompt_sweep"
    for case in ("person_seed307", "fastmotion_seed308", "watch_seed309"):
        base = prompt_sweep / case
        result[case] = {
            "bf16": base / "bf16.mp4",
            "top0": base / "top0_nvfp4.mp4",
            "top5": base / "top05pct_bf16.mp4",
            "top10": base / "top10pct_bf16.mp4",
            "top20": base / "top20pct_bf16.mp4",
            "random20": base / "random20.mp4",
        }
    components = samples / "rcm_restore_components"
    for case, variants in result.items():
        for component in ("attention", "ffn"):
            video = components / case / f"all_{component}_bf16.mp4"
            if video.is_file():
                variants[f"all_{component}"] = video
    return result


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    return cap


@torch.inference_mode()
def compare(
    reference: Path,
    candidate: Path,
    device: torch.device,
    *,
    include_temporal: bool = True,
) -> dict[str, float | int]:
    ref_cap, pred_cap = open_video(reference), open_video(candidate)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    temporal_lpips = (
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        if include_temporal else None
    )
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    sq_error = ref_power = abs_error = 0.0
    count = frames = 0
    temporal_deltas = []
    previous_ref = previous_pred = None
    while True:
        ok_ref, ref = ref_cap.read()
        ok_pred, pred = pred_cap.read()
        if ok_ref != ok_pred:
            raise RuntimeError(f"frame-count mismatch: {reference} vs {candidate}")
        if not ok_ref:
            break
        if ref.shape != pred.shape:
            raise RuntimeError(f"frame-shape mismatch: {reference} vs {candidate}: {ref.shape} != {pred.shape}")
        # Video codec returns BGR; reordering is unnecessary for scalar MSE but required for LPIPS.
        ref = torch.from_numpy(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)
        pred = torch.from_numpy(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)
        error = pred - ref
        sq_error += float(error.square().sum())
        ref_power += float(ref.square().sum())
        abs_error += float(error.abs().sum())
        count += ref.numel()
        frames += 1
        lpips.update(pred, ref)
        if include_temporal and previous_ref is not None:
            assert temporal_lpips is not None
            # Temporal-LPIPS difference (tLP): compare the *perceived change*
            # between adjacent candidate frames against that of the matched
            # BF16 trajectory.  It is a full-reference temporal-consistency
            # score; lower means closer temporal dynamics, not sharper frames.
            temporal_lpips.update(previous_ref, ref)
            ref_change = float(temporal_lpips.compute())
            temporal_lpips.reset()
            temporal_lpips.update(previous_pred, pred)
            pred_change = float(temporal_lpips.compute())
            temporal_lpips.reset()
            temporal_deltas.append(abs(pred_change - ref_change))
        previous_ref, previous_pred = ref, pred
        ssim.update(pred, ref)
    ref_cap.release(); pred_cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {reference}")
    mse = sq_error / count
    metrics: dict[str, float | int] = {
        "frames": frames,
        "pixels_per_frame": count // frames,
        "mse": mse,
        "nmse": sq_error / max(ref_power, 1e-30),
        "mae": abs_error / count,
        "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-30)),
        "ssim": float(ssim.compute()),
        "lpips_alex": float(lpips.compute()),
    }
    if include_temporal:
        metrics["temporal_lpips_delta"] = sum(temporal_deltas) / max(len(temporal_deltas), 1)
        metrics["temporal_lpips_delta_p90"] = float(torch.tensor(temporal_deltas).quantile(0.9)) if temporal_deltas else 0.0
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/reports/rcm_restore_video_similarity")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    for case, variants in cases().items():
        for variant, video in variants.items():
            if variant == "bf16":
                continue
            print(f"[{case}] {variant}", flush=True)
            row = {"case": case, "variant": variant, "reference": str(variants["bf16"]), "video": str(video)}
            row.update(compare(variants["bf16"], video, device))
            rows.append(row)
    definition = (
        "All metrics compare decoded RGB frames in [0,1] to the matched BF16 video. "
        "MSE/NMSE/MAE pool every frame and pixel; PSNR uses data range 1; SSIM and LPIPS-Alex are mean frame metrics. "
        "temporal_lpips_delta is the mean absolute difference between candidate and BF16 adjacent-frame LPIPS "
        "changes; p90 is its 90th percentile. Lower temporal-LPIPS indicates closer temporal dynamics, not spatial sharpness."
    )
    (out / "metrics.json").write_text(json.dumps({"definition": definition, "rows": rows}, indent=2) + "\n")
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
