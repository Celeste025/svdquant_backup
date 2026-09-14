#!/usr/bin/env python3
"""Evaluate signed Top/Bottom restores against existing paired baselines."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "results/samples/rcm_restore_video_comparison_all"
OUT = ROOT / "results/reports/rcm_signed_restore_top_bottom"


def video_path(case: str, variant: str) -> Path:
    suffix = {
        "bf16": "bf16",
        "nvfp4": "top0_nvfp4",
        "unsigned_top10": "top10_trajectory_actual",
        "signed_top10": "signed_top",
        "signed_bottom10": "signed_bottom",
    }[variant]
    return VIDEOS / f"{case}__{suffix}.mp4"


@torch.inference_mode()
def compare_video(reference: Path, candidate: Path, device: torch.device) -> dict[str, float | int]:
    ref_cap = cv2.VideoCapture(str(reference))
    pred_cap = cv2.VideoCapture(str(candidate))
    if not ref_cap.isOpened() or not pred_cap.isOpened():
        raise RuntimeError(f"could not open {reference} or {candidate}")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    err2 = ref2 = abs_err = 0.0
    count = frames = 0
    while True:
        ok_ref, ref = ref_cap.read()
        ok_pred, pred = pred_cap.read()
        if ok_ref != ok_pred:
            raise RuntimeError(f"frame-count mismatch: {reference} vs {candidate}")
        if not ok_ref:
            break
        ref_t = torch.from_numpy(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)
        pred_t = torch.from_numpy(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)
        error = pred_t - ref_t
        err2 += float(error.square().sum())
        ref2 += float(ref_t.square().sum())
        abs_err += float(error.abs().sum())
        count += ref_t.numel()
        frames += 1
        lpips.update(pred_t, ref_t)
        ssim.update(pred_t, ref_t)
    ref_cap.release()
    pred_cap.release()
    mse = err2 / count
    return {
        "frames": frames,
        "mse": mse,
        "nmse": err2 / max(ref2, 1e-30),
        "mae": abs_err / count,
        "psnr_db": 10 * math.log10(1 / max(mse, 1e-30)),
        "ssim": float(ssim.compute()),
        "lpips_alex": float(lpips.compute()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def latent_metrics() -> list[dict]:
    latent_dir = OUT / "latents"
    old_dir = ROOT / "results/reports/rcm_trajectory_actual_restore_top10/latents"
    reference = torch.load(old_dir / "bf16.pt", map_location="cpu", weights_only=True).float()
    variants = {
        "nvfp4": old_dir / "nvfp4.pt",
        "unsigned_top10": old_dir / "trajectory_actual_top10.pt",
        "signed_top10": latent_dir / "airplane_seed303__signed_top.pt",
        "signed_bottom10": latent_dir / "airplane_seed303__signed_bottom.pt",
    }
    ref2 = float(reference.square().sum())
    rows = []
    for variant, path in variants.items():
        value = torch.load(path, map_location="cpu", weights_only=True).float()
        error = value - reference
        err2 = float(error.square().sum())
        rows.append({
            "case": "airplane_seed303",
            "variant": variant,
            "latent_mse": err2 / reference.numel(),
            "latent_nmse": err2 / ref2,
            "latent_mae": float(error.abs().mean()),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cases = ("airplane_seed303", "person_seed307", "fastmotion_seed308", "watch_seed309")
    variants = ("nvfp4", "unsigned_top10", "signed_top10", "signed_bottom10")
    rows: list[dict] = []
    for case in cases:
        reference = video_path(case, "bf16")
        for variant in variants:
            candidate = video_path(case, variant)
            print(f"[{case}] {variant}", flush=True)
            row = {"case": case, "variant": variant, "reference": str(reference), "video": str(candidate)}
            row.update(compare_video(reference, candidate, device))
            rows.append(row)
    write_csv(OUT / "video_metrics.csv", rows)
    (OUT / "video_metrics.json").write_text(json.dumps(rows, indent=2))

    by_key = {(row["case"], row["variant"]): row for row in rows}
    changes: list[dict] = []
    for case in cases:
        base = by_key[(case, "nvfp4")]
        for variant in variants[1:]:
            row = by_key[(case, variant)]
            changes.append({
                "case": case,
                "variant": variant,
                "nmse_reduction_pct_vs_nvfp4": 100 * (base["nmse"] - row["nmse"]) / base["nmse"],
                "mse_reduction_pct_vs_nvfp4": 100 * (base["mse"] - row["mse"]) / base["mse"],
                "lpips_reduction_pct_vs_nvfp4": 100 * (base["lpips_alex"] - row["lpips_alex"]) / base["lpips_alex"],
                "psnr_delta_db_vs_nvfp4": row["psnr_db"] - base["psnr_db"],
                "ssim_delta_vs_nvfp4": row["ssim"] - base["ssim"],
            })
    write_csv(OUT / "changes_vs_nvfp4.csv", changes)
    (OUT / "changes_vs_nvfp4.json").write_text(json.dumps(changes, indent=2))

    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append({
            "variant": variant,
            "mean_video_nmse": float(np.mean([row["nmse"] for row in selected])),
            "geomean_video_nmse": float(np.exp(np.mean(np.log([row["nmse"] for row in selected])))),
            "mean_lpips_alex": float(np.mean([row["lpips_alex"] for row in selected])),
            "mean_psnr_db": float(np.mean([row["psnr_db"] for row in selected])),
            "mean_ssim": float(np.mean([row["ssim"] for row in selected])),
        })
    write_csv(OUT / "summary.csv", summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    latent_rows = latent_metrics()
    write_csv(OUT / "airplane_latent_metrics.csv", latent_rows)
    (OUT / "airplane_latent_metrics.json").write_text(json.dumps(latent_rows, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(cases))
    width = .24
    for index, variant in enumerate(variants[1:]):
        subset = [row for row in changes if row["variant"] == variant]
        axes[0].bar(x + (index - 1) * width, [row["nmse_reduction_pct_vs_nvfp4"] for row in subset], width, label=variant)
        axes[1].bar(x + (index - 1) * width, [row["lpips_reduction_pct_vs_nvfp4"] for row in subset], width, label=variant)
    for axis, title in zip(axes, ("Video NMSE reduction", "LPIPS reduction")):
        axis.axhline(0, color="black", lw=1)
        axis.set_xticks(x, [case.split("_seed")[0] for case in cases])
        axis.set_ylabel("improvement over NVFP4 (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_vs_nvfp4.png", dpi=180)
    plt.close(fig)
    print(f"saved evaluation to {OUT}", flush=True)


if __name__ == "__main__":
    main()
