#!/usr/bin/env python3
"""Frame-level BF16-paired LPIPS/tLP analysis for the fast-motion case."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "results" / "samples"
REFERENCE = SAMPLES / "rcm_restore_prompt_sweep" / "fastmotion_seed308" / "bf16.mp4"
VARIANTS = {
    "nvfp4": SAMPLES / "rcm_restore_prompt_sweep" / "fastmotion_seed308" / "top0_nvfp4.mp4",
    "top5": SAMPLES / "rcm_restore_prompt_sweep" / "fastmotion_seed308" / "top05pct_bf16.mp4",
    "top10": SAMPLES / "rcm_restore_prompt_sweep" / "fastmotion_seed308" / "top10pct_bf16.mp4",
    "top20": SAMPLES / "rcm_restore_prompt_sweep" / "fastmotion_seed308" / "top20pct_bf16.mp4",
    "all_attention": SAMPLES / "rcm_restore_components" / "fastmotion_seed308" / "all_attention_bf16.mp4",
    "all_ffn": SAMPLES / "rcm_restore_components" / "fastmotion_seed308" / "all_ffn_bf16.mp4",
}


def to_tensor(frame, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)


@torch.inference_mode()
def score_pair(metric, left: torch.Tensor, right: torch.Tensor) -> float:
    metric.update(left, right)
    score = float(metric.compute())
    metric.reset()
    return score


@torch.inference_mode()
def analyze_variant(candidate: Path, device: torch.device, metric) -> tuple[list[float], list[float]]:
    ref_cap, cand_cap = cv2.VideoCapture(str(REFERENCE)), cv2.VideoCapture(str(candidate))
    if not ref_cap.isOpened() or not cand_cap.isOpened():
        raise RuntimeError(f"Could not open {REFERENCE} or {candidate}")
    spatial, temporal = [], []
    previous_ref = previous_cand = None
    while True:
        ok_ref, ref = ref_cap.read()
        ok_cand, cand = cand_cap.read()
        if ok_ref != ok_cand:
            raise RuntimeError(f"Frame-count mismatch: {candidate}")
        if not ok_ref:
            break
        if ref.shape != cand.shape:
            raise RuntimeError(f"Frame-shape mismatch: {candidate}")
        ref_t, cand_t = to_tensor(ref, device), to_tensor(cand, device)
        spatial.append(score_pair(metric, cand_t, ref_t))
        if previous_ref is not None:
            ref_change = score_pair(metric, previous_ref, ref_t)
            cand_change = score_pair(metric, previous_cand, cand_t)
            temporal.append(abs(cand_change - ref_change))
        previous_ref, previous_cand = ref_t, cand_t
    ref_cap.release(); cand_cap.release()
    return spatial, temporal


def main() -> None:
    output = ROOT / "results" / "reports" / "fastmotion_temporal_lpips"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    traces, summary = {}, {}
    for name, path in VARIANTS.items():
        print(f"[{name}] {path}", flush=True)
        spatial, temporal = analyze_variant(path, device, metric)
        traces[name] = {"spatial_lpips": spatial, "temporal_lpips_delta": temporal}
        summary[name] = {
            "spatial_lpips_mean": sum(spatial) / len(spatial),
            "temporal_lpips_delta_mean": sum(temporal) / len(temporal),
            "temporal_lpips_delta_p90": float(torch.tensor(temporal).quantile(0.9)),
            "worst_transition": int(torch.tensor(temporal).argmax()) + 1,
            "worst_transition_tlp": max(temporal),
        }

    with (output / "per_frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "frame", "spatial_lpips", "transition_to_frame", "temporal_lpips_delta"))
        writer.writeheader()
        for name, values in traces.items():
            for frame, spatial in enumerate(values["spatial_lpips"]):
                writer.writerow({"variant": name, "frame": frame, "spatial_lpips": spatial, "transition_to_frame": "", "temporal_lpips_delta": ""})
            for frame, temporal in enumerate(values["temporal_lpips_delta"], start=1):
                writer.writerow({"variant": name, "frame": "", "spatial_lpips": "", "transition_to_frame": frame, "temporal_lpips_delta": temporal})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False, constrained_layout=True)
    for name, values in traces.items():
        axes[0].plot(range(len(values["spatial_lpips"])), values["spatial_lpips"], label=name, linewidth=1.5)
        axes[1].plot(range(1, len(values["temporal_lpips_delta"]) + 1), values["temporal_lpips_delta"], label=name, linewidth=1.5)
    axes[0].set(title="Fastmotion: per-frame LPIPS to matched BF16 (lower is better)", ylabel="LPIPS")
    axes[1].set(title="Fastmotion: Temporal-LPIPS difference per transition (lower is better)", xlabel="frame / transition end frame", ylabel="tLP")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=9)
    fig.savefig(output / "fastmotion_spatial_and_temporal_lpips.png", dpi=160)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
