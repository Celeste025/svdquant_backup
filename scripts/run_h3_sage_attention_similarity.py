#!/usr/bin/env python3
"""PSNR / LPIPS / relative-L2 for the 8-case MiniMax-H3 SageAttention subset."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import lpips
import torch

BASE = Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "h3_sage_attention_similarity.json"
CASE_IDS = [f"vbench2_{i:03d}" for i in range(8)]
VARIANTS = ("bf16", "svdquant", "svdquant_sageattn2", "svdquant_sageattn3")
PAIRS = (
    ("svdquant", "bf16"),
    ("svdquant_sageattn2", "bf16"),
    ("svdquant_sageattn3", "bf16"),
    ("svdquant_sageattn2", "svdquant"),
    ("svdquant_sageattn3", "svdquant"),
)


def video_path(case_id: str, case_index: int, variant: str) -> Path:
    if variant in {"bf16", "svdquant"}:
        return BASE / "cases" / case_id / f"{variant}.mp4"
    gpu = case_index // 2
    backend = "sageattn2" if variant == "svdquant_sageattn2" else "sageattn3"
    return BASE / f"{backend}_cases8" / f"gpu{gpu}" / "cases" / case_id / "svdquant.mp4"


@torch.inference_mode()
def compare(reference: Path, candidate: Path, device: torch.device, metric: lpips.LPIPS) -> dict[str, float | int]:
    ref_cap = cv2.VideoCapture(str(reference))
    pred_cap = cv2.VideoCapture(str(candidate))
    if not ref_cap.isOpened() or not pred_cap.isOpened():
        raise RuntimeError(f"could not open {reference} / {candidate}")
    sq_error = ref_power = abs_error = 0.0
    count = frames = 0
    lpips_sum = 0.0
    while True:
        ok_ref, ref = ref_cap.read()
        ok_pred, pred = pred_cap.read()
        if ok_ref != ok_pred:
            raise RuntimeError(f"frame-count mismatch: {reference} vs {candidate}")
        if not ok_ref:
            break
        if ref.shape != pred.shape:
            raise RuntimeError(f"frame-shape mismatch: {reference} vs {candidate}: {ref.shape} != {pred.shape}")
        ref_t = torch.from_numpy(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)
        pred_t = torch.from_numpy(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)
        error = pred_t - ref_t
        sq_error += float(error.square().sum())
        ref_power += float(ref_t.square().sum())
        abs_error += float(error.abs().sum())
        count += ref_t.numel()
        frames += 1
        lpips_sum += float(metric(pred_t.mul(2).sub_(1), ref_t.mul(2).sub_(1)))
    ref_cap.release()
    pred_cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {reference}")
    mse = sq_error / count
    return {
        "frames": frames,
        "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-30)),
        "lpips_alex": lpips_sum / frames,
        "rel_l2": math.sqrt(sq_error / max(ref_power, 1e-30)),
        "l2_abs": math.sqrt(sq_error),
        "mse": mse,
        "mae": abs_error / count,
    }


def summarize(rows: list[dict], label: str) -> dict:
    def stat(key: str) -> tuple[float, float]:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
        return float(values.mean()), float(values.std(unbiased=False))

    return {
        "label": label,
        "pairs": len(rows),
        "psnr_db_mean": stat("psnr_db")[0],
        "psnr_db_std": stat("psnr_db")[1],
        "lpips_mean": stat("lpips_alex")[0],
        "lpips_std": stat("lpips_alex")[1],
        "rel_l2_mean": stat("rel_l2")[0],
        "rel_l2_std": stat("rel_l2")[1],
        "per_case": [{"case": row["case"], "psnr_db": row["psnr_db"], "lpips_alex": row["lpips_alex"], "rel_l2": row["rel_l2"]} for row in rows],
    }


def main() -> None:
    device = torch.device("cuda")
    metric = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    for case_index, case_id in enumerate(CASE_IDS):
        for candidate_variant, reference_variant in PAIRS:
            candidate = video_path(case_id, case_index, candidate_variant)
            reference = video_path(case_id, case_index, reference_variant)
            row = {
                "case": case_id,
                "candidate": candidate_variant,
                "reference": reference_variant,
                "video": str(candidate),
            }
            row.update(compare(reference, candidate, device, metric))
            rows.append(row)
            print(f"{case_id} {candidate_variant} vs {reference_variant}: "
                  f"psnr={row['psnr_db']:.2f}dB lpips={row['lpips_alex']:.4f} rel_l2={row['rel_l2']:.4f}", flush=True)

    summaries = []
    for candidate_variant, reference_variant in PAIRS:
        group = [row for row in rows if row["candidate"] == candidate_variant and row["reference"] == reference_variant]
        key = f"{candidate_variant}|vs_{reference_variant}"
        summaries.append(summarize(group, key))
        no_case0 = [row for row in group if row["case"] != "vbench2_000"]
        summaries.append(summarize(no_case0, key + "|excl_case0"))

    csv_path = DEFAULT_OUTPUT.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    DEFAULT_OUTPUT.write_text(json.dumps({
        "definition": (
            "Decoded RGB frames in [0,1] against the reference video. PSNR pools all frames/pixels "
            "with data range 1; LPIPS is AlexNet, mean over frames (lower = closer); rel_l2 = "
            "||pred-ref||_2 / ||ref||_2 pooled over the whole video (lower = closer)."
        ),
        "rows": rows,
        "summary": summaries,
    }, indent=2) + "\n")
    print(f"saved {DEFAULT_OUTPUT} and {csv_path}", flush=True)
    for entry in summaries:
        print(f"{entry['label']}: psnr={entry['psnr_db_mean']:.2f}±{entry['psnr_db_std']:.2f}dB "
              f"lpips={entry['lpips_mean']:.4f}±{entry['lpips_std']:.4f} rel_l2={entry['rel_l2_mean']:.4f}±{entry['rel_l2_std']:.4f}")


if __name__ == "__main__":
    main()
