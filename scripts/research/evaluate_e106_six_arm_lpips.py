#!/usr/bin/env python3
"""Full-resolution paired RGB NMSE and LPIPS-Alex for E106 six-arm videos."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from summarize_e105_h3_output_interventions import frames, sha256


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("bf16", "svdquant", "permuted_error", "tail_restore", "rademacher_flat", "gaussian")
LABELS = ("BF16", "SVDQuant", "Permuted real error", "Restore >4 sigma", "Equal |error| +/-", "Gaussian error")


def model_sha256(metric: LearnedPerceptualImagePatchSimilarity) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(metric.net.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_poster(snapshots: list[tuple[int, tuple[np.ndarray, ...]]], rows: dict,
                destination: Path, case: str) -> None:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    w, h, band = 512, 288, 60
    one = (3 * w, 2 * (h + band))
    canvas = Image.new("RGB", (one[0], one[1] * len(snapshots) + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 9), f"{case}: 124 full-resolution frames vs BF16; lower is closer", font=font,
              fill=(25, 31, 41))
    for snapshot_index, (frame_id, image_set) in enumerate(snapshots):
        y_base = 42 + snapshot_index * one[1]
        for arm_index, (arm, label, frame) in enumerate(zip(ARMS, LABELS, image_set, strict=True)):
            x = (arm_index % 3) * w
            y = y_base + (arm_index // 3) * (h + band)
            canvas.paste(Image.fromarray(frame).resize((w, h), Image.Resampling.BILINEAR), (x, y + band))
            draw.rectangle((x, y, x + w - 1, y + band - 1), fill=(25, 31, 41))
            draw.text((x + 9, y + 3), f"{label}   frame {frame_id}", font=font, fill="white")
            values = rows[arm]
            draw.text((x + 9, y + 31),
                      f"RGB NMSE {values['rgb_nmse'] * 100:.2f}%   LPIPS-Alex {values['lpips_alex']:.4f}",
                      font=small_font, fill=(241, 245, 249))
    canvas.save(destination, quality=90, optimize=True)


@torch.inference_mode()
def evaluate(case: str, metric: LearnedPerceptualImagePatchSimilarity, model_hash: str) -> dict:
    directory = ROOT / "results/research/E106" / case
    source = json.loads((directory / "six_arm_summary.json").read_text())
    if source.get("format") != "h3-e106-six-arm-summary-v1" or source.get("case") != case or source.get("frames") != 124:
        raise RuntimeError("invalid E106 source summary")
    videos = {arm: Path(source["sources"][arm]) for arm in ARMS}
    for arm, path in videos.items():
        if not path.is_file() or sha256(path) != source["source_sha256"][arm]:
            raise RuntimeError(f"source video changed: {arm}")
    rows = {arm: {"squared_sum_uint8": 0, "lpips_per_frame": []} for arm in ARMS}
    ref_squared = 0
    snapshots: list[tuple[int, tuple[np.ndarray, ...]]] = []
    count = 0
    for frame_id, image_set in enumerate(zip_longest(*(frames(videos[arm]) for arm in ARMS))):
        if frame_id >= 124 or any(image is None or image.shape != (576, 1024, 3) for image in image_set):
            raise RuntimeError(f"missing, extra or malformed frame: {frame_id}")
        reference = image_set[0]
        ref_squared += int(np.square(reference.astype(np.int32)).sum(dtype=np.int64))
        candidate_batch = torch.from_numpy(np.stack(image_set[1:], axis=0)).permute(0, 3, 1, 2)
        candidate_batch = candidate_batch.to(device="cuda", dtype=torch.float32).div_(255)
        ref_batch = torch.from_numpy(reference.copy()).permute(2, 0, 1).unsqueeze(0)
        ref_batch = ref_batch.to(device="cuda", dtype=torch.float32).div_(255).expand_as(candidate_batch).contiguous()
        scores = metric(candidate_batch, ref_batch).detach().cpu().reshape(-1).tolist()
        metric.reset()
        if len(scores) != 5 or not all(math.isfinite(score) and score >= 0 for score in scores):
            raise RuntimeError(f"invalid LPIPS score at frame {frame_id}")
        if frame_id == 0:
            self_score = float(metric(ref_batch[:1], ref_batch[:1]).item())
            metric.reset()
            if abs(self_score) > 1e-7:
                raise RuntimeError("LPIPS identity check failed")
        for arm, frame, score in zip(ARMS[1:], image_set[1:], scores, strict=True):
            diff = frame.astype(np.int32) - reference.astype(np.int32)
            rows[arm]["squared_sum_uint8"] += int(np.square(diff).sum(dtype=np.int64))
            rows[arm]["lpips_per_frame"].append(float(score))
        if frame_id in (0, 30, 60, 90):
            snapshots.append((frame_id, image_set))
        count += 1
        if count % 20 == 0:
            print(f"{case}: LPIPS {count}/124 frames", flush=True)
    if count != 124 or len(snapshots) != 4 or ref_squared <= 0:
        raise RuntimeError("incomplete E106 LPIPS evaluation")
    rows["bf16"] = {"squared_sum_uint8": 0, "lpips_per_frame": [0.0] * 124}
    for arm in ARMS:
        row = rows[arm]
        row["rgb_nmse"] = row["squared_sum_uint8"] / ref_squared
        row["lpips_alex"] = float(np.mean(row["lpips_per_frame"]))
        row["frames"] = 124
        if arm != "bf16" and abs(row["rgb_nmse"] - source["rollout_distance_proxy"][arm]["rgb_nmse"]) > 1e-12:
            raise RuntimeError(f"RGB NMSE failed independent E106 reconciliation: {arm}")
    poster = directory / "six_arm_metrics_poster.jpg"
    report = directory / "six_arm_nmse_lpips.json"
    if poster.exists() or report.exists():
        raise FileExistsError("refusing to overwrite E106 metric figure/report")
    make_poster(snapshots, rows, poster, case)
    payload = {
        "format": "e106-h3-six-arm-nmse-lpips-v1", "case": case,
        "definition": {
            "reference": "same-case BF16 decoded MP4 RGB uint8, frame-index paired",
            "rgb_nmse": "sum over all frames, pixels and channels of (candidate-reference)^2 / sum(reference^2)",
            "lpips_alex": "TorchMetrics pretrained AlexNet, FP32, normalize=True for [0,1] RGB; arithmetic mean of all 124 full-resolution frame scores",
            "scope": "1024x576, 124 frames, no resizing/cropping/alignment for metrics; audio excluded; distances are not quality scores",
        },
        "source_summary": str((directory / "six_arm_summary.json").resolve()),
        "source_summary_sha256": sha256(directory / "six_arm_summary.json"),
        "source_videos": {arm: str(videos[arm].resolve()) for arm in ARMS},
        "source_sha256": source["source_sha256"],
        "environment": {"torch": torch.__version__, "torchmetrics": torchmetrics.__version__,
                        "gpu": torch.cuda.get_device_name(), "metric_state_sha256": model_hash},
        "reference_squared_sum_uint8": ref_squared, "rows": rows,
        "poster": str(poster.resolve()), "poster_sha256": sha256(poster),
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({arm: {"rgb_nmse": rows[arm]["rgb_nmse"], "lpips_alex": rows[arm]["lpips_alex"]}
                      for arm in ARMS}, ensure_ascii=False, indent=2), flush=True)
    print(f"COMPLETE {report} {poster}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("race", "clap0"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, reduction="none").cuda().eval()
    evaluate(args.case, metric, model_sha256(metric))


if __name__ == "__main__":
    main()
