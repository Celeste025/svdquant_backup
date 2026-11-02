#!/usr/bin/env python3
"""Frame-paired full-resolution RGB NMSE and LPIPS-Alex for E124 videos."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from summarize_e113_native_videos import frames, sha256


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/research/E124"
ARMS = ("bf16", "sage3_fixed", "split_mixed")
PAIRS = (("sage3_fixed", "bf16"), ("split_mixed", "bf16"),
         ("split_mixed", "sage3_fixed"))


def state_hash(metric: LearnedPerceptualImagePatchSimilarity) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(metric.net.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_case(case: dict, metric: LearnedPerceptualImagePatchSimilarity,
                  metric_hash: str, results: Path = RESULTS) -> dict:
    prompt_id = case["prompt_id"]
    output = results / f"video_metrics_p{prompt_id:03d}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved.get("format") != "e124-video-lpips-nmse-v1" or
                saved.get("prompt_id") != prompt_id or
                saved.get("metric_state_sha256") != metric_hash):
            raise RuntimeError(f"Existing result mismatch: {output}")
        for arm in ARMS:
            if saved["source_sha256"][arm] != case["arms"][arm]["video_sha256"]:
                raise RuntimeError(f"Existing source hash mismatch: {output}")
        return saved

    videos = {arm: Path(case["arms"][arm]["video"]) for arm in ARMS}
    for arm, path in videos.items():
        if not path.is_file() or sha256(path) != case["arms"][arm]["video_sha256"]:
            raise RuntimeError(f"Video missing or hash mismatch: p{prompt_id:03d} {arm}")

    pair_keys = [f"{a}_vs_{b}" for a, b in PAIRS]
    rows = {key: {"squared_sum_uint8": 0, "reference_squared_sum_uint8": 0,
                  "lpips_per_frame": []} for key in pair_keys}
    count = 0
    for count, image_set in enumerate(zip_longest(*(frames(videos[a]) for a in ARMS)), 1):
        if count > 124 or any(image is None or image.shape != (576, 1024, 3)
                              for image in image_set):
            raise RuntimeError(f"Frame mismatch: p{prompt_id:03d} frame {count - 1}")
        images = dict(zip(ARMS, image_set, strict=True))
        candidates = torch.from_numpy(np.stack([images[a] for a, _ in PAIRS]))
        references = torch.from_numpy(np.stack([images[b] for _, b in PAIRS]))
        candidates = candidates.permute(0, 3, 1, 2).to("cuda", torch.float32).div_(255)
        references = references.permute(0, 3, 1, 2).to("cuda", torch.float32).div_(255)
        values = metric(candidates, references).detach().cpu().flatten().tolist()
        metric.reset()
        if len(values) != len(PAIRS) or not all(math.isfinite(v) and v >= 0 for v in values):
            raise RuntimeError(f"Non-finite LPIPS: p{prompt_id:03d} frame {count - 1}")
        for (arm, ref), key, value in zip(PAIRS, pair_keys, values, strict=True):
            candidate = images[arm].astype(np.int32)
            reference = images[ref].astype(np.int32)
            rows[key]["squared_sum_uint8"] += int(np.square(candidate - reference).sum(dtype=np.int64))
            rows[key]["reference_squared_sum_uint8"] += int(np.square(reference).sum(dtype=np.int64))
            rows[key]["lpips_per_frame"].append(float(value))
        if count % 31 == 0:
            print(f"p{prompt_id:03d}: {count}/124 frames", flush=True)
    if count != 124:
        raise RuntimeError(f"Incomplete video: p{prompt_id:03d} has {count} frames")
    for row in rows.values():
        if row["reference_squared_sum_uint8"] <= 0:
            raise RuntimeError("Zero reference energy")
        row["rgb_nmse"] = row["squared_sum_uint8"] / row["reference_squared_sum_uint8"]
        row["lpips_alex"] = float(np.mean(row["lpips_per_frame"]))
    report = {
        "format": "e124-video-lpips-nmse-v1", "prompt_id": prompt_id,
        "seed": case["seed"], "frames": count,
        "scope": "Same-prompt, same-seed, frame-index paired decoded RGB8 MP4; 1024x576, 124 full-resolution frames; audio excluded",
        "rgb_nmse_definition": "sum((candidate-reference)^2)/sum(reference^2) over all frames, pixels and RGB channels",
        "lpips_definition": "TorchMetrics pretrained AlexNet; FP32 normalized [0,1] RGB; mean of per-frame scores; lower means closer, not necessarily better quality",
        "source_video": {arm: str(videos[arm]) for arm in ARMS},
        "source_sha256": {arm: case["arms"][arm]["video_sha256"] for arm in ARMS},
        "metric_state_sha256": metric_hash, "pairs": rows,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = json.loads((RESULTS / "summary.json").read_text())
    if (source.get("format") != "e124-h3-attention-12prompt-summary-v1" or
            len(source.get("cases", [])) != 12):
        raise RuntimeError("Unexpected E124 summary")
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none").cuda().eval()
    metric_hash = state_hash(metric)
    reports = []
    for case in source["cases"]:
        reports.append(evaluate_case(case, metric, metric_hash))
    means = {key: {
        "rgb_nmse_mean_of_prompts": float(np.mean([r["pairs"][key]["rgb_nmse"] for r in reports])),
        "lpips_alex_mean_of_prompts": float(np.mean([r["pairs"][key]["lpips_alex"] for r in reports])),
        "rgb_nmse_pooled": (sum(r["pairs"][key]["squared_sum_uint8"] for r in reports) /
                            sum(r["pairs"][key]["reference_squared_sum_uint8"] for r in reports)),
    } for key in reports[0]["pairs"]}
    result = {
        "format": "e124-video-lpips-nmse-summary-v1", "cases": len(reports),
        "frames_per_case": 124, "metric_state_sha256": metric_hash,
        "environment": {"torch": torch.__version__, "torchmetrics": torchmetrics.__version__,
                        "gpu": torch.cuda.get_device_name(args.gpu)},
        "mean_of_prompt_metrics": means,
        "per_prompt": [{"prompt_id": r["prompt_id"], "seed": r["seed"],
                        "pairs": {key: {"rgb_nmse": row["rgb_nmse"],
                                         "lpips_alex": row["lpips_alex"]}
                                  for key, row in r["pairs"].items()}}
                       for r in reports],
    }
    destination = RESULTS / "video_lpips_nmse_summary.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(means, indent=2), flush=True)
    print(f"COMPLETE {destination}", flush=True)


if __name__ == "__main__":
    main()
