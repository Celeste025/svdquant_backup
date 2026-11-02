#!/usr/bin/env python3
"""Validate E126 videos and measure frame-paired RGB NMSE and LPIPS-Alex."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

import av
import numpy as np
import torch
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E126"
ARMS = ("bf16", "sage3_fixed", "split_mixed")
PAIRS = (("sage3_fixed", "bf16"), ("split_mixed", "bf16"),
         ("split_mixed", "sage3_fixed"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height, float(stream.average_rate)) != (1344, 768, 24.0):
            raise RuntimeError(f"Unexpected media format: {path}")
        yield from (frame.to_ndarray(format="rgb24") for frame in container.decode(stream))


def state_hash(metric: LearnedPerceptualImagePatchSimilarity) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(metric.net.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def latent_nmse(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = candidate.float(), reference.float()
    return float((a - b).square().sum(dtype=torch.float64) /
                 b.square().sum(dtype=torch.float64))


@torch.inference_mode()
def evaluate_case(row: dict, manifest: dict,
                  metric: LearnedPerceptualImagePatchSimilarity,
                  weights_sha256: str) -> dict:
    case = row["id"]
    result_path = REPORT / f"video_metrics_{case}.json"
    denoise = {}
    media = {}
    for arm in ARMS:
        report_path = REPORT / case / f"{arm}_denoise.json"
        video_path = REPORT / case / f"{arm}_video.json"
        denoise[arm] = json.loads(report_path.read_text())
        media[arm] = json.loads(video_path.read_text())
        d, m = denoise[arm], media[arm]
        if (d.get("status") != "denoise_complete" or len(d.get("steps", [])) != 50 or
                (d.get("case"), d.get("prompt"), d.get("seed"), d.get("arm")) !=
                (case, row["prompt"], 42, arm) or
                d["settings"] != manifest["settings"] or
                m.get("frames") != 124 or m.get("width") != 1344 or
                m.get("height") != 768 or m.get("audio_streams") != 1 or
                m.get("denoise_report_sha256") != sha256(report_path) or
                sha256(Path(d["terminal_latents"])) != d["terminal_sha256"] or
                sha256(Path(m["video"])) != m["sha256"] or
                any(step["step"] != index or step["attention_main_calls"] != 50 or
                    step["attention_refiner_calls"] != 2 or
                    not math.isfinite(step["dit_seconds"])
                    for index, step in enumerate(d["steps"]))):
            raise RuntimeError(f"E126 arm incomplete or modified: {case}/{arm}")
    if len({json.dumps(d["initial_latent_sha256"], sort_keys=True)
            for d in denoise.values()}) != 1:
        raise RuntimeError(f"Initial latents do not match: {case}")
    if len({d["steps"][0]["main_tokens"] for d in denoise.values()}) != 1:
        raise RuntimeError(f"Attention token count differs across arms: {case}")
    paths = {arm: Path(media[arm]["video"]) for arm in ARMS}
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (existing.get("format") != "e126-h3-vbench5-video-metrics-v1" or
                existing.get("metric_state_sha256") != weights_sha256 or
                any(existing["source_sha256"][arm] != media[arm]["sha256"] for arm in ARMS)):
            raise RuntimeError(f"Existing metrics mismatch: {case}")
        return existing

    pair_keys = [f"{a}_vs_{b}" for a, b in PAIRS]
    metrics = {key: {"squared_sum_uint8": 0, "reference_squared_sum_uint8": 0,
                     "lpips_per_frame": []} for key in pair_keys}
    frame_count = 0
    for frame_count, image_set in enumerate(zip_longest(*(frames(paths[a]) for a in ARMS)), 1):
        if frame_count > 124 or any(image is None or image.shape != (768, 1344, 3)
                                    for image in image_set):
            raise RuntimeError(f"Malformed frame: {case} #{frame_count - 1}")
        images = dict(zip(ARMS, image_set, strict=True))
        candidates = torch.from_numpy(np.stack([images[a] for a, _ in PAIRS]))
        references = torch.from_numpy(np.stack([images[b] for _, b in PAIRS]))
        candidates = candidates.permute(0, 3, 1, 2).to("cuda", torch.float32).div_(255)
        references = references.permute(0, 3, 1, 2).to("cuda", torch.float32).div_(255)
        values = metric(candidates, references).detach().cpu().flatten().tolist()
        metric.reset()
        if len(values) != len(PAIRS) or not all(math.isfinite(v) and v >= 0 for v in values):
            raise RuntimeError(f"Invalid LPIPS: {case} #{frame_count - 1}")
        for (arm, ref), key, value in zip(PAIRS, pair_keys, values, strict=True):
            x, y = images[arm].astype(np.int32), images[ref].astype(np.int32)
            metrics[key]["squared_sum_uint8"] += int(np.square(x - y).sum(dtype=np.int64))
            metrics[key]["reference_squared_sum_uint8"] += int(np.square(y).sum(dtype=np.int64))
            metrics[key]["lpips_per_frame"].append(float(value))
        if frame_count % 31 == 0:
            print(f"{case}: {frame_count}/124 video-metric frames", flush=True)
    if frame_count != 124:
        raise RuntimeError(f"Incomplete video: {case} {frame_count} frames")
    for value in metrics.values():
        value["rgb_nmse"] = value["squared_sum_uint8"] / value["reference_squared_sum_uint8"]
        value["lpips_alex"] = float(np.mean(value["lpips_per_frame"]))
    latents = {arm: torch.load(denoise[arm]["terminal_latents"], map_location="cpu",
                               weights_only=True) for arm in ARMS}
    result = {
        "format": "e126-h3-vbench5-video-metrics-v1", "case": case,
        "prompt": row["prompt"], "seed": 42, "frames": frame_count,
        "main_tokens": denoise["bf16"]["steps"][0]["main_tokens"],
        "initial_latent_sha256": denoise["bf16"]["initial_latent_sha256"],
        "source_video": {arm: str(paths[arm]) for arm in ARMS},
        "source_sha256": {arm: media[arm]["sha256"] for arm in ARMS},
        "metric_state_sha256": weights_sha256,
        "definition": "Frame-index paired full-resolution RGB8; RGB NMSE=sum squared pixel error / BF16 pixel energy; LPIPS-Alex=mean TorchMetrics AlexNet score on normalized [0,1] frames; audio excluded; smaller is closer, not necessarily better quality",
        "pairs": metrics,
        "arms": {arm: {
            "dit_seconds_sum": float(sum(step["dit_seconds"] for step in denoise[arm]["steps"])),
            "peak_allocated_gib": denoise[arm]["peak_allocated_gib"],
            "terminal_latent_nmse_vs_bf16": {
                name: latent_nmse(latents[arm][f"{name}_latents"],
                                  latents["bf16"][f"{name}_latents"])
                for name in ("video", "audio")},
        } for arm in ARMS},
    }
    temporary = result_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((REPORT / "manifest.json").read_text())
    if manifest.get("format") != "e126-h3-vbench5-attention-manifest-v1":
        raise RuntimeError("Invalid E126 manifest")
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none").cuda().eval()
    weights_sha256 = state_hash(metric)
    cases = [evaluate_case(row, manifest, metric, weights_sha256)
             for row in manifest["prompts"]]
    summary = {
        "format": "e126-h3-vbench5-summary-v1", "cases": len(cases),
        "frames_per_case": 124, "settings": manifest["settings"],
        "metric_state_sha256": weights_sha256,
        "environment": {"torch": torch.__version__,
                        "torchmetrics": torchmetrics.__version__,
                        "gpu": torch.cuda.get_device_name(args.gpu)},
        "per_case": [{"case": case["case"], "prompt": case["prompt"],
                      "main_tokens": case["main_tokens"],
                      "arms": case["arms"],
                      "pairs": {key: {"rgb_nmse": value["rgb_nmse"],
                                       "lpips_alex": value["lpips_alex"]}
                                for key, value in case["pairs"].items()}}
                     for case in cases],
        "mean_of_case_metrics": {
            key: {"rgb_nmse": float(np.mean([case["pairs"][key]["rgb_nmse"]
                                               for case in cases])),
                  "lpips_alex": float(np.mean([case["pairs"][key]["lpips_alex"]
                                                for case in cases]))}
            for key in cases[0]["pairs"]},
        "mean_dit_seconds_sum": {arm: float(np.mean([case["arms"][arm]["dit_seconds_sum"]
                                                      for case in cases])) for arm in ARMS},
        "mean_terminal_latent_nmse_vs_bf16": {
            arm: {name: float(np.mean([
                case["arms"][arm]["terminal_latent_nmse_vs_bf16"][name]
                for case in cases])) for name in ("video", "audio")}
            for arm in ARMS},
    }
    destination = REPORT / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")
    print(json.dumps({"mean_of_case_metrics": summary["mean_of_case_metrics"],
                      "mean_dit_seconds_sum": summary["mean_dit_seconds_sum"]},
                     indent=2), flush=True)
    print(f"COMPLETE {destination}", flush=True)


if __name__ == "__main__":
    main()
