#!/usr/bin/env python3
"""Frame-paired RGB NMSE and LPIPS-Alex for E129 vs BF16 and E128 plain W4A4."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

import av
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from finalize_e129_h3_svdquant_attention import ARM, E124, E126, E129, ROOT, sha256

E128 = ROOT / "results/research/E128"
PAIRS = (("svdquant", "bf16"), ("plain_w4a4", "bf16"),
         ("svdquant", "plain_w4a4"))


def frames(path: Path, height: int, width: int):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height, float(stream.average_rate)) != (width, height, 24.0):
            raise RuntimeError(f"video format mismatch: {path}")
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def source(case: str, arm: str) -> tuple[Path, str]:
    folder = ((E124 if case.startswith("p") else E126) if arm == "bf16"
              else E128 if arm == "plain_w4a4" else E129)
    report_arm = "w4a4_bf16_attention" if arm == "plain_w4a4" else ARM if arm == "svdquant" else "bf16"
    info = json.loads((folder / case / f"{report_arm}_video.json").read_text())
    path = Path(info["video"])
    if not path.is_file() or sha256(path) != info["sha256"]:
        raise RuntimeError(f"missing/hash-mismatch video: {case}/{arm}")
    return path, info["sha256"]


@torch.inference_mode()
def evaluate(case: str, metric: LearnedPerceptualImagePatchSimilarity) -> dict:
    output = E129 / case / "video_metrics.json"
    sources = {arm: source(case, arm) for arm in ("bf16", "plain_w4a4", "svdquant")}
    source_hashes = {arm: digest for arm, (_, digest) in sources.items()}
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("format") != "e129-svdquant-video-metrics-v1" or saved.get("source_sha256") != source_hashes:
            raise RuntimeError(f"existing metric result conflicts: {output}")
        print(f"SKIP validated metrics: {case}", flush=True)
        return saved
    height, width = (576, 1024) if case.startswith("p") else (768, 1344)
    rows = {f"{a}_vs_{b}": {"squared_error": 0, "reference_energy": 0,
                            "lpips_per_frame": []} for a, b in PAIRS}
    paths = {arm: path for arm, (path, _) in sources.items()}
    count = 0
    for count, frame_set in enumerate(zip_longest(*(frames(paths[a], height, width)
                                                    for a in paths)), start=1):
        if count > 124 or any(frame is None or frame.shape != (height, width, 3)
                              for frame in frame_set):
            raise RuntimeError(f"frame mismatch: {case}, frame {count - 1}")
        images = dict(zip(paths, frame_set, strict=True))
        for a, b in PAIRS:
            key = f"{a}_vs_{b}"
            candidate = images[a].astype(np.int32)
            reference = images[b].astype(np.int32)
            rows[key]["squared_error"] += int(np.square(candidate - reference).sum(dtype=np.int64))
            rows[key]["reference_energy"] += int(np.square(reference).sum(dtype=np.int64))
            x = torch.from_numpy(images[a]).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32).div_(255)
            y = torch.from_numpy(images[b]).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32).div_(255)
            value = float(metric(x, y).item())
            metric.reset()
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"invalid LPIPS: {case} frame {count - 1}")
            rows[key]["lpips_per_frame"].append(value)
        if count % 31 == 0:
            print(f"{case}: {count}/124 frames", flush=True)
    if count != 124:
        raise RuntimeError(f"incomplete video: {case}, {count} frames")
    summary = {key: {"rgb_nmse": row["squared_error"] / row["reference_energy"],
                     "lpips_alex": float(np.mean(row["lpips_per_frame"]))}
               for key, row in rows.items()}
    result = {"format": "e129-svdquant-video-metrics-v1", "case": case,
              "frames": count, "height": height, "width": width,
              "definition": "Matched-frame decoded RGB8 NMSE and TorchMetrics AlexNet LPIPS; smaller is closer to reference, not necessarily better perceptual quality",
              "source_sha256": source_hashes, "summary": summary, "details": rows}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"case": case, "summary": summary}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True,
                                                   reduction="none").cuda().eval()
    evaluate(args.case, metric)


if __name__ == "__main__":
    main()
