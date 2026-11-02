#!/usr/bin/env python3
"""Paired decoded-frame metrics for E131 against BF16 and E128 W4A4."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[2]
def frames(path):
    with av.open(str(path)) as c:
        yield from (frame.to_ndarray(format="rgb24") for frame in c.decode(video=0))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("p016", "vbench_000"), default="p016")
    args = parser.parse_args()
    case = args.case
    reports = [("bf16", "E124" if case == "p016" else "E126", "bf16"),
               ("w4a4", "E128", "w4a4_bf16_attention"),
               ("recompute10", "E131", "w4a4_bf16_attention")]
    torch.set_num_threads(4)
    paths = {}
    initial = []
    for label, exp, arm in reports:
        directory = ROOT / "results/research" / exp / case
        info = json.loads((directory / f"{arm}_video.json").read_text())
        paths[label] = Path(info["video"])
        denoise = json.loads((directory / f"{arm}_denoise.json").read_text())
        initial.append(denoise["initial_latent_sha256"])
    assert initial[0] == initial[1] == initial[2]
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True,
                                                    reduction="none").cuda().eval()
    pairs = [("w4a4", "bf16"), ("recompute10", "bf16"),
             ("recompute10", "w4a4")]
    accum = {f"{a}_vs_{b}": {"sse": 0, "energy": 0, "lpips": []} for a, b in pairs}
    count = 0
    for count, triple in enumerate(zip(*(frames(p) for p in paths.values()), strict=True), 1):
        images = dict(zip(paths, triple, strict=True))
        expected_shape = (576, 1024, 3) if case == "p016" else (768, 1344, 3)
        assert all(x.shape == expected_shape for x in images.values())
        for a, b in pairs:
            row = accum[f"{a}_vs_{b}"]
            x, y = images[a], images[b]
            delta = x.astype(np.int32) - y.astype(np.int32)
            row["sse"] += int(np.square(delta).sum(dtype=np.int64))
            row["energy"] += int(np.square(y.astype(np.int32)).sum(dtype=np.int64))
            xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32).div_(255)
            yt = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32).div_(255)
            row["lpips"].append(float(metric(xt, yt).item()))
            metric.reset()
    assert count == 124
    summary = {k: {"rgb_nmse": v["sse"] / v["energy"],
                   "lpips_alex": float(np.mean(v["lpips"]))} for k, v in accum.items()}
    result = {"format": "e131-video-metrics-v1", "case": case, "frames": count,
              "initial_latent_sha256": initial[0], "sources": {k: str(v) for k, v in paths.items()},
              "summary": summary}
    out = ROOT / "results/research/E131" / f"{case}_video_metrics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
