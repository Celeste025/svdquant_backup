#!/usr/bin/env python3
"""Validate and compare the E125 clap triplet, then render a side-by-side preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from e124_video_lpips_nmse import evaluate_case, state_hash
from render_e124_triplet_previews import render
from summarize_e113_native_videos import sha256


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E125"
DATA = Path("/data1/models/svdquant-wjq/research/20261023/E125_h3_clap_attention")
ARMS = ("bf16", "sage3_fixed", "split_mixed")


def nmse(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((candidate.float() - reference.float()).square().sum(dtype=torch.float64) /
                 reference.float().square().sum(dtype=torch.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads((REPORT / "manifest.json").read_text())
    case = {"prompt_id": 161, "seed": 38001610, "arms": {}}
    denoise = {}
    latents = {}
    for arm in ARMS:
        denoise_path = REPORT / "clap0" / f"{arm}_denoise.json"
        media_path = REPORT / "clap0" / f"{arm}_video.json"
        denoise[arm] = json.loads(denoise_path.read_text())
        media = json.loads(media_path.read_text())
        if (denoise[arm].get("status") != "denoise_complete" or
                len(denoise[arm].get("steps", [])) != 20 or
                media.get("frames") != 124 or
                media.get("denoise_report_sha256") != sha256(denoise_path) or
                (denoise[arm]["prompt_id"], denoise[arm]["seed"], denoise[arm]["prompt"]) !=
                (161, 38001610, "A person is clapping") or
                denoise[arm]["settings"] != manifest["settings"] or
                sha256(Path(denoise[arm]["terminal_latents"])) != denoise[arm]["terminal_sha256"]):
            raise RuntimeError(f"Incomplete or mismatched E125 arm: {arm}")
        case["arms"][arm] = {"video": media["video"], "video_sha256": media["sha256"]}
        latents[arm] = torch.load(denoise[arm]["terminal_latents"], map_location="cpu",
                                  weights_only=True)
    if len({json.dumps(row["initial_latent_sha256"], sort_keys=True)
            for row in denoise.values()}) != 1:
        raise RuntimeError("Initial latents differ across E125 arms")

    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none").cuda().eval()
    metrics = evaluate_case(case, metric, state_hash(metric), REPORT)
    preview = DATA / "previews/clap0_triplet_video_only.mp4"
    if not preview.exists():
        render(case, preview)
    report = {
        "format": "e125-h3-clap-attention-summary-v1", "prompt_id": 161,
        "seed": 38001610, "settings": manifest["settings"],
        "initial_latent_sha256": denoise["bf16"]["initial_latent_sha256"],
        "arms": {arm: {
            "video": case["arms"][arm]["video"],
            "video_sha256": case["arms"][arm]["video_sha256"],
            "dit_seconds_sum": float(np.sum([step["dit_seconds"] for step in denoise[arm]["steps"]])),
            "terminal_latent_nmse_vs_bf16": {
                kind: nmse(latents[arm][f"{kind}_latents"],
                           latents["bf16"][f"{kind}_latents"])
                for kind in ("video", "audio")},
        } for arm in ARMS},
        "video_metrics": {key: {"rgb_nmse": row["rgb_nmse"],
                                "lpips_alex": row["lpips_alex"]}
                          for key, row in metrics["pairs"].items()},
        "preview_video_only": str(preview), "preview_sha256": sha256(preview),
    }
    destination = REPORT / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
