#!/usr/bin/env python3
"""Verify and summarize the 12-prompt, three-arm H3 attention generations."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import statistics

import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E124"
MANIFEST = REPORT / "manifest.json"


def nmse(x: torch.Tensor, y: torch.Tensor) -> float:
    a, b = x.float(), y.float()
    return float((a - b).square().sum(dtype=torch.float64) /
                 b.square().sum(dtype=torch.float64).clamp_min(1e-30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert sha256(Path(manifest["prompt_source"])) == manifest["prompt_source_sha256"]
    ids = [int(x) for group in manifest["groups"].values() for x in group]
    assert len(ids) == len(set(ids)) == 12
    prompts = {int(row["prompt_id"]): row for row in
               (json.loads(line) for line in
                Path(manifest["prompt_source"]).read_text().splitlines())}
    summary = {"format": "e124-h3-attention-12prompt-summary-v1",
               "prompt_source": manifest["prompt_source"],
               "manifest": str(MANIFEST), "cases": [], "missing": [],
               "status": "running"}
    for pid in ids:
        prompt = prompts[pid]
        case = f"p{pid:03d}"
        case_row = {"prompt_id": pid, "seed": int(prompt["seed"]),
                    "category": prompt["category"],
                    "source_duration": prompt["duration"], "arms": {}}
        loaded = {}
        for arm in manifest["arms"]:
            folder = REPORT / case
            denoise_path = folder / f"{arm}_denoise.json"
            video_path = folder / f"{arm}_video.json"
            if not denoise_path.is_file() or not video_path.is_file():
                summary["missing"].append(f"{case}/{arm}")
                continue
            denoise = json.loads(denoise_path.read_text())
            media = json.loads(video_path.read_text())
            assert denoise["status"] == "denoise_complete"
            assert denoise["prompt_id"] == pid and denoise["seed"] == int(prompt["seed"])
            assert denoise["prompt"] == prompt["prompt"]
            assert len(denoise["steps"]) == 20
            assert all(row["step"] == i and row["attention_main_calls"] == 50
                       and row["attention_refiner_calls"] == 2 and
                       math.isfinite(row["dit_seconds"])
                       for i, row in enumerate(denoise["steps"]))
            assert (media["frames"] == 124 and media["width"] == 1024 and
                    media["height"] == 576 and media["audio_streams"] == 1)
            path = Path(media["video"])
            assert path.is_file() and path.stat().st_size == media["bytes"]
            assert sha256(path) == media["sha256"]
            case_row["arms"][arm] = {
                "video": str(path), "video_bytes": media["bytes"],
                "video_sha256": media["sha256"],
                "initial_latent_sha256": denoise["initial_latent_sha256"],
                "denoise_seconds": denoise["completed_unix"] - denoise["started_unix"],
                "dit_seconds_sum": sum(row["dit_seconds"] for row in denoise["steps"]),
                "dit_seconds_mean": statistics.mean(row["dit_seconds"]
                                                      for row in denoise["steps"]),
            }
            loaded[arm] = torch.load(denoise["terminal_latents"],
                                     map_location="cpu", weights_only=True)
        if len(case_row["arms"]) == len(manifest["arms"]):
            first = case_row["arms"]["bf16"]["initial_latent_sha256"]
            assert all(row["initial_latent_sha256"] == first
                       for row in case_row["arms"].values())
            for arm in manifest["arms"]:
                case_row["arms"][arm]["terminal_latent_nmse_vs_bf16"] = {
                    name: nmse(loaded[arm][name + "_latents"],
                               loaded["bf16"][name + "_latents"])
                    for name in ("video", "audio")}
        summary["cases"].append(case_row)
    if not summary["missing"]:
        summary["status"] = "complete"
    path = REPORT / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "completed_cases":
                      sum(len(c["arms"]) == 3 for c in summary["cases"]),
                      "missing_arms": len(summary["missing"]),
                      "summary": str(path)}))


if __name__ == "__main__":
    main()
