#!/usr/bin/env python3
"""Validate, score, and render the paired E144 p025 text/detail videos."""
from __future__ import annotations

from itertools import zip_longest
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E144/p025"
BF16_REPORT = ROOT / "results/research/E124/p025"
ARM = "s70_fp4_20_int8_10"
OUT = Path("/data1/models/svdquant-wjq/research/20261101/E144_text_detail_draft/p025")


def frames(path: Path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


@torch.inference_mode()
def main() -> None:
    ref_d = json.loads((BF16_REPORT / "bf16_denoise.json").read_text())
    arm_d = json.loads((REPORT / f"{ARM}_denoise.json").read_text())
    ref_v = json.loads((BF16_REPORT / "bf16_video.json").read_text())
    arm_v = json.loads((REPORT / f"{ARM}_video.json").read_text())
    assert ref_d["prompt_id"] == arm_d["prompt_id"] == 25
    assert ref_d["seed"] == arm_d["seed"] == 2141
    assert ref_d["settings"] == arm_d["settings"]
    assert ref_d["initial_latent_sha256"] == arm_d["initial_latent_sha256"]
    assert len(ref_d["steps"]) == len(arm_d["steps"]) == 20
    assert arm_d["attention_router"]["dense_first_steps"] == 2
    assert arm_d["attention_router"]["dense_first_layers"] == 1
    assert arm_d["attention_router"]["sparse_config"]["sparsity_ratio"] == 0.7
    assert arm_d["attention_router"]["quant_config"]["nvfp4_ratio"] == 2 / 3
    assert arm_d["attention_router"]["quant_config"]["int8_ratio"] == 1 / 3
    paths = {"bf16": Path(ref_v["video"]), ARM: Path(arm_v["video"])}

    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).cuda().eval()
    sse = energy = 0
    lpips: list[float] = []
    selected: dict[int, dict[str, np.ndarray]] = {}
    for index, values in enumerate(zip_longest(*(frames(p) for p in paths.values()))):
        assert index < 124 and all(v is not None and v.shape == (576, 1024, 3) for v in values)
        ref, test = values
        delta = test.astype(np.int32) - ref.astype(np.int32)
        sse += int(np.square(delta).sum(dtype=np.int64))
        energy += int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
        ty = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
        tx = torch.from_numpy(test).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
        lpips.append(float(metric(tx, ty)))
        metric.reset()
        if index in (0, 31, 62, 93, 123):
            selected[index] = {"bf16": ref, ARM: test}
    assert len(lpips) == 124 and len(selected) == 5

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    sheet = Image.new("RGB", (1024, 5 * 320), "black")
    draw = ImageDraw.Draw(sheet)
    for row, (index, images) in enumerate(selected.items()):
        y = row * 320
        for col, (name, image) in enumerate(images.items()):
            x = col * 512
            sheet.paste(Image.fromarray(image).resize((512, 288)), (x, y + 32))
            label = "BF16" if name == "bf16" else "70% skip + 20% FP4 + 10% INT8"
            draw.text((x + 8, y + 5), f"frame {index}: {label}", font=small, fill="white")
    sheet_path = REPORT / "five_timepoints.jpg"
    sheet.save(sheet_path, quality=94)

    comparison = OUT / "p025_bf16_vs_draft20fp4_10int8.mp4"
    assert not comparison.exists(), comparison
    containers = [av.open(str(path)) for path in paths.values()]
    try:
        with av.open(str(comparison), "w", options={"movflags": "+faststart"}) as output:
            stream = output.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = 1024, 320, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            count = 0
            for count, pair in enumerate(zip_longest(*(c.decode(video=0) for c in containers)), 1):
                assert count <= 124 and all(frame is not None for frame in pair)
                canvas = Image.new("RGB", (1024, 320), "black")
                painter = ImageDraw.Draw(canvas)
                for col, (frame, label) in enumerate(zip(pair, ("BF16", "70% skip + 20% FP4 + 10% INT8"))):
                    canvas.paste(frame.to_image().resize((512, 288)), (col * 512, 32))
                    painter.text((col * 512 + 8, 5), label, font=font, fill="white")
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    output.mux(packet)
            assert count == 124
            for packet in stream.encode():
                output.mux(packet)
    finally:
        for container in containers:
            container.close()

    result = {
        "format": "e144-text-detail-evaluation-v1",
        "prompt_id": 25,
        "seed": 2141,
        "initial_latent_sha256": ref_d["initial_latent_sha256"],
        "schedule": {
            "dense_first_steps": 2,
            "dense_first_layers": 1,
            "expected_sparse_main_calls": 18 * 49,
            "skip_fraction": 0.7,
            "all_interaction_nvfp4_fraction": 0.2,
            "all_interaction_int8_fraction": 0.1,
        },
        "sources": {name: str(path) for name, path in paths.items()},
        "metrics": {
            "rgb_nmse": sse / energy,
            "lpips_alex_mean": float(np.mean(lpips)),
            "lpips_alex_median": float(np.median(lpips)),
            "lpips_alex_max": float(np.max(lpips)),
            "lpips_alex_per_frame": lpips,
        },
        "dit_seconds": {
            "bf16": float(sum(step["dit_seconds"] for step in ref_d["steps"])),
            ARM: float(sum(step["dit_seconds"] for step in arm_d["steps"])),
        },
        "comparison_video": str(comparison),
        "comparison_audio": "none; individual originals retain audio",
        "five_timepoints": str(sheet_path),
    }
    (REPORT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "metrics": {k: v for k, v in result["metrics"].items() if k != "lpips_alex_per_frame"}}, indent=2))


if __name__ == "__main__":
    main()
