#!/usr/bin/env python3
"""Score and render BF16/mixed/pure-INT8/pure-NVFP4 p025 videos."""
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
REPORT = ROOT / "results/research/E145/p025"
E124 = ROOT / "results/research/E124/p025"
E144 = ROOT / "results/research/E144/p025"
DATA = Path("/data1/models/svdquant-wjq/research/20261101/E145_text_detail_pure_precision/p025")
ARMS = ("s70_int8_30", "s70_nvfp4_30")


def frames(path: Path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


@torch.inference_mode()
def main() -> None:
    ref_d = json.loads((E124 / "bf16_denoise.json").read_text())
    mixed_d = json.loads((E144 / "s70_fp4_20_int8_10_denoise.json").read_text())
    reports = {arm: json.loads((REPORT / f"{arm}_denoise.json").read_text()) for arm in ARMS}
    for report in (mixed_d, *reports.values()):
        assert report["prompt_id"] == 25 and report["seed"] == 2141
        assert report["settings"] == ref_d["settings"]
        assert report["initial_latent_sha256"] == ref_d["initial_latent_sha256"]
        assert report["attention_router"]["dense_first_steps"] == 2
        assert report["attention_router"]["dense_first_layers"] == 1
    paths = {
        "bf16": Path(json.loads((E124 / "bf16_video.json").read_text())["video"]),
        "mixed20_10": Path(json.loads((E144 / "s70_fp4_20_int8_10_video.json").read_text())["video"]),
        **{arm: Path(json.loads((REPORT / f"{arm}_video.json").read_text())["video"]) for arm in ARMS},
    }

    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).cuda().eval()
    accum = {name: {"sse": 0, "energy": 0, "lpips": []} for name in paths if name != "bf16"}
    selected: dict[int, dict[str, np.ndarray]] = {}
    for index, values in enumerate(zip_longest(*(frames(path) for path in paths.values()))):
        assert index < 124 and all(v is not None and v.shape == (576, 1024, 3) for v in values)
        images = dict(zip(paths, values))
        ref = images["bf16"]
        ty = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
        for name, row in accum.items():
            delta = images[name].astype(np.int32) - ref.astype(np.int32)
            row["sse"] += int(np.square(delta).sum(dtype=np.int64))
            row["energy"] += int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
            tx = torch.from_numpy(images[name]).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
            row["lpips"].append(float(metric(tx, ty)))
            metric.reset()
        if index in (0, 31, 62, 93, 123):
            selected[index] = images
    assert all(len(row["lpips"]) == 124 for row in accum.values())

    labels = {
        "bf16": "BF16",
        "mixed20_10": "20% FP4 + 10% INT8",
        "s70_int8_30": "30% INT8",
        "s70_nvfp4_30": "30% NVFP4",
    }
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    sheet = Image.new("RGB", (1536, 5 * 248), "black")
    painter = ImageDraw.Draw(sheet)
    for row, (index, images) in enumerate(selected.items()):
        y = row * 248
        for col, (name, image) in enumerate(images.items()):
            x = col * 384
            sheet.paste(Image.fromarray(image).resize((384, 216)), (x, y + 32))
            painter.text((x + 6, y + 6), f"f{index} {labels[name]}", font=small, fill="white")
    sheet_path = REPORT / "four_arm_five_timepoints.jpg"
    sheet.save(sheet_path, quality=95)

    comparison = DATA / "p025_bf16_mixed_int8_nvfp4.mp4"
    assert not comparison.exists(), comparison
    containers = [av.open(str(path)) for path in paths.values()]
    try:
        with av.open(str(comparison), "w", options={"movflags": "+faststart"}) as output:
            stream = output.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = 1536, 248, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            count = 0
            for count, group in enumerate(zip_longest(*(c.decode(video=0) for c in containers)), 1):
                assert count <= 124 and all(frame is not None for frame in group)
                canvas = Image.new("RGB", (1536, 248), "black")
                draw = ImageDraw.Draw(canvas)
                for col, (name, frame) in enumerate(zip(paths, group)):
                    canvas.paste(frame.to_image().resize((384, 216)), (col * 384, 32))
                    draw.text((col * 384 + 6, 6), labels[name], font=small, fill="white")
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    output.mux(packet)
            assert count == 124
            for packet in stream.encode():
                output.mux(packet)
    finally:
        for container in containers:
            container.close()

    summary = {
        name: {
            "rgb_nmse": row["sse"] / row["energy"],
            "lpips_alex_mean": float(np.mean(row["lpips"])),
            "lpips_alex_median": float(np.median(row["lpips"])),
            "lpips_alex_max": float(np.max(row["lpips"])),
        }
        for name, row in accum.items()
    }
    result = {
        "format": "e145-text-detail-pure-precision-v1",
        "prompt_id": 25,
        "seed": 2141,
        "initial_latent_sha256": ref_d["initial_latent_sha256"],
        "sources": {name: str(path) for name, path in paths.items()},
        "summary_vs_bf16": summary,
        "dit_seconds": {
            "bf16": float(sum(x["dit_seconds"] for x in ref_d["steps"])),
            "mixed20_10": float(sum(x["dit_seconds"] for x in mixed_d["steps"])),
            **{name: float(sum(x["dit_seconds"] for x in report["steps"])) for name, report in reports.items()},
        },
        "prefix_note": "pure NVFP4 video phase uses INT8 prefix K/V and BF16 prefix queries; other quantized arms use INT8 prefix K/V and queries",
        "comparison_video": str(comparison),
        "comparison_audio": "none; individual originals retain audio",
        "five_timepoints": str(sheet_path),
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
