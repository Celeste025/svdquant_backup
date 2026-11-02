#!/usr/bin/env python3
"""Score and render p025 retained-20% DraftAttention2 variants."""
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
REPORT = ROOT / "results/research/E146/p025"
E124 = ROOT / "results/research/E124/p025"
E145 = ROOT / "results/research/E145/p025"
DATA = Path("/data1/models/svdquant-wjq/research/20261101/E146_text_detail_retained20/p025")


def frames(path: Path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


@torch.inference_mode()
def main() -> None:
    ref_d = json.loads((E124 / "bf16_denoise.json").read_text())
    int8_30_d = json.loads((E145 / "s70_int8_30_denoise.json").read_text())
    names = ("s80_int8_20", "s80_nvfp4_10_int8_10")
    reports = {name: json.loads((REPORT / f"{name}_denoise.json").read_text()) for name in names}
    for report in (int8_30_d, *reports.values()):
        assert report["prompt_id"] == 25 and report["seed"] == 2141
        assert report["settings"] == ref_d["settings"]
        assert report["initial_latent_sha256"] == ref_d["initial_latent_sha256"]
    paths = {
        "bf16": Path(json.loads((E124 / "bf16_video.json").read_text())["video"]),
        "int8_30": Path(json.loads((E145 / "s70_int8_30_video.json").read_text())["video"]),
        **{name: Path(json.loads((REPORT / f"{name}_video.json").read_text())["video"]) for name in names},
    }
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).cuda().eval()
    accum = {name: {"sse": 0, "energy": 0, "lpips": []} for name in paths if name != "bf16"}
    selected = {}
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
        "int8_30": "70 skip / 30 INT8",
        "s80_int8_20": "80 skip / 20 INT8",
        "s80_nvfp4_10_int8_10": "80 skip / 10 FP4 + 10 INT8",
    }
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    sheet = Image.new("RGB", (1536, 5 * 248), "black")
    draw = ImageDraw.Draw(sheet)
    for row, (index, images) in enumerate(selected.items()):
        y = row * 248
        for col, (name, image) in enumerate(images.items()):
            x = col * 384
            sheet.paste(Image.fromarray(image).resize((384, 216)), (x, y + 32))
            draw.text((x + 5, y + 6), f"f{index} {labels[name]}", font=font, fill="white")
    sheet_path = REPORT / "retained20_four_arm_five_timepoints.jpg"
    REPORT.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, quality=95)

    comparison = DATA / "p025_bf16_int8_30_int8_20_mix10_10.mp4"
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
                painter = ImageDraw.Draw(canvas)
                for col, (name, frame) in enumerate(zip(paths, group)):
                    canvas.paste(frame.to_image().resize((384, 216)), (col * 384, 32))
                    painter.text((col * 384 + 5, 6), labels[name], font=font, fill="white")
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
        "format": "e146-text-detail-retained20-v1",
        "prompt_id": 25,
        "seed": 2141,
        "initial_latent_sha256": ref_d["initial_latent_sha256"],
        "sources": {name: str(path) for name, path in paths.items()},
        "summary_vs_bf16": summary,
        "dit_seconds": {
            "bf16": float(sum(x["dit_seconds"] for x in ref_d["steps"])),
            "int8_30": float(sum(x["dit_seconds"] for x in int8_30_d["steps"])),
            **{name: float(sum(x["dit_seconds"] for x in report["steps"])) for name, report in reports.items()},
        },
        "comparison_video": str(comparison),
        "comparison_audio": "none; individual originals retain audio",
        "five_timepoints": str(sheet_path),
    }
    (REPORT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
