#!/usr/bin/env python3
"""Render synchronized video-only three-arm previews for E126."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from summarize_e126_h3_vbench5_attention import ARMS, REPORT, frames, sha256


DATA = Path("/data1/models/svdquant-wjq/research/20261023/E126_h3_vbench5_attention")
LABELS = ("BF16 SDPA", "Sage3 FP4 (mask fixed)", "Split mixed FP4/FP8")


def render(case: str) -> Path:
    sources = {}
    for arm in ARMS:
        meta = json.loads((REPORT / case / f"{arm}_video.json").read_text())
        path = Path(meta["video"])
        if not path.is_file() or sha256(path) != meta["sha256"] or meta["frames"] != 124:
            raise RuntimeError(f"Invalid video: {case}/{arm}")
        sources[arm] = path
    destination = DATA / "previews" / f"{case}_triplet_video_only.mp4"
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    font_file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_file), 16) if font_file.is_file() else ImageFont.load_default()
    writer = av.open(str(destination), "w")
    stream = writer.add_stream("libx264", rate=24)
    stream.width, stream.height = 1344, 280
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    count = 0
    try:
        for count, images in enumerate(zip_longest(*(frames(sources[arm]) for arm in ARMS)), 1):
            if count > 124 or any(image is None or image.shape != (768, 1344, 3)
                                  for image in images):
                raise RuntimeError(f"Bad frame: {case} #{count - 1}")
            canvas = Image.new("RGB", (1344, 280), (24, 28, 38))
            draw = ImageDraw.Draw(canvas)
            for index, (image, label) in enumerate(zip(images, LABELS, strict=True)):
                x = index * 448
                canvas.paste(Image.fromarray(image).resize((448, 256), Image.Resampling.BILINEAR),
                             (x, 24))
                draw.text((x + 8, 3), f"{case}: {label}", font=font,
                          fill=(245, 248, 255))
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(frame):
                writer.mux(packet)
        if count != 124:
            raise RuntimeError(f"Incomplete video: {case} {count} frames")
        for packet in stream.encode():
            writer.mux(packet)
    finally:
        writer.close()
    print(f"{case}: {count} preview frames -> {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Single case ID; omit for all five")
    args = parser.parse_args()
    manifest = json.loads((REPORT / "manifest.json").read_text())
    cases = [row["id"] for row in manifest["prompts"]]
    if args.case:
        if args.case not in cases:
            raise ValueError(args.case)
        cases = [args.case]
    for case in cases:
        render(case)


if __name__ == "__main__":
    main()
