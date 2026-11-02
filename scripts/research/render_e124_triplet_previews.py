#!/usr/bin/env python3
"""Make synchronized, video-only BF16/Sage3/split previews for E124."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from summarize_e113_native_videos import frames, sha256


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/research/E124/summary.json"
ARMS = ("bf16", "sage3_fixed", "split_mixed")
LABELS = ("BF16", "Sage3 FP4 (mask fixed)", "Split mixed FP4/FP8")


def render(case: dict, destination: Path) -> None:
    prompt_id = case["prompt_id"]
    if destination.exists():
        raise FileExistsError(destination)
    sources = [Path(case["arms"][arm]["video"]) for arm in ARMS]
    for arm, path in zip(ARMS, sources, strict=True):
        if not path.is_file() or sha256(path) != case["arms"][arm]["video_sha256"]:
            raise RuntimeError(f"Source missing or hash mismatch: p{prompt_id:03d} {arm}")

    font_file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_file), 18) if font_file.is_file() else ImageFont.load_default()
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = av.open(str(destination), mode="w")
    stream = writer.add_stream("libx264", rate=24)
    stream.width, stream.height = 1536, 320
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    count = 0
    try:
        for count, image_set in enumerate(zip_longest(*(frames(path) for path in sources)), 1):
            if count > 124 or any(image is None or image.shape != (576, 1024, 3)
                                  for image in image_set):
                raise RuntimeError(f"Malformed frame: p{prompt_id:03d} #{count - 1}")
            canvas = Image.new("RGB", (1536, 320), (24, 28, 38))
            draw = ImageDraw.Draw(canvas)
            for index, (image, label) in enumerate(zip(image_set, LABELS, strict=True)):
                x = index * 512
                canvas.paste(Image.fromarray(image).resize((512, 288), Image.Resampling.BILINEAR),
                             (x, 32))
                draw.text((x + 10, 5), f"p{prompt_id:03d}  {label}", font=font,
                          fill=(245, 248, 255))
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(frame):
                writer.mux(packet)
        if count != 124:
            raise RuntimeError(f"Incomplete triplet: p{prompt_id:03d} {count} frames")
        for packet in stream.encode():
            writer.mux(packet)
    finally:
        writer.close()
    print(f"p{prompt_id:03d}: {count} frames -> {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-ids", nargs="+", type=int, default=[16, 76, 42])
    args = parser.parse_args()
    summary = json.loads(SUMMARY.read_text())
    cases = {case["prompt_id"]: case for case in summary["cases"]}
    data_dir = Path(summary["cases"][0]["arms"]["bf16"]["video"]).parents[2]
    for prompt_id in args.prompt_ids:
        render(cases[prompt_id], data_dir / "previews" / f"p{prompt_id:03d}_triplet_video_only.mp4")


if __name__ == "__main__":
    main()
