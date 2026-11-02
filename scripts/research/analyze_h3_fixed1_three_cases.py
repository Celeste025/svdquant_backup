#!/usr/bin/env python3
"""E101: decoded-video error and labelled BF16/dynamic/fixed triplets.

These are free-rollout H.264 RGB distances, not teacher-forced DiT NMSE.
"""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def video_frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (1024, 576):
            raise ValueError(f"Unexpected dimensions in {path}: {stream.width}x{stream.height}")
        yield from (frame.to_ndarray(format="rgb24") for frame in container.decode(stream))


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("bf16", "dynamic", "fixed", "output", "poster", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.poster.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite a montage, poster, or report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    width, height, band, expected_frames = 1024, 576, 40, 124
    labels = ("BF16 original", "SVDQuant dynamic", "SVDQuant fixed scale=1")
    totals = {arm: {"l1": 0, "l2": 0} for arm in ("dynamic", "fixed")}
    energy = 0
    per_frame = []
    label_font = font(22)
    poster_rows = []
    output = av.open(str(args.output), mode="w")
    stream = output.add_stream("libx264", rate=24)
    stream.width = width * 3
    stream.height = height + band
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    try:
        for index, frame_set in enumerate(zip_longest(video_frames(args.bf16), video_frames(args.dynamic),
                                                     video_frames(args.fixed))):
            if any(frame is None for frame in frame_set):
                raise RuntimeError(f"Frame count mismatch at frame {index}")
            ref, dynamic, fixed = frame_set
            if any(frame.shape != (height, width, 3) for frame in frame_set):
                raise RuntimeError(f"Frame shape mismatch at frame {index}")
            energy_frame = int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
            energy += energy_frame
            item = {"frame": index}
            for arm, frame in (("dynamic", dynamic), ("fixed", fixed)):
                diff = frame.astype(np.int16) - ref.astype(np.int16)
                l1 = int(np.abs(diff).sum(dtype=np.int64))
                l2 = int(np.square(diff.astype(np.int32)).sum(dtype=np.int64))
                totals[arm]["l1"] += l1
                totals[arm]["l2"] += l2
                item[arm] = {"mae": l1 / diff.size, "mse": l2 / diff.size,
                             "nmse": l2 / energy_frame if energy_frame else None}
            per_frame.append(item)
            canvas = Image.new("RGB", (width * 3, height + band), (20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            for col, (label, frame) in enumerate(zip(labels, frame_set)):
                canvas.paste(Image.fromarray(frame), (col * width, band))
                draw.text((col * width + 14, 5), f"{label} | {args.case} | seed {args.seed}",
                          font=label_font, fill="white")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                output.mux(packet)
            if index in (0, 30, 60, 90):
                poster_rows.append(canvas.resize((width * 3 // 2, (height + band) // 2)))
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()
    if len(per_frame) != expected_frames or energy <= 0:
        raise RuntimeError(f"Expected {expected_frames} valid frames, got {len(per_frame)}")
    if len(poster_rows) != 4:
        raise RuntimeError("Incomplete poster frame selection")
    poster = Image.new("RGB", (poster_rows[0].width, sum(row.height for row in poster_rows)))
    top = 0
    for row in poster_rows:
        poster.paste(row, (0, top))
        top += row.height
    poster.save(args.poster, optimize=True)

    summary = {}
    for arm, sums in totals.items():
        mse = sums["l2"] / (expected_frames * width * height * 3)
        summary[arm] = {"mae": sums["l1"] / (expected_frames * width * height * 3),
                        "mse": mse, "nmse": sums["l2"] / energy,
                        "psnr_db": 10 * math.log10(255**2 / mse) if mse else None}
    payload = {"case": args.case, "seed": args.seed,
               "definition": "Pooled decoded H.264 RGB difference against separately generated BF16; "
                             "free-rollout visual distance, not teacher-forced DiT-output NMSE",
               "frames": expected_frames, "video_shape": [height, width, 3], "fps": 24,
               "inputs": {name: str(getattr(args, name)) for name in ("bf16", "dynamic", "fixed")},
               "montage": str(args.output), "poster": str(args.poster),
               "montage_audio": "none; use original MP4s for audio",
               "summary": summary, "per_frame": per_frame}
    args.report.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
