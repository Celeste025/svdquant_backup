#!/usr/bin/env python3
"""Stack sampled frames of several videos (rows) x timestamps (columns) into one PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image


def frames_of(path: Path, idxs: list[int]) -> dict[int, Image.Image]:
    out: dict[int, Image.Image] = {}
    want = set(idxs)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in want:
                out[i] = frame.to_image()
                want.discard(i)
            if not want:
                break
    missing = set(idxs) - set(out)
    if missing:
        raise RuntimeError(f"{path}: missing frames {sorted(missing)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True,
                        help="label=path (repeatable; rows in order)")
    parser.add_argument("--frames", default="12,60,118")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    idxs = [int(x) for x in args.frames.split(",")]
    rows = []
    for item in args.video:
        label, _, path = item.partition("=")
        frames = frames_of(Path(path), idxs)
        rows.append((label, [frames[i].resize((args.width, args.height)) for i in idxs]))

    canvas = Image.new("RGB", (args.width * len(idxs), args.height * len(rows)), "white")
    for r, (label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            canvas.paste(img, (c * args.width, r * args.height))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"saved {args.out} ({canvas.size[0]}x{canvas.size[1]}) rows={[r[0] for r in rows]}")


if __name__ == "__main__":
    main()
