#!/usr/bin/env python3
"""Compare decoded 8-bit RGB frames from E100's matched-seed H3 videos."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

import av
import numpy as np


def frames(path: Path):
    with av.open(str(path)) as container:
        yield from (frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = []
    totals = {"dynamic": {"abs": 0, "sq": 0}, "fixed": {"abs": 0, "sq": 0}}
    pixels = 0
    for frame_id, triplet in enumerate(zip_longest(frames(args.bf16), frames(args.dynamic), frames(args.fixed))):
        if any(frame is None for frame in triplet):
            raise RuntimeError(f"video frame counts differ at frame {frame_id}")
        bf16, dynamic, fixed = triplet
        if bf16.shape != (480, 832, 3) or dynamic.shape != bf16.shape or fixed.shape != bf16.shape:
            raise RuntimeError(f"unexpected frame shape at {frame_id}: {[x.shape for x in triplet]}")
        item = {"frame": frame_id}
        for name, image in (("dynamic", dynamic), ("fixed", fixed)):
            diff = image.astype(np.int16) - bf16.astype(np.int16)
            abs_sum = int(np.abs(diff).sum(dtype=np.int64))
            sq_sum = int(np.square(diff.astype(np.int32)).sum(dtype=np.int64))
            totals[name]["abs"] += abs_sum
            totals[name]["sq"] += sq_sum
            mse = sq_sum / diff.size
            item[name] = {"mae": abs_sum / diff.size, "mse": mse,
                          "psnr_db": 10 * math.log10(255**2 / mse) if mse else None}
        rows.append(item)
        pixels += bf16.size
    if len(rows) != 124:
        raise RuntimeError(f"expected 124 decoded video frames, found {len(rows)}")

    summary = {}
    for name, sums in totals.items():
        mse = sums["sq"] / pixels
        summary[name] = {"mae": sums["abs"] / pixels, "mse": mse,
                         "psnr_db": 10 * math.log10(255**2 / mse) if mse else None}
    payload = {"definition": "decoded H.264 8-bit RGB difference against independently generated BF16 video; not DiT-output NMSE",
               "frames": len(rows), "shape": [480, 832, 3], "summary": summary, "per_frame": rows,
               "inputs": {"bf16": str(args.bf16), "dynamic": str(args.dynamic), "fixed": str(args.fixed)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
