#!/usr/bin/env python3
"""Build five-arm H3 intervention contact videos and honest rollout-distance summaries."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("bf16", "svdquant", "tail_restore", "rademacher_flat", "gaussian")
LABELS = ("BF16", "SVDQuant", "Restore >4σ", "Equal |error| ±", "Gaussian error")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height, float(stream.average_rate)) != (1024, 576, 24.0):
            raise RuntimeError(f"unexpected dimensions or frame rate: {path}")
        yield from (frame.to_ndarray(format="rgb24") for frame in container.decode(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("race", "clap0"), required=True)
    args = parser.parse_args()
    case = args.case
    research = ROOT / "results/research"
    old = json.loads((research / "E101" / case / "video_pixel_comparison.json").read_text())
    sources = {"bf16": Path(old["inputs"]["bf16"]),
               "svdquant": Path(old["inputs"]["dynamic"])}
    intervention_stats = {}
    for arm in ARMS[2:]:
        media = json.loads((research / "E105" / case / f"{arm}_video.json").read_text())
        denoise_path = research / "E105" / case / f"{arm}_denoise.json"
        denoise = json.loads(denoise_path.read_text())
        if (media["case"], media["arm"], media["frames"], media["audio_streams"]) != (case, arm, 124, 1):
            raise RuntimeError(f"invalid video metadata: {arm}")
        if sha256(denoise_path) != media["denoise_report_sha256"]:
            raise RuntimeError(f"denoise report changed after video encoding: {arm}")
        if denoise["status"] != "denoise_complete" or len(denoise["steps"]) != 20:
            raise RuntimeError(f"incomplete denoising report: {arm}")
        per_modality = {}
        for modality in ("video", "audio"):
            rows = []
            for step, entry in enumerate(denoise["steps"]):
                if entry["step"] != step:
                    raise RuntimeError(f"out-of-order step: {arm}/{step}")
                row = entry["modalities"][modality]
                if not all(math.isfinite(row[key]) and row[key] > 0 for key in
                           ("target_nmse", "realized_nmse", "target_mse", "realized_mse")):
                    raise RuntimeError(f"invalid error metric: {arm}/{step}/{modality}")
                change = row["realized_mse"] / row["target_mse"] - 1
                if (arm == "tail_restore" and change > 1e-7) or (arm != "tail_restore" and abs(change) > 0.0051):
                    raise RuntimeError(f"intervention failed energy check: {arm}/{step}/{modality}")
                rows.append(row)
            per_modality[modality] = {
                "target_nmse_mean": float(np.mean([r["target_nmse"] for r in rows])),
                "realized_nmse_mean": float(np.mean([r["realized_nmse"] for r in rows])),
                "relative_mse_change_mean": float(np.mean([r["relative_mse_change"] for r in rows])),
                "relative_mse_change_range": [float(min(r["relative_mse_change"] for r in rows)),
                                              float(max(r["relative_mse_change"] for r in rows))],
            }
            if arm == "tail_restore":
                per_modality[modality]["repaired_fraction_mean"] = float(np.mean(
                    [r["repaired_fraction"] for r in rows]))
                per_modality[modality]["removed_error_energy_share_mean"] = float(np.mean(
                    [r["removed_error_energy_share"] for r in rows]))
        intervention_stats[arm] = per_modality
        sources[arm] = Path(media["video"])
        if sha256(sources[arm]) != media["sha256"]:
            raise RuntimeError(f"intervention media hash mismatch: {arm}")
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError("missing one of the five case videos")
    output_dir = Path("/data1/models/svdquant-wjq/research/20261016/E105_h3_output_interventions") / case
    montage = output_dir / f"{case}_five_arm_montage.mp4"
    poster = research / "E105" / case / "five_arm_poster.jpg"
    report = research / "E105" / case / "five_arm_summary.json"
    if any(path.exists() for path in (montage, poster, report)):
        raise FileExistsError("refusing to overwrite montage/poster/report")
    output_dir.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 19) if font_path.is_file() else ImageFont.load_default()
    w, h, band = 512, 288, 34
    canvas_size = (3 * w, 2 * (h + band))
    totals = {arm: {"abs": 0, "squared": 0} for arm in ARMS[1:]}
    ref_squared = 0
    captured = []
    output = av.open(str(montage), mode="w")
    stream = output.add_stream("libx264", rate=24)
    stream.width, stream.height = canvas_size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    count = 0
    try:
        for count, frame_set in enumerate(zip_longest(*(frames(sources[arm]) for arm in ARMS)), start=1):
            if any(frame is None for frame in frame_set) or count > 124:
                raise RuntimeError(f"mismatched or excess frame at index {count - 1}")
            ref = frame_set[0].astype(np.int16)
            ref_squared += int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
            for arm, frame in zip(ARMS[1:], frame_set[1:], strict=True):
                diff = frame.astype(np.int16) - ref
                totals[arm]["abs"] += int(np.abs(diff).sum(dtype=np.int64))
                totals[arm]["squared"] += int(np.square(diff.astype(np.int32)).sum(dtype=np.int64))
            canvas = Image.new("RGB", canvas_size, (24, 24, 24))
            draw = ImageDraw.Draw(canvas)
            for index, (label, frame) in enumerate(zip(LABELS, frame_set, strict=True)):
                x, y = (index % 3) * w, (index // 3) * (h + band)
                canvas.paste(Image.fromarray(frame).resize((w, h), Image.Resampling.BILINEAR),
                             (x, y + band))
                draw.text((x + 10, y + 6), label, font=font, fill="white")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                output.mux(packet)
            if count - 1 in (0, 30, 60, 90):
                captured.append(canvas)
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()
    if count != 124 or len(captured) != 4 or ref_squared <= 0:
        raise RuntimeError("incomplete five-arm montage")
    poster_canvas = Image.new("RGB", (canvas_size[0], canvas_size[1] * 4))
    for i, image in enumerate(captured):
        poster_canvas.paste(image, (0, i * canvas_size[1]))
    poster_canvas.save(poster, quality=88, optimize=True)
    denom = 124 * 576 * 1024 * 3
    proxies = {arm: {"rgb_mae": values["abs"] / denom,
                     "rgb_nmse": values["squared"] / ref_squared}
               for arm, values in totals.items()}
    payload = {"format": "h3-e105-five-arm-summary-v1", "case": case,
               "definition": "decoded H.264 RGB distance to separately generated BF16; free-rollout divergence, NOT perceptual quality",
               "sources": {arm: str(path.resolve()) for arm, path in sources.items()},
               "source_sha256": {arm: sha256(path) for arm, path in sources.items()},
               "frames": count, "fps": 24, "montage": str(montage.resolve()),
               "montage_sha256": sha256(montage), "montage_audio": "none; original arm videos contain audio",
               "poster": str(poster.resolve()), "poster_sha256": sha256(poster),
               "intervention_error_validation": intervention_stats,
               "rollout_distance_proxy": proxies}
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(proxies, indent=2, ensure_ascii=False))
    print(f"COMPLETE {montage} {poster} {report}")


if __name__ == "__main__":
    main()
