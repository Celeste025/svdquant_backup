#!/usr/bin/env python3
"""Validate E106 and render six-arm, time-aligned H3 comparisons."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from summarize_e105_h3_output_interventions import frames, sha256


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("bf16", "svdquant", "permuted_error", "tail_restore", "rademacher_flat", "gaussian")
LABELS = ("BF16", "SVDQuant", "Permuted real error", "Restore >4sigma", "Equal |error| +/-", "Gaussian error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("race", "clap0"), required=True)
    args = parser.parse_args()
    case = args.case
    research = ROOT / "results/research"
    old = json.loads((research / "E101" / case / "video_pixel_comparison.json").read_text())
    e105 = json.loads((research / "E105" / case / "five_arm_summary.json").read_text())
    denoise_path = research / "E106" / case / "permuted_error_denoise.json"
    video_path = research / "E106" / case / "permuted_error_video.json"
    denoise = json.loads(denoise_path.read_text())
    media = json.loads(video_path.read_text())
    if denoise["format"] != "h3-e106-error-permutation-denoise-v1" or denoise["status"] != "denoise_complete":
        raise RuntimeError("incomplete or invalid E106 denoising")
    if (denoise["case"], denoise["arm"], len(denoise["steps"])) != (case, "permuted_error", 20):
        raise RuntimeError("unexpected E106 case, arm, or step count")
    if (media["format"], media["case"], media["arm"], media["frames"], media["audio_streams"]) != (
        "h3-e106-error-permutation-video-v1", case, "permuted_error", 124, 1
    ) or media["denoise_report_sha256"] != sha256(denoise_path):
        raise RuntimeError("invalid E106 media metadata or denoising report hash")
    per_modality = {}
    for modality in ("video", "audio"):
        rows = []
        for step, entry in enumerate(denoise["steps"]):
            if entry["step"] != step:
                raise RuntimeError("out-of-order E106 step")
            row = entry["modalities"][modality]
            if (not row["fp32_values_preserved_before_bf16_cast"] or
                not all(math.isfinite(row[key]) and row[key] > 0 for key in
                        ("target_mse", "realized_mse", "target_nmse", "realized_nmse")) or
                abs(row["relative_mse_change"]) > 0.0051 or
                row["bf16_rounding_distortion_over_target_mse"] > 0.0051):
                raise RuntimeError(f"failed E106 numerical gate: {step}/{modality}")
            rows.append(row)
        per_modality[modality] = {
            "target_nmse_mean": float(np.mean([r["target_nmse"] for r in rows])),
            "realized_nmse_mean": float(np.mean([r["realized_nmse"] for r in rows])),
            "max_absolute_relative_mse_change": float(max(abs(r["relative_mse_change"]) for r in rows)),
            "max_bf16_rounding_distortion_over_target_mse": float(max(
                r["bf16_rounding_distortion_over_target_mse"] for r in rows)),
            "mean_original_vs_permuted_same_position_cosine": float(np.mean(
                [r["original_vs_permuted_same_position_cosine"] for r in rows])),
        }
    sources = {"bf16": Path(old["inputs"]["bf16"]),
               "svdquant": Path(old["inputs"]["dynamic"]),
               "permuted_error": Path(media["video"])}
    for arm in ARMS[3:]:
        sources[arm] = Path(e105["sources"][arm])
    if any(not path.is_file() for path in sources.values()) or sha256(sources["permuted_error"]) != media["sha256"]:
        raise RuntimeError("missing or hash-mismatched source video")
    for arm in ("bf16", "svdquant", *ARMS[3:]):
        if sha256(sources[arm]) != e105["source_sha256"][arm]:
            raise RuntimeError(f"E105 control video hash changed: {arm}")

    out_dir = Path("/data1/models/svdquant-wjq/research/20261017/E106_h3_error_permutation") / case
    montage = out_dir / f"{case}_six_arm_montage.mp4"
    poster = research / "E106" / case / "six_arm_poster.jpg"
    report = research / "E106" / case / "six_arm_summary.json"
    if any(path.exists() for path in (montage, poster, report)):
        raise FileExistsError("refusing to overwrite E106 montage/poster/summary")
    out_dir.mkdir(parents=True, exist_ok=True)
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
                canvas.paste(Image.fromarray(frame).resize((w, h), Image.Resampling.BILINEAR), (x, y + band))
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
        raise RuntimeError("incomplete E106 montage")
    poster_canvas = Image.new("RGB", (canvas_size[0], canvas_size[1] * 4))
    for index, frame in enumerate(captured):
        poster_canvas.paste(frame, (0, index * canvas_size[1]))
    poster_canvas.save(poster, quality=88, optimize=True)
    denominator = 124 * 576 * 1024 * 3
    proxies = {arm: {"rgb_mae": item["abs"] / denominator,
                     "rgb_nmse": item["squared"] / ref_squared}
               for arm, item in totals.items()}
    for arm in ("svdquant", *ARMS[3:]):
        if abs(proxies[arm]["rgb_nmse"] - e105["rollout_distance_proxy"][arm]["rgb_nmse"]) > 1e-12:
            raise RuntimeError(f"control metric did not reproduce E105: {arm}")
    payload = {
        "format": "h3-e106-six-arm-summary-v1", "case": case,
        "definition": "decoded H.264 RGB distance to separately generated BF16; free-rollout divergence, NOT perceptual quality",
        "sources": {arm: str(path.resolve()) for arm, path in sources.items()},
        "source_sha256": {arm: sha256(path) for arm, path in sources.items()},
        "frames": count, "fps": 24, "montage": str(montage.resolve()), "montage_sha256": sha256(montage),
        "montage_audio": "none; individual source MP4s contain audio",
        "poster": str(poster.resolve()), "poster_sha256": sha256(poster),
        "permutation_error_validation": per_modality,
        "rollout_distance_proxy": proxies,
    }
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"validation": per_modality, "rollout": proxies}, indent=2))
    print(f"COMPLETE {montage} {poster} {report}")


if __name__ == "__main__":
    main()
