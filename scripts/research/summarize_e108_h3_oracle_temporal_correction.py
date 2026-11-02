#!/usr/bin/env python3
"""Validate E108 and render aligned BF16/SVDQuant/oracle-correction videos."""
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
ARMS = ("bf16", "svdquant", "oracle_prev_error_k06")
LABELS = ("BF16", "SVDQuant", "Oracle previous error, k=0.6")


def check_denoise(denoise: dict, case: str) -> dict:
    if (denoise.get("format"), denoise.get("status"), denoise.get("case"), denoise.get("arm")) != (
        "h3-e108-oracle-previous-error-denoise-v1", "denoise_complete", case, ARMS[2]
    ) or len(denoise.get("steps", [])) != 20:
        raise RuntimeError("incomplete or invalid E108 denoising report")
    results = {}
    for modality in ("video", "audio"):
        rows = []
        for step, entry in enumerate(denoise["steps"]):
            if entry["step"] != step:
                raise RuntimeError(f"out-of-order E108 step: {step}")
            row = entry["modalities"][modality]
            if not all(math.isfinite(row[key]) and row[key] > 0 for key in (
                "target_mse", "target_nmse", "realized_mse", "realized_nmse",
                "reference_mean_square", "output_elements"
            )) or row["coefficient"] != 0.6:
                raise RuntimeError(f"nonfinite or unexpected E108 metric: {step}/{modality}")
            if step == 0 and (row["previous_error_available"] or
                              row["realized_mse"] != row["target_mse"]):
                raise RuntimeError(f"E108 step 0 was altered: {modality}")
            if step > 0 and (not row["previous_error_available"] or
                             row["bf16_rounding_distortion_over_target_mse"] > 0.01):
                raise RuntimeError(f"E108 prior-error or rounding gate failed: {step}/{modality}")
            rows.append(row)
        # Output shapes and energy denominators can differ by modality. Pool MSE
        # using the actual element counts, not an unweighted average of NMSEs.
        follow = rows[1:]
        n = sum(row["output_elements"] for row in follow)
        reference_energy = sum(row["reference_mean_square"] * row["output_elements"] for row in follow)
        raw_energy = sum(row["target_mse"] * row["output_elements"] for row in follow)
        corrected_energy = sum(row["realized_mse"] * row["output_elements"] for row in follow)
        if n <= 0 or reference_energy <= 0:
            raise RuntimeError("invalid E108 pooled denominator")
        results[modality] = {
            "step_0_target_nmse": rows[0]["target_nmse"],
            "step_0_realized_nmse": rows[0]["realized_nmse"],
            "steps_1_19_raw_nmse_pooled": raw_energy / reference_energy,
            "steps_1_19_corrected_nmse_pooled": corrected_energy / reference_energy,
            "steps_1_19_relative_error_energy_change": corrected_energy / raw_energy - 1,
            "steps_1_19_improved_count": sum(row["realized_mse"] < row["target_mse"] for row in follow),
            "steps_1_19_worsened_count": sum(row["realized_mse"] > row["target_mse"] for row in follow),
            "steps_1_19_mean_previous_error_cosine": float(np.mean([row["prior_error_cosine"] for row in follow])),
            "steps_1_19_max_bf16_rounding_distortion_over_raw_mse": max(
                row["bf16_rounding_distortion_over_target_mse"] for row in follow
            ),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("race", "clap0"), required=True)
    args = parser.parse_args()
    case = args.case
    research = ROOT / "results/research"
    old = json.loads((research / "E101" / case / "video_pixel_comparison.json").read_text())
    e105 = json.loads((research / "E105" / case / "five_arm_summary.json").read_text())
    denoise_path = research / "E108" / case / f"{ARMS[2]}_denoise.json"
    video_path = research / "E108" / case / f"{ARMS[2]}_video.json"
    denoise = json.loads(denoise_path.read_text())
    media = json.loads(video_path.read_text())
    stats = check_denoise(denoise, case)
    if (media.get("format"), media.get("case"), media.get("arm"),
        media.get("frames"), media.get("audio_streams")) != (
        "h3-e108-oracle-previous-error-video-v1", case, ARMS[2], 124, 1
    ) or media["denoise_report_sha256"] != sha256(denoise_path):
        raise RuntimeError("invalid E108 video metadata or denoising hash")
    sources = {"bf16": Path(old["inputs"]["bf16"]),
               "svdquant": Path(old["inputs"]["dynamic"]),
               ARMS[2]: Path(media["video"])}
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError("missing one of the three case videos")
    for arm in ARMS[:2]:
        if sha256(sources[arm]) != e105["source_sha256"][arm]:
            raise RuntimeError(f"control video hash changed: {arm}")
    if sha256(sources[ARMS[2]]) != media["sha256"]:
        raise RuntimeError("E108 generated video hash mismatch")

    out_dir = Path("/data1/models/svdquant-wjq/research/20261017/E108_h3_oracle_temporal_correction") / case
    montage = out_dir / f"{case}_three_arm_montage.mp4"
    poster = research / "E108" / case / "three_arm_poster.jpg"
    report = research / "E108" / case / "three_arm_summary.json"
    if any(path.exists() for path in (montage, poster, report)):
        raise FileExistsError("refusing to overwrite E108 montage/poster/summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    w, h, band = 512, 288, 34
    size = (3 * w, h + band)
    totals = {arm: {"abs": 0, "squared": 0} for arm in ARMS[1:]}
    reference_squared = 0
    captured = []
    output = av.open(str(montage), mode="w")
    stream = output.add_stream("libx264", rate=24)
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    count = 0
    try:
        for count, frame_set in enumerate(zip_longest(*(frames(sources[arm]) for arm in ARMS)), start=1):
            if any(frame is None for frame in frame_set) or count > 124:
                raise RuntimeError(f"mismatched or excess frame at index {count - 1}")
            ref = frame_set[0].astype(np.int16)
            reference_squared += int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
            for arm, frame in zip(ARMS[1:], frame_set[1:], strict=True):
                diff = frame.astype(np.int16) - ref
                totals[arm]["abs"] += int(np.abs(diff).sum(dtype=np.int64))
                totals[arm]["squared"] += int(np.square(diff.astype(np.int32)).sum(dtype=np.int64))
            canvas = Image.new("RGB", size, (24, 24, 24))
            draw = ImageDraw.Draw(canvas)
            for index, (label, frame) in enumerate(zip(LABELS, frame_set, strict=True)):
                x = index * w
                canvas.paste(Image.fromarray(frame).resize((w, h), Image.Resampling.BILINEAR), (x, band))
                draw.text((x + 10, 6), label, font=font, fill="white")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                output.mux(packet)
            if count - 1 in (0, 30, 60, 90):
                captured.append(canvas)
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()
    if count != 124 or len(captured) != 4 or reference_squared <= 0:
        raise RuntimeError("incomplete E108 montage")
    poster_canvas = Image.new("RGB", (size[0], size[1] * 4))
    for index, canvas in enumerate(captured):
        poster_canvas.paste(canvas, (0, index * size[1]))
    poster_canvas.save(poster, quality=88, optimize=True)
    denominator = 124 * 576 * 1024 * 3
    proxies = {arm: {"rgb_mae": value["abs"] / denominator,
                     "rgb_nmse": value["squared"] / reference_squared}
               for arm, value in totals.items()}
    if abs(proxies["svdquant"]["rgb_nmse"] - e105["rollout_distance_proxy"]["svdquant"]["rgb_nmse"]) > 1e-12:
        raise RuntimeError("SVDQuant control pixel NMSE did not reproduce E105")
    payload = {
        "format": "h3-e108-three-arm-summary-v1", "case": case,
        "definition": "decoded H.264 RGB distance to separately generated BF16; free-rollout divergence, NOT perceptual quality",
        "oracle_definition": "Q_t - 0.6*(Q_(t-1)-BF16_(t-1)) on the corrected arm's own trajectory; step 0 unchanged",
        "sources": {arm: str(path.resolve()) for arm, path in sources.items()},
        "source_sha256": {arm: sha256(path) for arm, path in sources.items()},
        "frames": count, "fps": 24, "montage": str(montage.resolve()),
        "montage_sha256": sha256(montage), "montage_audio": "none; individual source MP4s contain audio",
        "poster": str(poster.resolve()), "poster_sha256": sha256(poster),
        "paired_dit_output": stats, "rollout_distance_proxy": proxies,
    }
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"paired_dit_output": stats, "rollout_distance_proxy": proxies}, indent=2))
    print(f"COMPLETE {montage} {poster} {report}")


if __name__ == "__main__":
    main()
