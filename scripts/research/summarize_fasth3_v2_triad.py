"""Verify all nine FastH3 V2 videos and write an auditable run summary."""

import argparse
import hashlib
import json
from pathlib import Path

import av


ARMS = ("bf16", "ptq", "qad")
PROMPTS = (16, 11, 51)


def source_video(report: dict, comfy_output: Path) -> Path:
    if report["arm"] == "qad":
        return Path(report["request"]["output"]["output_path"])
    files = report["result"]["outputs"]["16"]["images"]
    if len(files) != 1:
        raise ValueError(f"Expected one saved video, got {len(files)}")
    record = files[0]
    if record["type"] != "output":
        raise ValueError(f"Unexpected output type: {record}")
    return comfy_output / record["subfolder"] / record["filename"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--comfy-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for prompt_id in PROMPTS:
        reference_prompt = None
        reference_seed = None
        for arm in ARMS:
            path = args.reports / f"p{prompt_id:03d}_{arm}_aligned.json"
            if arm == "qad":
                path = args.reports / f"p{prompt_id:03d}_qad.json"
            report = json.loads(path.read_text())
            if report["arm"] != arm or report["prompt_id"] != prompt_id:
                raise ValueError(f"Report identity mismatch: {path}")
            if arm == "qad":
                prompt = report["request"]["prompt"]
                seed = report["request"]["sampling"]["seed"]
                if not report.get("output_exists"):
                    raise ValueError(f"QAD report is not complete: {path}")
                if report["config"]["pipeline"]["experimental"]["VSA_sparsity"] != 0.8:
                    raise ValueError(f"QAD sparsity mismatch: {path}")
            else:
                prompt = report["workflow"]["5"]["inputs"]["prompt"]
                seed = report["workflow"]["11"]["inputs"]["noise_seed"]
                if report["result"]["status"]["status_str"] != "success":
                    raise ValueError(f"Comfy report is not complete: {path}")
                sparse = report["workflow"]["8"]["inputs"]
                if sparse["selection.keep_percent"] != 20.0 or sparse["start_percent"] != 0.0:
                    raise ValueError(f"Comfy VSA is not aligned: {path}")
            if reference_prompt is None:
                reference_prompt, reference_seed = prompt, seed
            elif prompt != reference_prompt or seed != reference_seed:
                raise ValueError(f"Prompt/seed mismatch for p{prompt_id}: {arm}")

            video = source_video(report, args.comfy_output)
            if not video.is_file():
                raise FileNotFoundError(video)
            with av.open(str(video)) as container:
                video_streams = container.streams.video
                audio_streams = container.streams.audio
                if len(video_streams) != 1 or len(audio_streams) != 1:
                    raise ValueError(f"Expected one video and one audio stream: {video}")
                video_stream = video_streams[0]
                frames = sum(1 for _ in container.decode(video=0))
                if (video_stream.width, video_stream.height, frames) != (1024, 576, 124):
                    raise ValueError(f"Unexpected video geometry: {video}")
                duration = container.duration / 1_000_000
            records.append({
                "prompt_id": prompt_id, "arm": arm, "seed": seed,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "video": str(video), "bytes": video.stat().st_size,
                "width": 1024, "height": 576, "frames_decoded": frames,
                "duration_s": duration, "audio_streams": 1,
                "report": str(path),
            })
    summary = {
        "records": records,
        "comparison_note": (
            "All arms use the same prompt text, seed, 1024x576, 124 frames, eight "
            "denoising forwards and 20% VSA keep rate. BF16/PTQ run through ComfyUI; "
            "QAD runs through FastVideo. Equal seeds do not prove identical initial "
            "noise or numerically paired outputs across runtimes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"VERIFIED {len(records)} videos; {args.output}")


if __name__ == "__main__":
    main()
