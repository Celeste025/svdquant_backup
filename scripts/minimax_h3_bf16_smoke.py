#!/usr/bin/env python3
"""Run a BF16 MiniMax-H3 FL2VA smoke inference with the pruned DiT checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = Path(os.environ.get("DIFFSYNTH_ROOT", ROOT / "third_party/DiffSynth-Studio"))
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=(
        "A cinematic close-up of a hummingbird hovering beside a red flower in a misty forest at dawn, "
        "natural wing motion, shallow depth of field, realistic documentary photography, no text or watermark."
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=39, help="Must follow 17n+5.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("height/width must be multiples of 32 and frames must satisfy 17n+5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    disk = {
        "offload_dtype": "disk", "offload_device": "disk", "onload_dtype": "disk", "onload_device": "disk",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }
    pipeline = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="Comfy-Org/MiniMax-H3", origin_file_pattern="diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors", **disk),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **disk),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **disk),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **disk),
        ],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / 1024**3 - 4,
    )
    video, audio = pipeline(prompt=args.prompt, height=args.height, width=args.width, num_frames=args.frames,
                            num_inference_steps=args.steps, seed=args.seed, tiled=True)
    write_video_audio(video=video, audio=audio, output_path=str(args.output), fps=24,
                      audio_sample_rate=pipeline.audio_vae.sample_rate)
    print(f"saved {args.output} ({len(video)} frames)", flush=True)


if __name__ == "__main__":
    main()
