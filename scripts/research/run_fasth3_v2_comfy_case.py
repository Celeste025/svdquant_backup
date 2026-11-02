"""Run one paired FastH3 V2 BF16/PTQ Comfy case using the official T2V graph.

The prompt and seed are read from the original JSONL by prompt_id. This is a
BF16 DiT versus NVFP4 DiT comparison with the same Comfy text encoder, VAEs,
sampler, and (optionally) VSA patch. The API server must already be running.
"""

import argparse
import json
import time
from pathlib import Path

import requests


FILENAMES = {
    "bf16": "fastvideo_fasth3_8step_v2_pruned_bf16.safetensors",
    "ptq": "fastvideo_fasth3_8step_v2_pruned_nvfp4.safetensors",
}


def find_prompt(path: Path, prompt_id: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["prompt_id"]) == prompt_id:
                return row
    raise ValueError(f"prompt_id {prompt_id} absent from {path}")


def graph(*, model_name: str, prompt: str, seed: int, width: int,
          height: int, frames: int, filename_prefix: str, vsa: bool) -> dict:
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": model_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames}},
        "6": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["1", 0], "shift_video": 10.0, "shift_audio": 3.0}},
        "7": {"class_type": "BasicGuider", "inputs": {
            "model": ["8", 0] if vsa else ["6", 0], "conditioning": ["5", 0]}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": ["8", 0] if vsa else ["6", 0],
            "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["7", 0], "sampler": ["10", 0],
            "sigmas": ["9", 0], "latent_image": ["5", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {
            "samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["12", 0], "vae": ["4", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {
            "images": ["13", 0], "audio": ["14", 0], "fps": 24.0}},
        "16": {"class_type": "SaveVideo", "inputs": {
            "video": ["15", 0], "filename_prefix": filename_prefix,
            "format": "mp4", "format.codec": "h264"}},
    }
    if vsa:
        nodes["8"] = {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["6", 0],
            "selection": "vsa", "selection.keep_percent": 20.0,
            "start_percent": 0.0, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 0,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False,
        }}
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=FILENAMES, required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--no-vsa", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    name = FILENAMES[args.arm]
    source = args.asset_root / args.arm / "diffusion_models" / name
    if not source.is_file():
        raise FileNotFoundError(source)
    target = args.comfy_root / "models" / "diffusion_models" / name
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"Model symlink points elsewhere: {target}")
    elif target.exists():
        raise FileExistsError(target)
    else:
        target.symlink_to(source.resolve())

    row = find_prompt(args.prompt_source, args.prompt_id)
    prefix = f"fasth3_v2_triad/p{args.prompt_id:03d}_{args.arm}_seed{row['seed']}"
    payload = {"prompt": graph(
        model_name=name, prompt=row["prompt"], seed=int(row["seed"]),
        width=args.width, height=args.height, frames=args.frames,
        filename_prefix=prefix, vsa=not args.no_vsa),
        "client_id": "fasth3-v2-triad"}
    response = requests.post(f"{args.server}/prompt", json=payload, timeout=60)
    response.raise_for_status()
    job = response.json()
    if "error" in job:
        raise RuntimeError(json.dumps(job, indent=2))
    print(f"Comfy queued {args.arm} prompt {args.prompt_id}: {job}", flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        time.sleep(15)
        history = requests.get(f"{args.server}/history/{job['prompt_id']}", timeout=60).json()
        if job["prompt_id"] in history:
            result = history[job["prompt_id"]]
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({
                "arm": args.arm, "prompt_id": args.prompt_id, "seed": row["seed"],
                "source": str(source), "width": args.width, "height": args.height,
                "frames": args.frames, "workflow": payload["prompt"],
                "result": result,
            }, indent=2, ensure_ascii=False) + "\n")
            if result.get("status", {}).get("status_str") != "success":
                raise RuntimeError(f"Comfy generation failed: {args.report}")
            print(f"COMPLETE {args.report}", flush=True)
            return
    raise TimeoutError(f"Comfy generation timed out after {args.timeout}s: {job['prompt_id']}")


if __name__ == "__main__":
    main()
