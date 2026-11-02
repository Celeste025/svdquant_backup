"""Generate one FastH3-QAD-v2 case from a local AdaLN-slim FastVideo root."""

import argparse
import json
import os
import time
from pathlib import Path


def find_prompt(path: Path, prompt_id: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["prompt_id"]) == prompt_id:
                return row
    raise ValueError(f"prompt_id {prompt_id} absent from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fp4-attention", action="store_true",
                        help="Use optional FP4 VSA extension; default BF16 VSA compares linears more fairly")
    args = parser.parse_args()

    table = args.model_root / "transformer" / "adaln_tables.pt"
    if not table.is_file():
        raise FileNotFoundError(table)
    os.environ["FASTVIDEO_H3_ADALN_TABLE"] = str(table)
    os.environ["FASTVIDEO_H3_VSA_FP4"] = "1" if args.fp4_attention else "0"
    os.environ.setdefault("FASTVIDEO_MINIMAX_H3_FUSIONS", "all")
    os.environ.setdefault("FASTVIDEO_NVFP4_MM_BACKEND", "cutlass")
    os.environ.setdefault("FASTVIDEO_H3_VAE_TILE_BATCH", "12")
    os.environ.setdefault("FASTVIDEO_VSA_TRITON", "1")
    os.environ.setdefault("FASTVIDEO_VSA_PTX_BLACKWELL", "0")
    os.environ.setdefault("FASTVIDEO_FA4", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from fastvideo import VideoGenerator

    row = find_prompt(args.prompt_source, args.prompt_id)
    contract = json.loads((args.model_root / "fastvideo_inference.json").read_text())
    if contract["transformer_forwards"] != 8 or contract["num_inference_steps"] != 9:
        raise ValueError("This runner requires the published eight-forward schedule")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model_path": str(args.model_root),
        "engine": {
            "num_gpus": 1,
            "use_fsdp_inference": False,
            "quantization": {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"},
            "parallelism": {"tp_size": 1, "sp_size": 1},
            "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False,
                        "image_encoder": False, "vae": True, "pin_cpu_memory": False,
                        "lazy_module_load": False},
            "compile": {"enabled": False, "vae_enabled": False},
        },
        "pipeline": {"workload_type": "t2v", "vae_tiling": True, "experimental": {
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "h3_vsa_fp4": {"enabled": args.fp4_attention, "prefix_precision": "fp8",
                           "two_level_p": True, "smooth_k": True},
            "VSA_sparsity": 0.8, "VSA_tile_size": 64,
            "h3_sequential_load": True, "inference_torch_compile": False,
            "vae_parallel_decode": False, "video_decode_backend": "h3-vae",
        }},
    }
    request = {
        "prompt": row["prompt"], "negative_prompt": "",
        "sampling": {"seed": int(row["seed"]), "height": args.height,
                     "width": args.width, "num_frames": args.frames, "fps": 24,
                     "num_inference_steps": 9, "guidance_scale": 1.0,
                     "batch_cfg": False},
        "output": {"output_path": str(args.output), "save_video": True,
                   "return_frames": False},
    }
    report = {"arm": "qad", "prompt_id": args.prompt_id, "seed": row["seed"],
              "model_root": str(args.model_root), "contract": contract,
              "config": config, "request": request}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    start = time.perf_counter()
    generator = VideoGenerator.from_config(config)
    report["load_s"] = time.perf_counter() - start
    print(f"QAD loaded in {report['load_s']:.1f}s", flush=True)
    start = time.perf_counter()
    generator.generate(request)
    report["generate_s"] = time.perf_counter() - start
    report["output_exists"] = args.output.is_file()
    report["output_bytes"] = args.output.stat().st_size if args.output.is_file() else None
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not report["output_exists"] or not report["output_bytes"]:
        raise RuntimeError("FastVideo returned without a nonempty MP4")
    print(f"COMPLETE {args.output} {report['generate_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
