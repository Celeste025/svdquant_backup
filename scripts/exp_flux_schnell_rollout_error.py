#!/usr/bin/env python3
"""Paired 4-step FLUX.1-schnell W4A4/BF16 rollout-output audit.

The final DiT output is captured online with a transformer hook.  FLUX calls
this a noise/flow prediction rather than rCM-Wan's velocity; this script labels
it ``dit_output`` and never assumes a scheduler parameterization.  Only scalar
per-step errors, final images, and a side-by-side PNG are saved.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DATA_ROOT", "/data/models/svdquant-wjq")

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["DATA_ROOT"])
MODEL_NAME = "FLUX.1-schnell"
CKPT = DATA / "ckpts" / "flux.1-schnell-int4-fast-s64-lowmem"
MODEL = DATA / "models" / MODEL_NAME
CALIB = DATA / "datasets/torch.bfloat16/flux.1-schnell/fmeuler4-g0/qdiff/s64"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_infer_module():
    # Its constants must see schnell-specific env vars during module import.
    os.environ.update({
        "FLUX_MODEL_PATH": str(MODEL),
        "FLUX_MODEL_CONFIG": "configs/model/flux.1-schnell.yaml",
        "FLUX_NUM_STEPS": "4",
        "FLUX_GUIDANCE": "0.0",
        "FLUX_CALIB_PATH": str(CALIB),
    })
    source = ROOT / "scripts" / "infer_bf16_vs_w4a4_one.py"
    spec = importlib.util.spec_from_file_location("flux_schnell_loader", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def value(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    elif hasattr(output, "sample"):
        output = output.sample
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"unexpected transformer output: {type(output)}")
    return output.detach().float().cpu().contiguous()


def metrics(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float | int]:
    got, ref = got.float(), ref.float()
    err = got - ref
    err2, ref2 = float(err.square().sum()), float(ref.square().sum())
    return {"numel": int(ref.numel()), "mse": err2 / max(ref.numel(), 1), "nmse": err2 / max(ref2, 1e-30)}


def make_montage(left: Path, right: Path, out: Path) -> None:
    a, b = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    header = 34
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height) + header), "white")
    canvas.paste(a, (0, header)); canvas.paste(b, (a.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "BF16", fill="black")
    draw.text((a.width + 8, 8), "W4A4 SVDQuant", fill="black")
    canvas.save(out)


@torch.inference_mode()
def run_once(pipe: Any, prompt: str, seed: int, height: int, width: int, out: Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    outputs: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    handle = pipe.transformer.register_forward_hook(lambda _m, _args, result: outputs.append(value(result)))
    try:
        result = pipe(
            prompt, height=height, width=width, num_inference_steps=4, guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            callback_on_step_end=lambda _p, _i, _t, kwargs: (latents.append(kwargs["latents"].detach().float().cpu().contiguous()) or kwargs),
            callback_on_step_end_tensor_inputs=["latents"],
        )
        result.images[0].save(out)
    finally:
        handle.remove()
    if len(outputs) != 4 or len(latents) != 4:
        raise RuntimeError(f"expected four DiT outputs and latents, got outputs={len(outputs)}, latents={len(latents)}")
    return outputs, latents


def clear_cuda() -> None:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/samples/flux_schnell_rollout_error")
    args = ap.parse_args()
    out = args.output_dir.resolve() / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if not (CKPT / "model.pt").is_file():
        raise FileNotFoundError(CKPT / "model.pt")
    loader = load_infer_module()
    bf16_path, quant_path = out / "bf16.png", out / "w4a4.png"
    print("[bf16] paired four-step generation", flush=True)
    bf = loader.load_bf16_pipeline()
    bf_outputs, bf_latents = run_once(bf, args.prompt, args.seed, args.height, args.width, bf16_path)
    # Explicitly delete the caller reference before loading the quantized pipeline.
    del bf
    clear_cuda()
    print("[w4a4] paired four-step generation", flush=True)
    quant = loader.load_quant_pipeline(CKPT)
    q_outputs, q_latents = run_once(quant, args.prompt, args.seed, args.height, args.width, quant_path)
    del quant
    clear_cuda()
    rows = []
    for step in range(4):
        row = {"step": step}
        row.update({f"dit_output_{k}": v for k, v in metrics(q_outputs[step], bf_outputs[step]).items()})
        row.update({f"latent_{k}": v for k, v in metrics(q_latents[step], bf_latents[step]).items()})
        rows.append(row)
    write_csv(out / "per_step_metrics.csv", rows)
    write_json(out / "run_config.json", {
        "model": MODEL_NAME, "model_path": str(MODEL), "checkpoint": str(CKPT),
        "prompt": args.prompt, "seed": args.seed, "height": args.height, "width": args.width,
        "steps": 4, "guidance": 0.0,
        "definition": "dit_output is the paired final transformer noise/flow prediction captured at each rollout step; latent is measured after its scheduler update. Both compare W4A4's own rollout to the paired BF16 rollout.",
    })
    make_montage(bf16_path, quant_path, out / "bf16_vs_w4a4.png")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
