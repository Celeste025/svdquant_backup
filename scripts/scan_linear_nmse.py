#!/usr/bin/env python3
"""Fast same-input, per-Linear PTQ error scan.

This intentionally does *not* use calibration caches or re-run the complete model
for every layer. It records a deterministic token sample at every BF16
``nn.Linear`` during the first DiT invocation, then feeds those exact inputs to
the identically named quantized modules. Each row is local quantization error,
not accumulated upstream error.

Examples:
  python scripts/scan_linear_nmse.py --model flux.1-dev
  python scripts/scan_linear_nmse.py --model all --max-tokens 32
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from linear_quant_context import UnsupportedExternalTransform, prepare_isolated_linear_input


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/data/models/svdquant-wjq")
PROMPT = "A red fox walking through a snowy forest at sunrise, cinematic natural light."
SEED = 42


class StopAfterFirstTransformer(Exception):
    """Private control flow after the first top-level transformer forward."""


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return first_tensor(value[0])
    if hasattr(value, "sample"):
        return value.sample
    raise TypeError(f"cannot get tensor from {type(value)!r}")


def token_sample(tensor: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Keep full rows for -1, otherwise evenly-spaced rows by feature vector."""
    flat = tensor.detach().reshape(-1, tensor.shape[-1])
    n = flat.shape[0] if max_tokens == -1 else min(max_tokens, flat.shape[0])
    if n == flat.shape[0]:
        picked = flat
    else:
        index = torch.linspace(0, flat.shape[0] - 1, n, device=flat.device).long()
        picked = flat.index_select(0, index)
    return picked.to(device="cpu", dtype=torch.bfloat16).contiguous().clone()


def metrics(reference: torch.Tensor, quantized: torch.Tensor) -> dict[str, float | list[int]]:
    ref, quantized = reference.float(), quantized.float()
    error = quantized - ref
    mse = error.square().mean()
    ref_power = ref.square().mean().clamp_min(1e-20)
    return {
        "mse": float(mse),
        "nmse": float(mse / ref_power),
        "relative_rmse": float(mse.sqrt() / ref_power.sqrt()),
        "mae": float(error.abs().mean()),
        "max_abs_error": float(error.abs().max()),
        "bf16_rms": float(ref_power.sqrt()),
        "quant_rms": float(quantized.square().mean().sqrt()),
        "cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), quantized.flatten(), dim=0)),
        "shape": list(reference.shape),
    }


@torch.inference_mode()
def capture_linear_io(transformer: torch.nn.Module, invoke, max_tokens: int) -> dict[str, dict[str, Any]]:
    """Capture first-call BF16 I/O samples from every primitive Linear."""
    captured: dict[str, dict[str, Any]] = {}
    hooks = []
    for name, module in transformer.named_modules():
        if not name or not isinstance(module, torch.nn.Linear):
            continue

        def hook(layer, args, output, layer_name=name):
            if layer_name in captured or not args or not torch.is_tensor(args[0]):
                return
            try:
                x, y = token_sample(args[0], max_tokens), token_sample(first_tensor(output), max_tokens)
                if x.shape[0] == y.shape[0]:
                    captured[layer_name] = {
                        "input": x, "output": y,
                        "in_features": int(layer.in_features), "out_features": int(layer.out_features),
                        "bf16_type": type(layer).__name__,
                    }
            except Exception as exc:
                captured[layer_name] = {"error": f"capture: {type(exc).__name__}: {exc}"}

        hooks.append(module.register_forward_hook(hook))
    root_hook = transformer.register_forward_hook(
        lambda _module, _args, _output: (_ for _ in ()).throw(StopAfterFirstTransformer())
    )
    try:
        invoke()
    except StopAfterFirstTransformer:
        pass
    finally:
        root_hook.remove()
        for hook in hooks:
            hook.remove()
    if not captured:
        raise RuntimeError("no Linear I/O captured; the first transformer call did not run")
    return captured


def load_flux(data: Path, name: str, steps: int, guidance: float):
    os.environ.update({
        "DATA_ROOT": str(data), "FLUX_MODEL_PATH": str(data / "models" / name),
        "FLUX_MODEL_CONFIG": f"configs/model/{name.lower()}.yaml", "FLUX_NUM_STEPS": str(steps),
        "FLUX_GUIDANCE": str(guidance),
        "FLUX_CALIB_PATH": str(data / "datasets" / "torch.bfloat16" / name.lower() /
                               ("fmeuler4-g0" if guidance == 0 else "fmeuler50-g3.5") / "qdiff" / "s64"),
    })
    helper = load_script("infer_bf16_vs_w4a4_one.py")
    kwargs = dict(prompt=PROMPT, height=1024, width=1024, num_inference_steps=steps,
                  guidance_scale=guidance, generator=torch.Generator("cuda").manual_seed(SEED))
    return helper, data / "ckpts" / f"{name.lower()}-int4-fast-s64-lowmem", kwargs


def load_wan(data: Path):
    os.environ["DATA_ROOT"] = str(data)
    helper = load_script("infer_wan_bf16_vs_w4a4_one.py")
    kwargs = dict(prompt=PROMPT, negative_prompt=helper.NEGATIVE, height=helper.HEIGHT, width=helper.WIDTH,
                  num_frames=helper.NUM_FRAMES, num_inference_steps=helper.STEPS,
                  guidance_scale=helper.GUIDANCE, generator=torch.Generator("cuda").manual_seed(SEED))
    return helper, data / "ckpts" / "wan2.1-1.3b-real-nvfp4-s16", kwargs


def load_rcm(data: Path):
    from diffusers import WanPipeline
    model, ckpt = data / "models" / "rcm-Wan2.1-T2V-1.3B-Diffusers", data / "ckpts" / "rcm-wan2.1-1.3b-real-nvfp4-s16"
    helper = load_script("infer_rcm_wan_4step.py")

    def build_bf16():
        return WanPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")

    def build_quant():
        pipe = WanPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
        helper.load_quantized_transformer(pipe, ckpt, model)
        return pipe

    def invoke(pipe):
        device, dtype = torch.device("cuda"), pipe.transformer.dtype
        embeds, _ = pipe.encode_prompt(prompt=PROMPT, do_classifier_free_guidance=False,
                                       max_sequence_length=512, device=device, dtype=dtype)
        latent = pipe.prepare_latents(1, pipe.transformer.config.in_channels, 480, 832, 77,
                                     torch.float32, device, torch.Generator(device).manual_seed(SEED)).to(torch.float64)
        time = math.sin(math.atan(80)) / (math.cos(math.atan(80)) + math.sin(math.atan(80)))
        pipe.transformer(hidden_states=(latent * time).to(dtype),
                         timestep=torch.tensor([time * 1000], device=device, dtype=dtype),
                         encoder_hidden_states=embeds, return_dict=False)
    return build_bf16, build_quant, invoke, ckpt


@torch.inference_mode()
def scan_one(model: str, data: Path, max_tokens: int) -> dict[str, Any]:
    """Load BF16/PTQ sequentially; only token samples remain after BF16 unload."""
    if model.startswith("flux"):
        name, steps, guidance = ("FLUX.1-dev", 50, 3.5) if model.endswith("dev") else ("FLUX.1-schnell", 4, 0.0)
        helper, ckpt, kwargs = load_flux(data, name, steps, guidance)
        bf16_pipe = helper.load_bf16_pipeline()
        captured = capture_linear_io(bf16_pipe.transformer, lambda: bf16_pipe(**kwargs, output_type="latent"), max_tokens)
        del bf16_pipe; gc.collect(); torch.cuda.empty_cache()
        quant_pipe = helper.load_quant_pipeline(ckpt)
    elif model == "wan2.1-1.3b":
        helper, ckpt, kwargs = load_wan(data)
        bf16_pipe = helper.load_bf16_pipeline()
        captured = capture_linear_io(bf16_pipe.transformer, lambda: bf16_pipe(**kwargs, output_type="latent"), max_tokens)
        del bf16_pipe; gc.collect(); torch.cuda.empty_cache()
        quant_pipe = helper.load_quant_pipeline(ckpt)
    elif model == "rcm-wan2.1-1.3b":
        build_bf16, build_quant, invoke, ckpt = load_rcm(data)
        bf16_pipe = build_bf16()
        captured = capture_linear_io(bf16_pipe.transformer, lambda: invoke(bf16_pipe), max_tokens)
        del bf16_pipe; gc.collect(); torch.cuda.empty_cache()
        quant_pipe = build_quant()
    else:
        raise ValueError(model)

    # PTQ loading builds the transformer on CUDA. The scan needs no whole-model
    # forward on the quantized side, so return it to host RAM before visiting the
    # layers one at a time.
    quant_pipe.transformer.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    quant_modules, rows = dict(quant_pipe.transformer.named_modules()), []
    for index, (name, record) in enumerate(captured.items(), start=1):
        base = {"name": name, "index": index, **{k: v for k, v in record.items() if k not in {"input", "output"}}}
        q_module = quant_modules.get(name)
        if "error" in record:
            rows.append(base); continue
        if q_module is None:
            rows.append(base | {"error": "missing matching quantized module"}); continue
        try:
            # Move one layer only: the scan remains viable for the 12B Flux models.
            q_module.to("cuda")
            q_input = record["input"].to("cuda")
            q_input, context = prepare_isolated_linear_input(quant_pipe.transformer, name, q_input)
            # Time/embed projections can deliberately retain FP32 weights.
            q_weight = getattr(q_module, "weight", None)
            if torch.is_tensor(q_weight) and q_weight.is_floating_point():
                q_input = q_input.to(dtype=q_weight.dtype)
            q_output = first_tensor(q_module(q_input)).detach().to("cpu", torch.bfloat16)
            if q_output.shape != record["output"].shape:
                raise RuntimeError(f"output shape {tuple(q_output.shape)} != BF16 {tuple(record['output'].shape)}")
            rows.append(base | {"quant_type": type(q_module).__name__, "input_context": context.mode, **metrics(record["output"], q_output)})
        except UnsupportedExternalTransform as exc:
            rows.append(base | {"quant_type": type(q_module).__name__, "error": f"unsupported_context: {exc}"})
        except Exception as exc:
            rows.append(base | {"quant_type": type(q_module).__name__, "error": f"forward: {type(exc).__name__}: {exc}"})
        finally:
            q_module.to("cpu")
            torch.cuda.empty_cache()
        if index % 25 == 0:
            print(f"[{model}] measured {index}/{len(captured)} layers", flush=True)
    del quant_pipe; gc.collect(); torch.cuda.empty_cache()
    measured = [row for row in rows if "nmse" in row]
    return {
        "model": model, "checkpoint": str(ckpt), "seed": SEED, "prompt": PROMPT,
        "max_tokens_per_linear": max_tokens,
        "definition": "BF16 output versus the matching quantized Linear in its reconstructed quantized input coordinate; direct Linear QDQ and low-rank hooks remain active. Unsupported ancestor transforms are excluded.",
        "n_captured": len(captured), "n_measured": len(measured), "n_failed": len(rows) - len(measured),
        "layers": sorted(rows, key=lambda row: float(row.get("nmse", -1)), reverse=True),
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = ["index", "name", "bf16_type", "quant_type", "input_context", "in_features", "out_features", "nmse", "mse", "relative_rmse", "mae", "max_abs_error", "bf16_rms", "quant_rms", "cosine", "shape", "error"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["layers"]:
            writer.writerow(row | {"shape": json.dumps(row.get("shape", []))})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "wan2.1-1.3b", "rcm-wan2.1-1.3b", "flux.1-dev", "flux.1-schnell"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="rows per Linear; use -1 to retain every token (much higher host-RAM use)",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "reports" / "linear_local_error")
    args = parser.parse_args()
    # Model helper scripts change cwd to DeepCompressor's diffusion directory.
    # Resolve this once so reports always land in the user-selected destination.
    args.out_dir = args.out_dir.resolve()
    if args.max_tokens == 0 or args.max_tokens < -1:
        parser.error("--max-tokens must be positive, or -1 for all tokens")
    models = ["wan2.1-1.3b", "rcm-wan2.1-1.3b", "flux.1-dev", "flux.1-schnell"] if args.model == "all" else [args.model]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        print(f"=== local Linear scan: {model} ===", flush=True)
        report = scan_one(model, args.data_root, args.max_tokens)
        json_path, csv_path = args.out_dir / f"{model}_linear_local_error.json", args.out_dir / f"{model}_linear_local_error.csv"
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        write_csv(report, csv_path)
        print(f"[{model}] {report['n_measured']}/{report['n_captured']} layers measured -> {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
