#!/usr/bin/env python3
"""Paired one-linear H3 SVDQuant activation-global-scale check on a cached call."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from minimax_h3_svdquant_common import load_h3_pipeline, nvfp4_qdq, tree_device


class Captured(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def capture(args: argparse.Namespace, target: Path) -> None:
    call = torch.load(args.calls, map_location="cpu", weights_only=False)[0]
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    layer_state = state["layers"][args.layer]
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    module = pipe.dit
    for part in args.layer.split("."):
        module = getattr(module, part)
    weight, _ = module.load_from_disk(torch.bfloat16, "cuda", assign=False)
    payload = {
        "format": "h3-e102-linear-input-weight-v1",
        "layer": args.layer,
        "prompt_id": 1,
        "seed": 22529,
        "step": 0,
        "input_source": str(args.calls.resolve()),
        "state_source": str(args.state.resolve()),
        "weight": weight.detach().cpu(),
        "smooth": layer_state["smooth"].cpu(),
        "a": layer_state["final_a"].cpu(),
        "b": layer_state["final_b"].cpu(),
    }
    del weight, state

    def intercept(_module, inputs):
        payload["input"] = inputs[0].detach().cpu()
        raise Captured()

    handle = module.register_forward_pre_hook(intercept)
    try:
        with torch.inference_mode():
            pipe.dit(*tree_device(call["args"], "cuda"), **tree_device(call["kwargs"], "cuda"))
    except Captured:
        pass
    finally:
        handle.remove()
    if "input" not in payload:
        raise RuntimeError("target input was not captured")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    print(f"captured {args.layer}: x={tuple(payload['input'].shape)}, W={tuple(payload['weight'].shape)}", flush=True)


@torch.inference_mode()
def evaluate(source: Path, output: Path) -> None:
    p = torch.load(source, map_location="cpu", weights_only=False)
    x = p["input"].reshape(-1, p["input"].shape[-1]).to("cuda")
    w = p["weight"].to("cuda")
    smooth = p["smooth"].to("cuda")
    a = p["a"].to("cuda")
    b = p["b"].to("cuda")
    xs = x / smooth
    global_scale = float(xs.float().abs().amax().item() / (6 * 448))
    residual = w * smooth - b @ a
    wq = nvfp4_qdq(residual)
    xq = {mode: nvfp4_qdq(xs, tensor_scale_mode=mode) for mode in ("dynamic", "fixed_one")}
    print(f"actual activation global scale={global_scale:.9g}; computing paired outputs", flush=True)

    totals = {mode: {"sq_error": 0.0, "sq_ref": 0.0, "sq_main_error": 0.0,
                     "sq_main_ref": 0.0, "sq_act_error": 0.0, "sq_act_ref": 0.0}
              for mode in xq}
    local = xs.float().reshape(-1, xs.shape[-1] // 16, 16).abs().amax(-1) / 6
    local_stats = {
        "min": float(local.min()), "median": float(local.median()),
        "max": float(local.max()), "under_fp8_min_normal_fixed_one": int((local < 2**-6).sum()),
        "under_fp8_min_subnormal_fixed_one": int((local < 2**-9).sum()),
        "groups": local.numel(),
    }
    del local
    for start in range(0, x.shape[0], 32):
        stop = min(start + 32, x.shape[0])
        ref = F.linear(x[start:stop], w)
        main_ref = F.linear(xs[start:stop], residual)
        correction = F.linear(F.linear(xs[start:stop], a), b)
        ref_f = ref.float()
        main_ref_f = main_ref.float()
        xs_f = xs[start:stop].float()
        for mode, quant_x in xq.items():
            main = F.linear(quant_x[start:stop], wq)
            actual = main + correction
            t = totals[mode]
            t["sq_error"] += float((actual.float() - ref_f).square().sum(dtype=torch.float64))
            t["sq_ref"] += float(ref_f.square().sum(dtype=torch.float64))
            t["sq_main_error"] += float((main.float() - main_ref_f).square().sum(dtype=torch.float64))
            t["sq_main_ref"] += float(main_ref_f.square().sum(dtype=torch.float64))
            t["sq_act_error"] += float((quant_x[start:stop].float() - xs_f).square().sum(dtype=torch.float64))
            t["sq_act_ref"] += float(xs_f.square().sum(dtype=torch.float64))
    results = {mode: {
        "svdquant_output_nmse": t["sq_error"] / t["sq_ref"],
        "main_branch_output_nmse": t["sq_main_error"] / t["sq_main_ref"],
        "activation_nmse": t["sq_act_error"] / t["sq_act_ref"],
    } for mode, t in totals.items()}
    report = {
        "format": "h3-e102-global-scale-paired-v1",
        "layer": p["layer"], "prompt_id": p["prompt_id"], "step": p["step"],
        "input_shape": list(x.shape), "weight_shape": list(w.shape),
        "activation_global_scale_dynamic": global_scale,
        "local_ideal_scale_stats": local_stats,
        "input_weight_capture": str(source.resolve()),
        "input_weight_capture_sha256": sha256(source),
        "definition": "same x, BF16 W, SVDQuant smooth/A/B and quantized residual W in both arms; activation quantization uses the existing dynamic versus fixed_one runtime recipes; BF16 Wx reference",
        "runtime_recipe_note": "dynamic FP4 codes divide by FP8-rounded local scale times FP32 tensor scale; fixed_one FP4 codes divide by the raw clamped local scale, then reconstruct using its FP8-rounded value",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--layer", default="blocks.0.mlp.fc1")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.capture.exists():
        capture(args, args.capture)
        gc.collect()
        torch.cuda.empty_cache()
    evaluate(args.capture, args.output)


if __name__ == "__main__":
    main()
