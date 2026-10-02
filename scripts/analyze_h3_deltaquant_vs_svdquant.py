#!/usr/bin/env python3
"""Compare SVDQuant-style and DeltaQuant-style activation quantization on one H3 layer.

Inputs:
  * per-step activations captured by scripts/collect_h3_single_layer_activations.py
    (flattened packed-sequence input of a single linear layer, one tensor per step);
  * the SVDQuant calibration state of the same layer
    (app_data/artifacts/variants/nvfp4-g10-r64/quant_state.pt) with smooth/final_a/final_b;
  * the original bf16 layer weight from the pruned DiT safetensors.

For every captured step the script evaluates, in fp32:
  ref    : y = x W^T
  SVDQ   : y = q4(x/s) R4^T + (x/s) (b a)^T           (NVFP4 A4, NVFP4 residual, fp rank)
  DeltaQ : y = [q8(core) + q4(delta)] R4^T + (x/s) (b a)^T
where q4/q8 reuse the runtime-aligned fake-quant primitives, R4 is the real NVFP4
residual W*s - b a, and the DeltaQuant cube decomposition operates on the video
token grid only (text/audio/pad fall back to plain q4(x/s) in both variants).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minimax_h3_svdquant_common import DATA_ROOT, nvfp4_qdq

FP8_E4M3_MAX = 448.0
CUBES = {"16": (4, 1, 4), "32": (4, 2, 4), "64": (4, 2, 8)}
DIT_PATH = DATA_ROOT / "models" / "Comfy-Org" / "MiniMax-H3" / "diffusion_models" / "minimax_h3_fl2va_pruned_bf16.safetensors"
STATE_PATH = DATA_ROOT / "artifacts" / "variants" / "nvfp4-g10-r64" / "quant_state.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT / "runs" / "h3-singlelayer-dq-vs-svdq")
    parser.add_argument("--layer", default="blocks.24.mlp.fc2")
    parser.add_argument("--dit-path", type=Path, default=DIT_PATH)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def fp8_qdq(tensor: torch.Tensor, *, group_size: int = 64) -> torch.Tensor:
    """Per-group FP8 (E4M3) fake quant along the last dimension."""
    if tensor.shape[-1] % group_size:
        raise ValueError(f"last dim {tensor.shape[-1]} is not divisible by {group_size}")
    shape = tensor.shape
    rows = tensor.reshape(-1, shape[-1]).float()
    out = torch.empty_like(rows)
    for start in range(0, rows.shape[0], 1024):
        stop = min(rows.shape[0], start + 1024)
        x = rows[start:stop].reshape(-1, shape[-1] // group_size, group_size)
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_E4M3_MAX
        q = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
        out[start:stop] = q.reshape(stop - start, shape[-1])
    return out.reshape(shape)


@torch.no_grad()
def deltaquant_activation(x_s: torch.Tensor, layout: dict, grid: dict, cube: tuple[int, int, int],
                          *, fp8_group: int = 64, q4: torch.Tensor | None = None) -> torch.Tensor:
    """Rebuild the activation as fp8 cube-mean core + fp4 delta on the video grid.

    Tokens outside the video segment (and video tokens in the cube remainder
    bands) keep the plain q4 fallback, identical to the SVDQuant variant.
    """
    device = x_s.device
    out = nvfp4_qdq(x_s) if q4 is None else q4.clone()
    v0, v1 = layout["video"]
    t_n, h_n, w_n = grid["t"], grid["h"], grid["w"]
    ct, ch, cw = cube
    n_t, n_h, n_w = t_n // ct, h_n // ch, w_n // cw
    if n_t == 0 or n_h == 0 or n_w == 0:
        return out

    t = torch.arange(t_n, device=device).view(-1, 1, 1)
    h = torch.arange(h_n, device=device).view(1, -1, 1)
    w = torch.arange(w_n, device=device).view(1, 1, -1)
    valid = ((t < n_t * ct) & (h < n_h * ch) & (w < n_w * cw)).reshape(-1)
    cube_id = ((t // ct) * n_h + (h // ch)) * n_w + (w // cw)
    cube_id = cube_id.reshape(-1)[valid]

    vx = x_s[v0:v1].float()
    vx_valid = vx[valid]
    k = n_t * n_h * n_w
    channels = vx.shape[-1]
    sums = torch.zeros(k, channels, device=device, dtype=torch.float32)
    sums.index_add_(0, cube_id, vx_valid)
    counts = torch.zeros(k, device=device, dtype=torch.float32)
    counts.index_add_(0, cube_id, torch.ones_like(cube_id, dtype=torch.float32))
    core = sums / counts.clamp_min(1.0).unsqueeze(1)
    core_q = fp8_qdq(core, group_size=fp8_group)
    delta_q = nvfp4_qdq(vx_valid - core[cube_id])
    recon = (core_q[cube_id] + delta_q).to(out.dtype)

    video_offsets = torch.arange(v1 - v0, device=device)[valid] + v0
    out[video_offsets] = recon
    return out


@torch.no_grad()
def run_case(payload: dict, weight: torch.Tensor, smooth: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
             device: str) -> dict:
    layout = payload["layout"]
    grid = payload["grid"]
    v0, v1 = layout["video"]
    x_w = weight.float().to(device)                    # [out, in]
    s = smooth.float().to(device)                      # [in]
    ba = (b.float().to(device) @ a.float().to(device))  # [out, in]
    w_s = x_w * s
    r4 = nvfp4_qdq(w_s - ba)

    steps_out = []
    for step in sorted(int(k) for k in payload["steps"]):
        x = payload["steps"][step].float().to(device)
        x_s = x / s
        ref = x @ x_w.t()
        rank = x_s @ ba.t()
        ref_power = float(ref.square().sum(dtype=torch.float64))

        q4 = nvfp4_qdq(x_s)
        y_svdq = q4 @ r4.t() + rank
        timestep = None
        if payload.get("timesteps"):
            raw = payload["timesteps"][step]
            if torch.is_tensor(raw):
                raw = raw.tolist()
            timestep = raw
        entry = {
            "step": step,
            "timestep": timestep,
            "svdquant": {
                "y_nmse": nmse(y_svdq, ref, ref_power),
                "act_nmse": float(act_nmse(q4, x_s)),
                "act_nmse_video": float(act_nmse(q4[v0:v1], x_s[v0:v1])),
            },
            "deltaquant": {},
        }
        for name, cube in CUBES.items():
            a_d = deltaquant_activation(x_s, layout, grid, cube, q4=q4)
            y_dq = a_d @ r4.t() + rank
            entry["deltaquant"][f"cube{name}"] = {
                "y_nmse": nmse(y_dq, ref, ref_power),
                "act_nmse": float(act_nmse(a_d, x_s)),
                "act_nmse_video": float(act_nmse(a_d[v0:v1], x_s[v0:v1])),
            }
        steps_out.append(entry)
        print(f"  step {step}: svdq y={entry['svdquant']['y_nmse']:.4%} "
              + " ".join(f"dq{name} y={entry['deltaquant'][f'cube{name}']['y_nmse']:.4%}" for name in CUBES), flush=True)
    return {"case": payload["case"], "grid": grid, "layout": layout, "steps": steps_out}


def nmse(y: torch.Tensor, ref: torch.Tensor, ref_power: float) -> float:
    diff = (y.double() - ref.double()).square().sum()
    return float((diff / max(ref_power, 1e-30)).item())


def act_nmse(q: torch.Tensor, x_s: torch.Tensor) -> float:
    diff = (q.double() - x_s.double()).square().sum()
    return float((diff / x_s.double().square().sum().clamp_min(1e-30)).item())


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    layer = state["layers"][args.layer]
    smooth = layer["smooth"].float()
    a, b = layer["final_a"].float(), layer["final_b"].float()
    print(f"state: rank={a.shape[0]} alpha={layer.get('alpha')} final_error={layer.get('final_error')}", flush=True)

    from safetensors import safe_open
    with safe_open(str(args.dit_path), framework="pt") as handle:
        weight = handle.get_tensor(f"{args.layer}.weight")
    print(f"weight {tuple(weight.shape)} {weight.dtype}", flush=True)

    results = {"layer": args.layer, "state": str(args.state), "cubes": {k: list(v) for k, v in CUBES.items()},
               "cases": {}}
    for path in sorted(args.input_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["layer"] != args.layer:
            raise ValueError(f"{path.name}: captured layer {payload['layer']} != {args.layer}")
        print(f"[{payload['case']}] {len(payload['steps'])} steps, video={payload['layout']['video']}", flush=True)
        results["cases"][payload["case"]] = run_case(payload, weight, smooth, a, b, device)

    summary = {"svdquant": {}, "deltaquant": {}}
    for variant in ("svdquant", "deltaquant"):
        keys = ["y_nmse", "act_nmse", "act_nmse_video"]
        entries = [e[variant] for case in results["cases"].values() for e in case["steps"]]
        if variant == "deltaquant":
            merged = {name: {k: [e[f"cube{name}"][k] for e in entries] for k in keys} for name in CUBES}
            summary[variant] = {name: {k: sum(v) / len(v) for k, v in dims.items()} for name, dims in merged.items()}
        else:
            summary[variant] = {k: sum(e[k] for e in entries) / len(entries) for k in keys}
    results["summary"] = summary

    out = args.output or (args.input_dir / f"compare_{args.layer.replace('.', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n=== mean over all steps/cases ===")
    s = summary["svdquant"]
    print(f"SVDQuant  : y_nmse={s['y_nmse']:.4%} act_nmse={s['act_nmse']:.4%} act_video={s['act_nmse_video']:.4%}")
    for name, dims in summary["deltaquant"].items():
        print(f"DeltaQ c{name:>2}: y_nmse={dims['y_nmse']:.4%} act_nmse={dims['act_nmse']:.4%} act_video={dims['act_nmse_video']:.4%}")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
