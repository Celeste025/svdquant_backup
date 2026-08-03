#!/usr/bin/env python3
"""Calibrate Wan SVDQuant fake-quant parameters with output-error search.

The calibration follows DeepCompressor's diffusion SVDQuant defaults:
AbsMax/AbsMax smoothing, the 39-candidate (20-grid) alpha/beta search,
shared rank-32 QKV/KV branches, alternating low-rank/residual quantization,
an output-MSE objective, and early stopping (at most 100 iterations).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
try:
    from diffusers import WanTransformer3DModel
except ImportError:
    WanTransformer3DModel = None


DEFAULT_MODEL = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
)


def fake_quant_s4(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    shape = x.shape
    groups = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-6).div_(7)
    return groups.div(scale).round_().clamp_(-7, 7).mul_(scale).reshape(shape)


def fake_quant_u4(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    shape = x.shape
    groups = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    scale = groups.amax(dim=-1, keepdim=True).clamp_min_(1e-6).div_(15)
    return groups.div(scale).round_().clamp_(0, 15).mul_(scale).reshape(shape)


def seed_for(name: str, candidate: int, iteration: int) -> int:
    digest = hashlib.sha256(f"{name}:{candidate}:{iteration}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def low_rank(weight: torch.Tensor, rank: int, *, niter: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    u, s, v = torch.svd_lowrank(weight, q=rank, niter=niter)
    up = u * s.unsqueeze(0)
    down = v.transpose(0, 1)
    return up, down


def candidate_pairs(num_grids: int = 20) -> list[tuple[float, float]]:
    choices = [i / num_grids for i in range(1, num_grids)]
    # DeepCompressor: alpha=0.5 (>0), beta=-2, GridSearch.
    return [(0.0, 0.0)] + [(a, 0.0) for a in choices] + [(a, 1.0 - a) for a in choices]


def module_groups() -> list[list[str]]:
    groups: list[list[str]] = []
    for block in range(30):
        p = f"blocks.{block}"
        groups.extend(
            [
                [f"{p}.attn1.to_q", f"{p}.attn1.to_k", f"{p}.attn1.to_v"],
                [f"{p}.attn1.to_out.0"],
                [f"{p}.attn2.to_q"],
                [f"{p}.attn2.to_k", f"{p}.attn2.to_v"],
                [f"{p}.attn2.to_out.0"],
                [f"{p}.ffn.net.0.proj"],
                [f"{p}.ffn.net.2"],
            ]
        )
    return groups


@torch.no_grad()
def calibrate_group(
    names: list[str],
    weights: list[torch.Tensor],
    x: torch.Tensor,
    *,
    group_size: int,
    rank: int,
    max_iters: int,
    search_niter: int,
    final_niter: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    group_name = "+".join(names)
    x = x.to(device=device, dtype=torch.float32)
    # DeepCompressor's pipeline.shift_activations patch shifts the GELU output
    # before FFN down projections by -GELU_min, enabling unsigned INT4.
    unsigned_activation = len(names) == 1 and names[0].endswith(".ffn.net.2")
    input_shift = 0.171875 if unsigned_activation else 0.0
    if input_shift:
        x = x + input_shift
    weight = torch.cat([w.to(device=device, dtype=torch.float32) for w in weights], dim=0)
    target = x @ weight.transpose(0, 1)
    target_energy = target.square().mean().clamp_min_(1e-12)
    x_span = x.abs().amax(dim=0).clamp_min_(1e-6)
    w_span = weight.abs().amax(dim=0).clamp_min_(1e-6)

    best = None
    search_records = []
    for candidate, (alpha, beta) in enumerate(candidate_pairs()):
        if alpha == 0 and beta == 0:
            smooth = torch.ones_like(x_span)
        else:
            smooth = x_span.pow(alpha).div_(w_span.pow(beta)).clamp_(1e-4, 1e4)
        xs = x / smooth
        xq = fake_quant_u4(xs, group_size) if unsigned_activation else fake_quant_s4(xs, group_size)
        ws = weight * smooth.unsqueeze(0)
        up, down = low_rank(
            ws, rank, niter=search_niter, seed=seed_for(group_name, candidate, 0)
        )
        qw = fake_quant_s4(ws - up @ down, group_size)
        prediction = xq @ qw.transpose(0, 1) + (xs @ down.transpose(0, 1)) @ up.transpose(0, 1)
        error = (prediction - target).square().mean().div(target_energy).item()
        search_records.append({"alpha": alpha, "beta": beta, "nmse": error})
        if best is None or error < best[0]:
            best = (error, alpha, beta, smooth.detach().clone())
        del xs, xq, ws, up, down, qw, prediction

    _, best_alpha, best_beta, smooth = best
    xs = x / smooth
    xq = fake_quant_u4(xs, group_size) if unsigned_activation else fake_quant_s4(xs, group_size)
    ws = weight * smooth.unsqueeze(0)
    qw: torch.Tensor | int = 0
    best_iter = None
    iteration_records = []
    for iteration in range(max_iters):
        up, down = low_rank(
            ws - qw,
            rank,
            niter=final_niter,
            seed=seed_for(group_name, -1, iteration),
        )
        candidate_qw = fake_quant_s4(ws - up @ down, group_size)
        prediction = (
            xq @ candidate_qw.transpose(0, 1)
            + (xs @ down.transpose(0, 1)) @ up.transpose(0, 1)
        )
        error = (prediction - target).square().mean().div(target_energy).item()
        iteration_records.append(error)
        if best_iter is None or error <= best_iter[0]:
            best_iter = (
                error,
                iteration,
                candidate_qw.detach().clone(),
                up.detach().clone(),
                down.detach().clone(),
            )
            qw = candidate_qw
        else:
            break
        del prediction

    error, chosen_iter, qw, up, down = best_iter
    state: dict[str, dict[str, torch.Tensor]] = {}
    offset = 0
    for name, original_weight in zip(names, weights, strict=True):
        out_features = original_weight.shape[0]
        state[name] = {
            "smooth": smooth.to(device="cpu", dtype=torch.bfloat16),
            "qweight": qw[offset : offset + out_features].to(device="cpu", dtype=torch.bfloat16),
            "down": down.to(device="cpu", dtype=torch.bfloat16),
            "up": up[offset : offset + out_features].to(device="cpu", dtype=torch.bfloat16),
            "input_shift": torch.tensor(input_shift, dtype=torch.bfloat16),
            "unsigned_activation": torch.tensor(unsigned_activation),
        }
        offset += out_features
    report = {
        "names": names,
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "search_nmse": best[0],
        "final_nmse": error,
        "chosen_iteration": chosen_iter,
        "iterations_run": len(iteration_records),
        "input_shift": input_shift,
        "unsigned_activation": unsigned_activation,
        "iteration_nmse": iteration_records,
        "search": search_records,
    }
    return state, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--calib", type=Path, default=Path("outputs/wan_svdquant_fake/calib_seed44_tokens.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/wan_svdquant_fake/calibrated_shards")
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--search-niter", type=int, default=1)
    parser.add_argument("--final-niter", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.calib, map_location="cpu", weights_only=False)
    inputs = payload["inputs"]
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    modules = dict(transformer.named_modules())
    groups = module_groups()
    selected = [(i, names) for i, names in enumerate(groups) if i % args.num_shards == args.shard_index]
    state = {}
    reports = []
    device = torch.device("cuda")
    for local_index, (global_index, names) in enumerate(selected, 1):
        print(
            f"[shard {args.shard_index} {local_index:02d}/{len(selected):02d}] "
            f"group {global_index:03d}: {' + '.join(names)}",
            flush=True,
        )
        reference_input = inputs[names[0]]
        if any(not torch.equal(reference_input, inputs[name]) for name in names[1:]):
            raise RuntimeError(f"grouped modules do not share identical calibration inputs: {names}")
        weights = [modules[name].weight.detach().cpu() for name in names]
        group_state, report = calibrate_group(
            names,
            weights,
            reference_input,
            group_size=args.group_size,
            rank=args.rank,
            max_iters=args.max_iters,
            search_niter=args.search_niter,
            final_niter=args.final_niter,
            device=device,
        )
        state.update(group_state)
        report["global_group_index"] = global_index
        reports.append(report)
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    torch.save(
        {
            "format": "wan-svdquant-calibrated-v1",
            "group_size": args.group_size,
            "rank": args.rank,
            "num_grids": 20,
            "max_iters": args.max_iters,
            "state": state,
        },
        args.output_dir / f"{stem}.pt",
    )
    (args.output_dir / f"{stem}.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"saved {len(state)} modules from {len(reports)} groups", flush=True)


if __name__ == "__main__":
    main()
