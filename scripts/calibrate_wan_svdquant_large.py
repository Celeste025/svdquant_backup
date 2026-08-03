#!/usr/bin/env python3
"""Run output-error SVDQuant calibration over the large memmap dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
try:
    from diffusers import WanTransformer3DModel
except ImportError:
    WanTransformer3DModel = None

from calibrate_wan_svdquant import (
    candidate_pairs,
    fake_quant_s4,
    fake_quant_u4,
    low_rank,
    module_groups,
    seed_for,
)


DEFAULT_MODEL = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
)


def fake_quant_signed(
    x: torch.Tensor, *, bits: int, group_size: int
) -> torch.Tensor:
    if bits == 4:
        return fake_quant_s4(x, group_size)
    if bits != 8:
        raise ValueError(f"unsupported weight bits: {bits}")
    shape = x.shape
    groups = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-6).div_(127)
    return groups.div(scale).round_().clamp_(-127, 127).mul_(scale).reshape(shape)


def open_bfloat16_memmap(root: Path, record: dict) -> torch.Tensor:
    shape = tuple(record["shape"])
    raw = np.memmap(root / record["path"], dtype=np.uint16, mode="r", shape=shape)
    return torch.from_numpy(raw).view(torch.bfloat16)


@torch.no_grad()
def input_absmax(
    inputs: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    input_shift: float,
) -> torch.Tensor:
    result = None
    for start in range(0, inputs.shape[0], batch_size):
        x = inputs[start : start + batch_size].to(device=device, dtype=torch.float32)
        if input_shift:
            x.add_(input_shift)
        span = x.abs().amax(dim=0)
        result = span if result is None else torch.maximum(result, span)
    return result.clamp_min_(1e-6)


@torch.no_grad()
def output_nmse(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    smooth: torch.Tensor,
    qweight: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    batch_size: int,
    group_size: int,
    unsigned_activation: bool,
    input_shift: float,
    device: torch.device,
    quantize_activation: bool = True,
) -> float:
    error_sum = torch.zeros((), device=device, dtype=torch.float64)
    target_sum = torch.zeros((), device=device, dtype=torch.float64)
    wt = weight.transpose(0, 1)
    qwt = qweight.transpose(0, 1)
    dt = down.transpose(0, 1)
    ut = up.transpose(0, 1)
    for start in range(0, inputs.shape[0], batch_size):
        x = inputs[start : start + batch_size].to(device=device, dtype=torch.float32)
        if input_shift:
            x.add_(input_shift)
        xs = x / smooth
        if quantize_activation:
            xq = (
                fake_quant_u4(xs, group_size)
                if unsigned_activation
                else fake_quant_s4(xs, group_size)
            )
        else:
            xq = xs
        target = x @ wt
        prediction = xq @ qwt + (xs @ dt) @ ut
        error_sum.add_((prediction - target).double().square().sum())
        target_sum.add_(target.double().square().sum())
    return error_sum.div_(target_sum.clamp_min_(1e-12)).item()


@torch.no_grad()
def calibrate_group(
    names: list[str],
    weights: list[torch.Tensor],
    inputs: torch.Tensor,
    *,
    batch_size: int,
    group_size: int,
    rank: int,
    max_iters: int,
    search_niter: int,
    final_niter: int,
    quantize_activation: bool,
    weight_bits: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    group_name = "+".join(names)
    unsigned_activation = len(names) == 1 and (
        names[0].endswith(".ffn.net.2") or names[0].endswith(".ffn.2")
    )
    input_shift = 0.171875 if unsigned_activation else 0.0
    weight = torch.cat([w.to(device=device, dtype=torch.float32) for w in weights], dim=0)
    x_span = input_absmax(
        inputs, batch_size=batch_size, device=device, input_shift=input_shift
    )
    w_span = weight.abs().amax(dim=0).clamp_min_(1e-6)

    best = None
    search_records = []
    for candidate, (alpha, beta) in enumerate(candidate_pairs()):
        if alpha == 0 and beta == 0:
            smooth = torch.ones_like(x_span)
        else:
            smooth = x_span.pow(alpha).div_(w_span.pow(beta)).clamp_(1e-4, 1e4)
        ws = weight * smooth.unsqueeze(0)
        up, down = low_rank(
            ws, rank, niter=search_niter, seed=seed_for(group_name, candidate, 0)
        )
        qw = fake_quant_signed(
            ws - up @ down, bits=weight_bits, group_size=group_size
        )
        error = output_nmse(
            inputs,
            weight,
            smooth,
            qw,
            up,
            down,
            batch_size=batch_size,
            group_size=group_size,
            unsigned_activation=unsigned_activation,
            quantize_activation=quantize_activation,
            input_shift=input_shift,
            device=device,
        )
        search_records.append({"alpha": alpha, "beta": beta, "nmse": error})
        if best is None or error < best[0]:
            best = (error, alpha, beta, smooth.detach().clone())

    _, best_alpha, best_beta, smooth = best
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
        candidate_qw = fake_quant_signed(
            ws - up @ down, bits=weight_bits, group_size=group_size
        )
        error = output_nmse(
            inputs,
            weight,
            smooth,
            candidate_qw,
            up,
            down,
            batch_size=batch_size,
            group_size=group_size,
            unsigned_activation=unsigned_activation,
            quantize_activation=quantize_activation,
            input_shift=input_shift,
            device=device,
        )
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

    error, chosen_iter, qw, up, down = best_iter
    state = {}
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
    return state, {
        "names": names,
        "num_input_tokens": inputs.shape[0],
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "search_nmse": best[0],
        "final_nmse": error,
        "chosen_iteration": chosen_iter,
        "iterations_run": len(iteration_records),
        "iteration_nmse": iteration_records,
        "input_shift": input_shift,
        "unsigned_activation": unsigned_activation,
        "quantize_activation": quantize_activation,
        "weight_bits": weight_bits,
        "search": search_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--calib-dir", type=Path, default=Path("outputs/wan_svdquant_calib_large")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wan_svdquant_calib_large/calibrated_shards"),
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--weight-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--search-niter", type=int, default=1)
    parser.add_argument("--final-niter", type=int, default=4)
    parser.add_argument(
        "--activation-mode", choices=("w4a4", "w4a16"), default="w4a4"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if WanTransformer3DModel is None:
        raise RuntimeError("diffusers is required for the Diffusers Wan calibrator")
    manifest = json.loads((args.calib_dir / "manifest.json").read_text())
    records = {record["group_index"]: record for record in manifest["groups"]}
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
        record = records[global_index]
        if record["names"] != names:
            raise RuntimeError(f"group metadata mismatch at {global_index}")
        # Transfer each calibration group once.  Keeping the full group in BF16
        # costs at most ~5.9 GiB for Wan's FFN-down inputs and avoids repeating
        # the same host-to-device copy for every smoothing/low-rank candidate.
        inputs = open_bfloat16_memmap(args.calib_dir, record).to(
            device=device, dtype=torch.bfloat16
        )
        weights = [modules[name].weight.detach().cpu() for name in names]
        group_state, report = calibrate_group(
            names,
            weights,
            inputs,
            batch_size=args.eval_batch_size,
            group_size=args.group_size,
            rank=args.rank,
            max_iters=args.max_iters,
            search_niter=args.search_niter,
            final_niter=args.final_niter,
            quantize_activation=args.activation_mode == "w4a4",
            weight_bits=args.weight_bits,
            device=device,
        )
        state.update(group_state)
        report["global_group_index"] = global_index
        reports.append(report)
        del inputs, weights, group_state
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    torch.save(
        {
            "format": "wan-svdquant-calibrated-v1",
            "group_size": args.group_size,
            "rank": args.rank,
            "weight_bits": args.weight_bits,
            "num_grids": 20,
            "max_iters": args.max_iters,
            "activation_mode": args.activation_mode,
            "calibration_manifest": str(args.calib_dir / "manifest.json"),
            "state": state,
        },
        args.output_dir / f"{stem}.pt",
    )
    (args.output_dir / f"{stem}.json").write_text(json.dumps(reports, indent=2))
    print(f"saved {len(state)} modules from {len(reports)} groups", flush=True)


if __name__ == "__main__":
    main()
