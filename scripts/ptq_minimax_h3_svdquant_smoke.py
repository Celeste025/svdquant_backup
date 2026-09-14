#!/usr/bin/env python3
"""Block-sequential MiniMax-H3 W4A4 + rank-32 SVDQuant smoke PTQ.

The search objective is one H3 block endpoint.  Thus fused QKV, fused SwiGLU,
packed attention metadata and modality-aware AdaLN gates all participate without
inventing a lossy generic batch representation.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn

from deepcompressor.nn.patch.lowrank import LowRankBranch
from minimax_h3_svdquant_common import (
    TARGET_SUFFIXES, aggregate_nmse, install_runtime_hooks, load_h3_pipeline,
    module_fingerprint, nmse, nvfp4_qdq, raw_call_to_block0, run_block,
    target_linears, temporary_quantized_linear, tree_device, validate_state_manifest,
)


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=Path("/home/wjq/workspace/svdquant-exp/results/calib/minimax_h3_smoke_2p8s"))
    p.add_argument("--output-dir", type=Path, default=Path("/home/wjq/workspace/svdquant-exp/results/checkpoints/minimax_h3_svdquant_smoke"))
    p.add_argument("--max-blocks", type=int, default=50)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--num-iters", type=int, default=2)
    p.add_argument("--num-grids", type=int, default=2)
    p.add_argument("--element-size", type=int, default=128)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_samples(cache_dir: Path) -> list[dict]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    paths = [cache_dir / x["file"] for x in manifest["samples"]]
    if len(paths) != 8 or len({(x["prompt_id"], x["step"]) for x in manifest["samples"]}) != 8:
        raise RuntimeError("Calibration cache must contain exactly 8 distinct prompt/timestep samples")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def capture_inputs(block: nn.Module, linear: nn.Linear, samples) -> list[torch.Tensor]:
    values = []
    handle = linear.register_forward_pre_hook(lambda _m, a: values.append(a[0].detach().cpu()))
    try:
        run_block(block, samples)
    finally:
        handle.remove()
    if len(values) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} captured activations, got {len(values)}")
    return values


@torch.no_grad()
def channel_absmax(inputs: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    value = None
    for x in inputs:
        current = x.to(device).float().abs().flatten(0, -2).amax(dim=0)
        value = current if value is None else torch.maximum(value, current)
    return value.clamp_min(1e-8)


def smoothing_candidates(linear: nn.Linear, inputs: list[torch.Tensor], num_grids: int) -> list[torch.Tensor]:
    if num_grids != 2:
        raise ValueError("The smoke recipe intentionally requires num_grids=2")
    x_max = channel_absmax(inputs, linear.weight.device)
    w_max = linear.weight.detach().float().abs().amax(dim=0).clamp_min(1e-8)
    candidates = []
    for alpha in (0.4, 0.6):
        scale = x_max.pow(alpha) / w_max.pow(1.0 - alpha)
        candidates.append(scale.clamp(1e-4, 1e4).to(linear.weight.dtype))
    return candidates


def block_error(block, samples, refs, linear, qweight, smooth, a=None, b=None, element_size=128):
    with temporary_quantized_linear(linear, qweight, smooth, a, b, element_size=element_size) as hooks:
        value = aggregate_nmse(run_block(block, samples), refs)
        if hooks.act.calls != len(samples):
            raise RuntimeError("activation QDQ hook did not fire once per calibration sample")
    return value


@torch.no_grad()
def optimize_linear(block, name, linear, samples, *, rank, num_iters, num_grids, element_size):
    refs = run_block(block, samples)
    if linear.weight.is_meta:
        linear.load_from_disk(torch.bfloat16, "cuda", assign=True)
    else:
        linear.to(device="cuda", dtype=torch.bfloat16)
    if hasattr(linear, "state"):
        linear.state = 2
    inputs = capture_inputs(block, linear, samples)
    w = linear.weight.detach()

    smooth_records = []
    for grid_id, smooth in enumerate(smoothing_candidates(linear, inputs, num_grids)):
        ws = w * smooth
        qweight = nvfp4_qdq(ws, element_size=element_size)
        error = block_error(block, samples, refs, linear, qweight, smooth, element_size=element_size)
        smooth_records.append((error, grid_id, smooth.detach().cpu()))
        del ws, qweight
        torch.cuda.empty_cache()
    smooth_error, smooth_id, smooth_cpu = min(smooth_records, key=lambda x: x[0])
    smooth = smooth_cpu.to(w)

    # Exact BF16 coordinate-change check before quantization.
    test_x = inputs[0].to(w.device).flatten(0, -2)[:128]
    equiv = nmse(torch.nn.functional.linear(test_x / smooth, w * smooth),
                 torch.nn.functional.linear(test_x, w))
    if equiv >= 5e-5:
        raise RuntimeError(f"{name}: smoothing BF16 equivalence failed: {equiv}")

    candidate_records = []
    ws = w * smooth
    for candidate_id in range(num_iters):
        torch.manual_seed(91021 + candidate_id)
        branch = LowRankBranch(linear.in_features, linear.out_features, rank=rank, weight=ws)
        a = branch.a.weight.detach()
        b = branch.b.weight.detach()
        qweight = nvfp4_qdq(ws - b @ a, element_size=element_size)
        error = block_error(block, samples, refs, linear, qweight, smooth, a, b, element_size=element_size)
        candidate_records.append((error, candidate_id, qweight.detach().cpu(), a.detach().cpu(), b.detach().cpu()))
        del branch, qweight, a, b
        torch.cuda.empty_cache()
    best_error, best_id, qweight_cpu, a_cpu, b_cpu = min(candidate_records, key=lambda x: x[0])
    linear.weight.data.copy_(qweight_cpu.to(linear.weight))
    runtime = install_runtime_hooks(linear, smooth_cpu, a_cpu, b_cpu, element_size=element_size)
    state = {
        "smooth": smooth_cpu, "a": a_cpu, "b": b_cpu,
        "smooth_grid": smooth_id, "lowrank_candidate": best_id,
        "smooth_block_nmse": smooth_error, "svdquant_block_nmse": best_error,
        "shape": list(w.shape), "activation_qdq": "dynamic-real-nvfp4",
        "weight_qdq": "real-nvfp4", "group_size": 16,
    }
    print(f"  {name}: smooth#{smooth_id}={smooth_error:.6g}, lr#{best_id}={best_error:.6g}", flush=True)
    return state, runtime


def main() -> None:
    args = args_parser()
    if not (1 <= args.max_blocks <= 50):
        raise ValueError("max-blocks must be in [1, 50]")
    if (args.rank, args.num_iters, args.num_grids, args.element_size) != (32, 2, 2, 128):
        raise ValueError("Smoke contract requires rank32, num_iters2, num_grids2, element_size128")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    targets = target_linears(dit, max_blocks=args.max_blocks)
    print(f"validated structure: 50 blocks; selected {len(targets)} / 200 main-block Linear", flush=True)
    if args.verify_only:
        print("verify-only passed", flush=True)
        return
    samples_raw = load_samples(args.cache_dir)

    target_ids = {id(module.weight) for _, module in targets}
    untouched_before = module_fingerprint(dit, excluded_ids=target_ids)
    # BF16 replay determinism is the cache replay reference without persisting logits.
    one = tree_device(samples_raw[0], "cuda")
    out1 = tuple(x.detach().cpu() for x in dit(*one["input_args"], **one["input_kwargs"]))
    for module in dit.modules():
        if module is not dit and hasattr(module, "offload"):
            module.offload()
    torch.cuda.empty_cache()
    out2 = tuple(x.detach().cpu() for x in dit(*one["input_args"], **one["input_kwargs"]))
    replay_nmse = max(nmse(a, b) for a, b in zip(out1, out2, strict=True))
    for module in dit.modules():
        if module is not dit and hasattr(module, "offload"):
            module.offload()
    torch.cuda.empty_cache()
    del out1, out2, one

    block_samples = [raw_call_to_block0(dit, sample) for sample in samples_raw]
    all_state = {
        "format": "minimax-h3-svdquant-smoke-v1",
        "config": {"rank": 32, "num_iters": 2, "num_grids": 2, "group_size": 16,
                   "element_size": 128, "sample_size": 8, "sample_batch_size": 1,
                   "quant": "real-NVFP4 W4A4", "objective": "H3 block OutputsError"},
        "cache_replay_nmse": replay_nmse, "layers": {},
    }
    runtimes = []
    for block_id in range(args.max_blocks):
        block = dit.blocks[block_id]
        print(f"block {block_id}/{args.max_blocks - 1}", flush=True)
        for suffix in TARGET_SUFFIXES:
            linear = block
            for part in suffix.split("."):
                linear = getattr(linear, part)
            name = f"blocks.{block_id}.{suffix}"
            state, runtime = optimize_linear(
                block, name, linear, block_samples, rank=args.rank, num_iters=args.num_iters,
                num_grids=args.num_grids, element_size=args.element_size,
            )
            all_state["layers"][name] = state
            runtimes.append(runtime)
        # Block-0 smoke gate: all four activation and LowRank hooks must have fired.
        outputs = run_block(block, block_samples)
        if block_id == 0:
            if len(runtimes) != 4 or any(r.act.calls == 0 or r.branch is None for r in runtimes):
                raise RuntimeError("block 0 hook trigger check failed")
            print("block 0 hook trigger check passed", flush=True)
        block_samples = [(out, kw) for out, (_, kw) in zip(outputs, block_samples, strict=True)]
        for suffix in TARGET_SUFFIXES:
            module = block
            for part in suffix.split("."):
                module = getattr(module, part)
            if hasattr(module, "offload"):
                module.offload()
        block_state = {**all_state, "completed_blocks": block_id + 1}
        torch.save(block_state, args.output_dir / f"progress_block_{block_id:02d}.pt")
        (args.output_dir / "progress.json").write_text(json.dumps({
            "completed_blocks": block_id + 1, "total_blocks": args.max_blocks,
            "last_block": block_id, "layers": len(all_state["layers"]),
        }, indent=2))
        gc.collect(); torch.cuda.empty_cache()

    validate_state_manifest(all_state, args.max_blocks * 4)
    untouched_after = module_fingerprint(dit, excluded_ids=target_ids)
    if untouched_before != untouched_after:
        changed = sorted(k for k in untouched_before if untouched_before[k] != untouched_after.get(k))
        raise RuntimeError(f"Non-target parameter fingerprint changed: {changed[:8]}")
    final_outputs = outputs
    if any(not torch.isfinite(x).all() for x in final_outputs):
        raise RuntimeError("NaN/Inf after final processed block")
    all_state["checks"] = {
        "structure_50_blocks": True, "selected_linears": args.max_blocks * 4,
        "bf16_cache_replay_nmse": replay_nmse, "block0_hooks": True,
        "untouched_fingerprint": True, "finite_block_output": True,
    }
    torch.save(all_state, args.output_dir / "quant_state.pt")
    (args.output_dir / "summary.json").write_text(json.dumps({
        **all_state["config"], **all_state["checks"], "state": "quant_state.pt",
    }, indent=2))
    print(f"PTQ smoke state saved: {args.output_dir / 'quant_state.pt'}", flush=True)


if __name__ == "__main__":
    main()
