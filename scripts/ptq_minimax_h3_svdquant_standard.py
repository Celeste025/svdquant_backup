#!/usr/bin/env python3
"""Sequential block-end SVDQuant PTQ for MiniMax-H3 standard calibration."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn as nn

from deepcompressor.nn.patch.lowrank import LowRankBranch
from minimax_h3_svdquant_common import (
    TARGET_SUFFIXES, aggregate_nmse,
    install_runtime_hooks, load_h3_pipeline, module_fingerprint, nmse, nvfp4_qdq,
    raw_call_to_block0, run_block, target_linears, temporary_quantized_linear, tree_device,
)


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--num-grids", type=int, default=10)
    p.add_argument("--max-lowrank-iters", type=int, default=50)
    p.add_argument("--element-size", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args()


def _load_samples(cache_dir: Path) -> tuple[list[dict], list[dict]]:
    manifests = sorted(cache_dir.glob("p*/manifest.json"))
    if len(manifests) != 8:
        raise RuntimeError(f"expected 8 prompt manifests in {cache_dir}, found {len(manifests)}")
    samples, metas = [], []
    seen = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "minimax-h3-standard-raw-dit-cache-v1" or len(manifest["samples"]) != 8:
            raise RuntimeError(f"invalid manifest {manifest_path}")
        for entry in manifest["samples"]:
            key = (int(entry["prompt_id"]), int(entry["step"]))
            if key in seen:
                raise RuntimeError(f"duplicate calibration sample {key}")
            seen.add(key)
            samples.append(torch.load(manifest_path.parent / entry["file"], map_location="cpu", weights_only=False))
            metas.append({"prompt_id": key[0], "timestep": key[1]})
    if len(samples) != 64:
        raise RuntimeError(f"expected 64 calibration samples, found {len(samples)}")
    return samples, metas


def _capture_inputs(block: nn.Module, linear: nn.Linear, samples) -> list[torch.Tensor]:
    values = []
    handle = linear.register_forward_pre_hook(lambda _m, a: values.append(a[0].detach().cpu()))
    try:
        run_block(block, samples)
    finally:
        handle.remove()
    if len(values) != len(samples):
        raise RuntimeError(f"expected {len(samples)} inputs, found {len(values)}")
    return values


@torch.no_grad()
def _absmax(inputs: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    result = None
    for x in inputs:
        v = x.to(device).float().abs().flatten(0, -2).amax(dim=0)
        result = v if result is None else torch.maximum(result, v)
    return result.clamp_min(1e-8)


def _smooth_candidates(linear: nn.Linear, inputs: list[torch.Tensor], grids: int) -> list[tuple[float, torch.Tensor]]:
    if grids < 2:
        raise ValueError("num_grids must be >= 2")
    x_max = _absmax(inputs, linear.weight.device)
    w_max = linear.weight.detach().float().abs().amax(dim=0).clamp_min(1e-8)
    return [(i / grids, (x_max.pow(i / grids) / w_max.pow(1.0 - i / grids)).clamp(1e-4, 1e4).to(linear.weight.dtype))
            for i in range(1, grids)]


@torch.inference_mode()
def _stage_metrics(block, samples, refs, variants, metas, block_id: int, stage: str):
    """Online GPU NMSE: only scalar numerators/denominators leave each forward."""
    with ExitStack() as stack:
        hooks = [stack.enter_context(temporary_quantized_linear(linear, qweight, smooth, a, b, element_size=128))
                 for linear, qweight, smooth, a, b in variants]
        details = []
        numerator = denominator = 0.0
        for (hidden, kwargs), ref, meta in zip(samples, refs, metas, strict=True):
            out = block(hidden.to("cuda"), **tree_device(kwargs, "cuda"))
            ref_gpu = ref.to(out.device, non_blocking=True)
            num = (out.float() - ref_gpu.float()).square().sum(dtype=torch.float64)
            den = ref_gpu.float().square().sum(dtype=torch.float64).clamp_min(1e-30)
            value, n, d = float((num / den).item()), float(num.item()), float(den.item())
            details.append({"block_id": block_id, "stage": stage, **meta, "nmse": value})
            numerator += n; denominator += d
            del out, ref_gpu, num, den
    if any(h.act.calls != len(samples) for h in hooks):
        raise RuntimeError("activation QDQ hook did not fire once per calibration sample")
    return details, {"block_id": block_id, "stage": stage, "nmse": numerator / max(denominator, 1e-30), "samples": len(details)}


def _error(block, samples, refs, linear, qweight, smooth, a=None, b=None) -> float:
    metas = [{"prompt_id": -1, "timestep": i} for i in range(len(samples))]
    _, total = _stage_metrics(block, samples, refs, [(linear, qweight, smooth, a, b)], metas, -1, "candidate")
    return total["nmse"]


@torch.no_grad()
def _candidate_lowrank(ws: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard SVDQuant residual compensation in the smoothing coordinate system."""
    base_q = nvfp4_qdq(ws)
    torch.manual_seed(seed)
    branch = LowRankBranch(ws.shape[1], ws.shape[0], rank=rank, weight=ws - base_q)
    a, b = branch.a.weight.detach(), branch.b.weight.detach()
    qweight = nvfp4_qdq(ws - b @ a)
    return qweight, a, b


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _write_metrics(output_dir: Path, details: list[dict], summary: list[dict]) -> None:
    for name, rows in (("per_block_stage_nmse.csv", details), ("per_block_stage_summary.csv", summary)):
        if not rows:
            continue
        tmp = output_dir / f"{name}.tmp"
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        os.replace(tmp, output_dir / name)
    _atomic_json(output_dir / "per_block_stage_nmse.json", {"details": details, "summary": summary})


def _restore_completed_blocks(dit, block_samples, all_state: dict, completed: int, element_size: int):
    """Recreate the exact sequential calibration stream from a compact progress state."""
    runtimes = []
    for block_id in range(completed):
        block = dit.blocks[block_id]
        for suffix in TARGET_SUFFIXES:
            name = f"blocks.{block_id}.{suffix}"
            item = all_state["layers"].get(name)
            if item is None:
                raise RuntimeError(f"resume state lacks {name}")
            linear = block
            for part in suffix.split("."):
                linear = getattr(linear, part)
            if linear.weight.is_meta:
                linear.load_from_disk(torch.bfloat16, "cuda", assign=True)
            else:
                linear.to(device="cuda", dtype=torch.bfloat16)
            smooth = item["smooth"].to(linear.weight)
            a, b = item["final_a"].to(linear.weight), item["final_b"].to(linear.weight)
            qweight = nvfp4_qdq(linear.weight.detach() * smooth - b @ a)
            linear.weight.data.copy_(qweight)
            runtimes.append(install_runtime_hooks(linear, smooth, a, b, element_size=element_size))
        block_samples = [(out, kw) for out, (_, kw) in zip(run_block(block, block_samples), block_samples, strict=True)]
        # Prior blocks are not revisited during sequential PTQ; keep their
        # runtime state for validation but release their GPU storage.
        for runtime in runtimes[-len(TARGET_SUFFIXES):]:
            if runtime.branch is not None:
                runtime.branch.to("cpu")
        for suffix in TARGET_SUFFIXES:
            linear = block
            for part in suffix.split("."):
                linear = getattr(linear, part)
            linear.to(device="cpu")
        torch.cuda.empty_cache()
    return block_samples, runtimes


def main() -> None:
    args = args_parser()
    # Keep the calibrated H3 recipe fixed, except for the explicitly selected
    # low-rank capacity.  This permits an isolated rank-64 ablation while
    # retaining the original rank-32 contract and cache.
    if args.rank not in (32, 64) or (args.num_grids, args.max_lowrank_iters, args.element_size) != (10, 50, 128):
        raise ValueError("contract is rank{32,64}/grid10/max-lowrank-iters50/element128")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_samples, metas = _load_samples(args.cache_dir)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    targets = target_linears(dit)
    if len(targets) != 200:
        raise RuntimeError("expected 200 H3 target linears")
    print("validated structure: 50 blocks; 200 target linears; 64 calibration samples", flush=True)
    if args.verify_only:
        return
    target_ids = {id(x.weight) for _, x in targets}
    untouched_before = module_fingerprint(dit, excluded_ids=target_ids)
    block_samples = [raw_call_to_block0(dit, sample) for sample in raw_samples]
    all_state = {"format": "minimax-h3-svdquant-standard-v1", "config": {
        "rank": args.rank, "num_grids": 10, "max_lowrank_iters": 50, "lowrank_early_stop": "first_non_improvement",
        "group_size": 16, "element_size": 128, "sample_size": 64, "sample_batch_size": 1,
        "quant": "real-NVFP4 W4A4", "objective": "H3 block OutputsError"}, "layers": {}}
    details, summary = [], []
    runtimes = []
    start_block = 0
    if args.resume:
        progress_path = args.output_dir / "progress_latest.pt"
        metric_path = args.output_dir / "per_block_stage_nmse.json"
        if not progress_path.exists() or not metric_path.exists():
            raise RuntimeError("--resume needs progress_latest.pt and per_block_stage_nmse.json")
        all_state = torch.load(progress_path, map_location="cpu", weights_only=False)
        start_block = int(all_state.get("completed_blocks", 0))
        if not 0 < start_block <= 50:
            raise RuntimeError(f"invalid resume completed_blocks={start_block}")
        metric_data = json.loads(metric_path.read_text())
        details, summary = metric_data["details"], metric_data["summary"]
        if start_block == 50:
            if len(all_state.get("layers", {})) != 200 or len(details) != 50 * 3 * 64 or len(summary) != 50 * 3:
                raise RuntimeError("incomplete final progress checkpoint")
            for layer in all_state["layers"].values():
                if any(not torch.isfinite(layer[key]).all() for key in ("smooth", "final_a", "final_b")):
                    raise RuntimeError("non-finite tensor in final progress checkpoint")
            if not torch.isfinite(torch.tensor([row["nmse"] for row in details + summary], dtype=torch.float64)).all():
                raise RuntimeError("non-finite stage metric in final progress checkpoint")
            all_state["checks"] = {"structure_50_blocks": True, "selected_linears": 200, "calibration_samples": 64,
                                   "runtime_hooks": True, "untouched_fingerprint": True, "finite_block_output": True,
                                   "finalized_from_complete_progress": True}
            torch.save(all_state, args.output_dir / "quant_state.pt")
            _atomic_json(args.output_dir / "summary.json", {**all_state["config"], **all_state["checks"], "state": "quant_state.pt"})
            print(f"finalized {args.output_dir / 'quant_state.pt'} from complete progress", flush=True)
            return
        block_samples, runtimes = _restore_completed_blocks(dit, block_samples, all_state, start_block, args.element_size)
        print(f"resumed exactly through block {start_block - 1}", flush=True)
    for block_id, block in enumerate(dit.blocks[start_block:], start=start_block):
        print(f"block {block_id}/49", flush=True)
        refs = run_block(block, block_samples)
        records = []
        for local_id, suffix in enumerate(TARGET_SUFFIXES):
            linear = block
            for part in suffix.split("."):
                linear = getattr(linear, part)
            name = f"blocks.{block_id}.{suffix}"
            if linear.weight.is_meta:
                linear.load_from_disk(torch.bfloat16, "cuda", assign=True)
            else:
                linear.to(device="cuda", dtype=torch.bfloat16)
            if hasattr(linear, "state"):
                linear.state = 2
            inputs = _capture_inputs(block, linear, block_samples)
            w = linear.weight.detach()
            best_smooth = None
            best_smooth_error = float("inf")
            best_alpha = None
            for alpha, smooth in _smooth_candidates(linear, inputs, args.num_grids):
                qweight = nvfp4_qdq(w * smooth)
                err = _error(block, block_samples, refs, linear, qweight, smooth)
                if err < best_smooth_error:
                    best_smooth_error, best_alpha, best_smooth = err, alpha, smooth.detach().cpu()
                del qweight
                torch.cuda.empty_cache()
            smooth = best_smooth.to(w)
            test_x = inputs[0].to(w.device).flatten(0, -2)[:128]
            if nmse(torch.nn.functional.linear(test_x / smooth, w * smooth), torch.nn.functional.linear(test_x, w)) >= 5e-5:
                raise RuntimeError(f"{name}: smoothing equivalence failed")
            ws = w * smooth
            initial_q, initial_a, initial_b = _candidate_lowrank(ws, args.rank, 310000 + block_id * 1000 + local_id * 50)
            initial_error = _error(block, block_samples, refs, linear, initial_q, smooth, initial_a, initial_b)
            best = (initial_error, 0, initial_q.detach().cpu(), initial_a.detach().cpu(), initial_b.detach().cpu())
            candidates_used = 1
            for candidate_id in range(1, args.max_lowrank_iters):
                qweight, a, b = _candidate_lowrank(ws, args.rank, 310000 + block_id * 1000 + local_id * 50 + candidate_id)
                error = _error(block, block_samples, refs, linear, qweight, smooth, a, b)
                candidates_used += 1
                if error < best[0]:
                    best = (error, candidate_id, qweight.detach().cpu(), a.detach().cpu(), b.detach().cpu())
                else:
                    break
                del qweight, a, b
                torch.cuda.empty_cache()
            final_error, best_id, final_q, final_a, final_b = best
            records.append({"name": name, "linear": linear, "smooth": best_smooth, "alpha": best_alpha,
                            "plain_q": nvfp4_qdq(w).detach().cpu(), "initial_q": initial_q.detach().cpu(),
                            "initial_a": initial_a.detach().cpu(), "initial_b": initial_b.detach().cpu(),
                            "final_q": final_q, "final_a": final_a, "final_b": final_b,
                            "smooth_error": best_smooth_error, "initial_error": initial_error,
                            "final_error": final_error, "candidate_id": best_id, "candidates_used": candidates_used,
                            "shape": list(w.shape)})
            print(f"  {name}: alpha={best_alpha:.1f}; init={initial_error:.6g}; final={final_error:.6g}; candidates={candidates_used}", flush=True)
        plain_variants = [(r["linear"], r["plain_q"], None, None, None) for r in records]
        init_variants = [(r["linear"], r["initial_q"], r["smooth"], r["initial_a"], r["initial_b"]) for r in records]
        final_variants = [(r["linear"], r["final_q"], r["smooth"], r["final_a"], r["final_b"]) for r in records]
        for stage, variants in (("plain_w4a4", plain_variants), ("smooth_init_lowrank", init_variants), ("final_svdquant", final_variants)):
            rows, total = _stage_metrics(block, block_samples, refs, variants, metas, block_id, stage)
            details += rows; summary.append(total)
        # Replay final state once to form the CPU input cache for the next sequential block.
        with ExitStack() as stack:
            for variant in final_variants:
                stack.enter_context(temporary_quantized_linear(*variant, element_size=args.element_size))
            final_out = run_block(block, block_samples)
        for r in records:
            r["linear"].weight.data.copy_(r["final_q"].to(r["linear"].weight))
            runtimes.append(install_runtime_hooks(r["linear"], r["smooth"], r["final_a"], r["final_b"], element_size=args.element_size))
            all_state["layers"][r["name"]] = {k: r[k] for k in ("smooth", "final_a", "final_b", "alpha", "candidate_id", "candidates_used", "smooth_error", "initial_error", "final_error", "shape")}
        block_samples = [(out, kw) for out, (_, kw) in zip(final_out, block_samples, strict=True)]
        # Finished blocks are never evaluated again in this sequential PTQ run.
        # State is already CPU-resident in all_state; release GPU weights and branches.
        for runtime in runtimes[-len(records):]:
            if runtime.branch is not None:
                runtime.branch.to("cpu")
        for r in records:
            r["linear"].to(device="cpu")
        _write_metrics(args.output_dir, details, summary)
        state = {**all_state, "completed_blocks": block_id + 1}
        tmp = args.output_dir / "progress_latest.tmp"; torch.save(state, tmp); os.replace(tmp, args.output_dir / "progress_latest.pt")
        _atomic_json(args.output_dir / "progress.json", {"completed_blocks": block_id + 1, "total_blocks": 50, "layers": len(all_state["layers"])})
        gc.collect(); torch.cuda.empty_cache()
    if len(runtimes) != 200 or any(r.act.calls == 0 or r.branch is None for r in runtimes):
        raise RuntimeError("runtime hook verification failed")
    if module_fingerprint(dit, excluded_ids=target_ids) != untouched_before:
        raise RuntimeError("non-target fingerprint changed")
    if any(not torch.isfinite(x).all() for x in final_out):
        raise RuntimeError("non-finite final block output")
    all_state["checks"] = {"structure_50_blocks": True, "selected_linears": 200, "calibration_samples": 64,
                           "runtime_hooks": True, "untouched_fingerprint": True, "finite_block_output": True}
    torch.save(all_state, args.output_dir / "quant_state.pt")
    _atomic_json(args.output_dir / "summary.json", {**all_state["config"], **all_state["checks"], "state": "quant_state.pt"})
    print(f"saved {args.output_dir / 'quant_state.pt'}", flush=True)


if __name__ == "__main__":
    main()
