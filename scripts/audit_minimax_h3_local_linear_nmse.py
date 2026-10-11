#!/usr/bin/env python3
"""Audit local H3 SVDQuant linear-output NMSE on cached BF16 calls.

For every selected cached call and every quantized linear, this measures the
same quantity used by the historical Wan/rCM local reports: both arms receive
the exact BF16 linear input and we compare the direct linear output before and
after installing the frozen W4A4+smoothing+low-rank SVDQuant branch.  The
calibration procedure itself is not changed and no checkpoint is written.
"""
from __future__ import annotations

import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F

from minimax_h3_svdquant_common import (
    H3_DIT_PATH, TARGET_SUFFIXES, nvfp4_qdq, raw_call_to_block0, tree_device,
)
from finetune_minimax_h3_r64_lowrank import linear_for, normal_tree
from minimax_h3_svdquant_common import load_h3_pipeline
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=8,
                        help="One cached timestep per prompt; default is the middle captured step.")
    return parser.parse_args()


def load_samples(cache_dir: Path, step: int):
    samples, meta = [], []
    for manifest_path in sorted(cache_dir.glob("p*/manifest.json"), key=lambda p: int(p.parent.name[1:])):
        manifest = json.loads(manifest_path.read_text())
        matches = [entry for entry in manifest["samples"] if int(entry["step"]) == step]
        if len(matches) != 1:
            raise RuntimeError(f"{manifest_path}: expected exactly one cached step {step}")
        entry = matches[0]
        samples.append(torch.load(manifest_path.parent / entry["file"], map_location="cpu", weights_only=False))
        meta.append({"prompt_id": int(entry["prompt_id"]), "step": int(entry["step"])})
    if len(samples) != 8:
        raise RuntimeError(f"expected eight prompt samples, got {len(samples)}")
    return samples, meta


def metrics(pred: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, int]:
    diff = pred.float() - ref.float()
    err2 = float(diff.square().sum(dtype=torch.float64).item())
    ref2 = float(ref.float().square().sum(dtype=torch.float64).item())
    return err2, ref2, ref.numel()


def bf16_weight(block_id: int, suffix: str) -> torch.Tensor:
    """Load an immutable source weight, independent of runtime offloading."""
    key = f"blocks.{block_id}.{suffix}.weight"
    with safe_open(str(H3_DIT_PATH), framework="pt", device="cpu") as checkpoint:
        if key.removesuffix("weight") + "bias" in checkpoint.keys():
            raise RuntimeError(f"unexpected bias for audited target {key}")
        return checkpoint.get_tensor(key).to("cuda")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    if state.get("format") != "minimax-h3-svdquant-standard-v1" or len(state.get("layers", {})) != 200:
        raise RuntimeError("state must be a complete standard H3 SVDQuant checkpoint")
    samples, metas = load_samples(args.cache_dir, args.step)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    block_samples = [(normal_tree(hidden), normal_tree(kwargs))
                     for hidden, kwargs in (raw_call_to_block0(pipe.dit, sample) for sample in samples)]
    rows: list[dict] = []

    for block_id, block in enumerate(pipe.dit.blocks):
        block.eval()
        block_weights = {suffix: bf16_weight(block_id, suffix) for suffix in TARGET_SUFFIXES}
        for sample_id, ((hidden, kwargs), meta) in enumerate(zip(block_samples, metas, strict=True)):
            # Capture the true BF16 activation/output pair while the block is
            # live.  Source weights are read from the immutable safetensors
            # file because H3 can unload module parameters to meta tensors.
            captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            handles = []
            for suffix in TARGET_SUFFIXES:
                linear = linear_for(block, suffix)
                def capture(module, call_args, output, suffix=suffix):
                    captured[suffix] = (call_args[0].detach(), output.detach())
                handles.append(linear.register_forward_hook(capture))
            try:
                next_hidden = block(hidden.to("cuda"), **tree_device(kwargs, "cuda"))
            finally:
                for handle in handles:
                    handle.remove()
            if set(captured) != set(TARGET_SUFFIXES):
                raise RuntimeError(f"block {block_id}: incomplete hooks {sorted(captured)}")
            for suffix in TARGET_SUFFIXES:
                x, ref = captured[suffix]
                weight = block_weights[suffix]
                layer = state["layers"][f"blocks.{block_id}.{suffix}"]
                smooth, a, b = (layer["smooth"].to(weight), layer["final_a"].to(weight), layer["final_b"].to(weight))
                qweight = nvfp4_qdq(weight * smooth - b @ a)
                # Exact deployed H3 SVDQuant calculation: Q/DQ(x / smooth)
                # through the FP4 main branch, plus the high-precision low-rank
                # compensation branch evaluated before activation Q/DQ.
                scaled_x = x / smooth
                # All four audited H3 target linears have no bias in the
                # source checkpoint; verify this through the safetensors keys.
                pred = F.linear(nvfp4_qdq(scaled_x), qweight)
                pred.add_(F.linear(F.linear(scaled_x, a), b))
                err2, ref2, numel = metrics(pred, ref)
                rows.append({"prompt_id": meta["prompt_id"], "step": meta["step"], "block": block_id,
                             "suffix": suffix, "layer": f"blocks.{block_id}.{suffix}",
                             "err2": err2, "ref2": ref2, "nmse": err2 / max(ref2, 1e-30), "numel": numel})
                del smooth, a, b, qweight, pred, scaled_x
            block_samples[sample_id] = (normal_tree(next_hidden), normal_tree(kwargs))
            del captured, next_hidden
        torch.cuda.empty_cache()
        print(f"block {block_id + 1}/50 complete", flush=True)

    pooled = sum(row["err2"] for row in rows) / sum(row["ref2"] for row in rows)
    equal = sum(row["nmse"] for row in rows) / len(rows)
    by_suffix = {}
    for suffix in TARGET_SUFFIXES:
        subset = [row for row in rows if row["suffix"] == suffix]
        by_suffix[suffix] = {
            "rows": len(subset),
            "pooled_nmse": sum(row["err2"] for row in subset) / sum(row["ref2"] for row in subset),
            "layer_sample_equal_nmse": sum(row["nmse"] for row in subset) / len(subset),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.output.write_text(json.dumps({
        "definition": "For each target linear, BF16 and SVDQuant receive the identical actual BF16 input captured during a BF16 block replay; NMSE is direct linear-output SSE divided by BF16-output energy.",
        "state": str(args.state.resolve()), "cache_dir": str(args.cache_dir.resolve()),
        "prompt_count": len(samples), "selected_step": args.step, "rows": len(rows),
        "pooled_nmse": pooled, "layer_sample_equal_nmse": equal, "by_suffix": by_suffix,
    }, indent=2))
    print(f"saved {args.output} pooled_nmse={pooled:.8g} equal_nmse={equal:.8g}", flush=True)


if __name__ == "__main__":
    main()
