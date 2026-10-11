#!/usr/bin/env python3
"""Blockwise low-rank adaptation of the MiniMax-H3 rank-64 SVDQuant state.

The quantized W4A4 main weights and smoothing vectors stay frozen.  This worker
updates only the existing SVDQuant low-rank branches against complete BF16 block
outputs on the original 64 cached calls.  It is deliberately sequential, so a
block sees the outputs of already adapted preceding blocks without requiring a
30B full-model backward graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from minimax_h3_svdquant_common import (
    TARGET_SUFFIXES, install_runtime_hooks, load_h3_pipeline, nvfp4_qdq,
    raw_call_to_block0, run_block, target_linears, tree_cpu, tree_device,
)


TRAIN_PROMPTS = (1, 11, 20, 25, 46, 48, 105, 116)
TRAIN_STEPS = (0, 3, 5, 8, 11, 14, 16, 19)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-state", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-passes", type=int, default=3)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(value: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(value: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def load_samples(cache_dir: Path) -> tuple[list[dict], list[dict]]:
    samples, meta = [], []
    manifests = sorted(cache_dir.glob("p*/manifest.json"), key=lambda x: int(x.parent.name[1:]))
    if [int(x.parent.name[1:]) for x in manifests] != list(TRAIN_PROMPTS):
        raise RuntimeError("calibration prompt contract does not match the registered 8-prompt train split")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        entries = sorted(manifest["samples"], key=lambda x: int(x["step"]))
        if tuple(int(x["step"]) for x in entries) != TRAIN_STEPS:
            raise RuntimeError(f"{manifest_path}: unexpected calibration timestep contract")
        for entry in entries:
            item = torch.load(manifest_path.parent / entry["file"], map_location="cpu", weights_only=False)
            samples.append(item)
            meta.append({"prompt_id": int(entry["prompt_id"]), "step": int(entry["step"])})
    if len(samples) != 64:
        raise RuntimeError(f"expected 64 training calls, got {len(samples)}")
    return samples, meta


def linear_for(block: torch.nn.Module, suffix: str) -> torch.nn.Linear:
    value: object = block
    for part in suffix.split("."):
        value = getattr(value, part)
    if not isinstance(value, torch.nn.Linear):
        raise TypeError(f"{suffix} is not Linear")
    return value


def base_layer(base: dict, block_id: int, suffix: str) -> dict:
    return base["layers"][f"blocks.{block_id}.{suffix}"]


def install_block(block: torch.nn.Module, base: dict, adapted: dict, block_id: int):
    runtimes = []
    for suffix in TARGET_SUFFIXES:
        linear = linear_for(block, suffix)
        name = f"blocks.{block_id}.{suffix}"
        initial = base_layer(base, block_id, suffix)
        current = adapted[name]
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        smooth = initial["smooth"].to(weight)
        qa, qb = initial["final_a"].to(weight), initial["final_b"].to(weight)
        linear.load_state_dict({"weight": nvfp4_qdq(weight * smooth - qb @ qa).detach().cpu()}, assign=True, strict=False)
        linear.disk_offload = False
        linear.offload_dtype = linear.onload_dtype = torch.bfloat16
        linear.offload_device = linear.onload_device = torch.device("cpu")
        linear.computation_dtype, linear.computation_device = torch.bfloat16, torch.device("cuda")
        linear.vram_limit, linear.state = 0.0, 1
        runtime = install_runtime_hooks(linear, smooth, current["final_a"].to(weight), current["final_b"].to(weight))
        assert runtime.branch is not None
        runtime.branch.to(device="cuda", dtype=torch.bfloat16)
        for parameter in runtime.branch.parameters():
            parameter.requires_grad_(True)
        runtimes.append((name, linear, runtime))
        del weight, smooth, qa, qb
    return runtimes


def unload_block(runtimes) -> None:
    for _name, linear, runtime in runtimes:
        runtime.remove()
        if runtime.branch is not None:
            runtime.branch.to("cpu")
        linear.to("cpu")
    torch.cuda.empty_cache()


def store_adapted(runtimes, adapted: dict) -> None:
    for name, _linear, runtime in runtimes:
        assert runtime.branch is not None
        adapted[name] = {
            "final_a": runtime.branch.a.weight.detach().float().cpu().clone(),
            "final_b": runtime.branch.b.weight.detach().float().cpu().clone(),
        }


def freeze_block(block: torch.nn.Module) -> None:
    block.eval()
    for parameter in block.parameters():
        parameter.requires_grad_(False)


def normal_tree(value):
    """Clone inference-mode cache tensors into ordinary autograd-safe tensors."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(normal_tree(x) for x in value)
    if isinstance(value, list):
        return [normal_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: normal_tree(v) for k, v in value.items()}
    return value


def materialize_frozen_parameters(block: torch.nn.Module) -> None:
    """DiffSynth disk loading uses inference tensors, which autograd cannot save."""
    for module in block.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is not None and parameter.is_inference():
                module._parameters[name] = torch.nn.Parameter(parameter.detach().clone(), requires_grad=False)


@torch.no_grad()
def capture_block_input(dit: torch.nn.Module, raw_sample: dict, block_id: int):
    """Capture a real block entrance from the complete BF16 DIT prefix."""
    captured = {}
    class Stop(RuntimeError):
        pass
    def hook(_module, args, kwargs):
        captured["hidden"] = tree_cpu(args[0])
        captured["kwargs"] = tree_cpu(kwargs)
        raise Stop()
    handle = dit.blocks[block_id].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        try:
            dit(*tree_device(raw_sample["input_args"], "cuda"), **tree_device(raw_sample["input_kwargs"], "cuda"))
        except Stop:
            pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"failed to capture block {block_id} entrance")
    return normal_tree(captured["hidden"]), normal_tree(captured["kwargs"])


def make_ft_state(base_path: Path, base_hash: str, adapted: dict, completed_passes: int) -> dict:
    return {
        "format": "minimax-h3-svdquant-ft-v1",
        "base_state_path": str(base_path.resolve()),
        "base_state_sha256": base_hash,
        "config": {
            "trainable": "LowRankBranch A/B only", "optimizer": "AdamW",
            "lr": 1e-5, "weight_decay": 0.0, "train_calls": 64,
            "train_prompts": list(TRAIN_PROMPTS), "train_steps": list(TRAIN_STEPS),
            "objective": "complete BF16 block-output teacher-energy-normalized MSE",
            "frozen": "W4A4 main weights and smoothing",
        },
        "completed_passes": completed_passes,
        "layers": adapted,
    }


def main() -> None:
    args = parse_args()
    if args.max_passes < 1 or args.lr <= 0:
        raise ValueError("invalid training arguments")
    if time.time() >= args.deadline_unix:
        raise RuntimeError("deadline already expired")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # SageAttention is inference-only in this environment and silently cuts the
    # qkv branch gradient.  Use autograd-capable PyTorch SDPA only for training;
    # deployment/evaluation retain the normal inference backend.
    from diffsynth.core.attention import attention as attention_impl
    attention_impl.ATTENTION_IMPLEMENTATION = "torch"
    base = torch.load(args.base_state, map_location="cpu", weights_only=False)
    if base.get("format") != "minimax-h3-svdquant-standard-v1" or len(base.get("layers", {})) != 200:
        raise RuntimeError("base state must be the complete rank-64 standard SVDQuant checkpoint")
    if base["config"].get("rank") != 64:
        raise RuntimeError("base state is not rank-64")
    raw_samples, sample_meta = load_samples(args.cache_dir)
    base_hash = sha256(args.base_state)
    adapted = {name: {"final_a": item["final_a"].float().cpu().clone(), "final_b": item["final_b"].float().cpu().clone()}
               for name, item in base["layers"].items()}
    completed_passes = 0
    if args.resume and (args.output_dir / "progress.pt").is_file():
        progress = torch.load(args.output_dir / "progress.pt", map_location="cpu", weights_only=False)
        if progress["base_state_sha256"] != base_hash:
            raise RuntimeError("resume base-state hash mismatch")
        adapted, completed_passes = progress["layers"], int(progress["completed_passes"])
    contract = {
        "base_state": str(args.base_state.resolve()), "base_state_sha256": base_hash,
        "cache_dir": str(args.cache_dir.resolve()), "train_prompts": list(TRAIN_PROMPTS),
        "train_steps": list(TRAIN_STEPS), "train_calls": len(raw_samples), "deadline_unix": args.deadline_unix,
        "started_unix": time.time(), "max_passes": args.max_passes, "lr": args.lr,
    }
    atomic_json(contract, args.output_dir / "contract.json")

    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    if len(target_linears(pipe.dit)) != 200:
        raise RuntimeError("expected 200 H3 target linears")
    block0_samples = [(normal_tree(hidden), normal_tree(kwargs))
                      for hidden, kwargs in (raw_call_to_block0(pipe.dit, sample) for sample in raw_samples)]
    metrics: list[dict] = []
    all_gradient_names: set[str] = set()

    # Two blocks are sufficient to validate the route before committing to the
    # long sequential run; normal execution then continues from block 0.
    preflight_inputs = {0: [block0_samples[0]], 49: [capture_block_input(pipe.dit, raw_samples[0], 49)]}
    preflight_blocks = (0, 49)
    for block_id in preflight_blocks:
        block = pipe.dit.blocks[block_id]
        freeze_block(block)
        inputs = preflight_inputs[block_id]
        refs = run_block(block, inputs)
        materialize_frozen_parameters(block)
        refs = [normal_tree(x) for x in refs]
        runtimes = install_block(block, base, adapted, block_id)
        params = [p for _n, _l, runtime in runtimes for p in runtime.branch.parameters()]
        out = block(inputs[0][0].to("cuda"), **tree_device(inputs[0][1], "cuda"))
        loss = (out.float() - refs[0].to("cuda").float()).square().sum() / refs[0].float().square().sum().clamp_min(1e-30)
        loss.backward()
        for name, _linear, runtime in runtimes:
            assert runtime.branch is not None
            if all(p.grad is not None and torch.isfinite(p.grad).all() for p in runtime.branch.parameters()):
                all_gradient_names.add(name)
        for p in params:
            p.grad = None
        unload_block(runtimes)
    if len(all_gradient_names) != 8:
        raise RuntimeError(f"preflight gradient failure: only {len(all_gradient_names)}/8 branch pairs finite: {sorted(all_gradient_names)}")
    if args.preflight_only:
        atomic_json({"preflight": "passed", "gradient_branches": sorted(all_gradient_names)}, args.output_dir / "preflight.json")
        return

    for pass_id in range(completed_passes, args.max_passes):
        samples = block0_samples
        for block_id, block in enumerate(pipe.dit.blocks):
            if time.time() >= args.deadline_unix:
                atomic_save({"base_state_sha256": base_hash, "layers": adapted, "completed_passes": pass_id,
                             "stopped": "deadline", "current_pass": pass_id, "current_block": block_id}, args.output_dir / "progress.pt")
                atomic_json({"status": "deadline_before_complete_pass", "completed_passes": pass_id,
                             "valid_ft_checkpoint": completed_passes >= 1, "ended_unix": time.time()}, args.output_dir / "status.json")
                return
            freeze_block(block)
            refs = run_block(block, samples)
            materialize_frozen_parameters(block)
            refs = [normal_tree(x) for x in refs]
            runtimes = install_block(block, base, adapted, block_id)
            params = [p for _n, _l, runtime in runtimes for p in runtime.branch.parameters()]
            before = [p.detach().float().cpu().clone() for p in params]
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, foreach=False)
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for (hidden, kwargs), ref in zip(samples, refs, strict=True):
                out = block(hidden.to("cuda"), **tree_device(kwargs, "cuda"))
                ref_gpu = ref.to("cuda")
                loss = (out.float() - ref_gpu.float()).square().sum() / ref_gpu.float().square().sum().clamp_min(1e-30)
                (loss / len(samples)).backward()
                total += float(loss.detach())
                del out, ref_gpu, loss
            finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            if not finite:
                unload_block(runtimes)
                raise RuntimeError(f"non-finite low-rank gradient at pass {pass_id} block {block_id}")
            optimizer.step()
            changed = any(not torch.equal(a, p.detach().float().cpu()) for a, p in zip(before, params, strict=True))
            if not changed:
                unload_block(runtimes)
                raise RuntimeError(f"optimizer did not change low-rank parameters at pass {pass_id} block {block_id}")
            store_adapted(runtimes, adapted)
            # Propagate the current student state to the next block before its
            # hooks are removed; this is the sequential blockwise contract.
            with torch.no_grad():
                next_samples = []
                for hidden, kwargs in samples:
                    next_samples.append((block(hidden.to("cuda"), **tree_device(kwargs, "cuda")).detach().cpu(), kwargs))
            metrics.append({"pass": pass_id + 1, "block": block_id, "mean_train_nmse": total / len(samples),
                            "seconds": time.time() - contract["started_unix"]})
            unload_block(runtimes)
            samples = next_samples
            atomic_save({"base_state_sha256": base_hash, "layers": adapted, "completed_passes": pass_id,
                         "current_pass": pass_id, "current_block": block_id + 1, "metrics": metrics}, args.output_dir / "progress.pt")
            atomic_json(metrics, args.output_dir / "block_metrics.json")
        completed_passes = pass_id + 1
        atomic_save(make_ft_state(args.base_state, base_hash, adapted, completed_passes), args.output_dir / "quant_state.pt")
        atomic_save({"base_state_sha256": base_hash, "layers": adapted, "completed_passes": completed_passes,
                     "current_pass": completed_passes, "current_block": 0, "metrics": metrics}, args.output_dir / "progress.pt")
    atomic_json({"status": "complete", "completed_passes": completed_passes, "valid_ft_checkpoint": True,
                 "ended_unix": time.time(), "metrics": str(args.output_dir / "block_metrics.json")}, args.output_dir / "status.json")


if __name__ == "__main__":
    main()
