#!/usr/bin/env python3
"""Quantize the 200 H3 main-block linears with the real ConvRot NVFP4 path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchao.quantization import quantize_

from minimax_h3_svdquant_common import load_h3_pipeline, module_fingerprint, nmse
from minimax_h3_convrot_common import (
    assert_target_structure, configure_offloaded_quant_linear, convrot_config,
    convrot_workdir, import_convrot, pack_weight, rotation_only_nmse, state_config,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args()


@torch.no_grad()
def quantize_one(name, linear, config):
    weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=True)
    x = torch.randn((4, linear.in_features), device="cuda", dtype=torch.bfloat16)
    rot_nmse = rotation_only_nmse(weight, x)
    if rot_nmse >= 5e-5:
        raise RuntimeError(f"{name}: rotation-only equivalence failed: {rot_nmse:.3e}")
    with convrot_workdir():
        quantize_(linear, config)
    from convrot.rotated_nvfp4_tensor import RotatedNVFP4Tensor
    if not isinstance(linear.weight, RotatedNVFP4Tensor):
        raise RuntimeError(f"{name}: ConvRot handler did not install RotatedNVFP4Tensor")
    y = torch.nn.functional.linear(x, linear.weight, linear.bias)
    if not torch.isfinite(y).all():
        raise RuntimeError(f"{name}: NVFP4 dispatch produced non-finite output")
    packed = pack_weight(linear.weight)
    configure_offloaded_quant_linear(linear)
    # ``to(cpu)`` exercises ConvRot's Tensor-subclass transfer path before the
    # state is serialized; it is also the form used by DiffSynth at inference.
    linear.to(device="cpu", dtype=torch.bfloat16)
    return rot_nmse, packed


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import_convrot()
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    targets = assert_target_structure(dit)
    target_ids = {id(linear.weight) for _, linear in targets}
    untouched_before = module_fingerprint(dit, excluded_ids=target_ids)
    config = convrot_config()

    # Mandatory isolated dispatch smoke test.  It uses block 0 qkv, then the
    # rest of block 0, before mutating the remaining model.
    selected = targets[:1] if args.verify_only else targets[:4]
    checks = []
    layer_states = {}
    for name, linear in selected:
        rot_nmse, packed = quantize_one(name, linear, config)
        checks.append({"name": name, "rotation_only_nmse": rot_nmse})
        layer_states[name] = packed
    if args.verify_only:
        (args.output_dir / "verify.json").write_text(json.dumps({"checks": checks, **state_config()}, indent=2))
        print("ConvRot isolated dispatch verification passed", flush=True)
        return

    for index, (name, linear) in enumerate(targets[4:], start=5):
        rot_nmse, packed = quantize_one(name, linear, config)
        checks.append({"name": name, "rotation_only_nmse": rot_nmse})
        layer_states[name] = packed
        if index % 20 == 0:
            print(f"ConvRot quantized {index}/200", flush=True)
            torch.cuda.empty_cache()

    if module_fingerprint(dit, excluded_ids={id(linear.weight) for _, linear in targets}) != untouched_before:
        raise RuntimeError("non-target module fingerprint changed")
    if any(not torch.isfinite(torch.tensor(row["rotation_only_nmse"])) for row in checks):
        raise RuntimeError("non-finite equivalence result")

    state = {**state_config(), "checks": {"structure_50_blocks": True, "rotation_only": True,
             "real_nvfp4_dispatch": True, "untouched_fingerprint": True},
             "layers": layer_states,
             "rotation_only_nmse_max": max(row["rotation_only_nmse"] for row in checks)}
    torch.save(state, args.output_dir / "quant_state.pt")
    (args.output_dir / "summary.json").write_text(json.dumps({k: v for k, v in state.items() if k != "layers"}, indent=2))
    print(f"saved {args.output_dir / 'quant_state.pt'}", flush=True)


if __name__ == "__main__":
    main()
