#!/usr/bin/env python3
"""Capture compact, representative MiniMax-H3 direct-linear tensor patterns."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from finetune_minimax_h3_r64_lowrank import linear_for, normal_tree
from minimax_h3_svdquant_common import H3_DIT_PATH, load_h3_pipeline, nvfp4_qdq, raw_call_to_block0, tree_device

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/calib/minimax_h3_svdquant_standard_8p64s"
STATE = ROOT / "results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
OUT = Path("/data1/models/svdquant-wjq/results/E087_h3_patterns")
EXAMPLES = ((20, 8, 47, "attn.qkv_proj"), (116, 8, 33, "attn.out_proj"))


def sample(prompt_id: int, step: int) -> dict:
    manifest = json.loads((CACHE / f"p{prompt_id}" / "manifest.json").read_text())
    entry = next(x for x in manifest["samples"] if x["step"] == step)
    return torch.load(CACHE / f"p{prompt_id}" / entry["file"], map_location="cpu", weights_only=False)


def source_weight(block: int, suffix: str) -> torch.Tensor:
    with safe_open(str(H3_DIT_PATH), framework="pt", device="cpu") as source:
        return source.get_tensor(f"blocks.{block}.{suffix}.weight").cuda()


def compact(x: torch.Tensor, rows: int = 260, cols: int = 300) -> torch.Tensor:
    x = x.float()
    return F.interpolate(x[None, None], size=(min(rows, x.shape[0]), min(cols, x.shape[1])), mode="area")[0, 0].cpu()


def sampled(x: torch.Tensor, rows: int = 260, cols: int = 300) -> torch.Tensor:
    """Exact raw values on an evenly spaced grid; no signed pooling/cancellation."""
    r = torch.linspace(0, x.shape[0] - 1, min(rows, x.shape[0]), device=x.device).round().long()
    c = torch.linspace(0, x.shape[1] - 1, min(cols, x.shape[1]), device=x.device).round().long()
    return x.index_select(0, r).index_select(1, c).float().cpu()


@torch.inference_mode()
def capture(pipe, spec: tuple[int, int, int, str], state: dict) -> tuple:
    prompt_id, step, target_block, suffix = spec
    hidden, kwargs = raw_call_to_block0(pipe.dit, sample(prompt_id, step))
    captured = {}
    for block_id, block in enumerate(pipe.dit.blocks[:target_block + 1]):
        handle = None
        if block_id == target_block:
            def hook(_module, args, output):
                captured["x"], captured["ref"] = args[0].detach().clone(), output.detach().clone()
            handle = linear_for(block, suffix).register_forward_hook(hook)
        try:
            hidden = block(hidden.to("cuda"), **tree_device(kwargs, "cuda"))
        finally:
            if handle is not None:
                handle.remove()
        if block_id != target_block:
            hidden = normal_tree(hidden)
    x, ref = captured["x"], captured["ref"]
    weight = source_weight(target_block, suffix)
    layer = state["layers"][f"blocks.{target_block}.{suffix}"]
    smooth, a, b = (layer[k].to(weight) for k in ("smooth", "final_a", "final_b"))
    qweight = nvfp4_qdq(weight * smooth - b @ a)
    scaled = x / smooth
    main = F.linear(nvfp4_qdq(scaled), qweight)
    low_rank = F.linear(F.linear(scaled, a), b)
    full = main + low_rank
    main_error, full_error = main - ref, full - ref
    ref2 = ref.float().square().sum()
    main2, full2 = main_error.float().square().sum(), full_error.float().square().sum()
    dot = (main_error.float() * low_rank.float()).sum()
    panels = {
        "weight": compact(weight),
        "weight_residual": compact((qweight + b @ a) / smooth - weight),
        "input": compact(x),
        "smoothed_input": compact(scaled),
        "bf16_output": compact(ref),
        "main_error": sampled(main_error),
        "low_rank": sampled(low_rank),
        "full_error": sampled(full_error),
    }
    metrics = {
        "bf16_check_nmse": float((F.linear(x, weight).float().sub(ref.float()).square().sum() / ref2).item()),
        "main_nmse": float((main2 / ref2).item()),
        "full_nmse": float((full2 / ref2).item()),
        "main_rrmse": float((main2 / ref2).sqrt().item()),
        "full_rrmse": float((full2 / ref2).sqrt().item()),
        "low_rank_rms_over_output_rms": float((low_rank.float().square().sum() / ref2).sqrt().item()),
        "main_rms_over_output_rms": float((main.float().square().sum() / ref2).sqrt().item()),
        "full_rms_over_output_rms": float((full.float().square().sum() / ref2).sqrt().item()),
        "main_projection_onto_true_output": float((main.float() * ref.float()).sum().div(ref2).item()),
        "low_rank_projection_onto_true_output": float((low_rank.float() * ref.float()).sum().div(ref2).item()),
        "full_projection_onto_true_output": float((full.float() * ref.float()).sum().div(ref2).item()),
        "low_rank_error_projection_cancel_fraction": float((-dot / main2).item()),
        "error_energy_reduction_fraction": float((1 - full2 / main2).item()),
    }
    return {"spec": spec, "panels": panels, "metrics": metrics}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    for spec in EXAMPLES:
        prompt_id, step, block, suffix = spec
        data = capture(pipe, spec, state)
        torch.save(data, OUT / f"h3_linear_pattern_data_p{prompt_id}_s{step}_b{block}_{suffix.replace('.', '_')}.pt")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
