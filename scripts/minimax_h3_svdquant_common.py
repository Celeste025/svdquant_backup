"""Shared utilities for the MiniMax-H3 SVDQuant smoke experiment.

This module deliberately keeps the H3 integration small and explicit.  H3 uses
one packed text/video/audio sequence, fused QKV and fused SwiGLU.  Treating the
leading token axis as a conventional batch (as the generic diffusion adapter
does) corrupts ``cu_seqlens``.  Calibration samples are therefore replayed one
at a time and candidate errors are measured at the complete H3 block endpoint;
this naturally includes packed attention and modality-specific AdaLN gates.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = Path(os.environ.get("DIFFSYNTH_ROOT", ROOT / "third_party/DiffSynth-Studio"))
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

from deepcompressor.nn.patch.lowrank import LowRankBranch
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiTBlock
from diffsynth.models.minimax_h3_dit_comfy import MiniMaxH3DiTComfyPruned
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig


PROMPT_FILE = Path(os.environ.get("MINIMAX_H3_PROMPT_FILE", ROOT / "data/minimax_h3_prompts.jsonl"))
DIT_MODEL_ID = "Comfy-Org/MiniMax-H3"
DIT_PATTERN = "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
H3_MODEL_ID = "MiniMax/MiniMax-H3"
TARGET_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
FP4_VALUES = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
              0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def tree_map(value: Any, fn) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree_map(v, fn) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(v, fn) for v in value)
    if isinstance(value, list):
        return [tree_map(v, fn) for v in value]
    return value


def tree_cpu(value: Any) -> Any:
    return tree_map(value, lambda x: x.detach().cpu())


def tree_device(value: Any, device: torch.device | str) -> Any:
    return tree_map(value, lambda x: x.to(device=device, non_blocking=True))


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.float() - b.float()).square().sum(dtype=torch.float64)
    den = b.float().square().sum(dtype=torch.float64).clamp_min(1e-30)
    return float((num / den).item())


def read_prompt(prompt_id: int, path: Path = PROMPT_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["prompt_id"]) == int(prompt_id):
                return row
    raise KeyError(f"prompt_id={prompt_id} not found in {path}")


def disk_config() -> dict[str, Any]:
    return {
        "offload_dtype": "disk", "offload_device": "disk",
        "onload_dtype": "disk", "onload_device": "disk",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }


def load_h3_pipeline(*, full: bool, reserve_gib: float = 4.0, dit_disk: bool = True) -> MiniMaxH3Pipeline:
    """Load the local pruned checkpoint; ``full=False`` loads only the DiT."""
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
    cfg = disk_config()
    dit_cfg = cfg if dit_disk else {
        "offload_dtype": torch.bfloat16, "offload_device": "cpu",
        "onload_dtype": torch.bfloat16, "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }
    models = [ModelConfig(model_id=DIT_MODEL_ID, origin_file_pattern=DIT_PATTERN, **dit_cfg)]
    processor = None
    if full:
        models += [
            ModelConfig(model_id=H3_MODEL_ID, origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **cfg),
            ModelConfig(model_id=H3_MODEL_ID, origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **cfg),
            ModelConfig(model_id=H3_MODEL_ID, origin_file_pattern="FL2VA/audio_vae/model.safetensors", **cfg),
        ]
        processor = ModelConfig(model_id=H3_MODEL_ID, origin_file_pattern="FL2VA/processor/")
    total_gib = torch.cuda.mem_get_info("cuda")[1] / 1024**3
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", model_configs=models,
        processor_config=processor, vram_limit=max(1.0, total_gib - reserve_gib),
    )
    if not isinstance(pipe.dit, MiniMaxH3DiTComfyPruned):
        raise TypeError(f"Expected MiniMaxH3DiTComfyPruned, got {type(pipe.dit)!r}")
    return pipe


def target_linears(dit: nn.Module, max_blocks: int = 50) -> list[tuple[str, nn.Linear]]:
    if len(dit.blocks) != 50:
        raise RuntimeError(f"Expected 50 H3 main blocks, found {len(dit.blocks)}")
    found: list[tuple[str, nn.Linear]] = []
    for block_id, block in enumerate(dit.blocks[:max_blocks]):
        if not isinstance(block, MiniMaxH3DiTBlock):
            raise TypeError(f"blocks.{block_id}: unexpected type {type(block)!r}")
        for suffix in TARGET_SUFFIXES:
            module: nn.Module = block
            for part in suffix.split("."):
                module = getattr(module, part)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"blocks.{block_id}.{suffix} is not Linear")
            if module.in_features % 16 or module.out_features % 16:
                raise RuntimeError(f"blocks.{block_id}.{suffix} shape is not divisible by 16")
            found.append((f"blocks.{block_id}.{suffix}", module))
    expected = max_blocks * 4
    if len(found) != expected:
        raise RuntimeError(f"Expected {expected} target linears, found {len(found)}")
    return found


def module_fingerprint(module: nn.Module, excluded_ids: set[int] | None = None) -> dict[str, tuple]:
    excluded_ids = excluded_ids or set()
    result = {}
    for name, p in module.named_parameters():
        if p.is_meta:
            continue
        if id(p) in excluded_ids:
            continue
        x = p.detach().float().reshape(-1)
        # Cheap deterministic fingerprint that does not clone the tensor.
        result[name] = (tuple(p.shape), float(x[:: max(1, x.numel() // 257)].sum().cpu()))
    return result


def _fp8_round_positive(scale: torch.Tensor) -> torch.Tensor:
    # E4M3FN is the scale format prescribed by real_nvfp4.yaml.
    return scale.clamp_min(torch.finfo(torch.float32).tiny).to(torch.float8_e4m3fn).float()


@torch.no_grad()
def nvfp4_qdq(tensor: torch.Tensor, *, group_size: int = 16, element_size: int = 128) -> torch.Tensor:
    """NVFP4-aligned fake quant: FP32 tensor scale, E4M3 group scale, E2M1 values.

    The last dimension is grouped by 16.  Work is chunked along flattened rows
    to avoid materializing another model-sized temporary tensor on GPU.
    """
    if tensor.shape[-1] % group_size:
        raise ValueError(f"last dim {tensor.shape[-1]} is not divisible by {group_size}")
    dtype, shape = tensor.dtype, tensor.shape
    rows = tensor.reshape(-1, shape[-1])
    out = torch.empty_like(rows)
    levels = torch.tensor(FP4_VALUES, device=tensor.device, dtype=torch.float32)
    # One FP32 tensor scale maps the largest ideal block scale into E4M3 range.
    absmax = tensor.detach().float().abs().amax().clamp_min(1e-12)
    tensor_scale = (absmax / (6.0 * 448.0)).clamp_min(1e-12)
    for start in range(0, rows.shape[0], element_size):
        stop = min(rows.shape[0], start + element_size)
        x = rows[start:stop].float().reshape(-1, shape[-1] // group_size, group_size)
        ideal = x.abs().amax(dim=-1, keepdim=True).div(6.0).clamp_min(1e-12)
        block_scale = _fp8_round_positive(ideal / tensor_scale) * tensor_scale
        z = x / block_scale
        # 15 values; explicit nearest-codebook rounding keeps E2M1 semantics.
        idx = (z.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        q = levels[idx] * block_scale
        out[start:stop] = q.reshape(stop - start, shape[-1]).to(dtype)
    return out.reshape(shape)


class DynamicActivationQDQ:
    def __init__(self, smooth: torch.Tensor | None, element_size: int = 128):
        self.smooth = smooth
        self.element_size = element_size
        self.calls = 0

    def __call__(self, _module: nn.Module, args: tuple[Any, ...]):
        x = args[0]
        if self.smooth is not None:
            x = x / self.smooth.to(device=x.device, dtype=x.dtype)
        self.calls += 1
        return (nvfp4_qdq(x, element_size=self.element_size), *args[1:])


@dataclass
class RuntimeHooks:
    act: DynamicActivationQDQ
    handles: list[Any]
    branch: LowRankBranch | None = None

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def install_runtime_hooks(
    linear: nn.Linear, smooth: torch.Tensor | None, a: torch.Tensor | None, b: torch.Tensor | None,
    *, element_size: int = 128,
) -> RuntimeHooks:
    act = DynamicActivationQDQ(smooth=smooth, element_size=element_size)
    handles = [linear.register_forward_pre_hook(act)]
    branch = None
    if a is not None and b is not None:
        branch = LowRankBranch(linear.in_features, linear.out_features, rank=a.shape[0])
        branch.a.weight.data.copy_(a.to(branch.a.weight))
        branch.b.weight.data.copy_(b.to(branch.b.weight))
        branch.to(device=linear.weight.device, dtype=linear.weight.dtype)

        def add_branch(_module, inputs, output):
            return output + branch(inputs[0])

        handles.append(linear.register_forward_hook(add_branch))
    return RuntimeHooks(act=act, handles=handles, branch=branch)


@contextmanager
def temporary_quantized_linear(
    linear: nn.Linear, qweight: torch.Tensor, smooth: torch.Tensor | None,
    a: torch.Tensor | None = None, b: torch.Tensor | None = None, *, element_size: int = 128,
):
    original = linear.weight.detach().cpu().clone()
    linear.weight.data.copy_(qweight.to(linear.weight))
    hooks = install_runtime_hooks(linear, smooth, a, b, element_size=element_size)
    try:
        yield hooks
    finally:
        hooks.remove()
        linear.weight.data.copy_(original.to(linear.weight))
        del original


def serialize_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"input_args": tree_cpu(args), "input_kwargs": tree_cpu(kwargs)}


class StopAtBlock(RuntimeError):
    pass


@torch.inference_mode()
def raw_call_to_block0(dit: nn.Module, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def hook(_module, args, kwargs):
        captured["hidden"] = args[0].detach().cpu()
        captured["kwargs"] = tree_cpu(kwargs)
        raise StopAtBlock()

    handle = dit.blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
    call = tree_device(sample, "cuda")
    try:
        try:
            dit(*call["input_args"], **call["input_kwargs"])
        except StopAtBlock:
            pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("block 0 pre-hook did not fire")
    return captured["hidden"], captured["kwargs"]


@torch.inference_mode()
def run_block(block: nn.Module, samples: list[tuple[torch.Tensor, dict[str, Any]]]) -> list[torch.Tensor]:
    device = torch.device("cuda")
    outputs = []
    for hidden, kwargs in samples:
        outputs.append(block(hidden.to(device), **tree_device(kwargs, device)).detach().cpu())
    return outputs


def aggregate_nmse(outputs: Iterable[torch.Tensor], references: Iterable[torch.Tensor]) -> float:
    num = den = 0.0
    for out, ref in zip(outputs, references, strict=True):
        num += float((out.float() - ref.float()).square().sum(dtype=torch.float64))
        den += float(ref.float().square().sum(dtype=torch.float64))
    return num / max(den, 1e-30)


def validate_state_manifest(state: dict[str, Any], expected_layers: int) -> None:
    if state.get("format") != "minimax-h3-svdquant-smoke-v1":
        raise ValueError("Unsupported quantization state format")
    if len(state.get("layers", {})) != expected_layers:
        raise ValueError(f"Expected {expected_layers} layer states, got {len(state.get('layers', {}))}")
    cfg = state["config"]
    assert cfg["group_size"] == 16 and cfg["rank"] == 32
    assert cfg["num_grids"] == 2 and cfg["num_iters"] == 2
