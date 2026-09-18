"""ConvRot/TorchAO helpers for the pruned MiniMax-H3 DiT.

ConvRot itself is kept in its original repository.  This adapter only bridges
its Tensor subclass to DiffSynth's disk-offloaded ``AutoWrappedLinear``.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from minimax_h3_svdquant_common import module_fingerprint, nmse, target_linears

# The optional ConvRot checkout is deliberately external to this repository.
# An explicit environment variable keeps the adapter portable across servers
# while retaining the historical workstation location as a compatibility
# fallback for existing experiments.
CONVROT_ROOT = Path(os.environ.get("CONVROT_ROOT", "/home/wjq/workspace/ConvRot")).expanduser()
ROT_SIZE = 256


def import_convrot() -> None:
    """Register ConvRot/TorchAO dispatches from its unmodified source tree."""
    if not CONVROT_ROOT.is_dir():
        raise FileNotFoundError(
            f"ConvRot checkout not found at {CONVROT_ROOT}; set CONVROT_ROOT to "
            "the optional ConvRot source tree."
        )
    if str(CONVROT_ROOT) not in sys.path:
        sys.path.insert(0, str(CONVROT_ROOT))
    import convrot  # noqa: F401


@contextmanager
def convrot_workdir():
    """ConvRot's extension builder uses source paths relative to its root."""
    previous = Path.cwd()
    os.chdir(CONVROT_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def convrot_config():
    import_convrot()
    from convrot.config import ConvRotNVFP4Config
    return ConvRotNVFP4Config(
        rot_size=ROT_SIZE,
        use_dynamic_per_tensor_scale=True,
        use_triton_kernel=True,
    )


def warmup_convrot_extension() -> None:
    """Compile or load ConvRot once while relative CUDA sources resolve."""
    import_convrot()
    from convrot.ops import hadamard_rotate
    with convrot_workdir():
        hadamard_rotate(torch.ones((1, ROT_SIZE), device="cuda", dtype=torch.bfloat16), ROT_SIZE)
    torch.cuda.synchronize()


def configure_offloaded_quant_linear(linear: nn.Linear) -> None:
    """Keep a packed tensor authoritative instead of reloading BF16 from disk."""
    linear.disk_offload = False
    linear.offload_dtype = linear.onload_dtype = torch.bfloat16
    linear.offload_device = linear.onload_device = torch.device("cuda")
    linear.preparing_dtype = linear.computation_dtype = torch.bfloat16
    linear.preparing_device = linear.computation_device = torch.device("cuda")
    linear.vram_limit = 0.0
    linear.state = 0


def assert_target_structure(dit: nn.Module) -> list[tuple[str, nn.Linear]]:
    targets = target_linears(dit)
    if len(targets) != 200:
        raise RuntimeError(f"expected 200 targets, got {len(targets)}")
    bad = [(name, linear.in_features) for name, linear in targets if linear.in_features % ROT_SIZE]
    if bad:
        raise RuntimeError(f"ConvRot rot_size={ROT_SIZE} incompatible inputs: {bad[:4]}")
    return targets


@torch.no_grad()
def rotation_only_nmse(weight: torch.Tensor, x: torch.Tensor) -> float:
    import_convrot()
    from convrot.ops import hadamard_rotate
    ref = torch.nn.functional.linear(x, weight)
    with convrot_workdir():
        got = torch.nn.functional.linear(hadamard_rotate(x, ROT_SIZE), hadamard_rotate(weight, ROT_SIZE))
    return nmse(got, ref)


def state_config() -> dict:
    return {
        "format": "minimax-h3-convrot-nvfp4-v1",
        "targets": 200,
        "rot_size": ROT_SIZE,
        "weight": "TorchAO real NVFP4",
        "activation": "dynamic per-tensor NVFP4",
        "use_triton_kernel": True,
        "source": str(CONVROT_ROOT),
    }


def pack_weight(weight: torch.Tensor) -> dict:
    """Serialize NVFP4 constituents directly; Parameter serialization strips wrappers."""
    from convrot.rotated_nvfp4_tensor import RotatedNVFP4Tensor
    if not isinstance(weight, RotatedNVFP4Tensor):
        raise TypeError(f"expected RotatedNVFP4Tensor, got {type(weight)!r}")
    base = weight.nvfp4_weight
    return {
        "qdata": base.qdata, "scale": base.scale,
        "block_size": base.block_size, "orig_dtype": base.orig_dtype,
        "per_tensor_scale": base.per_tensor_scale,
        "act_per_tensor_scale": base.act_per_tensor_scale,
        "is_swizzled_scales": base.is_swizzled_scales,
        "use_triton_kernel": base.use_triton_kernel,
        "act_quant_kwargs": base.act_quant_kwargs,
        "rot_size": weight.rot_size,
    }


def unpack_weight(state: dict, device: str | torch.device = "cuda") -> torch.Tensor:
    from convrot.rotated_nvfp4_tensor import RotatedNVFP4Tensor
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    def move(x):
        return None if x is None else x.to(device)
    base = NVFP4Tensor(
        move(state["qdata"]), move(state["scale"]), state["block_size"], state["orig_dtype"],
        move(state.get("per_tensor_scale")), move(state.get("act_per_tensor_scale")),
        state.get("is_swizzled_scales", False), state.get("use_triton_kernel", False),
        state.get("act_quant_kwargs"),
    )
    return RotatedNVFP4Tensor(base, int(state["rot_size"]))
