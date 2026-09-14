"""Recreate ancestor-owned quantization transforms for an isolated Linear.

DeepCompressor deliberately registers some transforms (notably SmoothQuant) on
the parent attention module.  Calling a child ``nn.Linear`` directly bypasses
those hooks, so a local comparison must replay the applicable parent input
processors first.  This module is intentionally conservative: an unknown
external processor is reported as unsupported instead of producing a number in
the wrong coordinate system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class UnsupportedExternalTransform(RuntimeError):
    """The isolated Linear path cannot be reconstructed safely."""


@dataclass(frozen=True)
class InputContext:
    labels: tuple[str, ...]

    @property
    def mode(self) -> str:
        return "identity" if not self.labels else "+".join(self.labels)


def _ancestor_names(name: str) -> list[str]:
    """Return root-to-parent module paths, excluding the Linear itself."""
    pieces = name.split(".")
    return ["."] + [".".join(pieces[:count]) for count in range(1, len(pieces))]


def _processor_from_hook(hook: Any) -> Any | None:
    return getattr(hook, "processor", None)


def _is_direct_qkv_input(name: str, ancestor_name: str, ancestor: torch.nn.Module) -> bool:
    """Whether an attention parent pre-hook reaches this projection input."""
    if ancestor_name == "." or not name.startswith(ancestor_name + "."):
        return False
    local = name[len(ancestor_name) + 1 :]
    if local not in {"to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj"}:
        return False
    # Diffusers uses ``cross_attention_dim=query_dim`` even for ordinary
    # self-attention when the constructor was passed ``None``. It therefore
    # cannot identify the input route. ``is_cross_attention`` is the proper
    # module-level flag. Wan attn2 is an exception: its cross K/V path is
    # expressed through ``added_kv_proj_dim`` and the module flag remains
    # false, so retain the explicit structural rule for that family.
    #
    # A parent hook changes ``hidden_states`` (Q) but not the separate
    # ``encoder_hidden_states`` passed to a cross-attention K/V projection.
    if ".attn2." in name or bool(getattr(ancestor, "is_cross_attention", False)):
        return local in {"to_q", "q_proj"}
    return True


def _input_packager_name(processor: Any) -> str:
    packager = getattr(processor, "input_packager", None)
    return type(packager).__name__ if packager is not None else "SimpleInputPackager(default)"


def _can_process_tensor(processor: Any, tensor: torch.Tensor) -> bool:
    """Check the processor's declared channel dimension where available."""
    channels_dim = getattr(processor, "channels_dim", None)
    if channels_dim is None:
        return True
    return int(channels_dim) % tensor.ndim == tensor.ndim - 1


@torch.inference_mode()
def prepare_isolated_linear_input(
    root: torch.nn.Module, name: str, tensor: torch.Tensor
) -> tuple[torch.Tensor, InputContext]:
    """Replay supported ancestor pre-hooks on one captured Linear input.

    A captured BF16 Linear input is before any quant-only ancestor transform.
    Direct child invocation therefore needs each applicable processor in the
    same root-to-leaf order.  We only support DeepCompressor tensor processors
    that expose ``process(tensor)`` and address the final (feature) dimension.
    Packagers other than simple/keyed are rejected explicitly.
    """
    value, labels = tensor, []
    for ancestor_name in _ancestor_names(name):
        ancestor = root if ancestor_name == "." else root.get_submodule(ancestor_name)
        for hook in ancestor._forward_pre_hooks.values():
            processor = _processor_from_hook(hook)
            if processor is None:
                continue
            processor_name = type(processor).__name__
            # A parent attention hook changes the input of direct Q/K/V only;
            # to_out receives the later attention result in another coordinate.
            if processor_name == "ActivationSmoother" and not _is_direct_qkv_input(name, ancestor_name, ancestor):
                continue
            process = getattr(processor, "process", None)
            packager_name = _input_packager_name(processor)
            if not callable(process):
                raise UnsupportedExternalTransform(
                    f"{name}: {ancestor_name} has external {processor_name} without process(tensor)"
                )
            if packager_name not in {"SimpleInputPackager", "KeyedInputPackager", "SimpleInputPackager(default)"}:
                raise UnsupportedExternalTransform(
                    f"{name}: {ancestor_name} uses unsupported input packager {packager_name} ({processor_name})"
                )
            if not _can_process_tensor(processor, value):
                raise UnsupportedExternalTransform(
                    f"{name}: {ancestor_name} {processor_name} channels_dim does not address Linear features"
                )
            # A parent processor can target a different argument (e.g. context
            # rather than hidden states).  For a keyed packager we can only use
            # it when its scale clearly describes this Linear's feature axis.
            scale = getattr(processor, "smooth_scale", None)
            if torch.is_tensor(scale) and scale.numel() != value.shape[-1]:
                raise UnsupportedExternalTransform(
                    f"{name}: {ancestor_name} {processor_name} scale length {scale.numel()} != input features {value.shape[-1]}"
                )
            try:
                value = process(value)
            except Exception as exc:  # preserve the layer and hook identity in report
                raise UnsupportedExternalTransform(
                    f"{name}: failed replaying {processor_name} on {ancestor_name}: {type(exc).__name__}: {exc}"
                ) from exc
            labels.append(f"{ancestor_name}:{processor_name}")
    return value, InputContext(tuple(labels))


def audit_isolated_linear_context(root: torch.nn.Module, name: str, in_features: int) -> dict[str, Any]:
    """Describe whether direct measurement has a reconstructable input path."""
    probe = torch.zeros((1, 1, in_features), device=next(root.parameters()).device, dtype=torch.bfloat16)
    try:
        _, context = prepare_isolated_linear_input(root, name, probe)
        return {"status": "supported", "input_context": context.mode}
    except UnsupportedExternalTransform as exc:
        return {"status": "unsupported", "reason": str(exc)}
