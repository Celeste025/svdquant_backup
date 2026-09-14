import os
import typing as tp
from collections import OrderedDict

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from deepcompressor.data.cache import TensorsCache

from ..nn.struct import DiffusionAttentionStruct, DiffusionFeedForwardStruct, DiffusionModelStruct
from .config import DiffusionQuantConfig

__all__ = [
    "get_needs_inputs_fn",
    "get_needs_outputs_fn",
    "wrap_joint_attn",
    "wrap_wan_gated",
    "maybe_wrap_wan_gated",
    "maybe_wan_eval_inputs",
]


def _wan_gated_enabled() -> bool:
    """Gated OutputsError is on by default for local Wan PTQ; set DEEPCOMPRESSOR_WAN_GATED=0 to disable."""
    return os.environ.get("DEEPCOMPRESSOR_WAN_GATED", "1") not in ("0", "false", "False")


def wrap_joint_attn(attn: nn.Module, /, *, indexes: int | tuple[int, ...] = 1) -> tp.Callable:
    if isinstance(indexes, int):

        def eval(*args, **kwargs) -> torch.Tensor:
            return attn(*args, **kwargs)[indexes]

    else:

        def eval(*args, **kwargs) -> tuple[torch.Tensor, ...]:
            tensors = attn(*args, **kwargs)
            result = torch.concat([tensors[i] for i in indexes], dim=-2)
            return result

    return eval


def wrap_wan_gated(
    module: nn.Module | tp.Callable,
    block: WanTransformerBlock,
    which: tp.Literal["msa", "ffn"],
) -> tp.Callable:
    """Wrap a Wan branch so OutputsError sees residual-scaled outputs ``y * gate``.

    Wan applies gates outside attn1 / FFN:
      hs = hs + attn1_out * gate_msa
      hs = hs + ffn_out * c_gate_msa
    ``temb`` must be provided in kwargs (popped before calling ``module``).
    """
    assert which in ("msa", "ffn")
    gate_idx = 2 if which == "msa" else 5
    # Attention accepts rotary_emb; Linear / FeedForward do not.
    keep_rotary = which == "msa" and not isinstance(module, nn.Linear)

    def eval(*args, **kwargs) -> torch.Tensor:
        if "temb" not in kwargs:
            raise KeyError("wrap_wan_gated requires per-sample 'temb' in eval kwargs")
        temb = kwargs.pop("temb")
        if not keep_rotary:
            kwargs.pop("rotary_emb", None)
        else:
            # Ensure RoPE freqs land on the same device as Attention inputs.
            rot = kwargs.get("rotary_emb", None)
            if isinstance(rot, torch.Tensor) and args:
                ref = args[0] if isinstance(args[0], torch.Tensor) else kwargs.get("hidden_states")
                if isinstance(ref, torch.Tensor) and rot.device != ref.device:
                    kwargs["rotary_emb"] = rot.to(device=ref.device, non_blocking=True)
            elif isinstance(rot, torch.Tensor) and "hidden_states" in kwargs:
                ref = kwargs["hidden_states"]
                if isinstance(ref, torch.Tensor) and rot.device != ref.device:
                    kwargs["rotary_emb"] = rot.to(device=ref.device, non_blocking=True)
        y = module(*args, **kwargs)
        if not isinstance(y, torch.Tensor):
            y = y[0]
        if isinstance(temb, torch.Tensor) and temb.device != y.device:
            temb = temb.to(device=y.device, non_blocking=True)
        gate = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[gate_idx]
        return y * gate.type_as(y)

    return eval


def _wan_gate_wrap_kind(
    parent: DiffusionAttentionStruct | DiffusionFeedForwardStruct | None,
    field_name: str,
    eval_module: tp.Any = None,
) -> tp.Literal["msa", "ffn"] | None:
    """Return which Wan gate to apply for this eval target, or None if ungated."""
    if parent is None:
        return None
    block = getattr(getattr(parent, "parent", None), "module", None)
    if not isinstance(block, WanTransformerBlock):
        return None

    if isinstance(parent, DiffusionAttentionStruct):
        if not parent.is_self_attn():
            return None
        if field_name.endswith("o_proj"):
            return "msa"
        if field_name.endswith(("q_proj", "k_proj", "v_proj")) or field_name == "":
            # Gate full Attention eval (module / struct / already-wrapped callable).
            # Do NOT gate a lone Q/K/V Linear used only for span stats.
            if isinstance(eval_module, nn.Linear):
                return None
            return "msa"
        return None

    if isinstance(parent, DiffusionFeedForwardStruct):
        if field_name in ("down_proj", "up_proj"):
            return "ffn"
        return None

    return None


def maybe_wrap_wan_gated(
    eval_module: tp.Any,
    parent: DiffusionAttentionStruct | DiffusionFeedForwardStruct | None,
    field_name: str,
) -> tp.Any:
    """Conditionally wrap Wan attn1 / FFN eval modules for gated OutputsError.

    - attn1 (self-attn): wrap with ``gate_msa`` when evaluating the Attention or ``o_proj``
    - FFN ``down_proj``: wrap Linear with ``c_gate_msa``
    - FFN ``up_proj``: wrap the full FeedForward (same input as up_proj) with ``c_gate_msa``
    - attn2 / other fields: unchanged
    - Disabled when ``DEEPCOMPRESSOR_WAN_GATED=0`` (ungated baseline PTQ).
    """
    if not _wan_gated_enabled():
        return eval_module
    kind = _wan_gate_wrap_kind(parent, field_name, eval_module)
    if kind is None:
        return eval_module
    block = parent.parent.module
    assert isinstance(block, WanTransformerBlock)
    if isinstance(parent, DiffusionFeedForwardStruct) and field_name == "up_proj":
        # Avoid multiplying intermediate (up) activations by hidden-dim gate.
        return wrap_wan_gated(parent.module, block, "ffn")
    return wrap_wan_gated(eval_module, block, kind)


def maybe_wan_eval_inputs(
    inputs: TensorsCache | None,
    parent: DiffusionAttentionStruct | DiffusionFeedForwardStruct | None,
    field_name: str,
    eval_module: tp.Any = None,
) -> TensorsCache | None:
    """Attach Wan ``temb`` / ``rotary_emb`` onto *eval* inputs only.

    Span / act statistics still use the original ``inputs`` (must stay ``num_tensors==1``).
    Gate kwargs are stored on the Wan block by ``_attach_wan_gate_eval_kwargs``.
    """
    if inputs is None:
        return None
    if not _wan_gated_enabled():
        return inputs
    kind = _wan_gate_wrap_kind(parent, field_name, eval_module)
    if kind is None:
        return inputs
    block = parent.parent.module
    gate = getattr(block, "_dc_wan_gate", None)
    if not isinstance(gate, dict) or "temb" not in gate:
        raise RuntimeError(
            f"Wan gated eval for {getattr(parent, 'name', parent)}/{field_name} "
            "needs block._dc_wan_gate['temb'] from calib attach"
        )
    ref = inputs.front().data[0]
    temb = gate["temb"]
    if temb.data[0].shape[0] != ref.shape[0]:
        raise RuntimeError(
            f"Wan gate kwargs batch mismatch for {getattr(parent, 'name', parent)}/{field_name}: "
            f"temb_batch={temb.data[0].shape[0]}, act_batch={ref.shape[0]}"
        )
    tensors = OrderedDict(inputs.tensors)
    tensors["temb"] = temb
    # Full Attention replay needs RoPE; Linear o_proj / FFN do not.
    needs_rotary = False
    if kind == "msa":
        if field_name.endswith("o_proj"):
            needs_rotary = False
        elif eval_module is not None and not isinstance(eval_module, nn.Linear):
            needs_rotary = True
        elif field_name.endswith(("q_proj", "k_proj", "v_proj")) or field_name == "":
            needs_rotary = True
    if needs_rotary and "rotary" in gate:
        rot = gate["rotary"]
        if rot.data[0].shape[0] != ref.shape[0]:
            raise RuntimeError(
                f"Wan rotary batch mismatch for {getattr(parent, 'name', parent)}/{field_name}: "
                f"rotary_batch={rot.data[0].shape[0]}, act_batch={ref.shape[0]}"
            )
        tensors["rotary_emb"] = rot
    return TensorsCache(tensors)


def get_needs_inputs_fn(
    model: DiffusionModelStruct, config: DiffusionQuantConfig
) -> tp.Callable[[str, nn.Module], bool]:
    """Get function that checks whether the module needs to cache inputs.

    Args:
        model (`DiffusionModelStruct`):
            The diffused model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.

    Returns:
        `Callable[[str, nn.Module], bool]`:
            The function that checks whether the module needs to cache inputs.
    """

    needs_inputs_names = set()
    for module_key, module_name, _, parent, field_name in model.named_key_modules():
        if (config.enabled_wgts and config.wgts.is_enabled_for(module_key)) or (
            config.enabled_ipts and config.ipts.is_enabled_for(module_key)
        ):
            if isinstance(parent, DiffusionAttentionStruct):
                if field_name.endswith("o_proj"):
                    needs_inputs_names.add(module_name)
                elif field_name in ("q_proj", "k_proj", "v_proj"):
                    needs_inputs_names.add(parent.q_proj_name)
                    if parent.parent.parallel and parent.idx == 0:
                        needs_inputs_names.add(parent.parent.name)
                    else:
                        needs_inputs_names.add(parent.name)
                elif field_name in ("add_q_proj", "add_k_proj", "add_v_proj"):
                    needs_inputs_names.add(parent.add_k_proj_name)
                    if parent.parent.parallel and parent.idx == 0:
                        needs_inputs_names.add(parent.parent.name)
                    else:
                        needs_inputs_names.add(parent.name)
                else:
                    raise RuntimeError(f"Unknown field name: {field_name}")
            elif isinstance(parent, DiffusionFeedForwardStruct):
                if field_name == "up_proj":
                    needs_inputs_names.update(parent.up_proj_names[: parent.config.num_experts])
                elif field_name == "down_proj":
                    needs_inputs_names.update(parent.down_proj_names[: parent.config.num_experts])
                else:
                    raise RuntimeError(f"Unknown field name: {field_name}")
            else:
                needs_inputs_names.add(module_name)

    def needs_inputs(name: str, module: nn.Module) -> bool:
        return name in needs_inputs_names

    return needs_inputs


def get_needs_outputs_fn(
    model: DiffusionModelStruct, config: DiffusionQuantConfig
) -> tp.Callable[[str, nn.Module], bool]:
    """Get function that checks whether the module needs to cache outputs.

    Args:
        model (`DiffusionModelStruct`):
            The diffused model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.

    Returns:
        `Callable[[str, nn.Module], bool]`:
            The function that checks whether the module needs to cache outputs.
    """

    # TODO: Implement the function that checks whether the module needs to cache outputs.

    def needs_outputs(name: str, module: nn.Module) -> bool:
        return False

    return needs_outputs
