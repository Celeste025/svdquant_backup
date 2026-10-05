"""E007: convert an already loaded legacy Wan SVDQuant graph to native NVFP4.

This is a correctness implementation, not a fast activation packer. ``legacy``
asks the original DeepCompressor quantizer for its codes/scales. Smoothing and
BF16 low-rank hooks retain their order; only the large residual GEMM changes.
No weight scales are recomputed, and no original residual Parameter is retained.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn
import torch.nn.functional as F


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Logical [rows,K/16] E4M3 -> SM120 128x4/32x4x4 storage."""
    rows, cols = scales.shape
    rp, cp = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    padded = torch.zeros((rp, cp), device=scales.device, dtype=scales.dtype)
    padded[:rows, :cols] = scales
    return (padded.view(rp // 128, 128, cp // 4, 4).permute(0, 2, 1, 3)
            .reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous())


def unswizzle_scales(storage: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    rp, cp = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    return (storage.reshape(-1, 32, 4, 4).transpose(1, 2)
            .reshape(rp // 128, cp // 4, 128, 4).permute(0, 2, 1, 3)
            .reshape(rp, cp)[:rows, :cols])


def _pack_values(values: torch.Tensor) -> torch.Tensor:
    """Encode already chosen E2M1 values; this function does no rounding."""
    levels = values.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    absolute = values.abs().contiguous()
    indices = torch.bucketize(absolute, levels).clamp_max(7)
    if not torch.equal(levels[indices], absolute):
        raise ValueError("quantizer returned non-E2M1 values; cannot pack faithfully")
    code = (indices + (values < 0).to(torch.int64) * 8).to(torch.uint8)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()


@dataclass
class PackedNVFP4:
    packed: torch.Tensor  # [rows,K/2] uint8, low nibble first
    scales: torch.Tensor | None  # optional logical [rows,K/16] E4M3 oracle view
    global_scale: torch.Tensor  # one FP32 decode multiplier
    swizzled_scales: torch.Tensor
    original_shape: tuple[int, ...]
    recipe: str

    @torch.inference_mode()
    def decode(self, dtype: torch.dtype = torch.bfloat16, chunk_rows: int = 128) -> torch.Tensor:
        """Independent signed-nibble LUT reconstruction, no encoder logic."""
        rows, k2 = self.packed.shape
        scales = self.scales
        if scales is None:
            scales = unswizzle_scales(self.swizzled_scales, rows, k2 * 2 // 16)
        out = torch.empty((rows, k2 * 2), device=self.packed.device, dtype=dtype)
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                            -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                           device=self.packed.device, dtype=torch.float32)
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            p = self.packed[start:stop]
            codes = torch.stack((p & 15, p >> 4), -1).reshape(stop-start, k2*2)
            effective = scales[start:stop].float() * self.global_scale
            out[start:stop] = (lut[codes.long()] * effective.repeat_interleave(16, -1)).to(dtype)
        return out.reshape(self.original_shape)


def _packet(values: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor,
            original_shape: tuple[int, ...], recipe: str) -> PackedNVFP4:
    rows, k = values.shape
    if k % 32 or rows < 1:
        raise ValueError(f"unsupported packed activation/weight shape {values.shape}")
    global_scale = global_scale.to(device=values.device, dtype=torch.float32).reshape(-1)
    if global_scale.numel() != 1 or not bool(torch.isfinite(global_scale).all()) or not bool((global_scale > 0).all()):
        raise ValueError("NVFP4 requires a positive finite scalar global; all-zero legacy input needs separate handling")
    scales = scales.reshape(rows, k // 16).to(device=values.device, dtype=torch.float32)
    sf = scales.to(torch.float8_e4m3fn)
    if not torch.equal(sf.float(), scales) or not bool((scales >= 0).all()):
        raise ValueError("block scales cannot be represented exactly as nonnegative E4M3")
    packed = _pack_values(values)
    return PackedNVFP4(packed, sf, global_scale, swizzle_scales(sf), original_shape, recipe)


@torch.inference_mode()
def pack_activation_legacy(x: torch.Tensor, *, quantizer, chunk_rows: int = 128) -> PackedNVFP4:
    """Slow exact-code oracle using the actual loaded legacy quantizer.

    Intentionally do not request qdata and data together: the old implementation
    aliases its qdata while dequantizing in place. Separate calls are required
    for a legacy-QDQ reference. Global reduction sees the original full shape.
    """
    if x.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 input after smoothing, got {x.dtype}")
    result = quantizer.quantize(x, return_with_dequant=False, return_with_quant=True)
    if result.qdata is None or result.scale is None:
        raise ValueError("legacy activation quantizer did not return codes/scales")
    scales = result.scale.state_dict("sf", device=x.device)
    if set(scales) != {"sf.0", "sf.1"}:
        raise ValueError(f"expected exactly two NVFP4 scale levels, got {list(scales)}")
    if result.zero is not None and bool(torch.as_tensor(result.zero).ne(0).any()):
        raise ValueError("nonzero quantizer zero point is not supported")
    flat = result.qdata.reshape(-1, x.shape[-1])
    # remove_zero() may have changed effective scale without changing leaves.
    # Native zero-SF is equivalent only where all selected codes are zero.
    native_effective = scales['sf.1'].float() * scales['sf.0'].float()
    effective = result.scale.data
    mismatch = effective != native_effective
    if bool(mismatch.any()):
        changed_nonzero = mismatch.reshape(-1, x.shape[-1] // 16, 1) & flat.reshape(-1, x.shape[-1] // 16, 16).ne(0)
        if bool(changed_nonzero.any()):
            raise ValueError("legacy effective scale differs from scale leaves at nonzero codes; no faithful packed expression")
    return _packet(flat, scales['sf.1'], scales['sf.0'], tuple(x.shape), 'legacy_codes_and_legacy_scales')


@torch.inference_mode()
def pack_activation_rne(x: torch.Tensor, *, quantizer, chunk_rows: int = 128) -> PackedNVFP4:
    """E2M1 RNE diagnostic, keeping the SAME legacy E4M3/global scales.

    This is deliberately not advertised as a complete hardware-RNE quantizer.
    It changes only E2M1 tie-breaking and isolates that effect.
    """
    base = pack_activation_legacy(x, quantizer=quantizer, chunk_rows=chunk_rows)
    flat = x.reshape(-1, x.shape[-1])
    levels = x.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
    even = torch.arange(8, device=x.device) % 2 == 0
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(flat.shape[0], start + chunk_rows)
        effective = base.scales[start:stop].float() * base.global_scale
        z = flat[start:stop].float().reshape(-1, x.shape[-1] // 16, 16) / effective.unsqueeze(-1)
        z = torch.where(effective.unsqueeze(-1) == 0, 0., z).reshape(stop-start, -1)
        dist = (z.abs().unsqueeze(-1) - levels).abs()
        nearest = dist == dist.amin(-1, keepdim=True)
        low = nearest.to(torch.int8).argmax(-1)
        preferred = nearest & even
        index = torch.where(preferred.any(-1), preferred.to(torch.int8).argmax(-1), low)
        value = levels[index] * torch.where(z < 0, -1., 1.)
        base.packed[start:stop] = _pack_values(value)
    base.recipe = 'e2m1_rne_with_fixed_legacy_scales'
    return base


ActivationPacker = Callable[..., PackedNVFP4]
PACKERS = {'legacy': pack_activation_legacy, 'rne': pack_activation_rne}


class _ConditionalQuantizerHook:
    """Keep original prehook position, enabling QDQ only for exact replay."""
    def __init__(self, original):
        self.original = original

    def __call__(self, module, args, kwargs):
        if module.execution_mode == 'legacy_qdq':
            return self.original(module, args, kwargs)
        return None


class NativeWanLinear(nn.Module):
    """Packed residual linear; existing hooks still implement smoothing/LR."""
    def __init__(self, packet: PackedNVFP4, bias: torch.Tensor | None, quantizer,
                 *, activation_packer: str | ActivationPacker = 'legacy', chunk_rows: int = 128):
        super().__init__()
        self.in_features = packet.packed.shape[1] * 2
        self.out_features = packet.packed.shape[0]
        self.register_buffer('weight_packed', packet.packed)
        # Only one SF layout stays resident; decode reconstructs a logical view.
        self.register_buffer('weight_scales', None)
        self.register_buffer('weight_global', packet.global_scale)
        self.register_buffer('weight_scales_swizzled', packet.swizzled_scales)
        self.register_buffer('bias', None if bias is None else bias.detach().clone())
        self.activation_quantizer = quantizer
        self.activation_packer = activation_packer
        self.chunk_rows = chunk_rows
        self.execution_mode = 'native'

    def weight_packet(self) -> PackedNVFP4:
        return PackedNVFP4(self.weight_packed, self.weight_scales, self.weight_global,
                           self.weight_scales_swizzled, (self.out_features, self.in_features), 'saved_weight_codes')

    def pack_input(self, x: torch.Tensor, *, packer: str | ActivationPacker | None = None) -> PackedNVFP4:
        selected = self.activation_packer if packer is None else packer
        function = PACKERS[selected] if isinstance(selected, str) else selected
        return function(x, quantizer=self.activation_quantizer, chunk_rows=self.chunk_rows)

    @torch.inference_mode()
    def main_from_packet(self, activation: PackedNVFP4, *, mode: str = 'native', include_bias: bool = False) -> torch.Tensor:
        bias = self.bias if include_bias else None
        if mode == 'native':
            recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
            y = F.scaled_mm(
                activation.packed.view(torch.float4_e2m1fn_x2),
                self.weight_packed.view(torch.float4_e2m1fn_x2).t(),
                [activation.swizzled_scales, activation.global_scale], recipe,
                [self.weight_scales_swizzled, self.weight_global], recipe,
                swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                bias=bias, output_dtype=torch.bfloat16)
        elif mode == 'packed_qdq':
            x = activation.decode(chunk_rows=self.chunk_rows).reshape(-1, self.in_features)
            w = self.weight_packet().decode(chunk_rows=self.chunk_rows)
            y = F.linear(x, w, bias)
        else:
            raise ValueError(f"unknown main mode {mode}")
        return y.reshape(*activation.original_shape[:-1], self.out_features)

    @torch.inference_mode()
    def main_forward(self, x: torch.Tensor, *, mode: str = 'native', include_bias: bool = False,
                     packer: str | ActivationPacker | None = None) -> torch.Tensor:
        """Pure residual branch: no hooks, no LR, optional bias, already-smoothed x."""
        return self.main_from_packet(self.pack_input(x, packer=packer), mode=mode, include_bias=include_bias)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution_mode == 'legacy_qdq':
            # Original ProcessHook has already transformed x at its old position.
            # Reconstruction is temporary; no BF16 weight remains on the module.
            return F.linear(x, self.weight_packet().decode(chunk_rows=self.chunk_rows), self.bias)
        return self.main_forward(x, mode=self.execution_mode, include_bias=True)


@contextmanager
def native_execution_mode(transformer: nn.Module, mode: str) -> Iterator[None]:
    if mode not in {'native', 'legacy_qdq', 'packed_qdq'}:
        raise ValueError(mode)
    modules = [(m, m.execution_mode) for m in transformer.modules() if isinstance(m, NativeWanLinear)]
    try:
        for module, _ in modules:
            module.execution_mode = mode
        yield
    finally:
        for module, previous in modules:
            module.execution_mode = previous


def native_bypass(transformer: nn.Module):
    """Run exact old QDQ hook + standard Linear semantics with restored BF16 W."""
    return native_execution_mode(transformer, 'legacy_qdq')


def _description(hook) -> dict:
    result = {'type': f'{type(hook).__module__}.{type(hook).__qualname__}', 'object_id': id(hook)}
    if hasattr(hook, 'processor'):
        result['processor_type'] = f'{type(hook.processor).__module__}.{type(hook.processor).__qualname__}'
    if hasattr(hook, 'branch'):
        result['branch_id'] = id(hook.branch)
        result['branch_a_id'] = id(hook.branch.a)
    return result


def _transfer_hooks(old: nn.Linear, new: NativeWanLinear, quant_hook) -> dict:
    pre = [(key, hook, key in old._forward_pre_hooks_with_kwargs) for key, hook in old._forward_pre_hooks.items()]
    post = [(key, hook, key in old._forward_hooks_with_kwargs, key in old._forward_hooks_always_called)
            for key, hook in old._forward_hooks.items()]
    unique = {id(h): h for _, h, *_ in [*pre, *post]}
    # Remove DeepCompressor owner->module references, not just dictionary entries.
    for hook in unique.values():
        if hasattr(hook, 'handles') and hasattr(hook, 'remove'):
            hook.remove(old)
    old._forward_pre_hooks.clear(); old._forward_pre_hooks_with_kwargs.clear()
    old._forward_hooks.clear(); old._forward_hooks_with_kwargs.clear(); old._forward_hooks_always_called.clear()
    manifest = {'pre': [], 'post': []}
    for old_key, hook, with_kwargs in pre:
        delegated = _ConditionalQuantizerHook(hook) if hook is quant_hook else hook
        handle = new.register_forward_pre_hook(delegated, with_kwargs=with_kwargs)
        if hasattr(hook, 'handles'):
            hook.handles[new].append(handle)
        manifest['pre'].append(dict(old_handle=old_key, new_handle=handle.id, with_kwargs=with_kwargs,
                                    conditional_qdq=hook is quant_hook, **_description(hook)))
    for old_key, hook, with_kwargs, always_call in post:
        handle = new.register_forward_hook(hook, with_kwargs=with_kwargs, always_call=always_call)
        if hasattr(hook, 'handles'):
            hook.handles[new].append(handle)
        manifest['post'].append(dict(old_handle=old_key, new_handle=handle.id, with_kwargs=with_kwargs,
                                     always_call=always_call, **_description(hook)))
    return manifest


@torch.inference_mode()
def _recover_saved_weight(weight: torch.Tensor, global_scale: torch.Tensor, scales: torch.Tensor,
                          *, chunk_rows: int) -> tuple[PackedNVFP4, dict]:
    rows, k = weight.shape
    if weight.dtype != torch.bfloat16 or rows % 32 or k % 32:
        raise ValueError(f"unsupported saved weight {weight.shape}, {weight.dtype}")
    global_scale = global_scale.to(device=weight.device, dtype=torch.float32).reshape(1)
    scales = scales.to(device=weight.device, dtype=torch.float32).reshape(rows, k // 16)
    # Positive magnitude nearest search is only code RECOVERY here. The saved
    # weight is already a BF16 reconstruction of a chosen quantization code.
    levels = weight.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
    packed = torch.empty((rows, k // 2), device=weight.device, dtype=torch.uint8)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        effective = scales[start:stop].unsqueeze(-1) * global_scale
        z = weight[start:stop].float().reshape(stop-start, k // 16, 16) / effective
        z = torch.where(effective == 0, 0., z).reshape(stop-start, k)
        indices = (z.abs().unsqueeze(-1) - levels).abs().argmin(-1)
        values = levels[indices] * torch.where(z < 0, -1., 1.)
        packed[start:stop] = _pack_values(values)
    sf = scales.to(torch.float8_e4m3fn)
    if not torch.equal(sf.float(), scales) or not bool((global_scale > 0).all()):
        raise ValueError('saved scales are not a finite exact E4M3/FP32 NVFP4 pair')
    packet = PackedNVFP4(packed, sf, global_scale, swizzle_scales(sf), tuple(weight.shape), 'saved_weight_codes')
    reconstructed = packet.decode(chunk_rows=chunk_rows)
    equal = torch.equal(reconstructed, weight)
    changed = int((reconstructed != weight).sum())
    maxabs = float((reconstructed.float() - weight.float()).abs().max())
    if not equal:
        raise ValueError(f'saved weight roundtrip failed: changed={changed}, max_abs={maxabs}')
    return packet, {'shape': list(weight.shape), 'elements': weight.numel(), 'exact': equal,
                    'changed_elements': changed, 'max_abs': maxabs, 'bf16_bytes': weight.numel()*2,
                    'packed_bytes': packed.numel(), 'logical_scale_bytes': sf.numel(),
                    'swizzled_scale_bytes': packet.swizzled_scales.numel()}


@torch.inference_mode()
def convert_wan_transformer_to_native(transformer: nn.Module, checkpoint_dir: str | Path, *,
                                      activation_packer: str | ActivationPacker = 'legacy',
                                      chunk_rows: int = 128) -> dict:
    """Convert the graph AFTER the historical load_quantized_transformer.

    All 300 weight roundtrips/prehook checks complete before graph mutation.
    No calibration or repeated smoothing/subtraction is performed.
    """
    from deepcompressor.app.diffusion.quant.quantizer import DiffusionActivationQuantizer
    from deepcompressor.utils.hooks import AccumBranchHook, ProcessHook
    checkpoint_dir = Path(checkpoint_dir)
    states = torch.load(checkpoint_dir/'wgts.pt', map_location='cpu', weights_only=False, mmap=True)
    scale_state = torch.load(checkpoint_dir/'scale.pt', map_location='cpu', weights_only=False, mmap=True)
    saved_model = torch.load(checkpoint_dir/'model.pt', map_location='cpu', weights_only=False, mmap=True)
    if len(states) != 300:
        raise ValueError(f'E007 expects 300 saved target linears, got {len(states)}')
    staged = []
    report = {'implementation': 'native_torch_gemm_legacy_smoothing_bf16_lr',
              'activation_packer': activation_packer if isinstance(activation_packer, str) else repr(activation_packer),
              'performance_status': 'correctness_only_slow_reference_packer',
              'target_count': len(states), 'weights': [], 'hooks': {}, 'parent_hooks_preserved': {}}
    for name in states:
        module = transformer.get_submodule(name)
        if type(module) is not nn.Linear:
            raise TypeError(f'{name}: only standard nn.Linear can be faithfully replayed, got {type(module)}')
        if module._backward_hooks or module._backward_pre_hooks:
            raise ValueError(f'{name}: training/backward hooks are outside inference scope')
        hooks = list(module._forward_pre_hooks.values())
        quant_hooks = [h for h in hooks if isinstance(h, ProcessHook) and isinstance(h.processor, DiffusionActivationQuantizer)]
        branches = [h for h in hooks if isinstance(h, AccumBranchHook)]
        if len(quant_hooks) != 1 or len(branches) != 1:
            raise ValueError(f'{name}: expected one activation QDQ and one LR prehook, got {len(quant_hooks)}, {len(branches)}')
        qhook = quant_hooks[0]
        if not qhook.pre or qhook.post or not qhook.activated or not qhook.processor.is_enabled():
            raise ValueError(f'{name}: unexpected QDQ hook status')
        if hooks.index(branches[0]) >= hooks.index(qhook):
            raise ValueError(f'{name}: LR hook must capture input BEFORE activation QDQ')
        if branches[0].tensor is not None:
            raise ValueError(f'{name}: cannot convert during an active forward')
        cpu_saved = saved_model[name+'.weight']
        if not torch.equal(module.weight.detach().cpu(), cpu_saved):
            raise ValueError(f'{name}: loaded residual weight differs from model.pt (double transform or wrong checkpoint)')
        zero = scale_state[name+'.weight.zero']
        if bool(torch.as_tensor(zero).ne(0).any()):
            raise ValueError(f'{name}: nonzero weight zero-point')
        packet, entry = _recover_saved_weight(module.weight, scale_state[name+'.weight.scale.0'],
                                              scale_state[name+'.weight.scale.1'], chunk_rows=chunk_rows)
        replacement = NativeWanLinear(packet, module.bias, qhook.processor,
                                      activation_packer=activation_packer, chunk_rows=chunk_rows)
        replacement.train(module.training)
        for attr in ['in_smooth_cache_key', 'out_smooth_cache_key']:
            if hasattr(module, attr):
                setattr(replacement, attr, getattr(module, attr))
        entry['name'] = name
        report['weights'].append(entry)
        staged.append((name, module, replacement, qhook))
    target_names = set(states)
    for name, module in transformer.named_modules():
        if name not in target_names and (module._forward_pre_hooks or module._forward_hooks):
            report['parent_hooks_preserved'][name] = {
                'pre': [dict(handle_id=key, **_description(h)) for key, h in module._forward_pre_hooks.items()],
                'post': [dict(handle_id=key, **_description(h)) for key, h in module._forward_hooks.items()]}
    for name, old, replacement, qhook in staged:
        report['hooks'][name] = _transfer_hooks(old, replacement, qhook)
        parent_name, _, child_name = name.rpartition('.')
        transformer.get_submodule(parent_name)._modules[child_name] = replacement
        # Also clear original Parameters so any stale external module reference
        # cannot retain 2.8 GB of BF16 residual weights after conversion.
        old.register_parameter('weight', None)
        old.register_parameter('bias', None)
    report['exact_roundtrip_count'] = sum(r['exact'] for r in report['weights'])
    report['total_weight_elements'] = sum(r['elements'] for r in report['weights'])
    report['old_bf16_residual_bytes_removed'] = sum(r['bf16_bytes'] for r in report['weights'])
    report['packed_weight_bytes'] = sum(r['packed_bytes'] for r in report['weights'])
    report['resident_weight_scale_bytes'] = sum(r['swizzled_scale_bytes'] for r in report['weights'])
    report['resident_logical_weight_scale_bytes'] = 0
    del staged, saved_model, scale_state, states
    gc.collect()
    return report
