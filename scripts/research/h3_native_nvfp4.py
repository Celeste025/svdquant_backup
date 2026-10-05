"""E009 H3 legacy-recipe native NVFP4 linear and offline-export loader.

Only the 200 main-block linears are replaced. This module owns no pipeline,
offloading, attention, or serving implementation. The slow activation encoder
is a correctness reference, not a performance implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from wan_native_nvfp4 import PackedNVFP4, swizzle_scales, unswizzle_scales

FORMAT = 'h3-native-nvfp4-legacy-v1'
SUFFIXES = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')
TARGET_NAMES = tuple(f'blocks.{block}.{suffix}' for block in range(50) for suffix in SUFFIXES)
RECIPE = {
    'input_dtype': 'bfloat16', 'group_size': 16, 'rank': 32,
    'global': 'max(max(abs(x)),1e-12)/2688 then clamp_min(1e-12), FP32',
    'group_scale': 'max(amax(group)/6,1e-12)/global, clamp FP32 tiny then E4M3FN RNE',
    'e2m1': 'E005 signed nibble convention; ties to numerically smaller value; negative rounded zero uses nibble8',
    'zero_sf': 'E005 canonical nibble0 for SF0; reject nonzero-input SF0; independent old QDQ numeric roundtrip required',
    'residual': 'BF16(W*s) - BF16(B@A), BF16 result; original GPU expression',
    'lr': '1.0 * BF16(BF16(x/s) @ A.T @ B.T), two BF16 F.linear calls',
    'main': 'packed NVFP4 F.scaled_mm with E4M3 block scale and FP32 tensor globals',
    'out': 'BF16 main (including original bias) + BF16 LR',
}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


@torch.inference_mode()
def pack_activation_legacy(x: torch.Tensor, *, chunk_rows: int = 128) -> PackedNVFP4:
    """Exact H3 common.py arithmetic, independent packed representation.

    No BF16 conversion is inserted before codebook rounding. Global reduction
    sees the complete tensor even though the group encoder is row-chunked.
    This deliberately does not call the different DeepCompressor/Wan recipe.
    """
    if x.dtype != torch.bfloat16 or x.ndim < 2 or x.shape[-1] % 32:
        raise ValueError(f'Expected BF16 [...,K] with K multiple of 32; got {x.dtype} {x.shape}')
    if chunk_rows < 1 or not bool(torch.isfinite(x).all()):
        raise ValueError('H3 pack requires finite input and positive chunk_rows')
    shape = tuple(x.shape)
    flat = x.reshape(-1, shape[-1])
    rows, k = flat.shape
    if rows < 1:
        raise ValueError('Empty NVFP4 matrix')
    g = (x.float().abs().amax().clamp_min(1e-12)/(6.0*448.0)).clamp_min(1e-12)
    scales = torch.empty((rows, k//16), device=x.device, dtype=torch.float8_e4m3fn)
    codes = torch.empty((rows, k//2), device=x.device, dtype=torch.uint8)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.],
                          device=x.device, dtype=torch.float32)
    underflow = torch.zeros((), device=x.device, dtype=torch.int64)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start+chunk_rows)
        grouped = flat[start:stop].float().reshape(-1, k//16, 16)
        amax = grouped.abs().amax(-1, keepdim=True)
        ideal = amax.div(6.0).clamp_min(1e-12)
        sf = (ideal/g).clamp_min(torch.finfo(torch.float32).tiny).to(torch.float8_e4m3fn)
        effective = sf.float()*g
        z = grouped/effective
        # E005 canonical signed zeros. Numerically identical to the historical
        # signed15 argmin, but a rounded negative zero keeps its sign nibble.
        z = torch.where(effective == 0, torch.zeros_like(z), z)
        distance = (z.abs().unsqueeze(-1)-levels).abs()
        nearest = distance == distance.amin(-1, keepdim=True)
        low = nearest.to(torch.int8).argmax(-1)
        high = 7-nearest.flip(-1).to(torch.int8).argmax(-1)
        index = torch.where(z < 0, high, low)
        chosen = (index+(z < 0).to(torch.int64)*8).to(torch.uint8).reshape(stop-start, k)
        codes[start:stop] = chosen[:, ::2] | (chosen[:, 1::2] << 4)
        scales[start:stop] = sf.squeeze(-1)
        underflow += ((amax != 0) & (sf.float() == 0)).sum()
    if bool(underflow != 0):
        raise ValueError('Nonzero H3 groups underflow E4M3 SF: outside validated legacy packed domain')
    if not bool(torch.isfinite(scales.float()).all()):
        raise ValueError('Nonfinite H3 block scales')
    return PackedNVFP4(codes, None, g.reshape(1), swizzle_scales(scales), shape,
                       'h3_legacy_fp32_clamped_global_e4m3_rne_e2m1_ties_lower')


@torch.inference_mode()
def packet_statistics(packet: PackedNVFP4) -> dict:
    rows, k2 = packet.packed.shape
    sf = unswizzle_scales(packet.swizzled_scales, rows, k2*2//16).float()
    return {'global_scale': float(packet.global_scale), 'groups': sf.numel(),
            'zero_sf': int((sf == 0).sum()),
            'nonzero_subnormal_sf': int(((sf > 0) & (sf < 2**-6)).sum()),
            'scales_at_448': int((sf == 448).sum()),
            'packed_sha256': tensor_sha256(packet.packed),
            'global_sha256': tensor_sha256(packet.global_scale),
            'swizzled_sf_sha256': tensor_sha256(packet.swizzled_scales)}


class NativeH3Linear(nn.Module):
    """Resident packed residual plus exact old smoothing/BF16 LR order."""
    def __init__(self, packet: PackedNVFP4, smooth: torch.Tensor,
                 a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None,
                 *, activation_packer: str | Callable = 'legacy', chunk_rows: int = 128):
        super().__init__()
        self.in_features = packet.packed.shape[1]*2
        self.out_features = packet.packed.shape[0]
        if tuple(a.shape) != (32, self.in_features) or tuple(b.shape) != (self.out_features, 32):
            raise ValueError('Expected rank32 H3 branch shapes')
        if smooth.numel() != self.in_features:
            raise ValueError('Invalid smoothing shape')
        if any(t.dtype != torch.bfloat16 for t in (smooth, a, b)):
            raise ValueError('Smooth and LR must be BF16')
        if not bool(torch.isfinite(smooth).all()) or not bool((smooth > 0).all()):
            raise ValueError('Smooth must be positive and finite')
        if not all(bool(torch.isfinite(t).all()) for t in (a, b)):
            raise ValueError('Nonfinite BF16 LR')
        self.register_buffer('weight_packed', packet.packed)
        self.register_buffer('weight_scales_swizzled', packet.swizzled_scales)
        self.register_buffer('weight_global', packet.global_scale)
        self.register_buffer('smooth', smooth.reshape(-1))
        self.register_buffer('lr_a', a)
        self.register_buffer('lr_b', b)
        self.register_buffer('bias', bias)
        self.activation_packer = activation_packer
        self.chunk_rows = chunk_rows
        self.execution_mode = 'native'

    def weight_packet(self) -> PackedNVFP4:
        return PackedNVFP4(self.weight_packed, None, self.weight_global,
                           self.weight_scales_swizzled,
                           (self.out_features, self.in_features), 'h3_exported_legacy_residual')

    def pack_input(self, x_s: torch.Tensor, *, packer: str | Callable | None = None) -> PackedNVFP4:
        """Pure pack interface: caller supplies input AFTER BF16 smoothing."""
        selected = self.activation_packer if packer is None else packer
        if selected == 'legacy':
            return pack_activation_legacy(x_s, chunk_rows=self.chunk_rows)
        if callable(selected):
            return selected(x_s, chunk_rows=self.chunk_rows)
        raise ValueError(f'Unknown H3 packer {selected!r}')

    @torch.inference_mode()
    def main_from_packet(self, activation: PackedNVFP4, *, mode: str = 'native',
                         include_bias: bool = False) -> torch.Tensor:
        bias = self.bias if include_bias else None
        if mode == 'native':
            recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
            out = F.scaled_mm(
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
            out = F.linear(x, w, bias)
        else:
            raise ValueError(mode)
        return out.reshape(*activation.original_shape[:-1], self.out_features)

    def main_forward(self, x_s: torch.Tensor, *, mode: str = 'native',
                     include_bias: bool = False, packer=None) -> torch.Tensor:
        return self.main_from_packet(self.pack_input(x_s, packer=packer), mode=mode,
                                     include_bias=include_bias)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(f'H3 input dtype changed to {x.dtype}')
        x_s = x/self.smooth
        if self.execution_mode == 'legacy_qdq':
            # Import only for reference mode; no historical module mutation.
            from minimax_h3_svdquant_common import nvfp4_qdq
            qx = nvfp4_qdq(x_s, element_size=self.chunk_rows)
            branch = 1.0 * F.linear(F.linear(x_s, self.lr_a), self.lr_b)
            main = F.linear(qx, self.weight_packet().decode(chunk_rows=self.chunk_rows), self.bias)
        elif self.execution_mode in ('native', 'packed_qdq'):
            packet = self.pack_input(x_s)
            branch = 1.0 * F.linear(F.linear(x_s, self.lr_a), self.lr_b)
            main = self.main_from_packet(packet, mode=self.execution_mode, include_bias=True)
        else:
            raise ValueError(self.execution_mode)
        return main + branch

    @classmethod
    def from_export(cls, payload: dict, *, device='cuda', activation_packer='legacy', chunk_rows=128):
        if payload['format'] != FORMAT or payload['recipe'] != RECIPE:
            raise ValueError('Unsupported H3 export recipe')
        tensors = {k: v.to(device=device) if torch.is_tensor(v) else v
                   for k, v in payload['tensors'].items()}
        packet = PackedNVFP4(tensors['weight_packed'], None, tensors['weight_global'],
                             tensors['weight_scales_swizzled'], tuple(payload['shape']), 'h3_exported_legacy_residual')
        return cls(packet, tensors['smooth'], tensors['lr_a'], tensors['lr_b'], tensors['bias'],
                   activation_packer=activation_packer, chunk_rows=chunk_rows)


@contextmanager
def native_execution_mode(dit: nn.Module, mode: str):
    if mode not in ('native', 'packed_qdq', 'legacy_qdq'):
        raise ValueError(mode)
    modules = [(m, m.execution_mode) for m in dit.modules() if isinstance(m, NativeH3Linear)]
    try:
        for module, _ in modules:
            module.execution_mode = mode
        yield
    finally:
        for module, previous in modules:
            module.execution_mode = previous


def native_bypass(dit: nn.Module):
    return native_execution_mode(dit, 'legacy_qdq')


@torch.inference_mode()
def install_native_h3(dit: nn.Module, export_dir: str | Path, *, device='cuda',
                      activation_packer='legacy', chunk_rows: int = 128,
                      verify_hashes: bool = True) -> dict:
    """Replace only 200 main linears; caller owns non-target residency.

    Replaced modules have old Parameters cleared so stale wrapper references
    cannot retain the old GPU weights. No hooks or injected LoRA are accepted.
    """
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir/'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('format') != FORMAT or manifest.get('recipe') != RECIPE:
        raise ValueError('Incomplete or incompatible H3 export')
    rows = manifest['layers']
    if len(rows) != 200 or {row['name'] for row in rows} != set(TARGET_NAMES):
        raise ValueError('Export does not contain the exact 200 main-block targets')
    report = {'target_count': 0, 'exact_roundtrip_count': 0, 'layers': [],
              'manifest_sha256': file_sha256(export_dir/'manifest.json'),
              'scope': '200 main-block linears only; caller owns non-target residency'}
    for row in rows:
        name = row['name']
        old = dit.get_submodule(name)
        if not isinstance(old, nn.Linear) or old._forward_pre_hooks or old._forward_hooks:
            raise ValueError(f'{name}: requires original Linear without runtime hooks')
        if getattr(old, 'lora_A_weights', []):
            raise ValueError(f'{name}: unexpected hotloaded LoRA')
        path = export_dir/row['file']
        if verify_hashes and file_sha256(path) != row['file_sha256']:
            raise ValueError(f'{name}: export SHA mismatch')
        if not row['roundtrip']['exact'] or row['roundtrip']['changed_elements'] != 0:
            raise ValueError(f'{name}: failed legacy weight roundtrip')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload['name'] != name or tuple(payload['shape']) != (old.out_features, old.in_features):
            raise ValueError(f'{name}: name/shape mismatch')
        replacement = NativeH3Linear.from_export(payload, device=device,
                                                  activation_packer=activation_packer, chunk_rows=chunk_rows)
        parent, attr = name.rsplit('.', 1)
        setattr(dit.get_submodule(parent), attr, replacement)
        old_type = f'{type(old).__module__}.{type(old).__qualname__}'
        old.weight = None
        old.bias = None
        report['layers'].append({'name': name, 'old_type': old_type, 'old_weight_parameter_cleared': True,
                                  'file_sha256': row['file_sha256'], 'roundtrip': row['roundtrip']})
        report['target_count'] += 1
        report['exact_roundtrip_count'] += 1
        del old, replacement, payload
    return report
