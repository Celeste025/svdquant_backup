"""E014 plain_h3_recipe: original BF16 weights, no smoothing or LR branch.

This is an engineering control, not a new quantizer or standard E2M1-RNE PTQ.
It shares the frozen H3 activation recipe and native scaled_mm ABI with E009.
Only the 200 main-block linears are replaced; caller owns model residency.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from h3_native_nvfp4 import TARGET_NAMES, file_sha256
from h3_nvfp4_zero_sf_compat import (
    COMPATIBILITY, pack_activation_fast, pack_activation_legacy,
)
from wan_native_nvfp4 import PackedNVFP4

FORMAT = 'h3-plain-native-nvfp4-v1'
RECIPE = {
    'label': 'plain_h3_recipe', 'input_dtype': 'bfloat16', 'group_size': 16,
    'rank': 0, 'smoothing': None,
    'weight_source': 'Original BF16 checkpoint W, without rescaling or low-rank subtraction',
    'global': 'max(max(abs(x)),1e-12)/2688 then clamp_min(1e-12), FP32',
    'group_scale': 'max(amax(group)/6,1e-12)/global, clamp FP32 tiny then E4M3FN RNE',
    'e2m1': 'E005 signed nibble convention; ties to numerically smaller value; negative rounded zero uses nibble8',
    'zero_sf': COMPATIBILITY,
    'main': 'packed NVFP4 F.scaled_mm with E4M3 block scale and FP32 tensor globals',
    'out': 'BF16 native main including unchanged original bias; no additional branch',
}
TENSOR_KEYS = {'weight_packed', 'weight_scales_swizzled', 'weight_global', 'bias'}


class PlainNativeH3Linear(nn.Module):
    """Resident original-W packet; no full BF16 W is kept or decoded at runtime."""
    def __init__(self, packet: PackedNVFP4, bias=None, *,
                 activation_packer=pack_activation_fast, chunk_rows=1024):
        super().__init__()
        self.out_features, packed_k = packet.packed.shape
        self.in_features = packed_k * 2
        if (tuple(packet.original_shape) != (self.out_features, self.in_features)
                or self.in_features % 32 or self.out_features % 32
                or packet.packed.dtype != torch.uint8
                or packet.swizzled_scales.dtype != torch.float8_e4m3fn
                or packet.global_scale.dtype != torch.float32
                or packet.global_scale.numel() != 1):
            raise ValueError('Invalid H3 original-weight packet metadata')
        expected_sf = ((self.out_features+127)//128)*128 * ((self.in_features//16+3)//4)*4
        if packet.swizzled_scales.numel() != expected_sf:
            raise ValueError('Invalid swizzled weight scale storage size')
        if not bool(torch.isfinite(packet.global_scale).all()) or not bool((packet.global_scale > 0).all()):
            raise ValueError('Invalid original-weight global scale')
        if bias is not None and (tuple(bias.shape) != (self.out_features,)
                                 or bias.dtype != torch.bfloat16):
            raise ValueError('Original bias shape/dtype mismatch')
        if not callable(activation_packer) or chunk_rows < 1:
            raise ValueError('Callable activation packer and positive chunk_rows required')
        self.register_buffer('weight_packed', packet.packed)
        self.register_buffer('weight_scales_swizzled', packet.swizzled_scales)
        self.register_buffer('weight_global', packet.global_scale.reshape(1))
        self.register_buffer('bias', bias)
        self.activation_packer = activation_packer
        self.chunk_rows = chunk_rows
        self.execution_mode = 'native'

    def weight_packet(self):
        return PackedNVFP4(self.weight_packed, None, self.weight_global,
                           self.weight_scales_swizzled,
                           (self.out_features, self.in_features), 'plain_h3_recipe_original_W')

    def pack_input(self, raw_x, *, packer=None):
        """Input is the unmodified original Linear input, never smoothed."""
        if raw_x.dtype != torch.bfloat16 or raw_x.shape[-1] != self.in_features:
            raise ValueError('Plain H3 activation dtype/last dimension mismatch')
        return (self.activation_packer if packer is None else packer)(raw_x, chunk_rows=self.chunk_rows)

    @torch.inference_mode()
    def main_from_packet(self, activation, *, mode='native', include_bias=False):
        if activation.packed.shape[1]*2 != self.in_features:
            raise ValueError('Activation packet K mismatch')
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
            # Explicit oracle only. The native forward never decodes weights.
            x = activation.decode(chunk_rows=self.chunk_rows).reshape(-1, self.in_features)
            out = F.linear(x, self.weight_packet().decode(chunk_rows=self.chunk_rows), bias)
        else:
            raise ValueError(mode)
        return out.reshape(*activation.original_shape[:-1], self.out_features)

    def main_forward(self, raw_x, *, mode='native', include_bias=False, packer=None):
        return self.main_from_packet(self.pack_input(raw_x, packer=packer),
                                     mode=mode, include_bias=include_bias)

    def forward(self, raw_x):
        return self.main_forward(raw_x, mode=self.execution_mode, include_bias=True)

    @classmethod
    def from_export(cls, payload, *, device='cuda', activation_packer=pack_activation_fast,
                    chunk_rows=1024):
        if payload['format'] != FORMAT or payload['recipe'] != RECIPE or set(payload['tensors']) != TENSOR_KEYS:
            raise ValueError('Unsupported plain H3 export format/recipe/tensors')
        tensors = {key: value.to(device=device) if torch.is_tensor(value) else value
                   for key, value in payload['tensors'].items()}
        packet = PackedNVFP4(tensors['weight_packed'], None, tensors['weight_global'],
                             tensors['weight_scales_swizzled'], tuple(payload['shape']), 'plain_h3_recipe_original_W')
        return cls(packet, tensors['bias'], activation_packer=activation_packer, chunk_rows=chunk_rows)


@contextmanager
def plain_execution_mode(dit, mode):
    if mode not in ('native', 'packed_qdq'):
        raise ValueError(mode)
    modules = [(m, m.execution_mode) for m in dit.modules() if isinstance(m, PlainNativeH3Linear)]
    try:
        for module, _ in modules:
            module.execution_mode = mode
        yield
    finally:
        for module, previous in modules:
            module.execution_mode = previous


def install_plain_h3(dit, export_dir, *, activation_packer=pack_activation_fast,
                     chunk_rows=1024, device='cuda', verify_hashes=True):
    """Install 200 original-W packets and clear stale original Parameter refs."""
    export_dir = Path(export_dir)
    manifest_path = export_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['layers']
    if (manifest.get('status') != 'complete' or manifest.get('format') != FORMAT
            or manifest.get('recipe') != RECIPE or manifest.get('target_count') != 200
            or manifest.get('exact_roundtrip_count') != 200
            or len(rows) != 200 or {r['name'] for r in rows} != set(TARGET_NAMES)):
        raise ValueError('Incomplete/incompatible 200-target plain H3 export')
    report = {'target_count': 0, 'exact_roundtrip_count': 0, 'layers': [],
              'format': FORMAT, 'recipe': RECIPE, 'export_dir': str(export_dir.resolve()),
              'manifest_sha256': file_sha256(manifest_path),
              'source_checkpoint': manifest['source_checkpoint'], 'sources': manifest['sources'],
              'scope': '200 original-W main linears; 8 refiner and all other modules unchanged'}
    # Validate all old modules before making any replacements.
    for name in TARGET_NAMES:
        old = dit.get_submodule(name)
        if (not isinstance(old, nn.Linear) or old._forward_pre_hooks or old._forward_hooks
                or getattr(old, 'lora_A_weights', []) or old.weight is None
                or old.weight.device.type == 'meta' or getattr(old, 'disk_offload', False)):
            raise ValueError(f'{name}: requires resident original Linear without hooks/LoRA/offload')
    for row in rows:
        name = row['name']
        old = dit.get_submodule(name)
        path = export_dir/row['file']
        if path.resolve().parent != export_dir.resolve():
            raise ValueError('Export layer path escapes export directory')
        if verify_hashes and file_sha256(path) != row['file_sha256']:
            raise ValueError(f'{name}: export file SHA mismatch')
        if not row['roundtrip']['exact'] or row['roundtrip']['changed_elements']:
            raise ValueError(f'{name}: failed original-W QDQ roundtrip')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if (payload['name'] != name or tuple(payload['shape']) != (old.out_features, old.in_features)
                or payload['source_weight_sha256'] != row['source_weight_sha256']
                or (payload['tensors']['bias'] is None) != (old.bias is None)):
            raise ValueError(f'{name}: payload/source/bias mismatch')
        if old.bias is not None and not torch.equal(old.bias.detach().cpu(), payload['tensors']['bias']):
            raise ValueError(f'{name}: changed original bias')
        replacement = PlainNativeH3Linear.from_export(payload, device=device,
                              activation_packer=activation_packer, chunk_rows=chunk_rows)
        parent, attr = name.rsplit('.', 1)
        setattr(dit.get_submodule(parent), attr, replacement)
        old.weight = None
        old.bias = None
        report['layers'].append({'name': name, 'file_sha256': row['file_sha256'],
                                'roundtrip': row['roundtrip'], 'old_weight_parameter_cleared': True})
        report['target_count'] += 1
        report['exact_roundtrip_count'] += 1
        del old, replacement, payload
    return report
