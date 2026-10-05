"""Plain Wan14B native NVFP4 installation, one temporary FP32 matrix at a time.

No training, smoothing or LR is introduced. Input weights must already be the
BF16 teacher values. No original-module dictionary, FP32 master, or state_dict
is retained. The caller must also discard any external old-weight references.
Do not call model.to(dtype=...) after conversion: tensor globals stay FP32.
CPU --check-only validates real meta-model structure, not GPU arithmetic.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from torch import nn
from wan_mainweight_qad import RECIPE, SUFFIXES, PlainPackedWanLinear, pack_nvfp4
from wan_nvfp4_fastpack import collect_fastpack_checks


def describe_targets(model: nn.Module) -> dict:
    """Return names/scalars only; valid for real or meta Wan14B modules."""
    blocks = len(model.blocks)
    if blocks != 40:
        raise ValueError(f'Expected the actual 40-block Wan14B graph, got {blocks}')
    names = tuple(f'blocks.{i}.{suffix}' for i in range(blocks) for suffix in SUFFIXES)
    discovered = {name for name, module in model.named_modules()
                  if name.startswith('blocks.') and isinstance(module, nn.Linear)}
    if discovered != set(names):
        raise ValueError('Main-block Linear names differ from the 10-per-block Wan coverage')
    rows = []
    for name in names:
        module = model.get_submodule(name)
        if type(module) is not nn.Linear or module.weight is None:
            raise ValueError(f'{name}: expected an original standard Linear')
        n, k = module.out_features, module.in_features
        if n % 32 or k % 32 or tuple(module.weight.shape) != (n, k):
            raise ValueError(f'{name}: unsupported native weight shape')
        rows.append(dict(name=name, shape=[n,k], elements=n*k, weight_dtype=str(module.weight.dtype),
                         device=str(module.weight.device), has_bias=module.bias is not None))
    return dict(block_count=blocks, target_count=len(rows), targets=rows,
                total_weight_elements=sum(r['elements'] for r in rows))


def _non_target_identity(model, prefixes):
    # Numeric identities only, never retained Tensor/Parameter references.
    return {(kind,name): (id(t),tuple(t.shape),str(t.dtype),str(t.device))
            for kind, iterator in [('parameter',model.named_parameters()), ('buffer',model.named_buffers())]
            for name,t in iterator if not name.startswith(prefixes)}


def _memory(device):
    return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_since_last_reset_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_since_last_reset_bytes=torch.cuda.max_memory_reserved(device))


@torch.no_grad()
def install_plain_native(model: nn.Module) -> dict:
    """Replace all 400 main linears in place and return a JSON-safe manifest.

    Only one weight FP32 temporary exists at a time. Every packet's domain flags
    are checked before that layer is replaced. On failure, earlier replacements
    remain installed: stop/discard that run; this function does not roll back.
    It does not reset CUDA peaks or empty the allocator. Reported peaks inherit
    the caller's last reset; current allocated/reserved before/after are explicit.
    Use collect_fastpack_checks() around later inference for batched flag reads.
    """
    description = describe_targets(model)
    for module in model.modules():
        if module._forward_pre_hooks or module._forward_hooks or module._backward_pre_hooks or module._backward_hooks:
            raise ValueError('Plain installation requires the original unhooked teacher graph')
    del module
    devices = set()
    for row in description['targets']:
        layer = model.get_submodule(row['name'])
        if layer.weight.dtype != torch.bfloat16 or layer.weight.device.type != 'cuda':
            raise ValueError(f"{row['name']}: require already-cast CUDA BF16 teacher weights")
        if layer.bias is not None and (layer.bias.dtype != torch.bfloat16 or layer.bias.device != layer.weight.device):
            raise ValueError(f"{row['name']}: bias dtype/device differs from BF16 teacher")
        devices.add(layer.weight.device)
    del layer
    if len(devices) != 1:
        raise ValueError('This thin installer supports a single resident device')
    device = next(iter(devices))
    prefixes = tuple(row['name']+'.' for row in description['targets'])
    untouched = _non_target_identity(model, prefixes)
    started = time.monotonic()
    report = dict(implementation='wan14b_plain_native_layerwise_pack', status='running', **description,
        recipe=dict(RECIPE), weight_source='already-BF16 teacher -> temporary FP32 -> frozen legacy pack',
        training=False, smoothing=False, low_rank=False, memory_before=_memory(device), layers=[],
        peak_scope='Since caller last reset; installer does not reset peaks', weight_pack_calls=0)
    for spec in description['targets']:
        name = spec['name']; original = model.get_submodule(name)
        memory_before = _memory(device)
        temporary = original.weight.detach().to(dtype=torch.float32).contiguous()
        temporary_bytes = temporary.numel()*temporary.element_size()
        with collect_fastpack_checks() as checks:
            packet = pack_nvfp4(temporary)
        if len(checks) != 1:
            raise RuntimeError(f'{name}: expected exactly one checked weight pack')
        del temporary, checks
        n, k = spec['shape']
        if (packet.global_scale.dtype != torch.float32 or packet.global_scale.shape != (1,) or
            packet.packed.dtype != torch.uint8 or tuple(packet.packed.shape) != (n,k//2) or
            packet.swizzled_scales.dtype != torch.float8_e4m3fn or
            packet.swizzled_scales.numel() != ((n+127)//128*128)*((k//16+3)//4*4)):
            raise RuntimeError(f'{name}: packed ABI differs')
        replacement = PlainPackedWanLinear(packet, original.bias).train(original.training)
        row = dict(name=name, shape=[n,k], original_weight_bytes=original.weight.numel()*original.weight.element_size(),
            temporary_fp32_bytes=temporary_bytes, packed_bytes=packet.packed.numel(),
            scale_bytes=packet.swizzled_scales.numel(), global_bytes=packet.global_scale.numel()*4,
            bias_bytes=0 if replacement.bias is None else replacement.bias.numel()*replacement.bias.element_size(),
            memory_before=memory_before)
        parent_name, leaf = name.rsplit('.',1)
        model.get_submodule(parent_name)._modules[leaf] = replacement
        # External references to an old module must not keep its old Parameters.
        original.register_parameter('weight', None)
        original.register_parameter('bias', None)
        del packet, replacement, original
        row['memory_after'] = _memory(device)
        report['layers'].append(row); report['weight_pack_calls'] += 1
    if _non_target_identity(model, prefixes) != untouched:
        raise RuntimeError('A non-target parameter/buffer identity, shape, dtype or device changed')
    counts = native_call_counts(model)
    if counts['target_count'] != description['target_count'] or counts['total_native_calls'] != 0:
        raise RuntimeError('Unexpected installed module count or pre-inference GEMM')
    report.update(status='complete', non_target_tensor_count=len(untouched), non_targets_same_objects=True,
        memory_after=_memory(device), seconds_including_pack_checks=time.monotonic()-started,
        old_weight_bytes_released=sum(r['original_weight_bytes'] for r in report['layers']),
        packed_resident_bytes=sum(r['packed_bytes']+r['scale_bytes']+r['global_bytes']+r['bias_bytes'] for r in report['layers']),
        max_temporary_fp32_bytes=max(r['temporary_fp32_bytes'] for r in report['layers']), initial_calls=counts)
    return report


def native_call_counts(model: nn.Module, *, reset: bool = False) -> dict:
    """Read per-module native GEMM counters; optionally reset AFTER reading."""
    counts = {}
    for name, module in model.named_modules():
        if isinstance(module, PlainPackedWanLinear):
            counts[name] = int(module.native_calls)
            if reset:
                module.native_calls = 0
    return dict(target_count=len(counts), total_native_calls=sum(counts.values()), per_layer=counts)


def cpu_structure_check(config_path: Path) -> dict:
    """Actual Diffusers meta graph, no checkpoint tensor loads or CUDA queries."""
    if os_cuda_visible():
        raise ValueError('Hide CUDA for the structure check')
    from diffusers import WanTransformer3DModel
    config = json.loads(config_path.read_text())
    with torch.device('meta'):
        model = WanTransformer3DModel.from_config(config)
    report = describe_targets(model)
    assert report['target_count'] == 400 and report['total_weight_elements'] == 14050918400
    shapes = Counter(tuple(row['shape']) for row in report['targets'])
    assert shapes == {(5120,5120):320, (13824,5120):40, (5120,13824):40}
    assert all(p.device.type == 'meta' for p in model.parameters())
    assert not torch.cuda.is_initialized()
    return dict(status='complete', scope='CPU meta structure only; GPU packing/GEMM not tested',
        config=str(config_path.resolve()), target_count=report['target_count'], block_count=report['block_count'],
        total_weight_elements=report['total_weight_elements'],
        shapes={str(k):v for k,v in shapes.items()}, model_weights_loaded=False, cuda_initialized=False,
        full_master_coexistence_bytes_avoided=report['total_weight_elements']*4,
        maximum_single_fp32_temporary_bytes=max(r['elements'] for r in report['targets'])*4)


def os_cuda_visible():
    import os
    return os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(cpu_structure_check(args.config), indent=2))
