"""Load a complete Wan14B rank32 SVDQuant checkpoint, then pack one layer at a time.

No calibration or new quantization is performed. The caller supplies a fresh
CUDA BF16 teacher graph and the actual quant_config used for the checkpoint.
All five checkpoint files are required; a resource-pilot partial cache is not a
checkpoint. No complete Wan14B SVD checkpoint or GPU forward has yet been tested
with this adapter. Failure can leave a partially changed graph: discard it.

Minimal entry:
    q, config_info = build_quant_config(original_model_dir)
    info = install_svd_native(transformer, complete_checkpoint_dir, quant_config=q)
    with collect_fastpack_checks() as flags:
        output = transformer(...)

Set DEEPCOMPRESSOR_WAN_GATED=0. Do not cast the model dtype after installation;
keep FP32 globals, original smoothing, and BF16 LR/bias/output semantics.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
import wan_native_nvfp4 as native
import wan_nvfp4_fastpack as fast
from wan14b_plain_native import describe_targets

ROOT = Path(__file__).resolve().parents[2]
DC = ROOT / 'third_party/deepcompressor'
EXAMPLES = DC / 'examples/diffusion'
collect_fastpack_checks = fast.collect_fastpack_checks


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _imports():
    if str(DC) not in sys.path:
        sys.path.insert(0, str(DC))


def build_quant_config(model_dir: str | Path):
    """Current shared r32/g10 recipe only; does not infer provenance from a directory.

    The inherited YAML contains 1.3B evaluation defaults. They are overridden for
    metadata; this helper never runs a scheduler or loads calibration samples.
    Pass the actual future PTQ quant_config instead if its recipe changes.
    """
    _imports()
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    files = ['configs/model/wan2.1-1.3b.yaml', 'configs/svdquant/real_nvfp4.yaml',
             'configs/svdquant/wan_s16.yaml']
    argv = [*files, '--pipeline-name=wan2.1-14b', '--pipeline-path='+str(model_dir),
            '--skip-eval', '--skip-gen', '--eval-num-gpus=1', '--eval-num-frames=81',
            '--eval-num-steps=50', '--eval-guidance-scale=5.0',
            '--eval-protocol=unipc50-g5.0-f81-shift3']
    cwd, old_argv = Path.cwd(), sys.argv
    try:
        os.chdir(EXAMPLES); sys.argv = ['wan14b-svd-native', *argv]
        config, _, unused_configs, unused_args, unknown = DiffusionPtqRunConfig.get_parser().parse_known_args()
        _require(not unknown and not unused_args, f'Unparsed config: {unknown}, {unused_args}')
    finally:
        os.chdir(cwd); sys.argv = old_argv
    _require(not config.pipeline.shift_activations, 'Activation shifting is outside this adapter')
    _validate_recipe(config.quant)
    return config.quant, dict(argv=argv, config_files=[str(EXAMPLES/p) for p in files],
                             unused_configs=unused_configs, resolved_config=config.dump())


def _validate_recipe(q):
    _require(q.enabled_wgts and q.enabled_ipts and not q.enabled_opts and
             not q.enabled_rotation and not q.needs_acts_quantizer_cache and not q.ipts.static,
             'Require dynamic W4A4 inputs, no output quantization/rotation/activation calibration')
    _require(q.wgts.enabled_low_rank and q.wgts.low_rank.rank == 32 and
             not q.wgts.low_rank.exclusive, 'Require the shared rank32 LR recipe')
    _require(q.enabled_smooth_proj and not q.enabled_smooth_attn and q.smooth.proj.num_grids == 10,
             'Require the existing projection smoothing recipe')
    _require(q.develop_dtype == torch.float32, 'Require FP32 quantization development')
    for c in (q.wgts, q.ipts):
        _require(str(c.dtype) == 'sfp4_e2m1_all' and c.zero_domain is None and
                 len(c.scale_dtypes) == 2 and c.scale_dtypes[0] == torch.float32 and
                 str(c.scale_dtypes[1]) == 'sfp8_e4m3_nan' and len(c.group_shapes) == 2 and
                 tuple(c.group_shapes[0][:2]) == (-1, -1) and
                 tuple(c.group_shapes[1][:2]) == (1, 16), 'Require the saved FP32/E4M3/group16 NVFP4 recipe')


def _structure(model):
    _imports()
    from deepcompressor.app.diffusion.nn.struct import DiffusionAttentionStruct, DiffusionModelStruct
    import deepcompressor.app.diffusion.dataset.calib as calib
    description = describe_targets(model)
    kind = type(model.blocks[0].attn1)
    if kind not in DiffusionAttentionStruct._factories:
        DiffusionAttentionStruct.register_factory(kind, DiffusionAttentionStruct._default_construct)
    if kind not in calib._ATTN_TYPES:
        calib._ATTN_TYPES = (*calib._ATTN_TYPES, kind)
    structure = DiffusionModelStruct.construct(model)
    _require(structure.module is model and structure.num_blocks == 40, '14B structure mismatch')
    return structure, description


def _groups(model):
    """Existing shared QKV/KV and singleton LR/smoothing keys, shapes only."""
    groups = {}
    suffix_groups = [('attn1.to_q', 'attn1.to_k', 'attn1.to_v'), ('attn1.to_out.0',),
                     ('attn2.to_q',), ('attn2.to_k', 'attn2.to_v'), ('attn2.to_out.0',),
                     ('ffn.net.0.proj',), ('ffn.net.2',)]
    for i in range(40):
        for suffixes in suffix_groups:
            names = [f'blocks.{i}.{suffix}' for suffix in suffixes]
            shapes = [tuple(model.get_submodule(n).weight.shape) for n in names]
            _require(len({s[1] for s in shapes}) == 1, 'Shared LR input dimensions differ')
            groups[names[0]] = dict(names=names, in_features=shapes[0][1], out_features=sum(s[0] for s in shapes))
    return groups


def _read_checkpoint(model, checkpoint_dir, names, q):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    files = {kind: checkpoint_dir/f'{kind}.pt' for kind in ('model', 'scale', 'wgts', 'smooth', 'branch')}
    _require(all(p.is_file() for p in files.values()), 'Require complete model/scale/wgts/smooth/branch.pt')
    # Trusted local PTQ artifacts. No device-tag restoration and no second GPU model.
    state = {kind: torch.load(p, map_location='cpu', weights_only=False, mmap=True) for kind,p in files.items()}
    _require(set(state['wgts']) == set(names), 'Expected complete 400-target weight settings, not a pilot cache')
    current = model.state_dict()
    _require(set(current) == set(state['model']), 'Complete model state coverage differs')
    _require(all(tuple(t.shape) == tuple(state['model'][n].shape) for n,t in current.items()), 'Saved model shapes differ')
    del current
    groups = _groups(model)
    _require(set(state['branch']) == set(groups), 'Expected all 280 shared/singleton rank32 branches')
    _require(set(groups) <= set(state['smooth']), 'Missing saved smoothing; calibration is forbidden')
    _require(state['smooth'].get('proj.fuse_when_possible', True) == q.smooth.proj.fuse_when_possible,
             'Smoothing fusion recipe differs')
    for name,g in groups.items():
        branch, smooth = state['branch'][name], state['smooth'][name]
        _require(set(branch) == {'a.weight', 'b.weight'} and
                 tuple(branch['a.weight'].shape) == (32,g['in_features']) and
                 tuple(branch['b.weight'].shape) == (g['out_features'],32), f'LR shape differs: {name}')
        _require(isinstance(smooth, torch.Tensor) and smooth.numel() == g['in_features'] and
                 bool(torch.isfinite(smooth).all()) and bool((smooth > 0).all()), f'Invalid smooth scale: {name}')
    for name in names:
        w = state['model'][name+'.weight']
        _require(w.dtype == torch.bfloat16, f'Saved residual is not BF16: {name}')
        _require(all(name+'.weight.'+key in state['scale'] for key in ('scale.0','scale.1','zero')),
                 f'Missing saved NVFP4 scales: {name}')
        _require(state['scale'][name+'.weight.scale.0'].dtype == torch.float32 and
                 state['scale'][name+'.weight.scale.0'].numel() == 1 and
                 state['scale'][name+'.weight.scale.1'].numel() == w.numel()//16, f'Scale shape differs: {name}')
    receipt = {kind: dict(file=str(p), resolved=str(p.resolve()), bytes=p.stat().st_size,
                         mtime_ns=p.stat().st_mtime_ns) for kind,p in files.items()}
    return state, receipt


class NativeSVDWanLinear(native.NativeWanLinear):
    """Only adds a counter; GEMM, bias, QDQ oracle and hooks remain the old code."""
    native_calls = 0

    def main_from_packet(self, activation, *, mode='native', include_bias=False):
        if mode == 'native':
            self.native_calls += 1
        return super().main_from_packet(activation, mode=mode, include_bias=include_bias)


def native_call_counts(model):
    per_layer = {n:int(m.native_calls) for n,m in model.named_modules() if isinstance(m, NativeSVDWanLinear)}
    return dict(target_count=len(per_layer), total_native_calls=sum(per_layer.values()), per_layer=per_layer)


@torch.inference_mode()
def install_svd_native(transformer, checkpoint_dir, *, quant_config, chunk_rows=128):
    """Fresh CUDA BF16 teacher -> legacy loaded graph -> layerwise packed graph.

    Config and all five files must come from a complete matching 14B PTQ run.
    Preserves saved non-target state and all parent smooth/LR hooks. Weight
    code recovery/roundtrip never recomputes scales. This is not transactional.
    Memory numbers inherit the caller's last peak reset; no reset/empty_cache is
    added by the converter (the reused legacy loader calls empty_cache itself).
    """
    _validate_recipe(quant_config)
    _imports()
    from deepcompressor.app.diffusion.quant import smooth_diffusion, load_diffusion_weights_state_dict, quantize_diffusion_activations
    from deepcompressor.app.diffusion.quant.quantizer import DiffusionActivationQuantizer
    from deepcompressor.app.diffusion.quant.utils import _wan_gated_enabled
    from deepcompressor.utils.hooks import AccumBranchHook, ProcessHook
    _require(not _wan_gated_enabled(), 'Set DEEPCOMPRESSOR_WAN_GATED=0 for this ungated recipe')
    structure, description = _structure(transformer)
    names = [r['name'] for r in description['targets']]
    for module in transformer.modules():
        _require(not module._forward_pre_hooks and not module._forward_hooks and
                 not module._backward_pre_hooks and not module._backward_hooks, 'Require a fresh unhooked teacher')
    del module
    devices = {p.device for p in transformer.parameters()}
    _require(len(devices) == 1 and next(iter(devices)).type == 'cuda', 'Require a single resident CUDA model')
    _require(all(transformer.get_submodule(n).weight.dtype == torch.bfloat16 for n in names), 'Teacher must already be BF16')
    state, files = _read_checkpoint(transformer, checkpoint_dir, names, quant_config)
    device = next(iter(devices)); started = time.monotonic()
    def memory():
        return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
                    peak_allocated_since_last_reset_bytes=torch.cuda.max_memory_allocated(device))
    report = dict(implementation='wan14b_svd_native_layerwise', target_count=400, block_count=40,
                  checkpoint_files=files, checkpoint_hash_policy='stat only; bind provenance in the calling worker',
                  memory_before=memory(), weights=[], hooks={}, activation_packer='frozen_wan_fastpack',
                  low_rank=32, saved_weight_scales_recomputed=False)
    # These are the same three load-path operations in DeepCompressor ptq().
    # Nonempty complete smooth/branch states and needs_acts_quantizer_cache=False
    # ensure neither smoothing/LR fitting nor activation data collection occurs.
    smooth_diffusion(structure, quant_config, smooth_cache=state['smooth'])
    load_diffusion_weights_state_dict(structure, quant_config, state_dict=state['model'], branch_state_dict=state['branch'])
    quantize_diffusion_activations(structure, quant_config)
    del structure
    report['legacy_loaded_memory'] = memory()
    targets = set(names)
    report['parent_hooks_preserved'] = {n: dict(
        pre=[dict(handle_id=k, **native._description(h)) for k,h in m._forward_pre_hooks.items()],
        post=[dict(handle_id=k, **native._description(h)) for k,h in m._forward_hooks.items()])
        for n,m in transformer.named_modules() if n not in targets and (m._forward_pre_hooks or m._forward_hooks)}
    for name in names:
        old = transformer.get_submodule(name)
        _require(type(old) is nn.Linear, f'Unexpected loaded module: {name}')
        hooks = list(old._forward_pre_hooks.values())
        quant = [h for h in hooks if isinstance(h, ProcessHook) and isinstance(h.processor, DiffusionActivationQuantizer)]
        branches = [h for h in hooks if isinstance(h, AccumBranchHook)]
        _require(len(quant) == len(branches) == 1, f'Expected one QDQ and one LR prehook: {name}')
        qhook, branch = quant[0], branches[0]
        _require(qhook.pre and not qhook.post and qhook.activated and qhook.processor.is_enabled() and
                 hooks.index(branch) < hooks.index(qhook) and branch.tensor is None, f'Hook order/status differs: {name}')
        _require(branch.branch.rank == 32 and branch.branch.a.weight.dtype == branch.branch.b.weight.dtype == torch.bfloat16,
                 f'LR dtype/rank differs: {name}')
        fast.validate_quantizer_contract(qhook.processor)
        _require(torch.equal(old.weight.detach().cpu(), state['model'][name+'.weight']), f'Loaded residual differs: {name}')
        if old.bias is not None:
            _require(old.bias.dtype == torch.bfloat16 and torch.equal(old.bias.detach().cpu(), state['model'][name+'.bias']),
                     f'Bias differs: {name}')
        _require(not bool(torch.as_tensor(state['scale'][name+'.weight.zero']).ne(0).any()), f'Nonzero zero point: {name}')
        packet, row = native._recover_saved_weight(old.weight, state['scale'][name+'.weight.scale.0'],
            state['scale'][name+'.weight.scale.1'], chunk_rows=chunk_rows)
        new = NativeSVDWanLinear(packet, old.bias, qhook.processor, activation_packer=fast.pack_activation_fast,
                                chunk_rows=chunk_rows).train(old.training)
        for attr in ('in_smooth_cache_key', 'out_smooth_cache_key'):
            if hasattr(old, attr):
                setattr(new, attr, getattr(old, attr))
        report['hooks'][name] = native._transfer_hooks(old, new, qhook)
        parent, leaf = name.rsplit('.', 1)
        transformer.get_submodule(parent)._modules[leaf] = new
        old.register_parameter('weight', None); old.register_parameter('bias', None)
        row.update(name=name, weight_global_dtype=str(new.weight_global.dtype),
                   bias_dtype=None if new.bias is None else str(new.bias.dtype))
        report['weights'].append(row)
        del packet, old, new, qhook, quant, branches, branch, hooks
    report.update(status='complete', exact_roundtrip_count=sum(r['exact'] for r in report['weights']),
        old_bf16_residual_bytes_removed=sum(r['bf16_bytes'] for r in report['weights']),
        packed_weight_bytes=sum(r['packed_bytes'] for r in report['weights']),
        resident_weight_scale_bytes=sum(r['swizzled_scale_bytes'] for r in report['weights']),
        resident_logical_weight_scale_bytes=0, total_weight_elements=sum(r['elements'] for r in report['weights']),
        initial_calls=native_call_counts(transformer), memory_after=memory(), seconds=time.monotonic()-started)
    _require(report['exact_roundtrip_count'] == report['initial_calls']['target_count'] == 400 and
             report['initial_calls']['total_native_calls'] == 0, 'Incomplete native installation')
    del state
    gc.collect()
    return report


def cpu_check(model_dir):
    """Actual meta model/config/struct only; no checkpoint reads or GPU arithmetic."""
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'Hide CUDA for CPU check')
    from diffusers import WanTransformer3DModel
    q, config_info = build_quant_config(model_dir)
    with torch.device('meta'):
        model = WanTransformer3DModel.from_config(json.loads((Path(model_dir)/'transformer/config.json').read_text()))
    structure, description = _structure(model)
    groups = _groups(model)
    actual = {name for key,name,module,_,_ in structure.named_key_modules()
              if isinstance(module, nn.Linear) and name.startswith('blocks.') and
              q.wgts.is_enabled_for(key) and q.ipts.is_enabled_for(key)}
    _require(actual == {r['name'] for r in description['targets']} and len(groups) == 280, '400-target/280-group coverage differs')
    _require(all(p.device.type == 'meta' for p in model.parameters()) and not torch.cuda.is_initialized(), 'CPU-only check violated')
    return dict(status='complete', block_count=40, target_count=len(actual), lr_group_count=len(groups),
                weight_elements=description['total_weight_elements'], config=config_info,
                model_weights_loaded=False, checkpoint_loaded=False, cuda_initialized=False,
                scope='Meta/config structure only; complete14B checkpoint loading, roundtrip, hooks and GPU arithmetic remain untested')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--check-only', action='store_true', required=True)
    args = p.parse_args()
    print(json.dumps(cpu_check(args.model_dir), indent=2))
