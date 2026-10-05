#!/usr/bin/env python3
"""E009 bounded two-layer smoke, then 200-layer legacy H3 residual export.

All tensors/artifacts are immutable after success. No full-model performance
measurement or calibration is performed. GPU weight reconstruction uses the
exact historical BF16 expression; every exported W must roundtrip exactly.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
os.environ.setdefault('DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'torch')
from minimax_h3_svdquant_common import (
    H3_DIT_PATH, load_h3_pipeline, raw_call_to_block0, tree_device,
    nvfp4_qdq, install_runtime_hooks,
)
from h3_native_nvfp4 import (
    FORMAT, RECIPE, TARGET_NAMES, NativeH3Linear, pack_activation_legacy,
    packet_statistics, file_sha256, tensor_sha256, unswizzle_scales,
)


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def atomic_torch(value, path):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(value, tmp)
    tmp.replace(path)


def metrics(got, ref):
    err = ref2 = 0.
    max_abs = 0.
    changed = 0
    gf, rf = got.reshape(-1), ref.reshape(-1)
    for start in range(0, rf.numel(), 1 << 20):
        g, r = gf[start:start+(1 << 20)].float(), rf[start:start+(1 << 20)].float()
        diff = g-r
        err += float(diff.square().sum(dtype=torch.float64))
        ref2 += float(r.square().sum(dtype=torch.float64))
        max_abs = max(max_abs, float(diff.abs().max()))
        changed += int((g != r).sum())
    return {'exact': changed == 0, 'changed_elements': changed, 'elements': ref.numel(),
            'err2': err, 'ref2': ref2, 'nmse': err/max(ref2, 1e-30), 'max_abs': max_abs}


@torch.inference_mode()
def build_layer(old, state, name, chunk_rows):
    if not getattr(old, 'disk_offload', False):
        raise RuntimeError(f'{name}: exporter expects untouched disk-backed source Linear')
    w, bias = old.load_from_disk(torch.bfloat16, 'cuda', assign=False)
    if bias is not None:
        raise RuntimeError(f'{name}: expected bias-free H3 main linear')
    smooth, a, b = [state[name][key].to(device=w.device, dtype=w.dtype)
                    for key in ('smooth', 'final_a', 'final_b')]
    residual = w*smooth-b@a
    packet = pack_activation_legacy(residual, chunk_rows=chunk_rows)
    original_qdq = nvfp4_qdq(residual, element_size=chunk_rows)
    decoded = packet.decode(chunk_rows=chunk_rows)
    if not bool(torch.isfinite(original_qdq).all()) or not bool(torch.isfinite(decoded).all()):
        raise RuntimeError(f'{name}: nonfinite weight reference/roundtrip')
    roundtrip = metrics(decoded, original_qdq)
    if not roundtrip['exact']:
        raise RuntimeError(f'{name}: independent decoded weight is not old QDQ: {roundtrip}')
    module = NativeH3Linear(packet, smooth, a, b, chunk_rows=chunk_rows)
    row = {'name': name, 'shape': list(w.shape), 'roundtrip': roundtrip,
           'quant_stats': packet_statistics(packet), 'source_weight_sha256': tensor_sha256(w),
           'residual_bf16_sha256': tensor_sha256(residual),
           'legacy_qdq_weight_sha256': tensor_sha256(original_qdq),
           'bias': None, 'state_alpha_field_used': False,
           'lr_multiplier': 1.0, 'reason_alpha_not_used': 'Historical loader does not consume state alpha and constructs LowRankBranch(alpha=1)'}
    del w, residual, decoded
    return module, original_qdq, row


def export_payload(module, name):
    return {'format': FORMAT, 'recipe': RECIPE, 'name': name,
            'shape': [module.out_features, module.in_features],
            'tensors': {key: None if tensor is None else tensor.detach().cpu().contiguous()
                        for key, tensor in [('weight_packed', module.weight_packed),
                                            ('weight_scales_swizzled', module.weight_scales_swizzled),
                                            ('weight_global', module.weight_global),
                                            ('smooth', module.smooth), ('lr_a', module.lr_a),
                                            ('lr_b', module.lr_b), ('bias', module.bias)]}}


def release_disk_linear(old):
    # Free any weights kept by the capture forward. Its DiskMap owns CPU data,
    # not GPU Parameters; subsequent reads use load_from_disk(assign=False).
    if getattr(old, 'disk_offload', False):
        old.to('meta')
        old.state = 0


@torch.inference_mode()
def smoke_layer(module, original_qdq, raw_x_cpu, row, args):
    # The artifact retains complete packed shape for later fused/fastpack tests.
    raw = raw_x_cpu.cuda()
    xs = raw/module.smooth
    fused_path = args.output_dir/'fused_inputs'/f"{row['name']}.pt"
    logical_sf = unswizzle_scales(module.weight_scales_swizzled,
                                  module.out_features, module.in_features//16)
    fused = {'format': 'E009_h3_fused_input_v1', 'layer': row['name'], 'prompt_id': 1,
             'step': 0, 'upstream': 'Full-shape BF16 H3 block0, explicit torch SDPA',
             'raw_input_sha256': tensor_sha256(raw), 'x_s_sha256': tensor_sha256(xs),
             'x_s': xs.cpu(), 'a': module.lr_a.cpu(), 'b': module.lr_b.cpu(),
             'smooth': module.smooth.cpu(), 'weight_packed': module.weight_packed.cpu(),
             'weight_scales_logical': logical_sf.cpu(),
             'weight_scales_swizzled': module.weight_scales_swizzled.cpu(),
             'weight_global': module.weight_global.cpu(), 'recipe': RECIPE,
             'source_weight_sha256': row['source_weight_sha256']}
    atomic_torch(fused, fused_path)
    artifact = {'path': str(fused_path), 'sha256': file_sha256(fused_path),
                'input_shape': list(raw.shape), 'smoothed_sha256': fused['x_s_sha256']}
    del fused, logical_sf, xs
    # A fixed prefix is a contract check, not a representative error/quality sample.
    x = raw[:512].contiguous()
    del raw
    xs = x/module.smooth
    activation = module.pack_input(xs)
    qa_ref = nvfp4_qdq(xs, element_size=args.chunk_rows)
    activation_roundtrip = metrics(activation.decode(chunk_rows=args.chunk_rows), qa_ref)
    if not activation_roundtrip['exact']:
        raise RuntimeError(f"{row['name']}: activation roundtrip failed")
    oracle = module.main_from_packet(activation, mode='packed_qdq')
    native = module.main_from_packet(activation, mode='native')
    main_metric = metrics(native, oracle)
    if not bool(torch.isfinite(native).all()) or main_metric['nmse'] > 1e-4:
        raise RuntimeError(f"{row['name']}: native samepacket main gate failed {main_metric}")
    # Independently exercise the existing hooks, including pre-QDQ LR capture.
    original = nn.Linear(module.in_features, module.out_features, bias=False,
                         device='meta', dtype=torch.bfloat16)
    original.weight = nn.Parameter(original_qdq, requires_grad=False)
    runtime = install_runtime_hooks(original, module.smooth, module.lr_a, module.lr_b,
                                    element_size=args.chunk_rows)
    try:
        old_output = original(x)
    finally:
        runtime.remove()
    module.execution_mode = 'legacy_qdq'
    bypass = module(x)
    bypass_metric = metrics(bypass, old_output)
    module.execution_mode = 'packed_qdq'
    packed_qdq = module(x)
    packed_metric = metrics(packed_qdq, old_output)
    if not bypass_metric['exact'] or not packed_metric['exact']:
        raise RuntimeError(f"{row['name']}: full linear old-hook replay failed: {bypass_metric} {packed_metric}")
    module.execution_mode = 'native'
    return {'layer': row['name'], 'input_scope': 'first 512 full-shape BF16-upstream tokens; contract only',
            'activation_roundtrip': activation_roundtrip, 'main_samepacket_native_vs_qdq': main_metric,
            'bypass_vs_original_hooks': bypass_metric, 'packed_qdq_vs_original_hooks': packed_metric,
            'fused_input_artifact': artifact, 'activation_packet': packet_statistics(activation)}


@torch.inference_mode()
def execute(args, report):
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing existing export directory {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    attn = importlib.import_module('diffsynth.core.attention.attention')
    if attn.ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('DIFFSYNTH_ATTENTION_IMPLEMENTATION must be torch')
    report.update(torch=torch.__version__, torch_file=torch.__file__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  attention='Explicit torch implementation; actual BF16 QKV audited in smoke',
                  environment={key: os.environ.get(key) for key in
                               ['CUDA_VISIBLE_DEVICES', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION',
                                'DIFFSYNTH_ROOT', 'MINIMAX_H3_DIT_PATH', 'SVDQUANT_DATA_ROOT']})
    paths = [Path(__file__), Path(__file__).with_name('h3_native_nvfp4.py'),
             Path(__file__).with_name('wan_native_nvfp4.py'),
             ROOT/'scripts/minimax_h3_svdquant_common.py', H3_DIT_PATH, args.state, args.sample,
             Path(importlib.import_module('diffsynth.models.minimax_h3_dit').__file__),
             Path(importlib.import_module('diffsynth.models.minimax_h3_dit_comfy').__file__)]
    print('E009 hashing model/state/source before GPU work', flush=True)
    report['files'] = {str(p): {'sha256': file_sha256(p), 'bytes': p.stat().st_size,
                               'mtime_ns': p.stat().st_mtime_ns} for p in paths}
    state = torch.load(args.state, map_location='cpu', weights_only=False, mmap=True)
    if state.get('format') != 'minimax-h3-svdquant-standard-v1' or set(state['layers']) != set(TARGET_NAMES):
        raise RuntimeError('Expected exact 200-layer historical H3 state')
    if state['config']['rank'] != 32 or state['config']['group_size'] != 16:
        raise RuntimeError('Unexpected old H3 state config')
    report['state_config'] = state['config']
    manifest = {'format': FORMAT, 'recipe': RECIPE, 'status': 'running_smoke',
                'target_count': 200, 'exact_roundtrip_count': 0, 'layers': [],
                'sources': report['files'], 'state_config': state['config'],
                'export_policy': 'GPU old residual expression; full legacy quantizer roundtrip per weight; no recalibration'}
    manifest_path = args.output_dir/'manifest.json'
    atomic_json(manifest, manifest_path)
    atomic_json(report, args.report)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    sample = torch.load(args.sample, map_location='cpu', weights_only=False)
    inputs = {}
    block = pipe.dit.blocks[0]
    handles = []
    smoke_names = ['blocks.0.attn.qkv_proj', 'blocks.0.mlp.fc2']
    for name in smoke_names:
        def capture(_module, inp, name=name):
            inputs[name] = inp[0].detach().cpu()
        handles.append(pipe.dit.get_submodule(name).register_forward_pre_hook(capture))
    actual_sdpa = {'calls': 0}
    sdpa_old = F.scaled_dot_product_attention
    def audited_sdpa(q, k, v, *pos, **kw):
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            raise RuntimeError('Smoke actual attention QKV changed from BF16')
        actual_sdpa['calls'] += 1
        return sdpa_old(q, k, v, *pos, **kw)
    F.scaled_dot_product_attention = audited_sdpa
    try:
        hidden, kwargs = raw_call_to_block0(pipe.dit, sample)
        block(hidden.cuda(), **tree_device(kwargs, 'cuda'))
    finally:
        F.scaled_dot_product_attention = sdpa_old
        for handle in handles:
            handle.remove()
    if actual_sdpa['calls'] < 1 or set(inputs) != set(smoke_names):
        raise RuntimeError('BF16-upstream smoke capture/attention audit failed')
    report['capture'] = {'actual_bf16_sdpa_calls': actual_sdpa['calls'],
                         'input_shapes': {k: list(v.shape) for k, v in inputs.items()},
                         'sample': str(args.sample), 'full_tokens': hidden.shape[0]}
    del hidden, kwargs, sample
    # Free weights retained by block0/refiner capture before sequential export.
    for module in pipe.dit.modules():
        if getattr(module, 'disk_offload', False) and isinstance(module, nn.Linear):
            release_disk_linear(module)
    gc.collect()
    torch.cuda.empty_cache()
    print('E009 begin two-layer native and old-hook replay gates', flush=True)
    report['smoke'] = []
    exported = {}

    def write_layer(module, row):
        filename = f"layers/{row['name']}.pt"
        path = args.output_dir/filename
        atomic_torch(export_payload(module, row['name']), path)
        row.update(file=filename, file_sha256=file_sha256(path), file_bytes=path.stat().st_size)
        exported[row['name']] = row
        manifest['layers'] = [exported[name] for name in TARGET_NAMES if name in exported]
        manifest['exact_roundtrip_count'] = len(exported)
        atomic_json(manifest, manifest_path)

    for name in smoke_names:
        begin = time.perf_counter()
        old = pipe.dit.get_submodule(name)
        module, qweight, row = build_layer(old, state['layers'], name, args.chunk_rows)
        smoke = smoke_layer(module, qweight, inputs.pop(name), row, args)
        report['smoke'].append(smoke)
        row['seconds_including_smoke'] = time.perf_counter()-begin
        write_layer(module, row)
        atomic_json(report, args.report)
        print(json.dumps({'smoke_passed': name, 'main_nmse': smoke['main_samepacket_native_vs_qdq']['nmse'],
                          'artifact': smoke['fused_input_artifact']['path']}), flush=True)
        del module, qweight, old
        gc.collect()
        torch.cuda.empty_cache()
    report['two_layer_gate'] = 'passed'
    if args.smoke_only:
        manifest['status'] = report['status'] = 'smoke_complete_not_full_export'
    else:
        manifest['status'] = 'exporting'
        atomic_json(manifest, manifest_path)
        print('E009 both smoke gates passed; exporting remaining weights', flush=True)
        for name in TARGET_NAMES:
            if name in exported:
                continue
            begin = time.perf_counter()
            old = pipe.dit.get_submodule(name)
            module, qweight, row = build_layer(old, state['layers'], name, args.chunk_rows)
            row['seconds_weight_contract'] = time.perf_counter()-begin
            write_layer(module, row)
            release_disk_linear(old)
            del module, qweight, old
            print(f"E009 exported {len(exported)}/200 {name} exact", flush=True)
        if len(exported) != 200:
            raise RuntimeError('Incomplete target export')
        manifest['status'] = report['status'] = 'complete'
    report['export'] = {'directory': str(args.output_dir), 'manifest': str(manifest_path),
                        'exact_roundtrip_count': len(exported),
                        'total_layer_file_bytes': sum(row['file_bytes'] for row in exported.values())}
    report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    atomic_json(manifest, manifest_path)
    report['export']['manifest_sha256'] = file_sha256(manifest_path)
    atomic_json(report, args.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt')
    parser.add_argument('--sample', type=Path, default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt')
    parser.add_argument('--output-dir', type=Path, default=Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export'))
    parser.add_argument('--report', type=Path, default=ROOT/'results/research/E009_h3_export.json')
    parser.add_argument('--chunk-rows', type=int, default=128)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f'Refusing to overwrite {args.report}')
    report = {'experiment': 'E009_h3_native_export', 'status': 'running',
              'format': FORMAT, 'recipe': RECIPE, 'start_unix': time.time(),
              'scope': 'Two real H3 block0 linear smoke gates then exact 200 weight export; no full-model benchmark or recalibration'}
    start = time.perf_counter()
    try:
        execute(args, report)
    except Exception:
        report['status'] = 'failed_stop_export'
        report['error'] = traceback.format_exc()
        report['elapsed_seconds'] = time.perf_counter()-start
        atomic_json(report, args.report)
        manifest_path = args.output_dir/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest['status'] = 'failed_stop_export'
            manifest['error'] = report['error']
            atomic_json(manifest, manifest_path)
        raise
    report['elapsed_seconds'] = time.perf_counter()-start
    atomic_json(report, args.report)
    print(json.dumps({'status': report['status'], 'export': report['export'],
                      'seconds': report['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
