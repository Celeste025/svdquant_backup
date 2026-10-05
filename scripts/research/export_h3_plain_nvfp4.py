#!/usr/bin/env python3
"""E014 streaming original-W NVFP4 export, with independent historical QDQ gates.

No full DiT is loaded. Export reads one original BF16 safetensors weight at a
time, checks all 200 numerical roundtrips, and releases its GPU tensors. The
four block0 smoke inputs are fixed-seed synthetic M512, not captured inputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'third_party/deepcompressor'))
os.environ.setdefault('DIFFSYNTH_ROOT', '/home/wjq/workspace/DiffSynth-Studio')
os.environ.setdefault('DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'torch')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from safetensors import safe_open
from minimax_h3_svdquant_common import nvfp4_qdq
from h3_native_nvfp4 import TARGET_NAMES, packet_statistics, file_sha256
from h3_nvfp4_zero_sf_compat import pack_activation_legacy, pack_activation_fast
from h3_plain_native_nvfp4 import FORMAT, RECIPE, TENSOR_KEYS, PlainNativeH3Linear

PLAN = ROOT/'research_state/06_experiments/E014_h3_plain_baseline_plan.md'
MANIFEST = PLAN.with_name('E014_h3_plain_baseline_manifest.json')
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E014')
REPORT_DIR = ROOT/'results/research/E014'
CHECKPOINT_SUFFIX = 'Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(report, indent=2)+'\n')
    temp.replace(path)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def metrics(got, ref):
    require(got.shape == ref.shape and got.dtype == ref.dtype, 'Metric shape/dtype mismatch')
    err2 = ref2 = max_abs = 0.
    changed = zero_sign = unexpected_bytes = 0
    gf, rf = got.reshape(-1), ref.reshape(-1)
    for start in range(0, gf.numel(), 1 << 20):
        g, r = gf[start:start+(1 << 20)], rf[start:start+(1 << 20)]
        require(bool(torch.isfinite(g).all()) and bool(torch.isfinite(r).all()), 'Nonfinite comparison')
        diff = g.float()-r.float()
        err2 += float(diff.double().square().sum())
        ref2 += float(r.double().square().sum())
        changed += int((g != r).sum())
        max_abs = max(max_abs, float(diff.abs().amax()))
        if g.dtype == torch.bfloat16:
            byte_diff = g.contiguous().view(torch.int16) != r.contiguous().view(torch.int16)
            zeros = byte_diff & (g == 0) & (r == 0)
            zero_sign += int(zeros.sum())
            unexpected_bytes += int((byte_diff & ~zeros).sum())
    return {'exact': changed == 0, 'changed_elements': changed, 'elements': got.numel(),
            'err2': err2, 'ref2': ref2, 'nmse': err2/max(ref2, 1e-300), 'max_abs': max_abs,
            'zero_sign_differences': zero_sign, 'nonzero_byte_differences': unexpected_bytes}


def provenance(args):
    protocol = json.loads(MANIFEST.read_text())
    require(protocol['experiment'] == 'E014', 'Wrong protocol')
    ref = protocol['asset_reference']
    require(file_sha256(ref['file']) == ref['sha256'], 'E010 full-hash asset inventory changed')
    inventory = json.loads(Path(ref['file']).read_text())
    require(inventory['status'] == 'complete', 'Incomplete asset audit')
    for path, row in inventory['assets'].items():
        st = Path(path).stat()
        require(st.st_size == row['bytes'] and st.st_mtime_ns == row['mtime_ns'], f'Asset stat changed: {path}')
    candidates = [p for p in inventory['assets'] if p.endswith(CHECKPOINT_SUFFIX)]
    require(len(candidates) == 1, 'Original BF16 checkpoint is ambiguous')
    checkpoint = Path(candidates[0])
    require(args.output_dir.resolve() == Path(protocol['plain_export_dir']).resolve(), 'Export dir differs from protocol')
    paths = [Path(__file__), Path(__file__).with_name('h3_plain_native_nvfp4.py'), PLAN, MANIFEST,
             ROOT/'scripts/minimax_h3_svdquant_common.py']
    paths += [Path(__file__).with_name(name) for name in (
        'h3_native_nvfp4.py', 'wan_native_nvfp4.py', 'h3_nvfp4_zero_sf_compat.py',
        'h3_nvfp4_fastpack.py', 'probe_h3_native_contract.py',
        'probe_h3_modality_pulse.py', 'smoke_nvfp4_systems_20261002.py')]
    sources = {str(p.resolve()): file_record(p) for p in paths}
    return {'format': FORMAT, 'recipe': RECIPE, 'sources': sources,
            'source_checkpoint': {'file': str(checkpoint), **inventory['assets'][str(checkpoint)]},
            'asset_inventory_reference': ref,
            'asset_validation': 'Established E010 full SHA reused; all asset size/mtime checked; no repeated model hash',
            'torch_version': torch.__version__}


def checkpoint_layout(path):
    rows = []
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        for name in TARGET_NAMES:
            key = name+'.weight'
            require(key in keys, f'Missing original weight {key}')
            view = handle.get_slice(key)
            shape = view.get_shape()
            require(view.get_dtype() == 'BF16' and len(shape) == 2 and all(v % 32 == 0 for v in shape),
                    f'Unsupported original weight {key}')
            bias_key = name+'.bias' if name+'.bias' in keys else None
            if bias_key:
                require(handle.get_slice(bias_key).get_shape() == [shape[0]]
                        and handle.get_slice(bias_key).get_dtype() == 'BF16', 'Unexpected bias contract')
            rows.append({'name': name, 'key': key, 'shape': shape, 'dtype': view.get_dtype(), 'bias_key': bias_key})
    require(len(rows) == 200, 'Expected 200 original weights')
    return rows


def payload_for(module, row):
    return {'format': FORMAT, 'recipe': RECIPE, 'name': row['name'], 'shape': row['shape'],
            'source_weight_sha256': row['source_weight_sha256'],
            'tensors': {key: None if getattr(module, key) is None else getattr(module, key).detach().cpu().contiguous()
                        for key in sorted(TENSOR_KEYS)}}


@torch.inference_mode()
def cpu_check(report):
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    torch.set_num_threads(6)
    layout = checkpoint_layout(report['source_checkpoint']['file'])
    with safe_open(report['source_checkpoint']['file'], framework='pt', device='cpu') as handle:
        actual_prefix = handle.get_slice(layout[0]['key'])[:32, :64].clone()
    generator = torch.Generator(device='cpu').manual_seed(2026100214)
    extreme = torch.full((32, 64), 1e-20, dtype=torch.bfloat16)
    extreme[0, 0] = 1e20
    fixtures = {'actual_bf16_weight_prefix': actual_prefix,
                'random': torch.randn((32, 64), generator=generator).bfloat16(),
                'all_zero': torch.zeros((32, 64), dtype=torch.bfloat16),
                'nonzero_groups_zero_sf': extreme}
    tests = []
    for name, x in fixtures.items():
        packet = pack_activation_legacy(x, chunk_rows=16)
        comparison = metrics(packet.decode(chunk_rows=16), nvfp4_qdq(x, element_size=16))
        require(comparison['exact'] and comparison['nonzero_byte_differences'] == 0, f'CPU fixture failed: {name}')
        tests.append({'name': name, 'input_shape': list(x.shape), 'input_sha256': tensor_sha(x),
                      'roundtrip': comparison, 'quant_stats': packet.zero_sf_statistics})
    weight = fixtures['random']
    bias = torch.linspace(-1, 1, 32, dtype=torch.bfloat16)
    module = PlainNativeH3Linear(pack_activation_legacy(weight, chunk_rows=16), bias,
                                 activation_packer=pack_activation_legacy, chunk_rows=16)
    row = {'name': 'cpu_fixture', 'shape': list(weight.shape), 'source_weight_sha256': tensor_sha(weight)}
    restored = PlainNativeH3Linear.from_export(payload_for(module, row), device='cpu',
                       activation_packer=pack_activation_legacy, chunk_rows=16)
    x = torch.randn((2, 3, 64), generator=generator).bfloat16()
    restored.execution_mode = 'packed_qdq'
    expected = torch.nn.functional.linear(nvfp4_qdq(x, element_size=16),
                                          nvfp4_qdq(weight, element_size=16), bias)
    require(torch.equal(restored(x), expected), 'CPU payload roundtrip/shape/bias failed')
    require(set(dict(restored.named_buffers())) == TENSOR_KEYS and not list(restored.parameters()),
            'Unexpected resident state or hidden original weight')
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', phase='check', checkpoint_layout=layout, target_count=len(layout),
                  cpu_fixtures=tests, payload_bias_leading_shape_exact=True, cuda_initialized=False,
                  scope='Header plus small CPU arithmetic; no native GEMM or full-model run')


def deadline(args):
    if args.deadline_unix is not None and time.time() >= args.deadline_unix:
        raise TimeoutError('Global E014 wall-clock deadline reached')
    if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > 60 * 1024**3:
        raise RuntimeError('E014 60 GiB allocated-memory budget exceeded')


@torch.inference_mode()
def smoke(module, row, chunk_rows, profile=False):
    seed = 2026100214 + TARGET_NAMES.index(row['name'])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn((512, module.in_features), generator=generator).bfloat16().cuda()
    packet = module.pack_input(raw)
    slow = pack_activation_legacy(raw, chunk_rows=chunk_rows)
    for a, b in ((packet.packed, slow.packed), (packet.global_scale, slow.global_scale),
                 (packet.swizzled_scales.view(torch.uint8), slow.swizzled_scales.view(torch.uint8))):
        require(torch.equal(a, b), 'Synthetic fast/independent-E005 packet mismatch')
    activation_metric = metrics(packet.decode(chunk_rows=chunk_rows), nvfp4_qdq(raw, element_size=chunk_rows))
    require(activation_metric['exact'] and activation_metric['nonzero_byte_differences'] == 0,
            'Synthetic activation roundtrip failed')
    oracle = module.main_from_packet(packet, mode='packed_qdq', include_bias=True)
    kernels = []
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            got = module.main_from_packet(packet, include_bias=True)
            torch.cuda.synchronize()
        kernels = sorted({event.name for event in prof.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA
                          and 'sm120' in event.name.lower() and 'e2m1' in event.name.lower()})
        require(bool(kernels), 'Missing actual SM120 E2M1 native-kernel evidence')
    else:
        got = module.main_from_packet(packet, include_bias=True)
    main = metrics(got, oracle)
    require(main['nmse'] <= 1e-4, f'Same-packet native arithmetic gate failed: {main}')
    return {'seed': seed, 'input_shape': list(raw.shape), 'input_sha256': tensor_sha(raw),
            'input_kind': 'fixed-seed synthetic, not captured model activation',
            'fast_vs_independent_e005_packet_bytes_exact': True, 'activation_roundtrip': activation_metric,
            'samepacket_main': main, 'native_kernel_names': kernels,
            'fastpack_checks': packet.zero_sf_statistics}


@torch.inference_mode()
def export(args, report):
    checked = json.loads(args.cpu_check.read_text())
    require(checked['status'] == 'complete' and checked['phase'] == 'check'
            and checked['sources'] == report['sources'] and checked['recipe'] == RECIPE
            and checked['source_checkpoint'] == report['source_checkpoint'], 'CPU/source provenance gate failed')
    require(not (args.output_dir/'manifest.json').exists(), 'Refusing existing export manifest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    deadline(args)
    require(torch.cuda.get_device_capability() == (12, 0), 'Expected SM120 native baseline')
    report.update(status='running', phase='export', target_count=0, exact_roundtrip_count=0, layers=[],
                  cpu_check=file_record(args.cpu_check), deadline_unix=args.deadline_unix,
                  gpu_name=torch.cuda.get_device_name(), export_dir=str(args.output_dir),
                  scope='Original-W streaming export; four synthetic same-code smoke tests, no model benchmark')
    save(report, args.report)
    save(report, args.output_dir/'manifest.json')
    with safe_open(report['source_checkpoint']['file'], framework='pt', device='cpu') as handle:
        for metadata in checked['checkpoint_layout']:
            deadline(args)
            name = metadata['name']
            started = time.monotonic()
            w_cpu = handle.get_tensor(metadata['key'])
            source_sha = tensor_sha(w_cpu)
            w = w_cpu.cuda()
            bias = handle.get_tensor(metadata['bias_key']).cuda() if metadata['bias_key'] else None
            packet = pack_activation_legacy(w, chunk_rows=args.chunk_rows)
            qdq = nvfp4_qdq(w, element_size=args.chunk_rows)
            decoded = packet.decode(chunk_rows=args.chunk_rows)
            roundtrip = metrics(decoded, qdq)
            require(roundtrip['exact'] and roundtrip['nonzero_byte_differences'] == 0,
                    f'{name}: packet decode differs from independent QDQ(original W)')
            row = {**metadata, 'source_weight_sha256': source_sha, 'roundtrip': roundtrip,
                   'legacy_qdq_weight_sha256': tensor_sha(qdq), 'quant_stats': packet_statistics(packet),
                   'encoder_stats': packet.zero_sf_statistics, 'bias_sha256': None if bias is None else tensor_sha(bias)}
            del w, w_cpu, qdq, decoded
            module = PlainNativeH3Linear(packet, bias, chunk_rows=args.chunk_rows)
            if name.startswith('blocks.0.'):
                row['smoke'] = smoke(module, row, args.chunk_rows, profile=name == TARGET_NAMES[0])
            payload = payload_for(module, row)
            path = args.output_dir/(name+'.pt')
            require(not path.exists(), f'Refusing existing payload {path}')
            temp = path.with_suffix('.tmp.pt')
            torch.save(payload, temp)
            temp.replace(path)
            row.update(file=path.name, file_sha256=file_sha256(path), file_bytes=path.stat().st_size,
                       seconds_including_validation=time.monotonic()-started)
            report['layers'].append(row)
            report['target_count'] += 1
            report['exact_roundtrip_count'] += int(roundtrip['exact'])
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
            save(report, args.report)
            save(report, args.output_dir/'manifest.json')
            print(f'E014 export {report["target_count"]}/200 {name}: numeric roundtrip exact', flush=True)
            del module, packet, payload, bias
            gc.collect()
            torch.cuda.empty_cache()
            deadline(args)
    require(report['target_count'] == report['exact_roundtrip_count'] == 200, 'Incomplete 200-weight export')
    report['status'] = 'complete'
    save(report, args.output_dir/'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['check', 'export'], required=True)
    parser.add_argument('--output-dir', type=Path, default=DATA/'plain_export')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--cpu-check', type=Path, default=REPORT_DIR/'export_cpu_check.json')
    parser.add_argument('--chunk-rows', type=int, default=128)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if args.report is None:
        args.report = REPORT_DIR/('export_cpu_check.json' if args.phase == 'check' else 'export.json')
    require(args.chunk_rows > 0, 'Positive chunk rows required')
    require(not args.report.exists(), f'Refusing to overwrite report {args.report}')
    report = {'experiment': 'E014', 'status': 'starting', 'phase': args.phase}
    started = time.monotonic()
    try:
        report.update(provenance(args))
        if args.phase == 'check':
            cpu_check(report)
        else:
            export(args, report)
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        save(report, args.report)
        print(json.dumps({'status': report['status'], 'report': str(args.report),
                          'seconds_total': report['seconds_total']}), flush=True)


if __name__ == '__main__':
    main()
