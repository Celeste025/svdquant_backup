#!/usr/bin/env python3
"""E009 matched resident H3 BF16 / old QDQ / native-fast deployment profile.

One arm per process, sequentially on the same idle GPU. Requires completed
correctness evidence. No full-model diagnostic hooks in timed forwards.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import gc
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
os.environ.setdefault('DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'torch')
from bench_h3_native_nvfp4 import make_h3_resident, DATA
from minimax_h3_svdquant_common import H3_DIT_PATH, load_h3_pipeline, tree_device, install_runtime_hooks
from probe_h3_modality_pulse import sha256, tensor_sha, save
from h3_native_nvfp4 import NativeH3Linear, TARGET_NAMES, install_native_h3

PROTOCOL = ROOT/'research_state/06_experiments/E009_h3_native_profile_protocol.md'
SAMPLE = ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt'
STAGES = {'bf16': 'resident_bf16', 'qdq': 'legacy_formula_reference', 'native': 'native'}
SF0_PROOF = ROOT/'results/research/E009_h3_sf0_domain_v2.json'
LR_DEPTH = ContextVar('e009_profile_lr_depth', default=0)


def distribution(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'values': values}


def check_endpoint(result, expected):
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError('Expected video/audio endpoint pair')
    cpu, digests = {}, {}
    for name, tensor in zip(('video', 'audio'), result, strict=True):
        cpu[name] = tensor.detach().cpu()
        digests[name] = tensor_sha(cpu[name])
        if not bool(torch.isfinite(cpu[name]).all()) or digests[name] != expected[name]:
            raise RuntimeError(f'{name} endpoint differs from completed E009: {digests[name]} != {expected[name]}')
    return cpu, digests


def resident_storage(model):
    tensors = [('registered_parameter', t) for t in model.parameters()]
    tensors += [('registered_buffer', t) for t in model.buffers()]
    for module in model.modules():
        for hook in module._forward_pre_hooks.values():
            branch = getattr(hook, 'branch', None)
            if isinstance(branch, nn.Module):
                tensors += [('hook_lr', t) for t in branch.parameters()]
            smooth = getattr(hook, 'smooth', None)
            if torch.is_tensor(smooth):
                tensors.append(('hook_smooth', smooth))
    seen, groups = set(), defaultdict(lambda: {'storages': 0, 'bytes': 0})
    for label, tensor in tensors:
        if not tensor.is_cuda:
            raise RuntimeError(f'Nonresident model storage: {label}, {tensor.device}')
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            groups[label]['storages'] += 1
            groups[label]['bytes'] += storage.nbytes()
            seen.add(key)
    return {'categories': dict(groups), 'unique_cuda_storage_bytes': sum(v['bytes'] for v in groups.values()),
            'definition': 'Unique registered and hook-owned model CUDA storage, excluding allocator/library state'}


@torch.inference_mode()
def install_resident_old_qdq(model):
    """Decode W ONCE at startup. Do not benchmark temporary per-call W decode."""
    runtimes = []
    for name in TARGET_NAMES:
        packed = model.get_submodule(name)
        if not isinstance(packed, NativeH3Linear):
            raise RuntimeError(f'{name}: expected native export before QDQ materialization')
        weight = packed.weight_packet().decode(chunk_rows=1024)
        linear = nn.Linear(packed.in_features, packed.out_features, bias=packed.bias is not None,
                           device='meta', dtype=torch.bfloat16)
        linear.weight = nn.Parameter(weight, requires_grad=False)
        if packed.bias is not None:
            linear.bias = nn.Parameter(packed.bias, requires_grad=False)
        runtime = install_runtime_hooks(linear, packed.smooth, packed.lr_a, packed.lr_b, element_size=1024)
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, linear)
        runtimes.append(runtime)
        del packed, linear, weight
    return runtimes


@contextmanager
def labels():
    """Only active in the independent profiler; arithmetic/order is unchanged.

    LR ranges include down/up and the historical scalar multiplier. The final
    BF16 main+LR add stays in other in both QDQ/native. Semantic ranges do not
    nest; the complete-forward parent is excluded from aggregation.
    """
    from minimax_h3_svdquant_common import DynamicActivationQDQ, nvfp4_qdq
    from deepcompressor.nn.patch.lowrank import LowRankBranch
    old = {'linear': F.linear, 'mm': F.scaled_mm, 'sdpa': F.scaled_dot_product_attention,
           'activation': DynamicActivationQDQ.__call__, 'branch': LowRankBranch.forward,
           'native': NativeH3Linear.forward}
    counts = Counter()

    @contextmanager
    def region(name):
        counts[name] += 1
        with torch.profiler.record_function('E009::'+name):
            yield

    def linear(*args, **kwargs):
        if LR_DEPTH.get():
            counts['lr_linear_calls'] += 1
            return old['linear'](*args, **kwargs)
        with region('main_bf16_gemm'):
            return old['linear'](*args, **kwargs)

    def mm(*args, **kwargs):
        with region('native_gemm'):
            return old['mm'](*args, **kwargs)

    def sdpa(q, k, v, *args, **kwargs):
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            raise RuntimeError('Actual H3 attention QKV must stay BF16')
        with region('attention'):
            return old['sdpa'](q, k, v, *args, **kwargs)

    def branch(self, *args, **kwargs):
        token = LR_DEPTH.set(1)
        try:
            with region('lr'):
                return old['branch'](self, *args, **kwargs)
        finally:
            LR_DEPTH.reset(token)

    def activation(self, module, args):
        self.clear()
        x = args[0]
        if self.smooth is not None:
            with region('smooth'):
                x = x/self.smooth.to(device=x.device, dtype=x.dtype)
        self.calls += 1
        with region('activation_qdq'):
            quantized = nvfp4_qdq(x, element_size=self.element_size)
        if self.branch is not None:
            self.branch_output = self.branch(x)
        return (quantized, *args[1:])

    def native(self, x):
        if self.execution_mode != 'native' or x.dtype != torch.bfloat16:
            raise RuntimeError('Profile native forward only accepts its frozen mode/dtype')
        with region('smooth'):
            x_s = x/self.smooth
        with region('pack'):
            packet = self.pack_input(x_s)
        token = LR_DEPTH.set(1)
        try:
            with region('lr'):
                correction = 1.0*F.linear(F.linear(x_s, self.lr_a), self.lr_b)
        finally:
            LR_DEPTH.reset(token)
        main = self.main_from_packet(packet, mode='native', include_bias=True)
        return main+correction

    F.linear, F.scaled_mm, F.scaled_dot_product_attention = linear, mm, sdpa
    DynamicActivationQDQ.__call__, LowRankBranch.forward, NativeH3Linear.forward = activation, branch, native
    try:
        yield counts
    finally:
        F.linear, F.scaled_mm, F.scaled_dot_product_attention = old['linear'], old['mm'], old['sdpa']
        DynamicActivationQDQ.__call__, LowRankBranch.forward, NativeH3Linear.forward = old['activation'], old['branch'], old['native']


def trace_summary(path):
    """Strict Chrome kernels, correlated to nonoverlapping CPU semantic ranges."""
    from summarize_wan_native_profile import aggregate
    events = json.loads(path.read_text())['traceEvents']
    annotations = [e for e in events if e.get('cat') == 'user_annotation' and
                   e.get('name', '').startswith('E009::') and e['name'] != 'E009::complete_forward']
    threads = defaultdict(list)
    for event in annotations:
        threads[(event['pid'], event['tid'])].append(event)
    index = {}
    for key, rows in threads.items():
        rows.sort(key=lambda e: e['ts'])
        if any(a['ts']+a['dur'] > b['ts']+1e-3 for a, b in zip(rows, rows[1:])):
            raise RuntimeError('Semantic CPU ranges overlap')
        index[key] = ([e['ts'] for e in rows], rows)
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    launches = {}
    for event in runtime:
        key = event.get('args', {}).get('correlation')
        if key is not None:
            if key in launches:
                raise RuntimeError('Duplicate launch correlation')
            launches[key] = event
    groups = defaultdict(list)
    kernels = [e for e in events if e.get('cat') == 'kernel']
    for event in kernels:
        launch = launches.get(event.get('args', {}).get('correlation'))
        if launch is None:
            raise RuntimeError('Kernel launch correlation missing')
        starts, rows = index.get((launch['pid'], launch['tid']), ([], []))
        i = bisect_right(starts, launch['ts'])-1
        category = rows[i]['name'] if i >= 0 and launch['ts'] < rows[i]['ts']+rows[i]['dur'] else 'other'
        groups[category].append(event)
    total = sum(e['dur'] for e in kernels)
    categories = [{'category': name, 'kernel_calls': len(rows), 'kernel_work_us': sum(e['dur'] for e in rows),
                    'fraction_of_kernel_work': sum(e['dur'] for e in rows)/total,
                    'kernels': aggregate(rows)} for name, rows in groups.items()]
    if abs(sum(row['kernel_work_us'] for row in categories)-total) > 1e-5:
        raise RuntimeError('Kernel work classification is not conservative')
    return {'strict_kernel_count': len(kernels), 'strict_kernel_work_us': total,
            'kernel_aggregate': aggregate(kernels), 'categories': sorted(categories, key=lambda r: -r['kernel_work_us']),
            'cpu_cuda_api_aggregate': aggregate(runtime),
            'copy_events': aggregate([e for e in events if e.get('cat') == 'gpu_memcpy']),
            'memset_events': aggregate([e for e in events if e.get('cat') == 'gpu_memset']),
            'event_category_counts': dict(Counter(e.get('cat', '<none>') for e in events)),
            'interpretation': 'Only cat=kernel is kernel work. GPU user annotations are excluded. Semantic labels do not nest. Kernel work is not critical-path latency; do not add parent/child or overlapping durations.'}


def correctness_binding(args, report):
    """Bind inherited completed stages to the explicit post-failure resume.

    The original full report is intentionally failed_stop. Only its three
    completed reference stages are inherited; native comes from the new run.
    """
    reference = json.loads(args.correctness.read_text())
    resume = json.loads(args.fast_proof.read_text())
    sf0 = json.loads(SF0_PROOF.read_text())
    if resume.get('status') != 'complete' or sf0.get('status') != 'complete':
        raise RuntimeError('Full native resume and real zero-SF domain proof must be complete')
    if not reference.get('resident_offload_exact') or not all(reference['stages'][s].get('complete') for s in
                                                             ('offload_bf16', 'resident_bf16', 'legacy_formula_reference')):
        raise RuntimeError('Inherited resident/BF16/QDQ reference stages incomplete')
    original_sha = sha256(args.correctness)
    bound_original = resume.get('source_failure_sha256')
    if bound_original is None:
        bound_original = resume.get('files', {}).get(str(args.correctness), {}).get('sha256')
    if bound_original != original_sha or sf0['source_failure_sha256'] != original_sha:
        raise RuntimeError('Resume/SF0 proof does not bind the exact original failed report')
    inherited = resume['inherited_reference']
    if inherited['sha256'] != original_sha or Path(inherited['file']) != args.correctness:
        raise RuntimeError('Inherited-reference identity changed')
    if inherited['source_files'] != reference['files']:
        raise RuntimeError('Inherited source manifest differs from original report')
    sf0_gate = sf0['legacy_qdq_vs_e005_decode']
    if not all(sf0_gate[k] for k in ('torch_equal', 'finite_common', 'finite_decoded')) or sf0_gate['other_byte_differences'] != 0:
        raise RuntimeError('Real zero-SF decode equivalence gate missing')
    slow = resume['stages']['native']
    fast = resume['fast_proof']
    if not slow.get('complete') or fast.get('status') != 'complete':
        raise RuntimeError('Resumed slow native stage incomplete')
    for name in ('video', 'audio'):
        if fast['endpoints'][name]['sha256'] != slow['endpoints'][name]['sha256'] or not fast['endpoints'][name]['finite']:
            raise RuntimeError(f'Full slow/fast {name} SHA mismatch')
    if len(slow['blocks']) != 50 or len(fast['blocks']) != 50:
        raise RuntimeError('Full slow/fast proof must cover all 50 blocks')
    for index, (a, b) in enumerate(zip(slow['blocks'], fast['blocks'], strict=True)):
        if a['index'] != index or b['index'] != index or a['sha256'] != b['sha256'] or not b['finite']:
            raise RuntimeError(f'Slow/fast block{index} SHA mismatch')
    checks = fast['fastpack_checks']
    if checks['checked_calls'] != 200 or checks['invalid_calls'] != 0:
        raise RuntimeError('Full fast proof does not validate 200 finite-domain checks')
    if fast['runtime_audit']['scaled_mm_calls'] != 200:
        raise RuntimeError('Full fast proof did not execute all 200 native GEMMs')
    if any(resume['failure_input_fast_byte_gate']['byte_mismatches'].values()):
        raise RuntimeError('Saved real failure input fast codes/scales/global mismatch')
    sources = dict(inherited['source_files'])
    for path, row in resume['files'].items():
        if path in sources and row['sha256'] != sources[path]['sha256']:
            raise RuntimeError(f'Contradictory inherited/new source hash: {path}')
        sources[path] = row
    for filename in ('h3_nvfp4_zero_sf_compat.py', 'h3_nvfp4_fastpack.py', 'h3_native_nvfp4.py',
                     'wan_native_nvfp4.py', 'probe_h3_native_contract.py'):
        path = Path(__file__).with_name(filename)
        if sources[str(path)]['sha256'] != sha256(path):
            raise RuntimeError(f'Full-model slow/fast source binding differs: {path}')
    report['correctness_binding'] = {
        'original_report_sha256': original_sha, 'original_status_retained': reference['status'],
        'inherited_complete_stages': ['offload_bf16', 'resident_bf16', 'legacy_formula_reference'],
        'resume_report': str(args.fast_proof), 'resume_report_sha256': sha256(args.fast_proof),
        'real_sf0_proof': str(SF0_PROOF), 'real_sf0_proof_sha256': sha256(SF0_PROOF),
        'slow_fast_all_50_blocks_and_two_endpoints_exact': True,
        'reference_fast_checks': checks,
    }
    selected = slow if args.arm == 'native' else reference['stages'][STAGES[args.arm]]
    return reference, resume, {name: value['sha256'] for name, value in selected['endpoints'].items()}


@torch.inference_mode()
def execute(args, report):
    reference, resume, expected = correctness_binding(args, report)
    report['expected_endpoint_sha256'] = expected
    report['fast_proof'] = {'path': str(args.fast_proof), 'sha256': sha256(args.fast_proof)}
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    attn = importlib.import_module('diffsynth.core.attention.attention')
    if attn.ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('Explicit torch SDPA implementation required')
    enabled = {'flash': torch.backends.cuda.flash_sdp_enabled(), 'math': torch.backends.cuda.math_sdp_enabled(),
               'mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(), 'cudnn': torch.backends.cuda.cudnn_sdp_enabled()}
    if enabled != reference['sdpa_enabled'] or torch.__version__ != reference['torch']:
        raise RuntimeError('Torch/SDPA dispatch configuration differs from correctness reference')
    report.update(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                  sdpa_enabled=enabled, attention='Same torch SDPA implementation and enabled backends as E009 correctness')
    startup = time.perf_counter()
    paths = [Path(__file__), PROTOCOL, args.correctness, args.fast_proof, SF0_PROOF, SAMPLE, H3_DIT_PATH,
             Path(__file__).with_name('bench_h3_native_nvfp4.py'), Path(__file__).with_name('h3_native_nvfp4.py'),
             Path(__file__).with_name('wan_native_nvfp4.py'), Path(__file__).with_name('h3_nvfp4_fastpack.py'),
             Path(__file__).with_name('h3_nvfp4_zero_sf_compat.py'), Path(__file__).with_name('probe_h3_native_contract.py'),
             Path(__file__).with_name('summarize_wan_native_profile.py'), ROOT/'scripts/minimax_h3_svdquant_common.py',
             args.export_dir/'manifest.json']
    report['files'] = {}
    for path in paths:
        digest = sha256(path)
        report['files'][str(path)] = {'sha256': digest, 'bytes': path.stat().st_size}
        previous = reference['files'].get(str(path))
        if previous and previous['sha256'] != digest:
            raise RuntimeError(f'Frozen E009 source drift: {path}')
    if sha256(args.export_dir/'manifest.json') != reference['export_manifest_sha256']:
        raise RuntimeError('Export manifest differs from correctness prerequisite')
    report['environment'] = {k: os.environ.get(k) for k in
                             ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'SVDQUANT_DATA_ROOT',
                              'DIFFSYNTH_ROOT', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'MINIMAX_H3_DIT_PATH']}
    sample = torch.load(SAMPLE, map_location='cpu', weights_only=False, mmap=True)
    call_args, call_kwargs = tree_device(sample['input_args'], 'cuda'), tree_device(sample['input_kwargs'], 'cuda')
    pipe = load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    model = pipe.dit.eval()
    report['resident_conversion'] = make_h3_resident(model)
    checks_factory = lambda: nullcontext([])
    if args.arm != 'bf16':
        report['native_installation'] = install_native_h3(model, args.export_dir, chunk_rows=1024)
    if args.arm == 'qdq':
        runtimes = install_resident_old_qdq(model)
        report['qdq_runtime_hooks'] = len(runtimes)
    elif args.arm == 'native':
        from h3_nvfp4_zero_sf_compat import pack_activation_fast, collect_fastpack_checks
        checks_factory = collect_fastpack_checks
        for module in model.modules():
            if isinstance(module, NativeH3Linear):
                module.activation_packer = pack_activation_fast
    # All modules are resident; prevent accidental disk loading from remaining references.
    from diffsynth.core.vram.layers import AutoWrappedLinear, AutoWrappedModule, AutoTorchModule
    if any(isinstance(m, AutoTorchModule) for m in model.modules()):
        raise RuntimeError('Offload wrapper remains in profile graph')
    def forbidden_disk(*args, **kwargs):
        raise RuntimeError('Disk model load attempted inside resident profile')
    AutoWrappedLinear.load_from_disk = AutoWrappedModule.load_from_disk = forbidden_disk
    gc.collect()
    torch.cuda.empty_cache()
    report['startup_seconds'] = time.perf_counter()-startup
    report['startup_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['resident_model_storage'] = resident_storage(model)
    save(report, args.output)

    def forward():
        return model(*call_args, **call_kwargs)

    def measured_forward():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with checks_factory() as checks:
            wall_start = time.perf_counter()
            start.record()
            result = forward()
            end.record()
            end.synchronize()
            forward_end = time.perf_counter()
        checked = time.perf_counter()
        if args.arm == 'native' and len(checks) != 200:
            raise RuntimeError(f'Expected 200 checked native calls, got {len(checks)}')
        return result, {'cuda_ms': start.elapsed_time(end),
                        'host_forward_ms_excluding_checks': (forward_end-wall_start)*1000,
                        'flag_validation_ms': (checked-forward_end)*1000,
                        'host_ms_including_checks': (checked-wall_start)*1000, 'fastpack_checks': len(checks),
                        'zero_sf_check_summary': getattr(checks, 'summary', None)}

    print(f'{args.arm}: warmup with strict video/audio SHA gate', flush=True)
    warm, timing = measured_forward()
    cpu, digests = check_endpoint(warm, expected)
    report['warmup'] = {'timing_excluded_from_steady': timing, 'endpoints_sha256': digests}
    snapshot = args.artifact_dir/f'{args.output.stem}.endpoints.pt'
    torch.save(cpu, snapshot)
    report['snapshot'] = {'path': str(snapshot), 'sha256': sha256(snapshot)}
    del warm, cpu, sample
    gc.collect()
    report['steady_before'] = {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()}
    torch.cuda.reset_peak_memory_stats()
    report['unprofiled_repeats'] = []
    for index in range(3):
        result, timing = measured_forward()
        cpu, digests = check_endpoint(result, expected)
        timing.update(repeat=index, endpoints_sha256=digests)
        report['unprofiled_repeats'].append(timing)
        del result, cpu
        save(report, args.output)
        print(f'{args.arm} repeat{index}: host_with_checks={timing["host_ms_including_checks"]:.3f}ms', flush=True)
    report['steady_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                               'after_allocated_bytes': torch.cuda.memory_allocated(),
                               'after_reserved_bytes': torch.cuda.memory_reserved(),
                               'scope': 'Three unprofiled forwards after warmup; excludes startup and profiler'}
    report['latency_ms'] = {key: distribution([r[key] for r in report['unprofiled_repeats']]) for key in
                            ['cuda_ms', 'host_forward_ms_excluding_checks', 'flag_validation_ms', 'host_ms_including_checks']}
    save(report, args.output)
    print(f'{args.arm}: one independent profiler pass', flush=True)
    torch.cuda.synchronize()
    with checks_factory() as checks:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, with_stack=False, profile_memory=False) as prof:
            with labels() as counts:
                with torch.profiler.record_function('E009::complete_forward'):
                    result = forward()
                    torch.cuda.synchronize()
        checks_start = time.perf_counter()
    check_ms = (time.perf_counter()-checks_start)*1000
    cpu, digests = check_endpoint(result, expected)
    del result, cpu
    expected_sdpa = reference['stages']['resident_bf16']['runtime_audit']['sdpa_calls']
    if counts['attention'] != expected_sdpa or counts['native_gemm'] != (200 if args.arm == 'native' else 0):
        raise RuntimeError(f'Profiler actual kernel-call contract differs: {counts}')
    if args.arm == 'native' and len(checks) != 200:
        raise RuntimeError('Profiler flags call count mismatch')
    trace = args.artifact_dir/f'{args.output.stem}.trace.json'
    prof.export_chrome_trace(str(trace))
    report['profile'] = trace_summary(trace)
    if args.arm == 'native' and not any('sm120' in row['name'] and 'e2m1' in row['name'] for row in report['profile']['kernel_aggregate']):
        raise RuntimeError('No actual SM120 NVFP4 kernel observed')
    report['profile'].update(counts=dict(counts), endpoint_sha256=digests, flag_validation_ms_outside_trace=check_ms,
                              zero_sf_check_summary=getattr(checks, 'summary', None),
                              trace={'path': str(trace), 'sha256': sha256(trace)})
    report['status'] = 'complete'
    save(report, args.output)
    print(json.dumps({'arm': args.arm, 'status': 'complete', 'latency_ms': report['latency_ms'],
                      'steady_memory': report['steady_memory']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['bf16', 'qdq', 'native'], required=True)
    parser.add_argument('--correctness', type=Path, default=ROOT/'results/research/E009_h3_full.json')
    parser.add_argument('--fast-proof', type=Path, required=True, help='Completed full slow/fast endpoint SHA proof JSON')
    parser.add_argument('--export-dir', type=Path, default=DATA/'legacy_export')
    parser.add_argument('--artifact-dir', type=Path, default=DATA/'profile')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--describe-only', action='store_true')
    args = parser.parse_args()
    if args.describe_only:
        print(PROTOCOL.read_text())
        return
    args.output = args.output or ROOT/f'results/research/E009_profile_{args.arm}.json'
    if args.output.exists() or any(args.artifact_dir.glob(args.output.stem+'*')):
        raise FileExistsError('Refusing to overwrite previous report/trace/snapshot')
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {'experiment': 'E009_resident_profile', 'status': 'running', 'arm': args.arm,
              'scope': 'One fixed p1 step0 H3 DiT call; original calibration sample; no generation/heldout quality claim',
              'policy': 'Same GPU sequential separate processes; 1 warmup, 3 repeats, 1 independent profiler',
              'qdq_baseline': 'Resident BF16 decoded weight once at startup plus original common activation QDQ/hooks',
              'profile_caveat': 'Strict Chrome kernel categories only; no user_annotation work double counting; profiler times are not steady latency',
              'sample': str(SAMPLE)}
    try:
        execute(args, report)
    except Exception:
        report['status'] = 'failed_no_performance_claim'
        report['error'] = traceback.format_exc()
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
