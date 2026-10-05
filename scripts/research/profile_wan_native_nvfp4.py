#!/usr/bin/env python3
"""E007 deployment profile: one arm per process, fixed complete Wan DiT call.

Run bf16 / qdq / native sequentially in separate processes on the same GPU.
One warmup, three unprofiled repeats, then one independent annotated profile.
No text encoder, VAE, per-layer CPU captures, free rollout, or quality claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'third_party/deepcompressor'))
from bench_wan_native_nvfp4 import (
    BASE, CACHE, CHECKPOINT, RCM, inspect_contract, metric, save, sha256,
    tensor_sha, tree,
)

PROTOCOL = ROOT/'research_state/06_experiments/E007_wan_native_profile_protocol.md'
CORRECTNESS = ROOT/'results/research/E007_wan_native_correctness.json'
FAST_PARITY = ROOT/'results/research/E007_fastpack_real_parity.json'
ARTIFACT_ROOT = Path('/data1/models/svdquant-wjq/research/20261002/E007/profile')
REFERENCE_STAGE = {'bf16': 'bf16', 'qdq': 'legacy_qdq', 'native': 'native_full'}
_LR_DEPTH = ContextVar('wan_profile_lr_depth', default=0)


def distribution(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values),
            'values': values}


def check_snapshot(output, expected_sha, stage):
    cpu = output.detach().cpu()
    if not bool(torch.isfinite(cpu).all()):
        raise RuntimeError(f'{stage}: nonfinite complete DiT endpoint')
    digest = tensor_sha(cpu)
    if digest != expected_sha:
        raise RuntimeError(f'{stage}: endpoint SHA differs from frozen E007: {digest} != {expected_sha}; no timing claim')
    return cpu, digest


def resident_model_storage(model):
    """Count unique CUDA storages, including modules owned only by old hooks.

    This is model storage, distinct from allocator-reserved memory and peak
    activations. Shared LR A storages are counted once. CUDA memory held by
    libraries, contexts and caching allocators is not attributed to parameters.
    """
    seen, categories = set(), defaultdict(lambda: {'storages': 0, 'bytes': 0})
    tensors = []
    for name, p in model.named_parameters():
        tensors.append(('registered_parameters', name, p))
    for name, b in model.named_buffers():
        tensors.append(('registered_buffers', name, b))
    seen_hooks = set()
    for path, module in model.named_modules():
        for hook in [*module._forward_pre_hooks.values(), *module._forward_hooks.values()]:
            hook = getattr(hook, 'original', hook)
            if id(hook) in seen_hooks:
                continue
            seen_hooks.add(id(hook))
            branch = getattr(hook, 'branch', None)
            if isinstance(branch, torch.nn.Module):
                for name, p in branch.named_parameters():
                    tensors.append(('hook_lr_parameters', path+'.'+name, p))
            processor = getattr(hook, 'processor', None)
            smooth = getattr(processor, 'smooth_scale', None)
            if torch.is_tensor(smooth):
                tensors.append(('hook_smoothing', path, smooth))
    for category, name, tensor in tensors:
        if not tensor.is_cuda:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        categories[category]['storages'] += 1
        categories[category]['bytes'] += storage.nbytes()
    return {'categories': dict(categories), 'unique_cuda_storage_bytes': sum(v['bytes'] for v in categories.values()),
            'definition': 'unique registered parameter/buffer plus hook-owned LR/smoothing CUDA storages; excludes allocator/library state'}


@contextmanager
def profile_labels():
    """Installed only for the independent profiler pass, never timed repeats."""
    from deepcompressor.calib.smooth import ActivationSmoother
    from deepcompressor.utils.hooks.branch import AccumBranchHook
    from wan_native_nvfp4 import NativeWanLinear
    old = {'linear': F.linear, 'scaled_mm': F.scaled_mm, 'sdpa': F.scaled_dot_product_attention,
           'smooth': ActivationSmoother.process,
           'lr_hook': AccumBranchHook.post_forward, 'pack': NativeWanLinear.pack_input}
    counts = defaultdict(int)

    def linear(*args, **kwargs):
        if _LR_DEPTH.get():
            counts['lr_bf16_gemm_calls'] += 1
            return old['linear'](*args, **kwargs)
        counts['main_bf16_gemm'] += 1
        with torch.profiler.record_function('E007::main_bf16_gemm'):
            return old['linear'](*args, **kwargs)

    def scaled_mm(*args, **kwargs):
        counts['native_gemm'] += 1
        with torch.profiler.record_function('E007::native_gemm'):
            return old['scaled_mm'](*args, **kwargs)

    def attention(q, k, v, *args, **kwargs):
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            raise RuntimeError('profile: actual attention QKV must remain BF16')
        counts['attention'] += 1
        with torch.profiler.record_function('E007::attention'):
            return old['sdpa'](q, k, v, *args, **kwargs)

    def smooth(self, *args, **kwargs):
        counts['smooth'] += 1
        with torch.profiler.record_function('E007::smooth'):
            return old['smooth'](self, *args, **kwargs)

    def lr_hook(self, *args, **kwargs):
        counts['lr'] += 1
        token = _LR_DEPTH.set(_LR_DEPTH.get()+1)
        try:
            with torch.profiler.record_function('E007::lr'):
                return old['lr_hook'](self, *args, **kwargs)
        finally:
            _LR_DEPTH.reset(token)

    def pack(self, *args, **kwargs):
        counts['pack'] += 1
        with torch.profiler.record_function('E007::pack'):
            return old['pack'](self, *args, **kwargs)

    F.linear, F.scaled_mm, F.scaled_dot_product_attention = linear, scaled_mm, attention
    ActivationSmoother.process = smooth
    AccumBranchHook.post_forward, NativeWanLinear.pack_input = lr_hook, pack
    try:
        yield counts
    finally:
        F.linear, F.scaled_mm, F.scaled_dot_product_attention = old['linear'], old['scaled_mm'], old['sdpa']
        ActivationSmoother.process = old['smooth']
        AccumBranchHook.post_forward, NativeWanLinear.pack_input = old['lr_hook'], old['pack']


def aggregate_profile(prof):
    kernels, runtime = defaultdict(list), defaultdict(list)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name].append(float(event.time_range.elapsed_us()))
        elif (event.name.startswith(('cuda', 'cuLaunch', 'cuGraph', 'cuMemcpy'))
              and not event.name.startswith('cuda::')):
            runtime[event.name].append(float(event.cpu_time_total))
    def rows(source):
        return sorted([{'name': name, 'calls': len(v), 'total_us': sum(v),
                        'mean_us': sum(v)/len(v), 'min_us': min(v), 'max_us': max(v)}
                       for name, v in source.items()], key=lambda x: x['total_us'], reverse=True)
    annotations, cpu = [], []
    for event in prof.key_averages():
        row = {'name': event.key, 'calls': int(event.count),
               'cpu_total_us': float(event.cpu_time_total), 'cpu_self_us': float(event.self_cpu_time_total),
               'device_total_us': float(event.device_time_total),
               'device_self_us': float(event.self_device_time_total)}
        if event.key.startswith('E007::'):
            annotations.append(row)
        else:
            cpu.append(row)
    ranges = [row for row in annotations if row['name'] != 'E007::complete_dit_forward']
    total_cuda_us = sum(sum(v) for v in kernels.values())
    classified_cuda_us = sum(row['device_total_us'] for row in ranges)
    return {'cuda_kernel_aggregate': rows(kernels), 'cuda_runtime_api_aggregate': rows(runtime),
            'annotation_ranges': sorted(annotations, key=lambda x: x['device_total_us'], reverse=True),
            'semantic_gpu_work': {'nonoverlapping_ranges': ranges, 'cuda_event_work_total_us': total_cuda_us,
                                  'unclassified_other_us': total_cuda_us-classified_cuda_us},
            'cpu_operator_top100_by_self': sorted(cpu, key=lambda x: x['cpu_self_us'], reverse=True)[:100],
            'interpretation': 'semantic subranges do not nest; complete_dit_forward is a parent total and must not be added. Do not add semantic ranges to their aten/CUDA children. CUDA duration sums measure work, not host critical-path latency; overlapping work cannot be extrapolated mechanically into speedup.'}


def provenance(report, reference, arm):
    paths = [Path(__file__), PROTOCOL, CORRECTNESS, Path(__file__).with_name('bench_wan_native_nvfp4.py'),
             Path(__file__).with_name('wan_native_nvfp4.py'), ROOT/'scripts/infer_rcm_wan_4step.py',
             CACHE, RCM/'config.json', RCM/'diffusion_pytorch_model.safetensors']
    if arm != 'bf16':
        paths += [CHECKPOINT/f'{name}.pt' for name in ['model', 'scale', 'wgts', 'smooth', 'branch']]
    if arm == 'native':
        paths += [FAST_PARITY, Path(__file__).with_name('wan_nvfp4_fastpack.py')]
    report['files'] = {}
    for path in paths:
        digest = sha256(path)
        report['files'][str(path)] = {'sha256': digest, 'bytes': path.stat().st_size,
                                    'resolved_path': str(path.resolve())}
        previous = reference['files'].get(str(path))
        if previous is not None and previous['sha256'] != digest:
            raise RuntimeError(f'E007 source/data drift at {path}')


@torch.inference_mode()
def execute(args, report):
    import diffusers
    from diffusers import WanPipeline, WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from infer_rcm_wan_4step import load_quantized_transformer
    import wan_native_nvfp4 as native_lib
    if diffusers.__version__ != '0.33.1' or torch.__version__ != '2.11.0+cu128':
        raise RuntimeError('profile requires the same historical torch 2.11.0+cu128 / Diffusers 0.33.1 environment')
    reference = json.loads(CORRECTNESS.read_text())
    if reference['status'] != 'complete':
        raise RuntimeError('full E007 correctness prerequisite is incomplete')
    report['expected_endpoint_sha256'] = expected = reference['stages'][REFERENCE_STAGE[args.arm]]['endpoint_sha256']
    startup = time.perf_counter()
    provenance(report, reference, args.arm)
    contract, payload = inspect_contract()
    if contract['input_sha256'] != reference['input_contract']['input_sha256']:
        raise RuntimeError('fixed input changed')
    report['input_contract'] = contract
    torch.set_num_threads(6); torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    report.update(torch=torch.__version__, diffusers=diffusers.__version__, torch_file=torch.__file__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  gpu_capability=list(torch.cuda.get_device_capability()),
                  attention_backend='torch SDPBackend.FLASH_ATTENTION, BF16 QKV; same as E007')
    report['environment'] = {key: os.environ.get(key) for key in
                             ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'SVDQUANT_DATA_ROOT',
                              'PYTORCH_CUDA_ALLOC_CONF', 'CUBLAS_WORKSPACE_CONFIG']}
    report['torch_settings'] = {'num_threads': torch.get_num_threads(),
                                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    print(f'{args.arm}: loading fixed rCM transformer', flush=True)
    model = WanTransformer3DModel.from_pretrained(RCM, torch_dtype=torch.bfloat16).cuda().eval()
    if len(model.blocks) != 30 or any(type(b.attn1.processor) is not WanAttnProcessor2_0 for b in model.blocks):
        raise RuntimeError('unexpected model/attention processor')
    if args.arm != 'bf16':
        pipe = WanPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, transformer=model,
                                           text_encoder=None, tokenizer=None, vae=None)
        load_quantized_transformer(pipe, CHECKPOINT, BASE)
        model = pipe.transformer.eval()
        del pipe
    checks_factory = lambda: nullcontext([])
    if args.arm == 'native':
        from wan_nvfp4_fastpack import pack_activation_fast, collect_fastpack_checks, validate_quantizer_contract
        parity = json.loads(FAST_PARITY.read_text())
        if parity['status'] != 'complete' or any(any(row['byte_mismatches'].values()) for row in parity['cases']):
            raise RuntimeError('fast packer real-input byte parity prerequisite failed')
        actual_hash = sha256(Path(__file__).with_name('wan_nvfp4_fastpack.py'))
        if parity['source_sha256']['wan_nvfp4_fastpack.py'] != actual_hash:
            raise RuntimeError('fast packer source differs from verified real parity source')
        conversion = native_lib.convert_wan_transformer_to_native(model, CHECKPOINT,
            activation_packer='legacy', chunk_rows=args.chunk_rows)
        if conversion['target_count'] != 300 or conversion['exact_roundtrip_count'] != 300:
            raise RuntimeError('300 exact saved-weight roundtrips required')
        names = []
        for name, module in model.named_modules():
            if isinstance(module, native_lib.NativeWanLinear):
                validate_quantizer_contract(module.activation_quantizer)
                module.activation_packer = pack_activation_fast
                names.append(name)
        if len(names) != 300:
            raise RuntimeError('all 300 activation quantizers must pass offline fast-recipe validation')
        report['conversion'] = conversion
        report['offline_fast_recipe_validation'] = {'count': len(names), 'module_names': names}
        checks_factory = collect_fastpack_checks
    call_args = tree(payload['input_args'], 'cuda')
    call_kwargs = tree(payload['input_kwargs'], 'cuda')
    cached_endpoint = payload['outputs'][0]
    del payload
    # Imports/extension loading for labels must not contaminate the later trace.
    # Entering this context performs no forward and immediately restores methods.
    with profile_labels():
        pass
    gc.collect(); torch.cuda.empty_cache()
    report['startup_seconds_including_hash_load_conversion'] = time.perf_counter()-startup
    report['startup_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save(report, args.output)

    def forward():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return model(*call_args, **call_kwargs)[0]

    def measured_forward():
        # All validation GPU kernels happen after end-event synchronization.
        # The deployment wall time INCLUDING those checks is reported as well.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with checks_factory() as checks:
            wall_start = time.perf_counter()
            start.record()
            result = forward()
            end.record(); end.synchronize()
            body_end = time.perf_counter()
            checks_start = time.perf_counter()
        scope_end = time.perf_counter()
        if args.arm == 'native' and len(checks) != 300:
            raise RuntimeError(f'expected 300 validated native pack calls, got {len(checks)}')
        return result, {'cuda_ms': start.elapsed_time(end),
                        'host_forward_ms_excluding_flag_validation': (body_end-wall_start)*1000,
                        'flag_validation_ms': (scope_end-checks_start)*1000,
                        'host_ms_including_checks': (scope_end-wall_start)*1000,
                        'fastpack_checks': len(checks)}

    print(f'{args.arm}: one full warmup and exact E007 endpoint SHA gate', flush=True)
    warm, warm_times = measured_forward()
    warm_cpu, digest = check_snapshot(warm, expected, 'warmup')
    if args.arm == 'bf16':
        # Reuse the warmup for this check: no additional BF16 validation forward.
        report['bf16_vs_original_cached_endpoint'] = metric(warm_cpu, cached_endpoint)
        if report['bf16_vs_original_cached_endpoint']['nmse'] > 1e-6:
            raise RuntimeError('BF16/cache correctness drift')
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = ARTIFACT_ROOT/(args.output.stem+'.endpoint.pt')
    if snapshot.exists():
        raise FileExistsError(f'refusing to overwrite endpoint artifact: {snapshot}')
    torch.save({'arm': args.arm, 'sample': str(CACHE), 'endpoint': warm_cpu,
                'endpoint_sha256': digest}, snapshot)
    report['warmup'] = {'timing_not_in_steady_summary': warm_times, 'endpoint_sha256': digest}
    report['snapshot'] = {'path': str(snapshot), 'sha256': sha256(snapshot)}
    del warm, warm_cpu, cached_endpoint
    gc.collect()
    report['resident_model_storage'] = resident_model_storage(model)
    report['steady_before'] = {'allocated_bytes': torch.cuda.memory_allocated(),
                               'reserved_bytes': torch.cuda.memory_reserved()}
    torch.cuda.reset_peak_memory_stats()
    report['unprofiled_repeats'] = []
    for index in range(3):
        result, timing = measured_forward()
        cpu, digest = check_snapshot(result, expected, f'unprofiled repeat {index}')
        timing.update(repeat=index, endpoint_sha256=digest)
        report['unprofiled_repeats'].append(timing)
        del result, cpu
        save(report, args.output)
        print(f'{args.arm}: repeat {index} wall_with_checks={timing["host_ms_including_checks"]:.3f} ms, CUDA={timing["cuda_ms"]:.3f} ms', flush=True)
    report['steady_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                               'after_allocated_bytes': torch.cuda.memory_allocated(),
                               'after_reserved_bytes': torch.cuda.memory_reserved(),
                               'scope': 'after one warmup, three unprofiled complete DiT calls including flag checks; excludes load/conversion/profiler'}
    report['latency_ms'] = {key: distribution([v[key] for v in report['unprofiled_repeats']])
                            for key in ['cuda_ms', 'host_forward_ms_excluding_flag_validation',
                                        'flag_validation_ms', 'host_ms_including_checks']}
    save(report, args.output)

    print(f'{args.arm}: independent single-forward annotated profiler (not latency data)', flush=True)
    torch.cuda.synchronize()
    profile_started = time.perf_counter()
    with checks_factory() as checks:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, with_stack=False, profile_memory=False) as prof:
            with profile_labels() as counts:
                with torch.profiler.record_function('E007::complete_dit_forward'):
                    profiled_result = forward()
                    torch.cuda.synchronize()
        flag_check_started = time.perf_counter()
    profile_checks_ms = (time.perf_counter()-flag_check_started)*1000
    if counts['attention'] != 60 or (args.arm == 'native' and (counts['native_gemm'] != 300 or len(checks) != 300)):
        raise RuntimeError(f'profile execution contract changed: {dict(counts)}, flags={len(checks)}')
    cpu, digest = check_snapshot(profiled_result, expected, 'annotated profiler')
    del profiled_result, cpu
    trace = ARTIFACT_ROOT/(args.output.stem+'.trace.json')
    if trace.exists():
        raise FileExistsError(f'refusing to overwrite trace: {trace}')
    prof.export_chrome_trace(str(trace))
    report['profile'] = aggregate_profile(prof)
    if args.arm == 'native' and not any('sm120' in row['name'] and 'e2m1' in row['name']
                                      for row in report['profile']['cuda_kernel_aggregate']):
        raise RuntimeError('profile did not observe an actual SM120 FP4 GEMM')
    report['profile'].update(counts=dict(counts), endpoint_sha256=digest,
                              trace={'path': str(trace), 'sha256': sha256(trace)},
                              flag_validation_ms_outside_profile=profile_checks_ms,
                              profiler_export_seconds_not_latency=time.perf_counter()-profile_started)
    report['status'] = 'complete'
    save(report, args.output)
    print(json.dumps({'arm': args.arm, 'status': report['status'], 'latency_ms': report['latency_ms'],
                      'steady_memory': report['steady_memory'],
                      'resident_model_storage': report['resident_model_storage']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['bf16', 'qdq', 'native'], required=True)
    parser.add_argument('--chunk-rows', type=int, default=1024)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--describe-only', action='store_true', help='print the frozen protocol without GPU initialization')
    args = parser.parse_args()
    if args.describe_only:
        print(PROTOCOL.read_text())
        return
    if args.output is None:
        args.output = ROOT/f'results/research/E007_profile_{args.arm}.json'
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite prior profile: {args.output}')
    report = {'experiment': 'E007_deployment_profile', 'arm': args.arm, 'status': 'partial',
              'sample': str(CACHE), 'warmup_calls': 1, 'unprofiled_repeats_planned': 3,
              'profile_calls': 1, 'process_policy': 'one arm per process, run all arms sequentially on the same GPU',
              'scope': 'single full DiT denoiser call; no text encoding, VAE, four-step rollout or video pipeline latency',
              'limitations': ['one calibration raw call, no quality/generalization or research-contribution claim',
                              'startup/JIT excluded from steady latency and reported separately',
                              'profiler annotations installed only in separate profiler forward',
                              'native finite-domain validation checked before accepting timings; cost also included in total host latency']}
    try:
        execute(args, report)
    except Exception:
        report['status'] = 'failed_no_performance_claim'
        report['error'] = traceback.format_exc()
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
