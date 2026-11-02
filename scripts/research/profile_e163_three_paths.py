#!/usr/bin/env python3
"""E163 Q128 full-API and phase comparison of proxy, first capture and reuse.

Uses the E154 validator's CLI, input captures and correctness checks. Full API
is timed without wrappers; instrumented calls are separate. Prefix runs on its
own stream, so phase sums must not be interpreted as critical-path latency.
"""
import importlib.util
import json
import statistics
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'e163_validator', ROOT / 'scripts/research/validate_e163_four_paths.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
ex = validation.executor
NAMES = {
    '_prepare_h3_sm120_inputs': 'qkv_prepare_including_optional_pool',
    'sm120_h3_draft_probability': 'draftmap_score',
    '_route_sm120_probability': 'route_selection',
    '_compact_exact_route': 'capture_route_compaction',
    'sm120_q128_dense_int8_importance_attention': 'dense_int8_plus_importance',
    'sm120_h3_materialize_route': 'route_materialize',
    'sm120_h3_materialize_packed_route': 'cached_route_materialize',
    '_run_sm120_phases': 'attention_main',
    'sm120_q128_prefix_int8_attention': 'prefix_parallel',
    'assemble_h3_k64_output': 'output_assembly',
}
original_measure = validation._measure
phase_report = {}
prepare_args = None
materialize_args = None

@torch.inference_mode()
def profiled_measure(function, *, warmup=1, repeats=3):
    global prepare_args, materialize_args
    full = original_measure(function, warmup=warmup, repeats=repeats)
    pending = []
    collected = {label: [] for label in NAMES.values()}
    collected_host = {label: [] for label in NAMES.values()}
    originals = {name: getattr(ex, name) for name in NAMES}
    def wrap(original, label):
        def instrumented(*args, **kwargs):
            global prepare_args, materialize_args
            if label == 'qkv_prepare_including_optional_pool':
                prepare_args = (args, kwargs.copy())
            if label == 'cached_route_materialize':
                materialize_args = (args, kwargs.copy())
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); host_begin = time.perf_counter()
            result = original(*args, **kwargs)
            host_end = time.perf_counter(); end.record()
            pending.append((label, begin, end))
            collected_host[label].append((host_end - host_begin) * 1000)
            return result
        return instrumented
    try:
        for name, original in originals.items():
            setattr(ex, name, wrap(original, NAMES[name]))
        for index in range(warmup + repeats):
            pending.clear()
            output = function()
            torch.cuda.synchronize()
            del output
            if index >= warmup:
                for label, begin, end in pending:
                    collected[label].append(begin.elapsed_time(end))
    finally:
        for name, original in originals.items():
            setattr(ex, name, original)
    phase_report[function.__name__] = {
        label: {'mean_ms': statistics.mean(values), 'samples_ms': values,
                'host_mean_ms': statistics.mean(collected_host[label])}
        for label, values in collected.items() if values
    }
    return full

if __name__ == '__main__':
    validation._measure = profiled_measure
    validation.main()
    # Replay the SAME legacy preparation operands with/without the fused pooling.
    # This isolates pooling's incremental cost from the different selected routes.
    if prepare_args is not None:
        args, kwargs = prepare_args
        pool_report = {}
        for enabled in (True, False, False, True):
            kwargs.update(has_pool=enabled, has_maxpool=enabled)
            samples = []
            for index in range(4):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); result = ex._prepare_h3_sm120_inputs(*args, **kwargs); end.record()
                end.synchronize(); del result
                if index: samples.append(begin.elapsed_time(end))
            pool_report.setdefault('pool_on' if enabled else 'pool_off', []).extend(samples)
        phase_report['paired_prepare_pool_toggle'] = {
            key: {'mean_ms': statistics.mean(values), 'samples_ms': values}
            for key, values in pool_report.items()
        }
    if materialize_args is not None:
        args, kwargs = materialize_args
        isolated = []
        host = []
        output_layout = None
        for index in range(24):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); started = time.perf_counter()
            result = ex.sm120_h3_materialize_packed_route(*args, **kwargs)
            if output_layout is None:
                output_layout = [
                    {'shape': list(t.shape), 'dtype': str(t.dtype),
                     'bytes': t.numel() * t.element_size()} for t in result
                ]
            host.append((time.perf_counter() - started) * 1000)
            end.record(); end.synchronize()
            if index >= 4: isolated.append(begin.elapsed_time(end))
            del result
        phase_report['isolated_cached_route_materialize'] = {
            'mean_ms': statistics.mean(isolated), 'samples_ms': isolated,
            'host_mean_ms': statistics.mean(host[4:]),
            'outputs': output_layout,
        }
    import sys
    target = Path(sys.argv[sys.argv.index('--output') + 1])
    report = {
        'gpu': torch.cuda.get_device_name(),
        'scope': 'CUDA event spans; full API measured separately without instrumentation; prefix overlaps main stream; 1 warmup, 3 measurements for recommended smoke CLI',
        'phases': phase_report,
    }
    target.with_name(target.stem + '_phases.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
