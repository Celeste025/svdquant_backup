#!/usr/bin/env python3
"""Small Q128 full-API and phase comparison of proxy routing versus route reuse.

Uses the E154 validator's CLI, input captures and correctness checks. Full API
is timed without wrappers; instrumented calls are separate. Prefix runs on its
own stream, so phase sums must not be interpreted as critical-path latency.
"""
import importlib.util
import json
import statistics
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'e154_validator', ROOT / 'scripts/research/validate_e154_exact_route_reuse.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
ex = validation.executor
NAMES = {
    '_prepare_h3_sm120_inputs': 'qkv_prepare_including_optional_pool',
    'sm120_h3_draft_probability': 'draftmap_score',
    '_route_sm120_probability': 'route_selection',
    'sm120_h3_materialize_route': 'route_materialize',
    'sm120_h3_materialize_packed_route': 'cached_route_materialize',
    '_run_sm120_phases': 'attention_main',
    'sm120_q128_prefix_int8_attention': 'prefix_parallel',
    'assemble_h3_k64_output': 'output_assembly',
}
original_measure = validation._measure
phase_report = {}
prepare_args = None

@torch.inference_mode()
def profiled_measure(function, *, warmup=1, repeats=3):
    global prepare_args
    full = original_measure(function, warmup=warmup, repeats=repeats)
    if function.__name__ == 'first_step':
        return full
    pending = []
    collected = {label: [] for label in NAMES.values()}
    originals = {name: getattr(ex, name) for name in NAMES}
    def wrap(original, label):
        def instrumented(*args, **kwargs):
            global prepare_args
            if label == 'qkv_prepare_including_optional_pool':
                prepare_args = (args, kwargs.copy())
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); result = original(*args, **kwargs); end.record()
            pending.append((label, begin, end))
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
        label: {'mean_ms': statistics.mean(values), 'samples_ms': values}
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
    import sys
    target = Path(sys.argv[sys.argv.index('--output') + 1])
    report = {
        'gpu': torch.cuda.get_device_name(),
        'scope': 'CUDA event spans; full API measured separately without instrumentation; prefix overlaps main stream; 1 warmup, 3 measurements for recommended smoke CLI',
        'phases': phase_report,
    }
    target.with_name(target.stem + '_phases.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
