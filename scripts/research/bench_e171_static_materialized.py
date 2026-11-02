#!/usr/bin/env python3
"""Test static physical-route reuse without changing production cache semantics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--input-size', choices=('480', '768'), required=True)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
name = 'validate_e169_480p_paths.py' if args.input_size == '480' else 'validate_e163_four_paths.py'
spec = importlib.util.spec_from_file_location('validator', ROOT/'scripts/research'/name)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
original_measure = validation._measure
static_report = {}

@torch.inference_mode()
def measure(function, *, warmup=5, repeats=20):
    baseline = original_measure(function, warmup=warmup, repeats=repeats)
    if function.__name__ != 'reuse_step':
        return baseline
    reference = function()
    original = validation.executor.sm120_h3_materialize_packed_route
    saved = {}
    calls = [0]
    def cached(*operands, **kwargs):
        signature = (tuple(x.data_ptr() for x in operands), tuple(sorted(kwargs.items())))
        if not saved:
            saved.update(signature=signature, output=original(*operands, **kwargs))
            calls[0] += 1
        assert saved['signature'] == signature
        return saved['output']
    try:
        validation.executor.sm120_h3_materialize_packed_route = cached
        candidate = function()
        assert torch.equal(reference, candidate), 'static physical route changed output'
        static = original_measure(function, warmup=warmup, repeats=repeats)
        static_report.update(baseline=baseline, static=static, bitwise_equal=True,
                             physical_route_materializations=calls[0],
                             physical_route_bytes=sum(x.numel()*x.element_size() for x in saved['output']))
    finally:
        validation.executor.sm120_h3_materialize_packed_route = original
    static_report['baseline_after'] = original_measure(function, warmup=warmup, repeats=repeats)
    return baseline

if __name__ == '__main__':
    validation._measure = measure
    validation.main()
    target = Path(sys.argv[sys.argv.index('--output')+1])
    target.with_name(target.stem+'_static.json').write_text(json.dumps(static_report, indent=2)+'\n')
    print(json.dumps(static_report, indent=2), flush=True)
