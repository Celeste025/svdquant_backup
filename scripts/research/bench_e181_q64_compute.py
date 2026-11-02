#!/usr/bin/env python3
"""Q128 quantization/routes with Q64 compute CTAs, only NV90/INT8-10 reuse."""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--input-size',choices=('480','768'),required=True)
a,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
name='validate_e169_480p_paths.py' if a.input_size=='480' else 'validate_e163_four_paths.py'
spec=importlib.util.spec_from_file_location('validation',ROOT/'scripts/research'/name)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
original_measure=v._measure
report={}
@torch.inference_mode()
def measure(function, *, warmup=5, repeats=20):
    baseline=original_measure(function,warmup=warmup,repeats=repeats)
    if function.__name__!='reuse_step':return baseline
    reference=function()
    original=v.executor._run_sm120_phases
    saved={}
    def q64(**kwargs):
        assert kwargs['query_block_size']==128
        assert kwargs['ratios'][0]>.89 and kwargs['ratios'][1]>.09 and kwargs['ratios'][3]==0
        if not saved:
            for key in ('block_ids','nvfp4_counts','middle_counts','fp16_counts'):
                tensor=kwargs[key]
                saved[key]=tensor.repeat_interleave(2,dim=2).contiguous() if tensor.ndim>=3 else tensor
        args=dict(kwargs);args.update(saved);args['query_block_size']=64
        operands=list(args['prepared_int8_operands'])
        operands[1]=operands[1].repeat_interleave(2,dim=-1).contiguous()
        args['prepared_int8_operands']=tuple(operands)
        return original(**args)
    try:
        v.executor._run_sm120_phases=q64
        candidate=function();assert torch.isfinite(candidate).all()
        delta=(candidate-reference).float()
        report.update(bitwise_equal=torch.equal(reference,candidate),
                      max_abs=float(delta.abs().max()),
                      nmse=float(delta.double().square().sum()/reference.double().square().sum()),
                      baseline=baseline,
                      q64_compute=original_measure(function,warmup=warmup,repeats=repeats))
    finally:v.executor._run_sm120_phases=original
    report['baseline_after']=original_measure(function,warmup=warmup,repeats=repeats)
    return baseline
v._measure=measure
v.main()
target=Path(sys.argv[sys.argv.index('--output')+1])
target.with_name(target.stem+'_q64_compute.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
