#!/usr/bin/env python3
"""Same NV90/INT8-10 tiles, separate phase kernels and LSE merge."""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--input-size',choices=('480','768'),required=True)
a,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
name='validate_e169_480p_paths.py' if a.input_size=='480' else 'validate_e163_four_paths.py'
spec=importlib.util.spec_from_file_location('v',ROOT/'scripts/research'/name)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
original_measure=v._measure;report={}
@torch.inference_mode()
def measure(function, *, warmup=5, repeats=20):
    baseline=original_measure(function,warmup=warmup,repeats=repeats)
    if function.__name__!='reuse_step':return baseline
    reference=function();original=v.executor._run_sm120_phases
    report['baseline']=baseline
    for compute_block in (128,64):
        saved={}
        def split(**kwargs):
            assert kwargs['query_block_size']==128 and kwargs['ratios'][0]>.89 and kwargs['ratios'][1]>.09
            args=dict(kwargs)
            if not saved:
                for key in ('block_ids','nvfp4_counts','middle_counts','fp16_counts'):
                    tensor=kwargs[key]
                    saved[key]=tensor.repeat_interleave(2,dim=2).contiguous() if compute_block==64 and tensor.ndim>=3 else tensor
                ids=saved['block_ids'];nv=saved['nvfp4_counts']
                columns=torch.arange(ids.size(-1),device=ids.device,dtype=torch.int64)
                indices=(nv.long().unsqueeze(-1)+columns).clamp_max(ids.size(-1)-1)
                saved['int8_ids']=torch.gather(ids,-1,indices).contiguous()
            for key in ('block_ids','nvfp4_counts','middle_counts','fp16_counts'):args[key]=saved[key]
            args['query_block_size']=compute_block
            if compute_block==64:
                operands=list(args['prepared_int8_operands'])
                operands[1]=operands[1].repeat_interleave(2,dim=-1).contiguous()
                args['prepared_int8_operands']=tuple(operands)
            args['prefix_phase']=None
            nv_args=dict(args);nv_args['ratios']=(1.,0.,0.,0.)
            out_nv,lse_nv=original(**nv_args)
            int_args=dict(args);int_args.update(ratios=(0.,1.,0.,0.),block_ids=saved['int8_ids'])
            out_int,lse_int=original(**int_args)
            weight=torch.sigmoid(lse_nv-lse_int).unsqueeze(-1)
            output=(out_nv.float()*weight+out_int.float()*(1.-weight)).to(out_nv.dtype)
            return output,None
        try:
            v.executor._run_sm120_phases=split
            candidate=function();assert torch.isfinite(candidate).all()
            delta=(candidate-reference).float()
            report[str(compute_block)]={
              'bitwise_equal':torch.equal(reference,candidate),'max_abs':float(delta.abs().max()),
              'nmse':float(delta.double().square().sum()/reference.double().square().sum()),
              'timing':original_measure(function,warmup=warmup,repeats=repeats)}
        finally:v.executor._run_sm120_phases=original
    report['baseline_after']=original_measure(function,warmup=warmup,repeats=repeats)
    return baseline
v._measure=measure;v.main()
target=Path(sys.argv[sys.argv.index('--output')+1])
target.with_name(target.stem+'_split.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
