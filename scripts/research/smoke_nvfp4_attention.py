#!/usr/bin/env python3
"""E006 prerequisite: official SM120 FP4 attention on synthetic inputs.
No model, latency, accuracy preservation, or novelty claim.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import torch
import torch.nn.functional as F


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)


def metric(got,ref):
    d=got.float()-ref.float();r=ref.float()
    return {'err2':d.square().sum(dtype=torch.float64).item(),
            'ref2':r.square().sum(dtype=torch.float64).item(),
            'nmse':(d.square().sum(dtype=torch.float64)/r.square().sum(dtype=torch.float64).clamp_min(1e-30)).item(),
            'maxabs':d.abs().max().item(),'finite':bool(got.isfinite().all())}


@torch.inference_mode()
def execute(report,args):
    torch.set_num_threads(4);torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False
    mod=importlib.import_module('flashinfer.nvfp4_attention_sm120')
    report.update(torch=torch.__version__,cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                  package_versions={n:importlib.metadata.version(n) for n in ['flashinfer-python','apache-tvm-ffi','cuda-python']},
                  environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','FLASHINFER_WORKSPACE_BASE','CUDA_HOME','MAX_JOBS']})
    src=Path(mod.__file__);report['source']={'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(report,args.output)
    def run(q,k,v):
        packed=mod.nvfp4_attention_sm120_quantize_qkv(q,k,v,per_block_mean=True)
        out=mod.nvfp4_attention_sm120_fwd(*packed,sm_scale=q.shape[-1]**-.5,causal=False,
               per_block_mean=True,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=k.shape[2])
        return out[:,:,:q.shape[2],:],packed
    for label,batch,n,zero in [('aligned',1,128,False),('unaligned',1,257,False),('short_zero',1,16,True),('two_sequences',2,129,False)]:
        start=time.time();shape=(batch,2,n,128)
        q,k,v=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        if zero:q.zero_();k.zero_();v.zero_()
        ref=F.scaled_dot_product_attention(q,k,v)
        print('start',label,shape,flush=True)
        got,packed=run(q,k,v);torch.cuda.synchronize()
        row={'case':label,'shape':list(shape),'vs_bf16_sdpa':metric(got,ref),
             'packed_shapes':[list(p.shape) for p in packed],
             'seconds_with_first_compile_not_latency':time.time()-start}
        if not row['vs_bf16_sdpa']['finite']:raise RuntimeError(f'nonfinite {label}')
        if zero and torch.count_nonzero(got):raise RuntimeError('all-zero input produced nonzero output')
        if not zero and row['vs_bf16_sdpa']['nmse']>.2:raise RuntimeError(f'gross attention mismatch: {row}')
        if label=='aligned':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                replay=mod.nvfp4_attention_sm120_fwd(*packed,sm_scale=128**-.5,causal=False,
                    per_block_mean=True,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=n)
                torch.cuda.synchronize()
            row['cuda_kernels']=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
            row['repeat_equal']=torch.equal(got,replay[:,:,:n,:])
            if not row['repeat_equal']:raise RuntimeError('packed replay is not deterministic')
        if batch==2:
            q2,k2,v2=[t.clone() for t in (q,k,v)]
            for t in (q2,k2,v2):t[1]*=3
            changed,_=run(q2,k2,v2)
            row['unmodified_sequence_maxdiff']=(got[0].float()-changed[0].float()).abs().max().item()
            if row['unmodified_sequence_maxdiff']!=0:raise RuntimeError('independent batch sequence changed')
        report['cases'].append(row);save(report,args.output)
        print(json.dumps(row),flush=True)
        del q,k,v,ref,got,packed
    report['status']='complete';report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
    save(report,args.output)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    report={'experiment':'E006_native_attention_prerequisite','status':'running','cases':[],
        'limitations':['synthetic inputs only','not native projection + attention interaction experiment','not a performance benchmark','official source quantization, no model quality claim']}
    try:execute(report,args)
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();save(report,args.output);raise

if __name__=='__main__':main()
