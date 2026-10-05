#!/usr/bin/env python3
"""E026: two attention calls on E018 block0 packets, replacing only correction.

The oracle does not quantize the actual official BF16 query mean any further.
It retains FP4 K loss. Q/K/V packets and P/PV implementation stay fixed; actual
softmax probabilities and their quantized codes necessarily change.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E026'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E026')
OLD = ROOT/'results/research/E018'
N, NP, H, D = 22539, 22656, 56, 128
SCALE = D**-.5


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def record(path):
    p=Path(path).absolute();return dict(file=str(p),bytes=p.stat().st_size,sha256=sha(p))


def save(path,r):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');temp.replace(path)


def trecord(t):
    import torch
    t=t.detach().cpu().contiguous()
    return dict(shape=list(t.shape),dtype=str(t.dtype),finite=bool(t.isfinite().all()),
                sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def load(ref):
    import torch
    assert record(ref['file'])==ref,ref['file']
    return torch.load(ref['file'],map_location='cpu',weights_only=True,mmap=True)


def inputs(report):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    import nvfp4_attention_fixed_scale as decoder
    reports={name:json.loads((OLD/name).read_text()) for name in ('capture_run.json','probe_run.json')}
    assert all(r['status']=='complete' for r in reports.values())
    capture=next(c for c in reports['capture_run.json']['cases'] if c['block']==0)
    probe=next(c for c in reports['probe_run.json']['cases'] if c['block']==0)
    refs={r['mode']:r for r in probe['outputs'] if r['shift']==0}
    raw=load(capture['artifact']);qp=load(refs['block_mean']['q_packet_artifact']);kv=load(probe['kv_packets'])
    for key in ('q','k','v','router_output'):
        assert raw[key].shape==(N,H,D) and raw[key].dtype==torch.bfloat16
    assert raw['scale']==SCALE and raw['valid_length']==N and raw['block']==0
    assert qp['q_fp4'].shape==(1,H,NP,D//2) and qp['q_scale'].shape==(1,H,NP,D//16)
    assert kv['k_fp4'].shape==(1,H,NP,D//2) and kv['k_scale'].shape==(1,H,NP,D//16)
    assert kv['v_fp4_t'].shape==(1,H,D,NP//2) and kv['v_scale_t'].shape==(1,H,D,NP//16)
    outputs={name:load(row['output']['artifact']) for name,row in refs.items()}
    assert all(t.shape==(N,H,D) and t.dtype==torch.bfloat16 for t in outputs.values())
    report.update(sources={name:record(path) for name,path in dict(runner=__file__,official=official.__file__,
        scale_decoder=decoder.__file__).items()},
        source_reports={name:record(OLD/name) for name in reports},
        inputs=dict(capture=capture['artifact'],q_packets=refs['block_mean']['q_packet_artifact'],kv_packets=probe['kv_packets'],
                    references={name:row['output'] for name,row in refs.items()}),
        geometry=dict(block=0,valid_length=N,padded_length=NP,heads=H,head_dim=D,query_groups=NP//128,softmax_scale=SCALE),
        source_case=raw['source_case'],source_tensor_shapes={k:list(v.shape) for k,v in {**qp,**kv}.items()})
    return raw,qp,kv,outputs


def decode_key(kv):
    """Same nibble/table decode as E019; reuse its verified SF/row layout helpers."""
    import torch
    from nvfp4_attention_fixed_scale import logical_scales,k_permutation
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
    inv=torch.argsort(k_permutation(NP,'cpu'))
    out=torch.empty((1,H,NP,D),dtype=torch.float32)
    for h in range(H):
        code=kv['k_fp4'][0,h]
        nib=torch.stack((code&15,code>>4),dim=-1).reshape(NP,D)
        sf=logical_scales(kv['k_scale'][0,h]).float()
        physical=levels[nib.long()]*sf.repeat_interleave(16,dim=-1)
        out[0,h]=physical.index_select(0,inv)
    return out


def metrics(value,reference,weights=None):
    """FP64 per-head and pooled statistics; heads are first, all valid rows included."""
    import torch
    totals=dict(error_energy=0.,reference_energy=0.,value_energy=0.,dot=0.,elements=0.)
    rows=[]
    def finish(s):
        den=(s['value_energy']*s['reference_energy'])**.5
        return dict(**s,nmse=s['error_energy']/s['reference_energy'] if s['reference_energy'] else None,
                    cosine=s['dot']/den if den else None,
                    rms_error=(s['error_energy']/s['elements'])**.5)
    for h in range(value.shape[0]):
        a=value[h].double();b=reference[h].double()
        w=1. if weights is None else weights.double().unsqueeze(-1)
        num=a.numel() if weights is None else float(weights.sum())*a.shape[-1]
        s=dict(error_energy=float(((a-b).square()*w).sum()),reference_energy=float((b.square()*w).sum()),
               value_energy=float((a.square()*w).sum()),dot=float((a*b*w).sum()),elements=float(num))
        rows.append(dict(head=h,**finish(s)))
        for k in totals:totals[k]+=s[k]
    return dict(pooled=finish(totals),per_head=rows)


def run(args,report,values):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','Root allocates physical GPU1'
    assert args.deadline_unix and 0<args.deadline_unix-time.time()<=600
    def budget():
        assert time.time()<args.deadline_unix,'E026 deadline exceeded'
        assert report['attention_calls']<=2
    budget();assert torch.cuda.get_device_capability()==(12,0)
    torch.cuda.set_per_process_memory_fraction(min(1.,20*2**30/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw,qp,kv,refs=values
    khat=decode_key(kv);assert bool(khat.isfinite().all())
    report['decoded_key']=trecord(khat)
    seg=tuple(raw[k].transpose(0,1).unsqueeze(0).contiguous().cuda() for k in ('q','k','v'))
    processed=official._preprocess_qkv(*seg,per_block_mean=True)
    original=processed[-1];del processed
    # Match official padded BF16 reduction, not CPU mean or ideal-real mean.
    mu=official._pad_seq_len_to_128(seg[0]).reshape(1,H,NP//128,128,D).mean(dim=3)
    oracle=torch.matmul(mu.float(),khat.cuda().transpose(-2,-1)).contiguous()
    assert original.shape==oracle.shape==(1,H,NP//128,NP)
    assert bool(original.isfinite().all()) and bool(oracle.isfinite().all())
    report['mean']=trecord(mu)
    report['corrections']={name:trecord(c) for name,c in (('original',original),('mean_times_k4',oracle))}
    # There is no tensor-wise global multiplier in this attention packet ABI.
    gpu={k:v.cuda() for k,v in {**qp,**kv}.items()}
    fixed=tuple(gpu[k] for k in ('q_fp4','k_fp4','v_fp4_t','q_scale','k_scale','v_scale_t'))
    kwargs=dict(sm_scale=SCALE,causal=False,per_block_mean=True,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)
    report['kernel_kwargs']={k:str(v) if k=='out_dtype' else v for k,v in kwargs.items()}
    assert not DATA.exists();DATA.mkdir(parents=True)
    report['outputs']={}
    for name,corr in (('original',original),('mean_times_k4',oracle)):
        budget();report['attention_calls']+=1
        out=official.nvfp4_attention_sm120_fwd(*fixed,corr,**kwargs)
        valid=out[0,:,:N,:].transpose(0,1).contiguous().cpu();assert bool(valid.isfinite().all())
        path=DATA/(name+'_output.pt');torch.save(valid,path)
        report['outputs'][name]=dict(artifact=record(path),tensor=trecord(valid),
            versus_bf16=metrics(valid.transpose(0,1),refs['bf16'].transpose(0,1)),
            versus_saved_block=metrics(valid.transpose(0,1),refs['block_mean'].transpose(0,1)))
        if name=='original':original_out=valid
        else:report['outputs'][name]['versus_current_original']=metrics(valid.transpose(0,1),original_out.transpose(0,1))
        save(args.output,report);del out
    report['existing_references']={name:metrics(refs[name].transpose(0,1),refs['bf16'].transpose(0,1))
                                    for name in ('block_mean','global_mean')}
    # Remove a row constant over actual keys. Weight the last Q group by its actual valid rows, derived below.
    old=original[0,:,:,:N].cpu();new=oracle[0,:,:,:N].cpu()
    # Center in FP64 so CPU BF16 reductions cannot masquerade as K4 loss.
    weights=(N-torch.arange(NP//128)*128).clamp(0,128)
    centered_old=old.double();centered_old-=centered_old.mean(-1,keepdim=True)
    centered_new=new.double();centered_new-=centered_new.mean(-1,keepdim=True)
    report['centered_correction']=metrics(centered_new,centered_old,weights)
    report['centered_correction']['softmax_scaled_rms_error']=report['centered_correction']['pooled']['rms_error']*SCALE
    report['centered_correction']['semantics']='FP64 row-center across N valid keys; Q-block statistics weighted by actual query count. Pre-softmax correction, before the one kernel softmax scale.'
    report.update(status='complete',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  complete_dit_calls=0,cuda_initialized=True)
    budget();torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or RD/(args.phase+'.json');assert not args.output.exists()
    report=dict(experiment='E026',phase=args.phase,status='running',attention_calls=0,complete_dit_calls=0,
        scope='Single existing H3 p36/s14/block0. Infinite-extra-precision mean oracle retains official BF16 mean, FP4 K, all six packets and original softmax/P/PV implementation. Actual P values change. No speed, quality or fusion claim.')
    started=time.time()
    try:
        import torch
        torch.set_num_threads(6)
        if args.phase=='check':assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        values=inputs(report)
        if args.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            check=json.loads((RD/'check.json').read_text());assert check['status']=='complete'
            assert check['sources']==report['sources'] and check['inputs']==report['inputs']
            with torch.inference_mode():run(args,report,values)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;save(args.output,report)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__':main()
