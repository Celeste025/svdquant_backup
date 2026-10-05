#!/usr/bin/env python3
"""E028 CPU-only oracle center spectra in three fixed geometries.

BF16 means/subtraction are reproduced on CPU, then all spectral statistics use
FP64. This does not claim byte equality with GPU reductions. Khat is decoded
from the actual E018 packet. No basis training, quantizer change or GPU work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E028'
OLD=ROOT/'results/research/E018'
H,N,NP,D,G=56,22539,22656,128,177
RANKS=(1,2,4,8,16,32,64,128)
GEOMETRIES=('euclidean','kc_score','k4_error_score')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def record(path):
    p=Path(path).absolute();return dict(file=str(p),bytes=p.stat().st_size,sha256=sha(p))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load(ref):
    import torch
    assert record(ref['file'])==ref,ref['file']
    return torch.load(ref['file'],map_location='cpu',weights_only=True,mmap=True)


def binding(report,budget):
    import torch
    import probe_h3_query_mean_k4 as decoder
    import nvfp4_attention_fixed_scale as layout
    capture=json.loads((OLD/'capture_run.json').read_text());probe=json.loads((OLD/'probe_run.json').read_text())
    assert capture['status']==probe['status']=='complete'
    report['sources']={k:record(p) for k,p in dict(runner=__file__,decoder=decoder.__file__,layout=layout.__file__).items()}
    report['source_reports']={p:record(OLD/p) for p in ('capture_run.json','probe_run.json')}
    report['inputs']=[]
    for block in (0,24,48):
        budget();c=next(r for r in capture['cases'] if r['block']==block);p=next(r for r in probe['cases'] if r['block']==block)
        raw=load(c['artifact']);kv=load(p['kv_packets'])
        assert raw['block']==block and raw['valid_length']==N
        for name in ('q','k'):assert raw[name].shape==(N,H,D) and raw[name].dtype==torch.bfloat16
        assert kv['k_fp4'].shape==(1,H,NP,D//2) and kv['k_scale'].shape==(1,H,NP,D//16)
        report['inputs'].append(dict(block=block,capture=c['artifact'],kv_packets=p['kv_packets'],source_case=raw['source_case']))
        del raw,kv
    # Bind the decoder that was actually exercised in completed E026.
    prior_path=ROOT/'results/research/E026/run.json';prior=json.loads(prior_path.read_text())
    assert prior['status']=='complete' and prior['sources']['runner']==report['sources']['decoder']
    assert prior['sources']['scale_decoder']==report['sources']['layout']
    report['decoder_validation']=record(prior_path)


def spectrum(delta,weights,cov):
    import torch
    weighted=delta*weights.sqrt().unsqueeze(-1)
    if cov is None:
        z=weighted;psd=None
    else:
        cov=(cov+cov.T)*.5
        eigen,basis=torch.linalg.eigh(cov)
        norm=float(eigen.abs().max())
        tolerance=64*torch.finfo(torch.float64).eps*D*norm
        minimum=float(eigen.min())
        assert minimum>=-tolerance, f'PSD violation beyond roundoff: {minimum}, tolerance {tolerance}'
        psd=dict(min_eigenvalue=minimum,max_eigenvalue=float(eigen.max()),negative_count=int((eigen<0).sum()),
                 negative_tolerance=tolerance,roundoff_rule='64 * FP64 epsilon * D * max(abs(eigenvalue))')
        z=(weighted@basis)*eigen.clamp_min(0).sqrt().unsqueeze(0)
    singular=torch.linalg.svdvals(z);energy=singular.square();total=float(energy.sum())
    assert bool(energy.isfinite().all())
    direct=float(z.square().sum())
    tail={str(rank):float(energy[rank:].sum()) for rank in RANKS}
    return dict(squared_singular_values=energy.tolist(),total_energy=total,psd=psd,
        spectral_vs_frobenius_relative_difference=abs(total-direct)/direct if direct else None,
        rank0_residual_fraction=1. if total else None,
        residual_energy=tail,residual_fraction={r:e/total if total else None for r,e in tail.items()})


def aggregate(heads):
    import torch
    total=sum(row['total_energy'] for row in heads);curves={}
    for rank in RANKS:
        key=str(rank);fractions=[row['residual_fraction'][key] for row in heads if row['total_energy']>0]
        values=torch.tensor(fractions,dtype=torch.float64)
        curves[key]=dict(energy_weighted_residual_fraction=sum(row['residual_energy'][key] for row in heads)/total if total else None,
            head_quantiles=None if not fractions else dict(zip(('min','p25','median','p90','max'),
                torch.quantile(values,torch.tensor([0.,.25,.5,.9,1.],dtype=torch.float64)).tolist())))
    return dict(total_energy=total,nonzero_energy_heads=sum(row['total_energy']>0 for row in heads),rank_curves=curves)


def run(args,report,budget):
    import torch
    from probe_h3_query_mean_k4 import decode_key
    weights=torch.full((G,),128.,dtype=torch.float64);weights[-1]=N-(G-1)*128
    assert weights[-1]==11 and int(weights.sum())==N
    report['layers']=[]
    for row in report['inputs']:
        budget();raw=load(row['capture']);kv=load(row['kv_packets']);khat=decode_key(kv)[0]
        layer=dict(block=row['block'],geometries={name:dict(heads=[]) for name in GEOMETRIES})
        report['layers'].append(layer)
        for h in range(H):
            budget();q=raw['q'][:,h].contiguous();k=raw['k'][:,h].contiguous()
            assert bool(q.isfinite().all()) and bool(k.isfinite().all()) and bool(khat[h].isfinite().all())
            # Match the official BF16 tensor operations and zero-padded Q geometry,
            # without asserting CPU and GPU reduction implementation equality.
            qpad=torch.nn.functional.pad(q,(0,0,0,NP-N))
            mu=qpad.reshape(G,128,D).mean(1);global_mu=qpad.mean(0,keepdim=True)
            kc=(k-k.mean(0,keepdim=True)).double()
            delta=mu.double()-global_mu.double()  # no further weighted Q recentering
            err=khat[h,:N].double()-kc
            kc-=kc.mean(0,keepdim=True);err-=err.mean(0,keepdim=True)
            covariance={'euclidean':None,'kc_score':kc.T@kc,'k4_error_score':err.T@err}
            for name,cov in covariance.items():
                result=spectrum(delta,weights,cov);result['head']=h
                if name!='euclidean':result['softmax_scaled_total_energy']=result['total_energy']/D
                layer['geometries'][name]['heads'].append(result)
        for name in GEOMETRIES:
            layer['geometries'][name]['aggregate']=aggregate(layer['geometries'][name]['heads'])
        write(args.output,report)
        print(json.dumps(dict(block=row['block'],status='complete',heads=H)),flush=True)
        del raw,kv,khat
    budget();assert not torch.cuda.is_initialized()
    report.update(status='complete',cuda_initialized=False,completed_layers=3,completed_head_geometries=3*H*3)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or RD/(args.phase+'.json');assert not args.output.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=300
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E028 CPU deadline exceeded')
    report=dict(experiment='E028',phase=args.phase,status='running',actual_gpu_calls=0,
        geometry=dict(heads=H,valid_tokens=N,padded_tokens=NP,groups=G,head_dim=D,tail_query_weight=11,ranks=list(RANKS)),
        arithmetic='CPU BF16 pad/mean/subtraction followed by FP64 covariance/eigh/SVD; not GPU byte replay. Khat comes from actual saved E018 packets.',
        formula='Z=sqrt(w)*DeltaMu*V*sqrt(Lambda), C=V*Lambda*V^T. Euclidean C=I; Kc/E are centered across valid keys only. E=Khat-Kc before that centering.',
        interpretation='Per-head/per-sample oracle representation spectrum. Current three layers share p36/s14. No fitted/deployable basis, quantizer, speed or quality claim. The changed-Q quantization error term is unmeasured; a broad Kc spectrum alone cannot refute full Q4 benefits.')
    try:
        import torch
        torch.set_num_threads(6);binding(report,budget)
        if args.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            checked=json.loads((RD/'check.json').read_text());assert checked['status']=='complete'
            assert checked['sources']==report['sources'] and checked['inputs']==report['inputs']
            with torch.inference_mode():run(args,report,budget)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;write(args.output,report)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__':main()
