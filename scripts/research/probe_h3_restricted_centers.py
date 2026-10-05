#!/usr/bin/env python3
"""E029: rank16, sample-adaptive query centers; fifteen fixed-packet forwards.

The same rounded BF16 center enters Q-center and center@Kc correction. KV is
never requantized. Full correction remains materialized: no deployment claim.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import probe_h3_query_mean_k4 as common

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E029'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E029')
OLD=ROOT/'results/research/E018'
H,N,NP,D,G,RANK=56,22539,22656,128,177,16
GEOMETRIES=('euclidean','kc_score','k4_error_score')
ARMS=('original_block','original_global',*GEOMETRIES)


def binding(report,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    import nvfp4_attention_fixed_scale as layout
    captured=json.loads((OLD/'capture_run.json').read_text());probed=json.loads((OLD/'probe_run.json').read_text())
    assert captured['status']==probed['status']=='complete'
    report['sources']={k:common.record(p) for k,p in dict(runner=__file__,common=common.__file__,layout=layout.__file__,official=official.__file__).items()}
    report['source_reports']={k:common.record(OLD/k) for k in ('capture_run.json','probe_run.json')}
    prior=ROOT/'results/research/E028/run.json';assert json.loads(prior.read_text())['status']=='complete'
    report['spectrum_reference']=common.record(prior);report['inputs']=[]
    for block in (0,24,48):
        budget();capture=next(r for r in captured['cases'] if r['block']==block)
        probe=next(r for r in probed['cases'] if r['block']==block)
        refs={r['mode']:r['output'] for r in probe['outputs'] if r['shift']==0}
        raw=common.load(capture['artifact']);kv=common.load(probe['kv_packets'])
        assert raw['block']==block and raw['valid_length']==N and raw['scale']==D**-.5
        for name in ('q','k','v'):assert raw[name].shape==(N,H,D) and raw[name].dtype==torch.bfloat16
        assert kv['k_fp4'].shape==(1,H,NP,D//2) and kv['k_scale'].shape==(1,H,NP,D//16)
        for ref in refs.values():
            value=common.load(ref['artifact']);assert value.shape==(N,H,D) and value.dtype==torch.bfloat16
        report['inputs'].append(dict(block=block,capture=capture['artifact'],kv_packets=probe['kv_packets'],references=refs,source_case=raw['source_case']))


def residual(delta,approx,weights,factor):
    a=(delta-approx)*weights.sqrt().unsqueeze(-1)
    b=delta*weights.sqrt().unsqueeze(-1)
    if factor is not None:a=a@factor;b=b@factor
    error=float(a.square().sum());total=float(b.square().sum())
    return dict(error_energy=error,total_energy=total,fraction=error/total if total else None)


def project(mu,global_mu,kc,khat,geometry,weights,budget):
    import torch
    aa=[];bb=[];deltas=[];factors=[];heads=[]
    for h in range(H):
        budget();delta=mu[h].double()-global_mu[h].double();weighted=delta*weights.sqrt().unsqueeze(-1)
        factor=None;psd=None
        if geometry!='euclidean':
            key=kc[h,:N].double()
            if geometry=='k4_error_score':key=khat[h,:N].double()-key
            key-=key.mean(0,keepdim=True)
            cov=key.T@key;cov=(cov+cov.T)*.5
            eigen,v=torch.linalg.eigh(cov);tol=64*torch.finfo(torch.float64).eps*D*float(eigen.abs().max())
            assert float(eigen.min())>=-tol,'PSD violation beyond roundoff'
            psd=dict(min_eigenvalue=float(eigen.min()),negative_count=int((eigen<0).sum()),negative_tolerance=tol)
            factor=v*eigen.clamp_min(0).sqrt().unsqueeze(0)
        z=weighted if factor is None else weighted@factor
        u,s,_=torch.linalg.svd(z,full_matrices=False);u=u[:,:RANK]
        a=u/weights.sqrt().unsqueeze(-1);b=u.T@weighted;delta_r=a@b
        energy=s.square();total=float(energy.sum());tail=float(energy[RANK:].sum())
        heads.append(dict(head=h,psd=psd,theoretical_tail_energy=tail,theoretical_total_energy=total,
            theoretical_tail_fraction=tail/total if total else None,
            ideal_projection=residual(delta,delta_r,weights,factor)))
        aa.append(a);bb.append(b);deltas.append(delta_r);factors.append(factor)
    return torch.stack(deltas),torch.stack(aa),torch.stack(bb),factors,heads


def run(args,report,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
    assert torch.cuda.get_device_capability()==(12,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,30*2**30/torch.cuda.get_device_properties(0).total_memory))
    module=official.get_nvfp4_attention_sm120_module()
    assert not DATA.exists();DATA.mkdir(parents=True)
    weights=torch.full((G,),128.,dtype=torch.float64);weights[-1]=11
    report['layers']=[]
    for source in report['inputs']:
        budget();raw=common.load(source['capture']);kv=common.load(source['kv_packets'])
        refs={name:common.load(r['artifact']) for name,r in source['references'].items()}
        khat=common.decode_key(kv)[0]
        segment=tuple(raw[key].transpose(0,1).unsqueeze(0).contiguous().cuda() for key in ('q','k','v'))
        qpad=official._pad_seq_len_to_128(segment[0])
        mu=qpad.reshape(1,H,G,128,D).mean(3)
        global_mu=qpad.mean(-2,keepdim=True)
        original=official._preprocess_qkv(*segment,per_block_mean=True)
        kc_gpu=original[1];kc=kc_gpu[0].cpu();mu_cpu=mu[0].cpu();global_cpu=global_mu[0].cpu()
        gpu_kv={name:t.cuda() for name,t in kv.items()}
        outdir=DATA/f'block{source["block"]}';outdir.mkdir()
        layer=dict(block=source['block'],arms={},actual_gpu_mu=common.trecord(mu),actual_gpu_global_mu=common.trecord(global_mu),
                   actual_gpu_kc=common.trecord(kc_gpu),center_source='Actual official GPU BF16 reductions; geometry subsequently CPU FP64.')
        report['layers'].append(layer)
        centers=dict(block_mean=mu_cpu,global_mean=global_cpu,k_mean=segment[1].mean(-2,keepdim=True)[0].cpu(),weights=weights,restricted={})
        for arm in ARMS:
            budget();projection=None
            if arm=='original_block':
                q_centered,_,_,correction=original;per_block=True
            elif arm=='original_global':
                q_centered,_,_,correction=official._preprocess_qkv(*segment,per_block_mean=False);per_block=False
            else:
                delta_r,a,b,factors,projection=project(mu_cpu,global_cpu,kc,khat,arm,weights,budget)
                # Reconstruct in FP32, then round ONCE. Both consumers use this BF16 c.
                c_float=global_mu.float()+delta_r.unsqueeze(0).float().cuda()
                c=c_float.to(torch.bfloat16)
                q_centered=(qpad.reshape(1,H,G,128,D)-c.unsqueeze(3)).reshape(1,H,NP,D).contiguous()
                correction=(c.float()@kc_gpu.transpose(-2,-1).float()).contiguous();per_block=True
                c32=c_float[0].cpu();c16=c[0].cpu()
                for h in range(H):
                    delta=mu_cpu[h].double()-global_cpu[h].double()
                    projection[h]['fp32_reconstruction']=residual(delta,c32[h].double()-global_cpu[h].double(),weights,factors[h])
                    projection[h]['post_bf16_center']=residual(delta,c16[h].double()-global_cpu[h].double(),weights,factors[h])
                    projection[h]['bf16_rounding_max_abs']=float((c16[h].float()-c32[h]).abs().max())
                centers['restricted'][arm]=dict(a=a,b=b,center_fp32=c32,center_bf16=c16)
                del delta_r,a,b,factors,c_float,c
            assert bool(q_centered.isfinite().all()) and bool(correction.isfinite().all())
            q_codes=torch.empty((1,H,NP,D//2),device='cuda',dtype=torch.uint8)
            q_sf=torch.empty((1,H,NP,D//16),device='cuda',dtype=torch.float8_e4m3fn)
            module.scaled_fp4_quant(q_centered,q_codes,q_sf,1)
            packed=(q_codes,gpu_kv['k_fp4'],gpu_kv['v_fp4_t'],q_sf,gpu_kv['k_scale'],gpu_kv['v_scale_t'],correction)
            kwargs=dict(sm_scale=D**-.5,causal=False,per_block_mean=per_block,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)
            assert report['attention_calls']<15;report['attention_calls']+=1
            output=official.nvfp4_attention_sm120_fwd(*packed,**kwargs)
            valid=output[0,:,:N,:].transpose(0,1).contiguous().cpu();assert bool(valid.isfinite().all())
            path=outdir/(arm+'_output.pt');torch.save(valid,path)
            row=dict(output=dict(artifact=common.record(path),tensor=common.trecord(valid)),projection=projection,
                kernel_kwargs={k:str(v) if k=='out_dtype' else v for k,v in kwargs.items()},
                metrics={name:common.metrics(valid.transpose(0,1),ref.transpose(0,1)) for name,ref in refs.items()})
            if projection is not None:
                total=sum(p['theoretical_total_energy'] for p in projection)
                row['energy_weighted_projection']=dict(total_energy=total,
                    theoretical_tail_fraction=sum(p['theoretical_tail_energy'] for p in projection)/total if total else None,
                    **{key:sum(p[key]['error_energy'] for p in projection)/total if total else None
                       for key in ('ideal_projection','fp32_reconstruction','post_bf16_center')})
            layer['arms'][arm]=row;common.save(args.output,report)
            print(json.dumps(dict(block=source['block'],arm=arm,status='complete',calls=report['attention_calls'])),flush=True)
            del packed,q_centered,correction,q_codes,q_sf,output,valid
        path=outdir/'centers_and_factors.pt';torch.save(centers,path);layer['centers']=common.record(path)
        common.save(args.output,report)
        del original,segment,qpad,kc_gpu,kc,mu,global_mu,mu_cpu,global_cpu,gpu_kv,khat,raw,kv,refs,centers
    budget();assert report['attention_calls']==15
    report.update(status='complete',cuda_initialized=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--gpu',type=int,default=0);p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or RD/(args.phase+'.json');assert not args.output.exists()
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=600
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E029 shared deadline')
    report=dict(experiment='E029',phase=args.phase,status='running',rank=RANK,attention_calls=0,actual_dit_calls=0,
        geometry=dict(heads=H,valid_length=N,padded_length=NP,groups=G,head_dim=D,tail_weight=11),
        arm_order=list(ARMS),scope='Current-sample rank16 center oracle; same rounded BF16 center in Q-c and c@Kc. Original KV packets and P/PV kernel unchanged. Full correction materialized; no speed/memory/quality claim. BF16 rounding can break exact rank. P values change naturally.')
    try:
        import torch
        torch.set_num_threads(6)
        if args.phase=='check':assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        binding(report,budget)
        if args.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            check=json.loads((RD/'check.json').read_text());assert check['status']=='complete'
            assert check['sources']==report['sources'] and check['inputs']==report['inputs']
            with torch.inference_mode():run(args,report,budget)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__':main()
