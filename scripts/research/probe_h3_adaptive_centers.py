#!/usr/bin/env python3
"""E031 fixed GPU basis-construction baselines and nine native attention calls.

Preparation timings exclude Q/K/V quantization and attention. The raw-QK scope
stops at Qcenter+T17, before A@T_basis creates full correction. Baseline products
are Qcenter+full correction, so these are representation-preparation costs,
not interchangeable end-to-end latencies or a predicted consumer speedup.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import probe_h3_query_mean_k4 as common

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E031'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E031')
H,N,NP,D,G,R=56,22539,22656,128,177,16
METHODS=('full_svd','range0','range1')
SEED=20261031


def binding(report,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    capture=json.loads((ROOT/'results/research/E018/capture_run.json').read_text())
    probe=json.loads((ROOT/'results/research/E018/probe_run.json').read_text())
    r29=json.loads((ROOT/'results/research/E029/run.json').read_text())
    r30=json.loads((ROOT/'results/research/E030/run.json').read_text())
    assert all(r['status']=='complete' for r in (capture,probe,r29,r30))
    report['sources']={k:common.record(p) for k,p in dict(runner=__file__,common=common.__file__,official=official.__file__).items()}
    report['references']={name:common.record(ROOT/('results/research/'+name)) for name in
        ('E018/capture_run.json','E018/probe_run.json','E029/run.json','E030/run.json')}
    report['inputs']=[]
    for block in (0,24,48):
        budget();c=next(r for r in capture['cases'] if r['block']==block);p=next(r for r in probe['cases'] if r['block']==block)
        a29=next(r['arms'] for r in r29['layers'] if r['block']==block);a30=next(r['arms'] for r in r30['layers'] if r['block']==block)
        refs={name:a29[name]['output'] for name in ('original_block','original_global','euclidean')}
        refs['transferred_factorable']=a30['factorable_fp32']['output']
        refs['bf16']=next(r['output'] for r in p['outputs'] if r['mode']=='bf16' and r['shift']==0)
        raw=common.load(c['artifact']);kv=common.load(p['kv_packets'])
        for key in ('q','k','v'):assert raw[key].shape==(N,H,D) and raw[key].dtype==torch.bfloat16
        assert kv['k_fp4'].shape==(1,H,NP,D//2)
        for ref in refs.values():assert common.load(ref['artifact']).shape==(N,H,D)
        report['inputs'].append(dict(block=block,capture=c['artifact'],kv_packets=p['kv_packets'],outputs=refs))
    generator=torch.Generator(device='cpu').manual_seed(SEED)
    omega=torch.randn((H,G,R),generator=generator,dtype=torch.float32)
    report['omega']=dict(seed=SEED,tensor=common.trecord(omega),distribution='CPU torch.randn FP32; independent head matrices in ONE frozen batch; same draw across layers/methods/repeats.')
    return omega


def means_and_key(q,k):
    import torch
    # Resident inputs are contiguous [H,N,D]; preserve official BF16 operations.
    kc=k-k.mean(-2,keepdim=True)
    qpad=torch.nn.functional.pad(q,(0,0,0,NP-N))
    kc=torch.nn.functional.pad(kc,(0,0,0,NP-N))
    mu=qpad.reshape(H,G,128,D).mean(2);global_mu=qpad.mean(-2,keepdim=True)
    return qpad,kc,mu,global_mu


def construct(mu,global_mu,omega,sqrtw,method):
    import torch
    delta=mu.float()-global_mu.float();x=delta*sqrtw
    if method=='full_svd':
        # Default CUDA driver, never the approximate gesvda driver.
        _,_,vh=torch.linalg.svd(x,full_matrices=False)
        b=vh[:,:R].contiguous()
    else:
        q0,_=torch.linalg.qr(x.transpose(-2,-1)@omega,mode='reduced')
        if method=='range1':q0,_=torch.linalg.qr(x.transpose(-2,-1)@(x@q0),mode='reduced')
        b=q0.transpose(-2,-1).contiguous()
    a=delta@b.transpose(-2,-1)
    center=global_mu.float()+a@b
    return dict(a=a,b=b,center32=center)


def prepare(q,k,omega,sqrtw,method):
    import torch
    if method in ('original_global','original_block'):
        # Strong baselines compute only their required Q mean.
        kc=torch.nn.functional.pad(k-k.mean(-2,keepdim=True),(0,0,0,NP-N))
        qpad=torch.nn.functional.pad(q,(0,0,0,NP-N))
        c=qpad.mean(-2,keepdim=True) if method=='original_global' else qpad.reshape(H,G,128,D).mean(2)
        centered=(qpad-c) if method=='original_global' else (qpad.reshape(H,G,128,D)-c.unsqueeze(2)).reshape(H,NP,D)
        correction=c.float()@kc.transpose(-2,-1).float()
        return dict(qcenter=centered.contiguous(),correction=correction.contiguous())
    qpad,kc,mu,global_mu=means_and_key(q,k)
    result=construct(mu,global_mu,omega,sqrtw,method)
    result['qcenter']=(qpad.float().reshape(H,G,128,D)-result['center32'].unsqueeze(2)).reshape(H,NP,D).to(torch.bfloat16).contiguous()
    result['t17']=torch.cat((global_mu.float(),result['b']),dim=1)@kc.transpose(-2,-1).float()
    return result


def tensor_sizes(value):
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size()) for k,v in value.items()}


def measure(fn,budget):
    import torch
    for _ in range(3):
        budget();value=fn();torch.cuda.synchronize();del value
    gc.collect();torch.cuda.synchronize()
    before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats();wall=[];cuda=[];products=None
    for _ in range(10):
        budget();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();t0=time.perf_counter();start.record()
        value=fn();end.record();end.synchronize();torch.cuda.synchronize()
        wall.append((time.perf_counter()-t0)*1000);cuda.append(start.elapsed_time(end))
        if products is None:products=tensor_sizes(value)
        del value  # No prior repeat's products remain live for the next repeat.
    peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    def stats(x):return dict(repeats=x,median=statistics.median(x),min=min(x),max=max(x))
    return dict(warmup=3,repeats=10,wall_ms=stats(wall),cuda_event_ms=stats(cuda),products=products,
        memory_bytes=dict(baseline=before,peak=peak,increment={k:peak[k]-before[k] for k in before}),
        lifetime='Input fixtures are resident throughout. Each returned product set is released outside its timer before the next invocation. Baseline is after warmup; reserved baseline includes warm allocator cache.')


def projection_stats(mu,global_mu,value):
    import torch
    weights=torch.full((G,1),128.,dtype=torch.float64);weights[-1]=11
    delta=mu.cpu().double()-global_mu.cpu().double()
    a=value['a'].cpu().double();b=value['b'].cpu().double();center=value['center32'].cpu().double()
    rows=[]
    for h in range(H):
        total=float((delta[h].square()*weights).sum())
        row=dict(head=h,total_energy=total,
            projection_energy=float(((delta[h]-a[h]@b[h]).square()*weights).sum()),
            center32_residual_energy=float(((mu[h].cpu().double()-center[h]).square()*weights).sum()),
            basis_orthogonality_max_abs=float((b[h]@b[h].T-torch.eye(R,dtype=torch.float64)).abs().max()))
        row['projection_fraction']=row['projection_energy']/total if total else None
        row['center32_residual_fraction']=row['center32_residual_energy']/total if total else None
        rows.append(row)
    total=sum(r['total_energy'] for r in rows)
    return dict(per_head=rows,total_energy=total,
        energy_weighted_projection_fraction=sum(r['projection_energy'] for r in rows)/total if total else None,
        energy_weighted_center32_residual_fraction=sum(r['center32_residual_energy'] for r in rows)/total if total else None)


def run(args,report,omega_cpu,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert torch.cuda.get_device_capability()==(12,0)
    assert not DATA.exists();DATA.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,30*2**30/torch.cuda.get_device_properties(0).total_memory))
    report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),tf32=False,svd_driver=None)
    path=DATA/'omega.pt';torch.save(omega_cpu,path);report['omega']['artifact']=common.record(path)
    omega=omega_cpu.cuda();w=torch.full((1,G,1),128.,device='cuda',dtype=torch.float32);w[:,-1]=11;sqrtw=w.sqrt()
    module=official.get_nvfp4_attention_sm120_module();report['layers']=[]
    for source in report['inputs']:
        budget();raw=common.load(source['capture']);kv=common.load(source['kv_packets'])
        refs={name:common.load(ref['artifact']) for name,ref in source['outputs'].items()}
        q=raw['q'].transpose(0,1).contiguous().cuda();k=raw['k'].transpose(0,1).contiguous().cuda()
        qpad,kc,mu,global_mu=means_and_key(q,k);del qpad,kc
        layer=dict(block=source['block'],timings={},outputs={});report['layers'].append(layer)
        for method in METHODS:
            layer['timings'][method]=dict(
                resident_means=measure(lambda:construct(mu,global_mu,omega,sqrtw,method),budget),
                resident_raw_qk=measure(lambda:prepare(q,k,omega,sqrtw,method),budget))
            common.save(args.output,report)
        for method in ('original_global','original_block'):
            layer['timings'][method]=dict(resident_raw_qk=measure(lambda:prepare(q,k,omega,sqrtw,method),budget))
        # Packing and native attention are outside EVERY preparation timer.
        gpu_kv={name:t.cuda() for name,t in kv.items()}
        for method in METHODS:
            budget();value=prepare(q,k,omega,sqrtw,method)
            correction=(value['t17'][:,:1]+value['a']@value['t17'][:,1:]).unsqueeze(0).contiguous()
            qcode=torch.empty((1,H,NP,D//2),device='cuda',dtype=torch.uint8)
            qsf=torch.empty((1,H,NP,D//16),device='cuda',dtype=torch.float8_e4m3fn)
            module.scaled_fp4_quant(value['qcenter'].unsqueeze(0),qcode,qsf,1)
            assert bool(value['qcenter'].isfinite().all()) and bool(correction.isfinite().all())
            kwargs=dict(sm_scale=D**-.5,causal=False,per_block_mean=True,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)
            assert report['attention_calls']<9;report['attention_calls']+=1
            out=official.nvfp4_attention_sm120_fwd(qcode,gpu_kv['k_fp4'],gpu_kv['v_fp4_t'],qsf,gpu_kv['k_scale'],gpu_kv['v_scale_t'],correction,**kwargs)
            valid=out[0,:,:N].transpose(0,1).contiguous().cpu();assert bool(valid.isfinite().all())
            path=DATA/f'block{source["block"]}_{method}_output.pt';torch.save(valid,path)
            factors=DATA/f'block{source["block"]}_{method}_factors.pt';torch.save({key:value[key].cpu() for key in ('a','b','center32')},factors)
            layer['outputs'][method]=dict(output=dict(artifact=common.record(path),tensor=common.trecord(valid)),factors=common.record(factors),
                projection=projection_stats(mu,global_mu,value),
                metrics={name:common.metrics(valid.transpose(0,1),ref.transpose(0,1)) for name,ref in refs.items()},
                kernel_kwargs={key:str(v) if key=='out_dtype' else v for key,v in kwargs.items()})
            common.save(args.output,report);del value,correction,qcode,qsf,out,valid
        del raw,kv,refs,q,k,mu,global_mu,gpu_kv
        gc.collect();torch.cuda.empty_cache()
    budget();assert report['attention_calls']==9
    report.update(status='complete',cuda_initialized=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or RD/(args.phase+'.json');assert not args.output.exists()
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=600
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E031 shared deadline')
    report=dict(experiment='E031',phase=args.phase,status='running',rank=R,attention_calls=0,actual_dit_calls=0,
        methods=list(METHODS),geometry=dict(heads=H,groups=G,valid_length=N,padded_length=NP,head_dim=D,tail_weight=11),
        timing_boundary='Scope1 resident BF16 mu/global→FP32 delta/X/basis/A/center32. Scope2 resident contiguous BF16 HND Q/K→official-order means/Kc→basis/A/center32/Qcenter/T17. No full correction, quantization, attention or KV packing in candidate timers. Baseline scope2 returns Qcenter+full original correction, not T17. Scopes are not additive or comparable to previous whole-pipeline timings.',
        scope='Fixed rank16 GPU full SVD or standard randomized range finder, zero/one power iteration; fixed independent per-head random matrices shared across cases and repeats. No new algorithm or consumer, deployment speed, or quality claim.')
    try:
        import torch
        torch.set_num_threads(6)
        if args.phase=='check':assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        omega=binding(report,budget)
        if args.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            checked=json.loads((RD/'check.json').read_text());assert checked['status']=='complete'
            assert checked['sources']==report['sources'] and checked['inputs']==report['inputs'] and checked['omega']==report['omega']
            with torch.inference_mode():run(args,report,omega,budget)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__':main()
