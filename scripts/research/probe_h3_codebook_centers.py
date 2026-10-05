#!/usr/bin/env python3
"""E032 fixed weighted K16 codebook: preparation costs and three attention calls.

The old native kernel still receives a materialized, gathered correction. Timed
candidate preparation stops at C/id/Qcenter/T16; it is not an implemented shared
table consumer or an end-to-end speed measurement. K-means is a standard baseline.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import probe_h3_query_mean_k4 as common
import probe_h3_adaptive_centers as previous

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E032'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E032')
PLAN = ROOT/'research_state/06_experiments/E032_codebook_center_plan.md'
H,N,NP,D,G,K,ITERATIONS = 56,22539,22656,128,177,16,6


def squared_distances(x, centers):
    # FP32 norm + batched MM; clamp only negative roundoff, no [H,G,K,D].
    return (x.square().sum(-1,keepdim=True) + centers.square().sum(-1).unsqueeze(1)
            - 2*(x@centers.transpose(-2,-1))).clamp_min(0)


def lloyd_update(delta, centers, weights):
    import torch
    ids = squared_distances(delta,centers).argmin(-1)
    assigned = torch.nn.functional.one_hot(ids,num_classes=K).float()*weights.unsqueeze(-1)
    counts = assigned.sum(1)
    updated = assigned.transpose(-2,-1)@delta / counts.clamp_min(1).unsqueeze(-1)
    # The clamp protects the unused branch; empty centers retain their old value.
    return torch.where((counts>0).unsqueeze(-1),updated,centers)


def construct(mu,global_mu,weights):
    import torch
    delta = mu.float()-global_mu.float()
    centers = torch.zeros_like(delta[:,:1])  # Initialization only, movable in Lloyd.
    minimum = squared_distances(delta,centers).squeeze(-1)
    for _ in range(K-1):
        index = (minimum*weights).argmax(-1)  # First index wins exact ties.
        selected = delta.gather(1,index[:,None,None].expand(-1,1,delta.shape[-1]))
        centers = torch.cat((centers,selected),dim=1)
        minimum = torch.minimum(minimum,squared_distances(delta,selected).squeeze(-1))
    for _ in range(ITERATIONS):
        centers = lloyd_update(delta,centers,weights)
    actual = (global_mu.float()+centers).to(torch.bfloat16)
    # Reassign to the actual rounded centers, without a seventh M-step.
    ids = squared_distances(mu.float(),actual.float()).argmin(-1).to(torch.int32)
    return dict(centers=actual.contiguous(),ids=ids.contiguous())


def prepare(q,k,weights):
    import torch
    qpad,kc,mu,global_mu = previous.means_and_key(q,k)
    result = construct(mu,global_mu,weights)
    selected = result['centers'].gather(1,result['ids'].long().unsqueeze(-1).expand(-1,-1,D))
    result['qcenter'] = (qpad.reshape(H,G,128,D)-selected.unsqueeze(2)).reshape(H,NP,D).contiguous()
    result['t16'] = (result['centers'].float()@kc.transpose(-2,-1).float()).contiguous()
    return result


def cpu_fixture():
    """Small deterministic structural check, not a GPU arithmetic claim."""
    import torch
    # Non-symmetric deterministic values avoid synthetic farthest-distance ties
    # whose last-bit ordering differs between norm+bmm and direct subtraction.
    mu = torch.sin(torch.arange(2*21*8).reshape(2,21,8).float()*1.713 + .2).to(torch.bfloat16)
    global_mu = mu.mean(1,keepdim=True)
    weights = torch.full((1,21),128.);weights[:,-1]=11
    result = construct(mu,global_mu,weights)
    direct = (mu.float().unsqueeze(2)-result['centers'].float().unsqueeze(1)).square().sum(-1)
    assert torch.equal(result['ids'].long(),direct.argmin(-1))
    # Independent small direct-distance Lloyd implementation, including weighted M.
    delta=mu.float()-global_mu.float();centers=torch.zeros((2,1,8))
    for _ in range(K-1):
        dist=(delta.unsqueeze(2)-centers.unsqueeze(1)).square().sum(-1).amin(-1)
        chosen=(dist*weights).argmax(-1)
        centers=torch.cat((centers,delta[torch.arange(2),chosen].unsqueeze(1)),1)
    for _ in range(ITERATIONS):
        ids=(delta.unsqueeze(2)-centers.unsqueeze(1)).square().sum(-1).argmin(-1)
        updated=centers.clone()
        for h in range(2):
            for c in range(K):
                mask=ids[h]==c
                if bool(mask.any()):
                    w=weights[0,mask]
                    updated[h,c]=(delta[h,mask]*w[:,None]).sum(0)/w.sum()
        centers=updated
    oracle=(global_mu.float()+centers).to(torch.bfloat16)
    assert torch.equal(result['centers'],oracle)
    # Every nonzero center is empty when all points are zero; it must survive M.
    points=torch.zeros((1,21,8));old=torch.arange(K).float()[None,:,None].expand(1,K,8).clone()
    assert torch.equal(lloyd_update(points,old,weights),old)
    zero=construct(torch.zeros_like(mu),torch.zeros_like(global_mu),weights)
    assert torch.count_nonzero(zero['centers'])==0 and torch.count_nonzero(zero['ids'])==0
    return dict(status='complete',small_direct_distance_reference=True,actual_bf16_reassignment=True,
                empty_centers_retained=True,degenerate_tie_chooses_first=True,
                scope='CPU structural and formula fixture only; no GPU/native evidence.')


def binding(report,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    files=dict(capture=ROOT/'results/research/E018/capture_run.json',
               probe=ROOT/'results/research/E018/probe_run.json',
               oracle=ROOT/'results/research/E029/run.json',
               transfer=ROOT/'results/research/E030/run.json',
               adaptive=ROOT/'results/research/E031/run.json')
    reports={name:json.loads(path.read_text()) for name,path in files.items()}
    assert all(value['status']=='complete' for value in reports.values())
    report['sources']={name:common.record(path) for name,path in dict(runner=__file__,common=common.__file__,
        previous=previous.__file__,official=official.__file__,plan=PLAN).items()}
    report['references']={name:common.record(path) for name,path in files.items()}
    report['inputs']=[]
    for block in (0,24,48):
        budget()
        capture=next(r for r in reports['capture']['cases'] if r['block']==block)
        probe=next(r for r in reports['probe']['cases'] if r['block']==block)
        oracle=next(r['arms'] for r in reports['oracle']['layers'] if r['block']==block)
        transfer=next(r['arms'] for r in reports['transfer']['layers'] if r['block']==block)
        adaptive=next(r['outputs'] for r in reports['adaptive']['layers'] if r['block']==block)
        refs={name:oracle[name]['output'] for name in ('original_block','original_global','euclidean')}
        refs['transferred_factorable']=transfer['factorable_fp32']['output']
        refs['adaptive_range1']=adaptive['range1']['output']
        refs['bf16']=next(r['output'] for r in probe['outputs'] if r['mode']=='bf16' and r['shift']==0)
        raw=common.load(capture['artifact']);kv=common.load(probe['kv_packets'])
        for key in ('q','k','v'):
            assert raw[key].shape==(N,H,D) and raw[key].dtype==torch.bfloat16
        assert raw['valid_length']==N and raw['scale']==D**-.5
        assert kv['k_fp4'].shape==(1,H,NP,D//2) and kv['k_scale'].shape==(1,H,NP,D//16)
        assert kv['v_fp4_t'].shape==(1,H,D,NP//2) and kv['v_scale_t'].shape==(1,H,D,NP//16)
        for ref in refs.values():
            out=common.load(ref['artifact']);assert out.shape==(N,H,D) and out.dtype==torch.bfloat16
        report['inputs'].append(dict(block=block,capture=capture['artifact'],kv_packets=probe['kv_packets'],outputs=refs))


def distortion(mu,global_mu,value):
    import torch
    mu=mu.cpu().double();global_mu=global_mu.cpu().double()
    centers=value['centers'].cpu().double();ids=value['ids'].cpu().long()
    weights=torch.full((G,),128.,dtype=torch.float64);weights[-1]=11
    rows=[]
    for h in range(H):
        counts=torch.bincount(ids[h],minlength=K)
        weighted=torch.bincount(ids[h],weights=weights,minlength=K)
        total=float(((mu[h]-global_mu[h]).square()*weights[:,None]).sum())
        residual=float(((mu[h]-centers[h,ids[h]]).square()*weights[:,None]).sum())
        rows.append(dict(head=h,global_residual_energy=total,postcast_residual_energy=residual,
            residual_fraction=residual/total if total else None,groups_per_center=counts.tolist(),
            valid_queries_per_center=weighted.tolist(),empty_centers=int((counts==0).sum())))
    total=sum(row['global_residual_energy'] for row in rows)
    return dict(per_head=rows,global_residual_energy=total,
        energy_weighted_residual_fraction=sum(row['postcast_residual_energy'] for row in rows)/total if total else None)


def run(args,report,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert torch.cuda.get_device_capability()==(12,0)
    assert not DATA.exists();DATA.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,30*2**30/torch.cuda.get_device_properties(0).total_memory))
    report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),tf32=False)
    weights=torch.full((1,G),128.,device='cuda',dtype=torch.float32);weights[:,-1]=11
    module=official.get_nvfp4_attention_sm120_module();report['layers']=[]
    for source in report['inputs']:
        budget();raw=common.load(source['capture']);kv=common.load(source['kv_packets'])
        refs={name:common.load(ref['artifact']) for name,ref in source['outputs'].items()}
        q=raw['q'].transpose(0,1).contiguous().cuda();k=raw['k'].transpose(0,1).contiguous().cuda()
        qpad,kc,mu,global_mu=previous.means_and_key(q,k);del qpad,kc
        layer=dict(block=source['block'],timings={});report['layers'].append(layer)
        layer['timings']['codebook']=dict(
            resident_means=previous.measure(lambda:construct(mu,global_mu,weights),budget),
            resident_raw_qk=previous.measure(lambda:prepare(q,k,weights),budget))
        for mode in ('original_global','original_block'):
            layer['timings'][mode]=dict(resident_raw_qk=previous.measure(
                lambda:previous.prepare(q,k,None,None,mode),budget))
        common.save(args.output,report)
        # All diagnostics, full correction gathering, packing and attention are untimed.
        value=prepare(q,k,weights)
        correction=value['t16'].gather(1,value['ids'].long().unsqueeze(-1).expand(-1,-1,NP)).unsqueeze(0).contiguous()
        assert bool(value['qcenter'].isfinite().all()) and bool(correction.isfinite().all())
        gpu_kv={name:t.cuda() for name,t in kv.items()}
        qcode=torch.empty((1,H,NP,D//2),device='cuda',dtype=torch.uint8)
        qsf=torch.empty((1,H,NP,D//16),device='cuda',dtype=torch.float8_e4m3fn)
        module.scaled_fp4_quant(value['qcenter'].unsqueeze(0),qcode,qsf,1)
        kwargs=dict(sm_scale=D**-.5,causal=False,per_block_mean=True,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)
        budget();assert report['attention_calls']<3;report['attention_calls']+=1
        out=official.nvfp4_attention_sm120_fwd(qcode,gpu_kv['k_fp4'],gpu_kv['v_fp4_t'],qsf,
            gpu_kv['k_scale'],gpu_kv['v_scale_t'],correction,**kwargs)
        valid=out[0,:,:N].transpose(0,1).contiguous().cpu();assert bool(valid.isfinite().all())
        path=DATA/f'block{source["block"]}_codebook_output.pt';torch.save(valid,path)
        small=DATA/f'block{source["block"]}_centers.pt'
        torch.save(dict(centers=value['centers'].cpu(),ids=value['ids'].cpu(),mu=mu.cpu(),global_mu=global_mu.cpu()),small)
        layer['output']=dict(artifact=common.record(path),tensor=common.trecord(valid))
        layer['centers']=common.record(small)
        layer['distortion']=distortion(mu,global_mu,value)
        layer['metrics']={name:common.metrics(valid.transpose(0,1),ref.transpose(0,1)) for name,ref in refs.items()}
        layer['kernel_kwargs']={key:str(v) if key=='out_dtype' else v for key,v in kwargs.items()}
        common.save(args.output,report)
        del raw,kv,refs,q,k,mu,global_mu,value,correction,gpu_kv,qcode,qsf,out,valid
        gc.collect();torch.cuda.empty_cache()
    budget();assert report['attention_calls']==3
    report.update(status='complete',cuda_initialized=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or RD/(args.phase+'.json');assert not args.output.exists()
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=600
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E032 shared deadline')
    report=dict(experiment='E032',phase=args.phase,status='running',clusters=K,lloyd_iterations=ITERATIONS,
        attention_calls=0,actual_dit_calls=0,
        geometry=dict(heads=H,groups=G,valid_length=N,padded_length=NP,head_dim=D,tail_weight=11),
        assignment='FP32 squared Euclidean norm+bmm, clamp_min0; unweighted E, weighted M; argmin/argmax first-index ties; final actual BF16 C reassignment without M.',
        initialization='Center0 is zero delta initially and movable. Fifteen weighted-farthest real rows. No RNG, selection or adaptive iterations.',
        timing_boundary='Scope1 resident BF16 mu/global to C BF16 and ids int32. Scope2 resident contiguous BF16 HND Q/K to means/Kc, C/id/Qcenter/T16. Init, six Lloyd iterations and rounded-center reassignment included. Full correction gather, packing, KV uploads and attention excluded. Original global/block raw scope remeasured with their required means only; they produce full correction instead of T16. Scopes are not additive.',
        scope='Standard weighted k-means representation baseline. Native accuracy still gathers full correction for the unmodified kernel; no new consumer or end-to-end speed/quality claim.')
    try:
        import torch
        torch.set_num_threads(6)
        if args.phase=='check':assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        binding(report,budget)
        if args.phase=='check':
            report['fixture']=cpu_fixture()
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            check=json.loads((RD/'check.json').read_text());assert check['status']=='complete'
            for key in ('sources','references','inputs'):assert check[key]==report[key],key
            with torch.inference_mode():run(args,report,budget)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__':main()
