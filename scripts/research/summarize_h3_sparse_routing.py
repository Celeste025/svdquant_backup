#!/usr/bin/env python3
"""E035 independent CPU reduction of six small score/mask artifacts.

No model, original QKV, router import, attention execution, or latency estimate.
FP32 softmax/CDF replays the recorded selector contract; aggregates use int64
or FP64. Heads/tiles belong to one state, not independent quality samples.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E035'
N, H, G, BS, PFX, V = 22539, 56, 211, 128, 11, 200


def record(path):
    path = Path(path).resolve(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): digest.update(chunk)
    return dict(file=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def geometry(torch):
    """Independent small integer construction, in (tile t,h,w)/(inner t,h,w) order."""
    groups = [list(range(lo,min(lo+128,813))) for lo in range(0,813,128)]
    groups += [list(range(813+lo,813+min(lo+128,414))) for lo in range(0,414,128)]
    for t0 in range(0,37,4):
        for h0 in range(0,18,4):
            for w0 in range(0,32,8):
                groups.append([1227+(t*18+h)*32+w for t in range(t0,min(t0+4,37))
                               for h in range(h0,min(h0+4,18)) for w in range(w0,min(w0+8,32))])
    valid=torch.tensor([len(g) for g in groups],dtype=torch.int64)
    partition=torch.tensor([i for group in groups for i in group],dtype=torch.int64)
    nonpad=torch.tensor([j*128+i for j,group in enumerate(groups) for i in range(len(group))])
    untile=torch.empty(N,dtype=torch.int64);untile[partition]=nonpad
    assert valid.shape==(G,) and int(valid.sum())==N and int(valid[:PFX].sum())==1227
    assert torch.equal(partition.sort().values,torch.arange(N))
    return dict(valid=valid,partition=partition,nonpad=nonpad,untile=untile)


def verify_tensors(torch, payload, receipts):
    for key,value in payload.items():
        if isinstance(value,torch.Tensor):
            row=receipts[key]
            assert list(value.shape)==row['shape'] and str(value.dtype)==row['dtype'],key
            digest=hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            assert digest==row['sha256'],key
            assert bool(value.isfinite().all()),key
        elif isinstance(value,dict): verify_tensors(torch,value,receipts[key])


def selector(torch,scores):
    probs=scores[:,PFX:,PFX:].softmax(-1)
    ordered,indices=probs.sort(dim=-1,descending=True,stable=True)
    cdf=ordered.cumsum(-1)
    # Original BSA keep rule, implemented differently from the capture router.
    keep=torch.ones_like(cdf,dtype=torch.bool)
    keep[...,1:]=cdf[...,:-1]<.9
    keep[...,:4]=True
    chosen=torch.zeros_like(keep).scatter(-1,indices,keep)
    mask=torch.ones((H,G,G),dtype=torch.bool);mask[:,PFX:,PFX:]=chosen
    crossing=(cdf>=.9).to(torch.int64).argmax(-1)+1
    assert bool((cdf[...,-1]>=.9).all())
    upper=cdf.gather(-1,(crossing-1).unsqueeze(-1)).squeeze(-1)
    lower=cdf.gather(-1,(crossing-2).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    lower=torch.where(crossing>1,lower,torch.zeros_like(lower))
    stats=dict(crossing_count=crossing,selected_video_count=keep.sum(-1),minimum_binding=crossing<4,
               cdf_lower_margin=.9-lower,cdf_upper_margin=upper-.9,
               selected_video_mass=(probs*chosen).sum(-1))
    return mask,probs,stats


def distribution(torch,value):
    x=value.double().reshape(-1)
    return dict(min=float(x.min()),median=float(x.median()),mean=float(x.mean()),max=float(x.max()),sum=float(x.sum()))


def arm_statistics(torch,mask,stats,valid,head=None):
    m=mask if head is None else mask[head:head+1]
    st=stats if head is None else {k:v[head:head+1] for k,v in stats.items()}
    count=m.sum(-1);keys=(m*valid.view(1,1,G)).sum(-1)
    pairs=(keys*valid.view(1,G)).sum()
    prefix_edges=m.shape[0]*(PFX*G+V*PFX)
    prefix_pairs=m.shape[0]*(1227*N+(N-1227)*1227)
    physical=int(m.sum())
    return dict(physical_block_edges=physical,video_video_block_edges=int(m[:,PFX:,PFX:].sum()),
        valid_token_pairs=int(pairs),padded_tile_pair_capacity=physical*BS*BS,
        dense_physical_edge_fraction=physical/(m.shape[0]*G*G),
        dense_valid_token_pair_fraction=int(pairs)/(m.shape[0]*N*N),
        forced_prefix_block_edges=prefix_edges,forced_prefix_valid_token_pairs=prefix_pairs,
        video_query_total_blocks=distribution(torch,count[:,PFX:]),
        video_query_valid_keys=distribution(torch,keys[:,PFX:]),
        selected_video_blocks=distribution(torch,st['selected_video_count']),
        partial_key_edges=int(m[:,:,valid<BS].sum()),minimum_binding_rows=int(st['minimum_binding'].sum()),
        cdf_lower_margin=distribution(torch,st['cdf_lower_margin']),
        cdf_upper_margin=distribution(torch,st['cdf_upper_margin']))


def paired_statistics(torch,a,b,valid,native_probs,head=None):
    if head is not None:
        a=a[head:head+1];b=b[head:head+1];native_probs=native_probs[head:head+1]
    added=(~a)&b;removed=a&(~b)
    ap,rp=added.sum(-1),removed.sum(-1);delta=ap-rp
    av=(added*valid.view(1,1,G)).sum(-1);rv=(removed*valid.view(1,1,G)).sum(-1)
    dv=av-rv;changed=(ap+rp)>0;same=(delta==0)&changed
    slots=torch.minimum(ap,rp)
    assert int(added[:,:PFX,:].sum()+added[:,:,:PFX].sum())==0
    assert int(removed[:,:PFX,:].sum()+removed[:,:,:PFX].sum())==0
    video_changed=changed[:,PFX:]
    nqueries=a.shape[0]*V
    physical_base=int(a.sum());token_base=int(((a*valid.view(1,1,G)).sum(-1)*valid).sum())
    net=int(delta.sum());netpairs=int((dv*valid).sum())
    out=dict(added_block_edges=int(ap.sum()),removed_block_edges=int(rp.sum()),net_block_edges=net,
        gross_budget_increase_block_edges=int(delta.clamp_min(0).sum()),
        gross_budget_decrease_block_edges=int((-delta).clamp_min(0).sum()),
        exchanged_edge_slots_across_all_rows=int(slots.sum()),
        added_valid_token_pairs=int((av*valid).sum()),removed_valid_token_pairs=int((rv*valid).sum()),
        net_valid_token_pairs=netpairs,net_physical_edge_fraction_of_bf16=net/physical_base,
        net_valid_token_pair_fraction_of_bf16=netpairs/token_base,
        changed_video_rows=int(video_changed.sum()),video_rows=nqueries,
        physical_budget_increased_rows=int((delta[:,PFX:]>0).sum()),
        physical_budget_decreased_rows=int((delta[:,PFX:]<0).sum()),
        physical_budget_equal_rows=int((delta[:,PFX:]==0).sum()),
        same_physical_budget_changed_rows=int(same.sum()),
        same_physical_budget_exchanged_edge_slots=int((ap*same).sum()),
        same_physical_budget_changed_valid_token_pairs=int((dv*same*valid).sum()),
        logical_key_budget_increased_rows=int((dv[:,PFX:]>0).sum()),
        logical_key_budget_decreased_rows=int((dv[:,PFX:]<0).sum()),
        logical_key_budget_equal_rows=int((dv[:,PFX:]==0).sum()),
        video_physical_count_delta=distribution(torch,delta[:,PFX:]),
        video_valid_key_count_delta=distribution(torch,dv[:,PFX:]),
        video_mask_jaccard=float((a[:,PFX:,PFX:]&b[:,PFX:,PFX:]).sum())/int((a[:,PFX:,PFX:]|b[:,PFX:,PFX:]).sum()),
        teacher_mask_mass_on_native_proxy=distribution(torch,(native_probs*a[:,PFX:,PFX:]).sum(-1)),
        native_mask_mass_on_native_proxy=distribution(torch,(native_probs*b[:,PFX:,PFX:]).sum(-1)))
    assert out['added_block_edges']==out['gross_budget_increase_block_edges']+out['exchanged_edge_slots_across_all_rows']
    assert out['removed_block_edges']==out['gross_budget_decrease_block_edges']+out['exchanged_edge_slots_across_all_rows']
    return out


def reduce(args,report,budget):
    import torch
    torch.set_num_threads(6)
    source=json.loads(args.report.read_text())
    assert source['status']=='complete' and source['saved_router_artifacts']==6
    assert source['executed_attention_calls']==source['executed_dit_calls']==0
    assert source['cuda_initialized'] is False and source['geometry']['threshold']==.9
    assert [c['block'] for c in source['cases']]==[0,24,48]
    assert record(source['sources']['runner']['file'])==source['sources']['runner']
    assert record(source['capture_report']['file'])==source['capture_report']
    capture=json.loads(Path(source['capture_report']['file']).read_text())
    assert capture['status']=='complete' and capture['actual_scaled_mm_calls']==3 and capture['actual_sdpa_calls']==102
    geo=geometry(torch);valid=geo['valid']
    report.update(router_report=record(args.report),router_source=source['sources']['runner'],capture_report=source['capture_report'],
        geometry=dict(blocks=G,block_size=BS,prefix_blocks=PFX,video_blocks=V,valid_length=N,
                      tile_shape=[4,4,8],video_grid=[37,18,32],valid_tokens_per_tile=valid.tolist()),
        capture_counts=dict(complete_dit_calls=capture['complete_dit_calls'],native_projections=3,sdpa_calls=102),cases=[])
    for case in source['cases']:
        budget();payloads={};stats={};probs={};masks={}
        for arm in ('bf16','native'):
            ref=case['arms'][arm]['artifact'];assert record(ref['file'])==ref
            payload=torch.load(ref['file'],map_location='cpu',weights_only=True,mmap=True)
            verify_tensors(torch,payload,case['arms'][arm]['tensors'])
            assert all(torch.equal(payload['geometry'][k],v) for k,v in geo.items())
            assert torch.equal(payload['query_valid_tokens'],valid) and torch.equal(payload['key_valid_tokens'],valid)
            assert payload['scores'].dtype==torch.float32 and payload['scores'].shape==(H,G,G)
            mask,prob,st=selector(torch,payload['scores'])
            assert torch.equal(mask,payload['mask'])
            assert torch.equal(mask.sum(-1),payload['selected_total_count'])
            assert torch.equal((mask*valid.view(1,1,G)).sum(-1),payload['selected_valid_keys'])
            for key,v in st.items():
                original=payload['video_query_stats'][key]
                assert torch.allclose(v.float(),original.float(),atol=2e-7,rtol=1e-6),key
            payloads[arm]=payload;masks[arm]=mask;probs[arm]=prob;stats[arm]=st
        total={a:arm_statistics(torch,masks[a],stats[a],valid) for a in masks}
        paired=paired_statistics(torch,masks['bf16'],masks['native'],valid,probs['native'])
        heads=[dict(head=h,arms={a:arm_statistics(torch,masks[a],stats[a],valid,h) for a in masks},
                    paired=paired_statistics(torch,masks['bf16'],masks['native'],valid,probs['native'],h)) for h in range(H)]
        net=torch.tensor([r['paired']['net_block_edges'] for r in heads])
        pairnet=torch.tensor([r['paired']['net_valid_token_pairs'] for r in heads])
        for arm in total:
            prior=case['arms'][arm]['summary']['total']
            assert total[arm]['physical_block_edges']==prior['active_physical_blocks']
            assert total[arm]['valid_token_pairs']==prior['valid_token_pairs']
        old=case['mask_comparison']['total']
        assert paired['added_block_edges']==old['added_blocks'] and paired['removed_block_edges']==old['removed_blocks']
        assert paired['net_valid_token_pairs']==old['valid_token_pair_delta']
        report['cases'].append(dict(block=case['block'],artifacts={a:case['arms'][a]['artifact'] for a in masks},
            verified_selector_masks=True,verified_integer_geometry=True,arms=total,paired=paired,per_head=heads,
            head_distribution=dict(physical_net=distribution(torch,net),physical_increased=int((net>0).sum()),
                physical_decreased=int((net<0).sum()),physical_unchanged=int((net==0).sum()),
                token_pair_net=distribution(torch,pairnet),token_pair_increased=int((pairnet>0).sum()),
                token_pair_decreased=int((pairnet<0).sum()),token_pair_unchanged=int((pairnet==0).sum())),
            qk_metrics_reported_not_recomputed={key:case[key] for key in ('q_native_vs_bf16','k_native_vs_bf16')}))
        save(args.output,report)
        print(json.dumps(dict(block=case['block'],net_blocks=paired['net_block_edges'],
            net_token_pairs=paired['net_valid_token_pairs'],same_budget_changed_rows=paired['same_physical_budget_changed_rows'])),flush=True)
    budget();assert not torch.cuda.is_initialized()
    report.update(status='complete',cuda_initialized=False,artifacts_reduced=6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=RD/'router_run.json')
    p.add_argument('--output',type=Path,default=RD/'independent_summary.json')
    args=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not args.output.exists(),'Preserve prior summary or failure'
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>180:raise TimeoutError('CPU reducer 180-second budget')
    report=dict(experiment='E035',component='independent_router_reduction',status='running',source=record(__file__),
        methods='Rebuild integer 3D tile partition; replay FP32 softmax and original BSA preceding-CDF<0.9 rule plus minimum4 from six saved scores; int64 edge/token costs and FP64 descriptive aggregates. No QKV/model reads.',
        limitations=['One already observed teacher state and three layers; heads/rows are correlated.',
            'Block-proxy probability over variable-size tiles is not full token-attention mass.',
            'Physical edges, padded tile-pair capacity and valid token pairs are workload descriptors, not measured latency or FLOPs.',
            'Same-count changed rows are descriptive subsets, not an executed budget-matched consumer intervention.',
            'No sparse consumer, generation, perceptual-quality or method-novelty conclusion.'])
    try:reduce(args,report,budget)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.monotonic()-started;save(args.output,report)


if __name__=='__main__':main()
