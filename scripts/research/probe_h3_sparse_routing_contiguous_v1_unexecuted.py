#!/usr/bin/env python3
"""E035 CPU diagnostic router; no attention kernel, CUDA, or product VSA claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import probe_h3_query_mean_k4 as common

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E035'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E035/router')
PLAN = ROOT / 'research_state/06_experiments/E035_sparse_router_projection_plan.md'
N, T, H, D, BS, G, PREFIX, VIDEO = 22539, 22592, 56, 128, 128, 177, 10, 167
P, MIN_KEEP = .9, 4
PREFIX_RANGES = dict(text=[0,813], audio=[813,1227], video=[1227,N], padding=[N,T])


def geometry():
    import torch
    valid = (N-torch.arange(G)*BS).clamp(0,BS).to(torch.int32)
    assert G == (N+BS-1)//BS and PREFIX == (1227+BS-1)//BS
    assert int(valid.sum()) == N and int(valid[-1]) == 11
    assert PREFIX*BS-1227 == 53 and T-N == 53
    return valid


def select_probabilities(probs, threshold=P, minimum=MIN_KEEP):
    """Smallest sorted prefix strictly exceeding threshold, then minimum count."""
    import torch
    assert bool(probs.isfinite().all()) and bool((probs >= 0).all())
    ordered, indices = torch.sort(probs,dim=-1,descending=True,stable=True)
    cdf = ordered.cumsum(-1)
    assert bool((cdf[..., -1] > threshold).all())
    crossing = (cdf <= threshold).sum(-1)+1
    selected = crossing.clamp(min=minimum,max=probs.shape[-1])
    rank = torch.arange(probs.shape[-1]).expand_as(indices)
    mask = torch.zeros_like(probs,dtype=torch.bool).scatter(-1,indices,rank < selected.unsqueeze(-1))
    upper = cdf.gather(-1,(crossing-1).unsqueeze(-1)).squeeze(-1)
    lower = cdf.gather(-1,(crossing-2).clamp(min=0).unsqueeze(-1)).squeeze(-1)
    lower = torch.where(crossing > 1,lower,torch.zeros_like(lower))
    selected_mass = (probs*mask).sum(-1)
    assert bool((selected >= minimum).all()) and bool((selected_mass > threshold).all())
    return mask,dict(crossing_count=crossing.to(torch.int32),selected_video_count=selected.to(torch.int32),
        minimum_binding=crossing < minimum,cdf_lower_margin=threshold-lower,
        cdf_upper_margin=upper-threshold,selected_video_mass=selected_mass)


def check_contract(report):
    import torch
    valid = geometry()
    for name,probs,expected in (
        ('uniform',torch.full((1,VIDEO),1/VIDEO,dtype=torch.float32),151),
        ('concentrated',torch.softmax(torch.tensor([[20.]+[0.]*(VIDEO-1)]),-1),4)):
        mask,stats = select_probabilities(probs)
        assert int(mask.sum()) == expected
        report.setdefault('selector_checks',[]).append(dict(name=name,selected=expected,
            retained_mass=float(stats['selected_video_mass'].item())))
    # Exactly representable boundary checks strict >, without prescribing tied identities.
    mask,stats = select_probabilities(torch.tensor([[.5,.25,.125,.125]]),threshold=.5,minimum=1)
    assert int(mask.sum()) == 2 and int(stats['crossing_count']) == 2
    report['geometry'] = dict(valid_length=N,total_projection_tokens=T,heads=H,head_dim=D,
        block_size=BS,blocks=G,last_block_valid_tokens=11,prefix_blocks=PREFIX,video_blocks=VIDEO,
        prefix_ranges=PREFIX_RANGES,prefix_overlap_video_tokens=53,
        query_key_valid_tokens=valid.tolist(),threshold=P,minimum_video_blocks=MIN_KEEP,
        forced_prefix_block_edges_per_head=PREFIX*G+VIDEO*PREFIX,
        forced_prefix_valid_token_pairs_per_head=(PREFIX*BS)*N+(N-PREFIX*BS)*(PREFIX*BS))


def pool_pair(raw,case,name,budget):
    """One chunked pass hashes full saved operands, pools valid rows, and compares Q/K."""
    import torch
    values = {arm:raw[arm][name] for arm in ('bf16','native')}
    pools = {arm:torch.empty((H,G,D),dtype=torch.float32) for arm in values}
    hashes = {arm:hashlib.sha256() for arm in values}
    totals = {key:torch.zeros(H,dtype=torch.float64) for key in ('error_energy','reference_energy','value_energy','dot')}
    maxima = torch.zeros(H,dtype=torch.float64)
    for arm,value in values.items():
        assert value.shape == (T,H,D) and value.dtype == torch.bfloat16
        receipt = case['tensors'][arm][name]
        assert receipt['shape'] == list(value.shape) and receipt['dtype'] == str(value.dtype)
    for lo in range(0,T,BS*8):
        budget(); hi = min(T,lo+BS*8); length = max(0,min(N,hi)-lo)
        chunks = {}
        for arm,value in values.items():
            chunk = value[lo:hi].contiguous()
            assert bool(chunk.isfinite().all())
            hashes[arm].update(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
            if length:
                current = chunk[:length].float(); chunks[arm] = current
                whole,tail = divmod(length,BS); group = lo//BS
                if whole:
                    pools[arm][:,group:group+whole] = current[:whole*BS].reshape(whole,BS,H,D).mean(1).transpose(0,1)
                if tail:pools[arm][:,group+whole] = current[whole*BS:].mean(0)
        if length:
            ref,value = chunks['bf16'].double(),chunks['native'].double()
            error = value-ref
            totals['error_energy'] += error.square().sum((0,2))
            totals['reference_energy'] += ref.square().sum((0,2))
            totals['value_energy'] += value.square().sum((0,2))
            totals['dot'] += (value*ref).sum((0,2))
            maxima = torch.maximum(maxima,error.abs().amax((0,2)))
    records = {}
    for arm in values:
        assert hashes[arm].hexdigest() == case['tensors'][arm][name]['sha256'], (case['block'],arm,name)
        assert bool(pools[arm].isfinite().all())
        records[arm] = dict(shape=[T,H,D],dtype='torch.bfloat16',sha256=hashes[arm].hexdigest(),finite=True)
    def finish(energies,max_abs,elements):
        return dict(**energies,nmse=energies['error_energy']/energies['reference_energy'] if energies['reference_energy'] else None,
            cosine=energies['dot']/(energies['reference_energy']*energies['value_energy'])**.5
                if energies['reference_energy']*energies['value_energy'] else None,
            max_abs=max_abs,rms_error=(energies['error_energy']/elements)**.5,elements=elements)
    rows = [dict(head=h,**finish({key:float(value[h]) for key,value in totals.items()},float(maxima[h]),N*D)) for h in range(H)]
    pooled = finish({key:float(value.sum()) for key,value in totals.items()},float(maxima.max()),N*H*D)
    return pools,dict(pooled=pooled,per_head=rows,scope='Native versus BF16, valid post-norm/RoPE operands only; no video-quality claim.'),records


def route(q,k,valid):
    import torch
    scores = (q@k.transpose(-2,-1))*(D**-.5)
    assert scores.dtype == torch.float32 and bool(scores.isfinite().all())
    probs = scores[:,PREFIX:,PREFIX:].softmax(-1)
    video_mask,stats = select_probabilities(probs)
    mask = torch.ones((H,G,G),dtype=torch.bool)
    mask[:,PREFIX:,PREFIX:] = video_mask
    payload = dict(scores=scores,mask=mask,query_valid_tokens=valid,key_valid_tokens=valid,
        selected_total_count=mask.sum(-1).to(torch.int32),
        selected_valid_keys=(mask*valid.view(1,1,G)).sum(-1).to(torch.int32),
        tail_selected=mask[:,:,-1].clone(),video_query_stats=stats)
    assert bool(mask[:,:PREFIX,:].all()) and bool(mask[:,:,:PREFIX].all())
    assert torch.equal(payload['selected_total_count'][:,PREFIX:],stats['selected_video_count']+PREFIX)
    assert bool((payload['selected_valid_keys'] <= N).all())
    return payload,probs


def distribution(value):
    value = value.double().reshape(-1)
    return dict(min=float(value.min()),median=float(value.median()),mean=float(value.mean()),max=float(value.max()),sum=float(value.sum()))


def route_summary(payload,valid):
    import torch
    mask = payload['mask']; stats = payload['video_query_stats']
    def summarize(h):
        m,c,v = mask[h],payload['selected_total_count'][h],payload['selected_valid_keys'][h]
        return dict(active_physical_blocks=int(m.sum()),video_video_blocks=int(m[...,PREFIX:,PREFIX:].sum()),
            valid_token_pairs=int((v*valid).sum()),selected_valid_keys=distribution(v),
            physical_blocks_per_query=distribution(c),video_blocks_per_video_query=distribution(stats['selected_video_count'][h]),
            tail_selected_query_blocks=int(payload['tail_selected'][h].sum()),
            minimum_binding_rows=int(stats['minimum_binding'][h].sum()),
            cdf_lower_margin=distribution(stats['cdf_lower_margin'][h]),
            cdf_upper_margin=distribution(stats['cdf_upper_margin'][h]))
    # int64 product avoids overflow in token-pair totals.
    payload['selected_valid_keys'] = payload['selected_valid_keys'].to(torch.int64)
    valid = valid.to(torch.int64)
    rows = [dict(head=h,**summarize(h)) for h in range(H)]
    total = summarize(slice(None))
    total['forced_prefix_physical_blocks'] = H*(PREFIX*G+VIDEO*PREFIX)
    total['forced_prefix_valid_token_pairs'] = H*((PREFIX*BS)*N+(N-PREFIX*BS)*(PREFIX*BS))
    total['dense_valid_token_pairs'] = H*N*N
    return dict(total=total,per_head=rows,
        row_statistics='Every row retained in artifact tensors; prefix query rows are dense and margin statistics apply only to the 167 video query rows.')


def comparison(bf16,native,native_probs,valid):
    import torch
    a,b = bf16['mask'],native['mask']
    rows = dict(intersection=(a&b).sum(-1).to(torch.int32),added=((~a)&b).sum(-1).to(torch.int32),
        removed=(a&(~b)).sum(-1).to(torch.int32),
        physical_budget_delta=native['selected_total_count']-bf16['selected_total_count'],
        valid_key_budget_delta=native['selected_valid_keys']-bf16['selected_valid_keys'],
        teacher_mask_mass_on_native_video_proxy=(native_probs*a[:,PREFIX:,PREFIX:]).sum(-1),
        native_mask_mass_on_native_video_proxy=(native_probs*b[:,PREFIX:,PREFIX:]).sum(-1))
    def summarize(h):
        added,removed,delta = rows['added'][h],rows['removed'][h],rows['physical_budget_delta'][h]
        changed = added+removed > 0
        return dict(intersection_blocks=int(rows['intersection'][h].sum()),added_blocks=int(added.sum()),removed_blocks=int(removed.sum()),
            changed_rows=int(changed.sum()),same_budget_changed_rows=int((changed&(delta==0)).sum()),
            budget_increased_rows=int((delta>0).sum()),budget_decreased_rows=int((delta<0).sum()),
            physical_budget_delta=distribution(delta),valid_key_budget_delta=distribution(rows['valid_key_budget_delta'][h]),
            valid_token_pair_delta=int((rows['valid_key_budget_delta'][h]*valid.to(torch.int64)).sum()),
            teacher_mass_on_native_proxy=distribution(rows['teacher_mask_mass_on_native_video_proxy'][h]),
            native_mass_on_native_proxy=distribution(rows['native_mask_mass_on_native_video_proxy'][h]))
    return rows,dict(total=summarize(slice(None)),per_head=[dict(head=h,**summarize(h)) for h in range(H)],
        boundary='Both masks are scored using the same native pooled video logits. These masses are proxy statistics, not full attention probability mass; BF16-derived mask is an intervention, not an oracle. No sparse attention or timing measured.')


def tensor_records(tree):
    import torch
    return {key:common.trecord(value) if isinstance(value,torch.Tensor) else tensor_records(value) for key,value in tree.items()}


def run(args,report,budget):
    import torch
    capture = json.loads(args.capture.read_text())
    assert capture['status'] == 'complete' and len(capture['cases']) == 3
    assert sorted(c['block'] for c in capture['cases']) == [0,24,48]
    checked = json.loads((RD/'router_check.json').read_text())
    assert checked['status'] == 'complete' and checked['sources'] == report['sources']
    report.update(capture_report=common.record(args.capture),cases=[],source_capture_status='complete',
        binding_scope='Capture artifact records retained from the completed capture report; full saved Q/K bytes independently hashed once while pooling. No rehash of raw x, kwargs, V, model weights, or whole capture file.')
    assert not args.data.exists();args.data.mkdir(parents=True)
    valid = geometry()
    for case in capture['cases']:
        budget()
        assert (case['valid_length'],case['total_tokens'],case['heads'],case['head_dim']) == (N,T,H,D)
        assert case['scale'] == D**-.5 and case['cu_seqlens'] == [0,N,T]
        path = Path(case['capture']['file']);assert path.stat().st_size == case['capture']['bytes']
        raw = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert raw['block'] == case['block'] and raw['valid_length'] == N and raw['total_tokens'] == T
        assert raw['scale'] == D**-.5 and raw['prefix'] == PREFIX_RANGES
        q,qerror,qrecords = pool_pair(raw,case,'q',budget)
        k,kerror,krecords = pool_pair(raw,case,'k',budget)
        budget()
        payloads,probabilities = {},{}
        for arm in ('bf16','native'):payloads[arm],probabilities[arm] = route(q[arm],k[arm],valid)
        row = dict(block=case['block'],capture=case['capture'],q_native_vs_bf16=qerror,k_native_vs_bf16=kerror,
            verified_operand_records=dict(q=qrecords,k=krecords),arms={})
        summaries = {arm:route_summary(payloads[arm],valid) for arm in payloads}
        paired,paired_summary = comparison(payloads['bf16'],payloads['native'],probabilities['native'],valid)
        payloads['native']['comparison_with_bf16'] = paired
        row['mask_comparison'] = paired_summary
        for arm,payload in payloads.items():
            budget();out = args.data/f'block{case["block"]}_{arm}.pt';torch.save(payload,out)
            row['arms'][arm] = dict(artifact=common.record(out),tensors=tensor_records(payload),summary=summaries[arm])
        report['cases'].append(row);common.save(args.output,report)
        print(json.dumps(dict(block=case['block'],active_blocks={arm:summaries[arm]['total']['active_physical_blocks'] for arm in summaries},
            changed_rows=paired_summary['total']['changed_rows'])),flush=True)
        del raw,q,k,payloads,probabilities,paired
    budget();assert not torch.cuda.is_initialized()
    report.update(status='complete',cuda_initialized=False,saved_router_artifacts=6,
        executed_attention_calls=0,executed_dit_calls=0,native_projection_calls_in_this_process=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--capture',type=Path,default=RD/'capture_run.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--deadline-unix',type=float)
    args=parser.parse_args();args.output=args.output or RD/('router_'+args.phase+'.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not args.output.exists(), 'Preserve existing result/failed attempt'
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0 < args.deadline_unix-started <= 180
    def budget():
        if args.deadline_unix and time.time() >= args.deadline_unix:raise TimeoutError('E035 CPU router deadline')
    report=dict(experiment='E035',component='router',phase=args.phase,status='running',
        sources=dict(runner=common.record(__file__),common=common.record(common.__file__),plan=common.record(PLAN)),
        scope='Diagnostic adapter: contiguous128, prefix-overlap blocks dense, FP32 valid-token mean/QK/video-only softmax, strict cumulative probability >0.9 with minimum4. Not the H3 VSA 3D or BSA model product. BF16 and native labels denote projection provenance, not different router arithmetic. No attention execution or latency claim.')
    try:
        import torch
        torch.set_num_threads(6)
        check_contract(report)
        if args.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            with torch.inference_mode():run(args,report,budget)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
