#!/usr/bin/env python3
"""E059 CPU source-intervention readout. The oracle is not a W4A16 model."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from summarize_h3_temporal_four_corner import load, sha

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E059'
POSITIONS=('source_teacher','next_teacher','next_shifted')


def energy(x):return float(x.square().sum())
def dot(a,b):return float((a*b).sum())
def cosine(a,b):
    denominator=(energy(a)*energy(b))**0.5
    return dot(a,b)/denominator if denominator else None
def centered(x):return x-x.mean(tuple(range(2,x.ndim)),keepdim=True)


def main():
    out=RD/'summary.json'
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();torch.set_num_threads(4)
    ep=RD/'evaluate.json';evaluation=json.loads(ep.read_text())
    assert evaluation['status']=='complete' and evaluation['complete_dit_calls']==12
    assert len(evaluation['cases'])==12
    old=json.loads((ROOT/'research_state/06_experiments/E015_h3_crossmodal_propagation_manifest.json').read_text())
    e015=json.loads((ROOT/'results/research/E015/evaluate_bf16.json').read_text())
    e057=json.loads((ROOT/'results/research/E057/evaluate_v2.json').read_text())
    index={(r['case_id'],r['position'],r['mode']):r for r in evaluation['cases']}
    assert len(index)==12
    rows=[];sources=[]
    for case in old['cases']:
        cid=case['id']
        refs={'source_teacher':case['e014_outputs']['bf16'],
              'next_teacher':next(r['artifact'] for r in e015['cases'] if r['id']==cid and r['corner']=='BB'),
              'next_shifted':next(r['artifact'] for r in e057['cases'] if r['id']==cid and r['source_arm']=='svd')}
        tensors={};case_sources={}
        for position in POSITIONS:
            records={'bf16':refs[position]}
            records.update({m:index[(cid,position,m)]['artifact'] for m in ('zero','oracle')})
            tensors[position]={m:load(rec)['velocities'] for m,rec in records.items()}
            case_sources[position]=records
        modalities={}
        for modality in ('video','audio'):
            stats={}
            for mode,transform in [('raw',lambda x:x),('channel_centered',centered)]:
                fields={};pointwise={}
                for position,values in tensors.items():
                    b=values['bf16'][modality].double()
                    n=transform(values['zero'][modality].double()-b)
                    o=transform(values['oracle'][modality].double()-b)
                    d=n-o
                    ne,oe,de=map(energy,(n,o,d));cross=2*dot(o,d)
                    assert abs(ne-oe-de-cross)<1e-9*max(1,ne)
                    fields[position]=(n,o,d)
                    pointwise[position]=dict(native_error_energy=ne,oracle_error_energy=oe,
                        removed_energy=de,twice_remaining_dot_removed=cross,
                        oracle_over_native_error_energy=oe/ne,
                        native_removed_cosine=cosine(n,d))
                n0,o0,d0=fields['source_teacher'];n1,o1,d1=fields['next_teacher']
                gn,go,gd=dot(n0,n1),dot(o0,o1),dot(d0,d1)
                mixed=dot(o0,d1)+dot(d0,o1)
                assert abs(gn-go-gd-mixed)<1e-9*max(1,abs(gn))
                pair=dict(native_adjacent_dot=gn,oracle_adjacent_dot=go,removed_adjacent_dot=gd,
                          mixed_adjacent_dot=mixed,native_cosine=cosine(n0,n1),
                          oracle_cosine=cosine(o0,o1),removed_cosine=cosine(d0,d1),
                          adjacent_dot_reduction=1-go/gn if gn>0 else None)
                nshift,oshift,_=fields['next_shifted']
                response_n=nshift-n1;response_o=oshift-o1
                stats[mode]=dict(pointwise=pointwise,teacher_pair=pair,
                    frozen_input_response=dict(native_energy=energy(response_n),oracle_energy=energy(response_o),
                         oracle_over_native=energy(response_o)/energy(response_n),
                         note='Both shifted inputs come from original native first step; not oracle rollout'))
            modalities[modality]=stats
        primary=modalities['video']['channel_centered']
        gate=(primary['teacher_pair']['adjacent_dot_reduction'] is not None
              and primary['teacher_pair']['adjacent_dot_reduction']>=0.5
              and all(primary['pointwise'][p]['oracle_over_native_error_energy']<=1.1 for p in POSITIONS[:2]))
        rows.append(dict(case_id=cid,modalities=modalities,primary_window_gate=gate))
        sources.append(dict(case_id=cid,records=case_sources))
    assert not torch.cuda.is_initialized()
    result=dict(experiment='E059',status='complete',seconds=time.monotonic()-started,
                cuda_initialized=False,new_model_forwards=0,script_sha256=sha(__file__),
                evaluation_sha256=sha(ep),cases=rows,sources=sources,
                both_window_gates=all(r['primary_window_gate'] for r in rows),
                limitation='Oracle total effect at fixed inputs; not unique W/A decomposition or deployment method')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',both_window_gates=result['both_window_gates'],
        rows=[dict(case=r['case_id'],gate=r['primary_window_gate'],
            video=r['modalities']['video']['channel_centered']) for r in rows]),indent=2))


if __name__=='__main__':main()
