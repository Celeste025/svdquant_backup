#!/usr/bin/env python3
"""E019 score-domain CPU math audit. No native-accumulator bitwise claim."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import summarize_h3_plain_baseline as util

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E019'
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E019/math_audit')
ARMS = ('no_p_fp64','e2_fp32sf','e2_e4sf','pnq_represented_lse','pnq_true_lse_negative','ceil_log2','sage_tile_sf1')
require = util.require


def e2_rne(value):
    boundaries = value.new_tensor([.25,.75,1.25,1.75,2.5,3.5,5.])
    levels = value.new_tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
    code = torch.bucketize(value.contiguous(), boundaries)
    at_tie = (value == boundaries[code.clamp(max=6)]) & (code < 7)
    code += (at_tie & (code % 2 == 1)).long()
    return code.to(torch.uint8), levels[code]


def rounded_scales(value):
    common = value.float()
    # All inputs are nonnegative and <=448, so native satfinite adds no clamp.
    require(bool(torch.isfinite(common).all()) and bool(((common>=0)&(common<=448)).all()), 'SF range')
    return common.double(), common.to(torch.float8_e4m3fn).double()


def reverse_running(tile_max):
    return tile_max.flip(-1).cummax(-1).values.flip(-1)


def events(unrounded, represented, valid_groups, group_mass):
    valid = valid_groups.expand_as(unrounded)
    masks = dict(normal=represented>=2**-6, subnormal=(represented>0)&(represented<2**-6),
                 zero=represented==0, reciprocal_floor=unrounded<1e-8)
    return {name:dict(count=((mask&valid).sum(-1)).tolist(),
                      true_softmax_mass=(group_mass*(mask&valid)).sum(-1).tolist()) for name,mask in masks.items()}


def contract_weights(z, split):
    """Construct seven full/split operators on the identical, fixed score rows."""
    rows,n = z.shape
    require(split%128==0 and 0<split<n and bool(torch.isfinite(z).all()), 'Split/scores invalid')
    tiles=(n+127)//128; padding=tiles*128-n; st=split//128
    zp=torch.nn.functional.pad(z,(0,padding),value=-float('inf')).reshape(rows,tiles,8,16)
    ag=zp.amax(-1); tile_max=ag.amax(-1); global_max=z.amax(-1)
    valid_groups=torch.isfinite(ag[:1]).reshape(1,-1)
    numerator=6*torch.exp(zp-torch.where(torch.isfinite(ag),ag,torch.zeros_like(ag))[...,None])
    codes,q=e2_rne(numerator); q=q.reshape(rows,tiles,8,16)
    true_unnormalized=torch.exp(z-global_max[:,None]); true_z=true_unnormalized.sum(-1)
    true_weights=true_unnormalized/true_z[:,None]
    group_mass=torch.nn.functional.pad(true_weights,(0,padding)).reshape(rows,-1,16).sum(-1)
    references=dict(full=reverse_running(tile_max),
        split=torch.cat((reverse_running(tile_max[:,:st]),reverse_running(tile_max[:,st:])),dim=1))
    raw={}; diagnostics={}
    for path,m in references.items():
        group_reference=m[:,:,None]
        common_scale=448*torch.exp(ag-group_reference)
        s32,s8=rounded_scales(common_scale)
        restore=torch.exp(m-global_max[:,None])[:,:,None,None]/2688
        raw['e2_fp32sf',path]=(q*s32[...,None]*restore).flatten(1)[:,:n]
        raw['e2_e4sf',path]=(q*s8[...,None]*restore).flatten(1)[:,:n]
        diagnostics[path]=dict(native_sf=events(common_scale.flatten(1),s8.flatten(1),valid_groups,group_mass))
        # An identical canonical per-group value is rescaled with ldexp, never
        # independently exp-recomputed for different integer reference paths.
        bg=torch.ceil(torch.where(torch.isfinite(ag),ag,torch.zeros_like(ag))/math.log(2)).long()
        bm=torch.ceil(m/math.log(2)).long(); br=torch.ceil(global_max/math.log(2)).long()
        canonical=torch.where(torch.isfinite(ag),448*torch.exp(ag-bg.double()*math.log(2)),torch.zeros_like(ag))
        cs=torch.ldexp(canonical,bg-bm[:,:,None]); _,cs8=rounded_scales(cs)
        relative=torch.ldexp(q*cs8[...,None],(bm-br[:,None])[:,:,None,None])
        correction=torch.exp(br.double()*math.log(2)-global_max)[:,None,None,None]/2688
        raw['ceil_log2',path]=(relative*correction).flatten(1)[:,:n]
        diagnostics[path]['ceil_sf']=events(cs.flatten(1),cs8.flatten(1),valid_groups,group_mass)
        inner=448*torch.exp(ag-tile_max[:,:,None]); _,inner8=rounded_scales(inner)
        # Paper SF1 is cast AFTER division by 2688. It only restores the
        # shared E2 codes; FP32 SF1 cannot feed back and change the codes.
        sf1=(torch.exp(tile_max-m)/2688).float().double()
        sage_restore=sf1*torch.exp(m-global_max[:,None])
        raw['sage_tile_sf1',path]=(q*inner8[...,None]*sage_restore[:,:,None,None]).flatten(1)[:,:n]
        diagnostics[path]['sage_inner_sf']=events(inner.flatten(1),inner8.flatten(1),valid_groups,group_mass)
        diagnostics[path]['sage_fp32_sf1_zero']=dict(count=(sf1==0).sum(-1).tolist(),
            true_softmax_mass=(group_mass.reshape(rows,tiles,8).sum(-1)*(sf1==0)).sum(-1).tolist())
    weights={}
    # Independent FP64 local softmax+natural-LSE merge baseline.
    lse=torch.stack([torch.logsumexp(part,dim=-1) for part in (z[:,:split],z[:,split:])],dim=1)
    merge=torch.softmax(lse,dim=1)
    weights['no_p_fp64']=dict(full=true_weights,split=torch.cat((torch.softmax(z[:,:split],-1)*merge[:,0,None],
                                                               torch.softmax(z[:,split:],-1)*merge[:,1,None]),dim=1))
    for arm in ('e2_fp32sf','e2_e4sf','ceil_log2','sage_tile_sf1'):
        weights[arm]={p:raw[arm,p]/true_z[:,None] for p in references}
    wfull,wsplit=raw['e2_e4sf','full'],raw['e2_e4sf','split']
    pnq_full=wfull/wfull.sum(-1,keepdim=True)
    weights['pnq_represented_lse']=dict(full=pnq_full,split=wsplit/wsplit.sum(-1,keepdim=True))
    zl=true_unnormalized[:,:split].sum(-1)/true_z
    zr=true_unnormalized[:,split:].sum(-1)/true_z
    weights['pnq_true_lse_negative']=dict(full=pnq_full,split=torch.cat((
        wsplit[:,:split]/wsplit[:,:split].sum(-1,keepdim=True)*zl[:,None],
        wsplit[:,split:]/wsplit[:,split:].sum(-1,keepdim=True)*zr[:,None]),dim=1))
    return weights,dict(shared_e2_codes=codes.flatten(1)[:,:n],true_weights=true_weights,
        true_local_lse=lse,true_merge_fractions=merge,reference_max=references),diagnostics


def output_stats(full, split, reference):
    e0=full-reference; delta=split-full; e1=split-reference
    energy=lambda x:x.square().sum(-1)
    ref2,e02,e12,d2=map(energy,(reference,e0,e1,delta)); cross=2*(e0*delta).sum(-1)
    residual=e12-e02-d2-cross
    require(bool((residual.abs()<=1e-10*(e12+e02+d2+cross.abs())+1e-25).all()),'Output energy identity')
    return dict(reference_energy=ref2,full_error_energy=e02,split_error_energy=e12,delta_energy=d2,
        cross2=cross,cross2_positive=cross.clamp(min=0),cross2_negative=cross.clamp(max=0),
        identity_residual=residual,max_abs_delta=delta.abs().amax(-1))


def metric_json(values):
    result={k:v.tolist() for k,v in values.items()}
    result['aggregate']={k:float(v.sum()) for k,v in values.items() if k!='max_abs_delta'}
    a=result['aggregate']; a.update(full_nmse=util.ratio(a['full_error_energy'],a['reference_energy']),
        split_nmse=util.ratio(a['split_error_energy'],a['reference_energy']),
        delta_nmse=util.ratio(a['delta_energy'],a['reference_energy']),
        delta_over_full_error=util.ratio(a['delta_energy'],a['full_error_energy']))
    return result


def apply_values(weights,v):
    return weights.reshape(len(v),-1,v.shape[1]).bmm(v).flatten(0,1)


def load_record(files, record, tensor_records=None):
    files.record(record); value=util.load_tensor_file(record['file'])
    if tensor_records is not None: require(util.signature(value)==tensor_records,'Stored tensor SHA mismatch')
    return value


def audit_native(files, case, z, v, no_p_output, split):
    native={k:load_record(files,row['artifact'],row['sample_tensors']) for k,row in case['native'].items()}
    require(set(native)=={'bf16_full','fp16_full','fp16_left','fp16_right'},'Native set differs')
    for name,val in native.items():
        require(tuple(val['output'].shape)==(8,9,128) and tuple(val['lse'].shape)==(8,9),'Native sample geometry')
        require(val['output'].dtype==(torch.bfloat16 if name=='bf16_full' else torch.float16) and val['lse'].dtype==torch.float32,'Native dtype')
    left,right=native['fp16_left'],native['fp16_right']
    alpha=torch.softmax(torch.stack((left['lse'].double(),right['lse'].double()),-1),-1)
    merged=(alpha[...,0,None]*left['output'].double()+alpha[...,1,None]*right['output'].double()).flatten(0,1)
    full=native['fp16_full']['output'].double().flatten(0,1)
    stats=metric_json(output_stats(full,merged,no_p_output))
    # Pure output-format floor: math partial outputs cast FP16 before merging.
    local_lse=torch.stack((torch.logsumexp(z[:,:split],-1),torch.logsumexp(z[:,split:],-1)),1)
    math_alpha=torch.softmax(local_lse,1)
    local_left=apply_values(torch.softmax(z[:,:split],-1),v[:,:split])
    local_right=apply_values(torch.softmax(z[:,split:],-1),v[:,split:])
    format_merge=math_alpha[:,0,None]*local_left.half().double()+math_alpha[:,1,None]*local_right.half().double()
    return (dict(native_fp16=stats,format_only_fp16=metric_json(output_stats(no_p_output.half().double(),format_merge,no_p_output)),
        bf16_to_fp16_full_format_comparison=metric_json(output_stats(native['bf16_full']['output'].double().flatten(0,1),
            full,no_p_output)),
        native_lse_vs_math_max_abs={k:float((native[k]['lse'].double().reshape(-1)-
            (torch.logsumexp(z,-1) if 'full' in k else local_lse[:,0 if 'left' in k else 1])).abs().max()) for k in native},
        caveat='Native differences include QK/PV FP32 accumulation, approximate exp, LSE FP32 and output FP16 rounding; not attributed solely to E4SF'),
        dict(native_fp16_merged=merged,native_fp16_full=full,format_only_fp16_merged=format_merge))


def run(args, report):
    files=util.Files(); probe,source=files.json(args.probe_report)
    report['source_probe']=source
    require([c['block'] for c in probe['cases']]==[0,24,48],'Layer set changed')
    require(probe['attention_calls']==12 and probe['complete_dit_calls']==0,'Native call allocation')
    args.data_dir.mkdir(parents=True,exist_ok=False); report['cases']=[]
    expected_queries=[1227+t*576+r for t in (2,18,34) for r in (0,287,575)]
    for case in probe['cases']:
        state=load_record(files,case['score_state'],case['score_state_tensors'])
        actual_correction=load_record(files,case['correction'],case['correction_tensor'])
        require(case['correction_e018_full_sha_exact'] is True and case['native']['bf16_full']['e018_bf16_full_exact'] is True,
                'Inherited E018 native/correction replay failed')
        require(torch.equal(state['correction_selected'],actual_correction[0,state['head_ids'],:,:22539]),
                'Selected correction differs from actual full tensor')
        require(state['valid_length']==22539 and state['split']==11264 and state['scale']==128**-.5,'Score geometry')
        require(state['head_ids'].tolist()==[0,8,16,24,32,40,48,55] and state['query_ids'].tolist()==expected_queries,'Fixed row selection')
        q,k,v=(state[n].double() for n in ('q_decoded','k_decoded','v_decoded'))
        correction=state['correction_selected'].double()
        require(q.shape==(8,9,128) and k.shape==v.shape==(8,22539,128) and correction.shape==(8,1,22539),'Decoded input shapes')
        z=((q.bmm(k.transpose(1,2))+correction)*state['scale']).flatten(0,1)
        weights,aux,events_out=contract_weights(z,state['split']); reference=apply_values(aux['true_weights'],v)
        matrix=dict(scores=z,**aux); case_out=dict(block=case['block'],head_ids=state['head_ids'].tolist(),
            query_ids=state['query_ids'].tolist(),score_state=case['score_state'],arms={},scale_events=events_out)
        for arm in ARMS:
            w=weights[arm]; full=apply_values(w['full'],v); split=apply_values(w['split'],v)
            m=output_stats(full,split,reference)
            stats=metric_json(m); stats['weights']=dict(full_mass_minus_one=(w['full'].sum(-1)-1).tolist(),
                split_mass_minus_one=(w['split'].sum(-1)-1).tolist(),l1_delta=(w['split']-w['full']).abs().sum(-1).tolist())
            case_out['arms'][arm]=stats
            matrix[arm]=dict(weights=w,full_output=full,split_output=split,output_stats=m)
        native,native_tensors=audit_native(files,case,z,v,reference,state['split'])
        case_out['native_comparison']=native; matrix.update(native_tensors)
        path=args.data_dir/f'block{case["block"]}_matrices.pt'; torch.save(matrix,path)
        case_out['matrices']=files.verify(path); report['cases'].append(case_out)
        print(f'E019 CPU block {case["block"]} complete',flush=True)
    report.update(status='complete',files=files.checked)


def self_test():
    mid=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],dtype=torch.float64)
    code,_=e2_rne(mid); require(code.tolist()==[0,2,2,4,4,6,6],'E2 ties-to-even')
    z=torch.cat((torch.zeros(128),torch.ones(128)*math.log(1.3))).double()[None]
    # Construct the analytic scores directly in FP64, not via a float32 log.
    z[:,128:]=math.log(1.3)
    w,aux,event=contract_weights(z,128)
    results={a:{p:float(w[a][p][:,:128].sum()) for p in ('full','split')} for a in ARMS}
    require(abs(results['e2_e4sf']['full']-352/(448/1.3+448))<1e-13,'Native-contract counterexample')
    require(abs(results['pnq_represented_lse']['full']-.44)<1e-13,'PNQ counterexample')
    require(abs(results['pnq_represented_lse']['split']-1/2.3)<1e-13,'Represented merge')
    require(torch.equal(aux['shared_e2_codes'],torch.full_like(aux['shared_e2_codes'],7)),'Counterexample E2 codes')
    for arm in ('no_p_fp64','ceil_log2'):
        require(float((w[arm]['full']-w[arm]['split']).abs().max())<1e-15,'Expected composable math control')
    uniform=torch.full((1,256),math.log(1.3),dtype=torch.float64)
    uniform_weights,_,_=contract_weights(uniform,128)
    require(abs(float(uniform_weights['ceil_log2']['full'].sum())-288*2/(448*1.3))<1e-13,
            'Canonical integer-reference ln2 must stay FP64')
    # Unequal within-shard distribution distinguishes intentionally wrong LSE
    # from represented-LSE even though both normalized results have mass one.
    z2=torch.cat((torch.linspace(-3.,.1,128,dtype=torch.float64),torch.linspace(-1.9,1.7,139,dtype=torch.float64)))[None]
    w2,a2,e2=contract_weights(z2,128)
    require(w2['pnq_represented_lse']['split'].shape==(1,267),'Partial tile mask')
    require(abs(float(w2['pnq_represented_lse']['split'].sum())-1)<1e-14,'PNQ represented mass')
    require(float((w2['pnq_represented_lse']['split']-w2['pnq_true_lse_negative']['split']).abs().max())>1e-8,'Wrong-merge negative control')
    require(sum(e2['full']['native_sf'][k]['count'][0] for k in ('normal','subnormal','zero'))==17,'No padding groups in SF stats')
    return dict(status='complete',analytic_counterexample=results,tests=['E2 midpoint RNE','native/PNQ analytic example',
        'noP and integer-reference composability','FP64 ln2 canonical scale restoration','correct/wrong represented merge distinguishable','tail mask/valid group event denominator'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--probe-report',type=Path,default=RD/'probe_run.json')
    parser.add_argument('--data-dir',type=Path,default=DATA); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); torch.set_num_threads(6)
    output=args.output or RD/('math_selftest.json' if args.self_test else 'partition_math_audit.json')
    require(not output.exists(),'Refusing to overwrite report'); started=time.monotonic()
    report=dict(experiment='E019',status='running',source={str(Path(__file__).resolve()):util.file_sha(__file__),
        str(Path(util.__file__).resolve()):util.file_sha(util.__file__)},methods=dict(threads=6,accumulation='FP64 CPU',
        score='(FP64 decoded Q @ decoded K.T + actual saved FP32 correction) * original scale; not native QK accumulator replay',
        codes='One shared E2 RNE code tensor from 6 exp(z-group16max), unaffected by scale-storage variants',
        native_sf='R32(448 exp(groupmax-running128max)); E4 RNE from exactly that common FP32 input',
        geometry='Fixed128 tiles,16microgroups,reverse traversal; split11264 keeps global groups; all117 tail pads masked',
        denominator='Mathematical FP64 unquantized or represented P; native is actually FP32 online row_sum',
        ceil='Per-group canonical reference plus ldexp; E4 subnormal/zero exceptions retained',
        sage='Paper SF1=R32(exp(tilemax-runningmax)/2688), innerE4=R8(R32(448exp(groupmax-tilemax))); shared codes; mathematical paper control',
        caveats=['Counterexamples establish possibility only; no quality/novel-method claim',
            'FP32SF1, exp/libm, casts and underflow are explicit mathematical boundaries',
            'No native reciprocal-floor feedback into shared E2 codes; floor events separately reported',
            'Rows/heads/layers correlated; no arbitrary significance threshold']))
    try:
        if args.self_test: report.update(self_test())
        else: run(args,report)
        require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    except BaseException as exc:
        report.update(status='failed_stop',error=repr(exc),traceback=traceback.format_exc()); raise
    finally:
        report.update(seconds_total=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized())
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(report['status'],report['seconds_total'],flush=True)

if __name__=='__main__': main()
