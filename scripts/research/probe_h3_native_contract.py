#!/usr/bin/env python3
"""E005: finite real-H3 NVFP4 contract audit, not a performance benchmark."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from minimax_h3_svdquant_common import H3_DIT_PATH,load_h3_pipeline,raw_call_to_block0,tree_device,nvfp4_qdq
from probe_h3_modality_pulse import summarize,partitions,sha256,save
from smoke_nvfp4_systems_20261002 import swizzle


@torch.inference_mode()
def quantize_pack(x,chunk=128):
    """Same rounded scales as historical QDQ; only E2M1 tie rule differs.

    Codes are independently decoded below. No intermediate BF16 conversion of z.
    Zero scale is explicitly counted, never hidden by a quantizer epsilon fix.
    """
    rows,k=x.shape
    g=(x.float().abs().amax().clamp_min(1e-12)/(6*448)).clamp_min(1e-12)
    sf=torch.empty((rows,k//16),device=x.device,dtype=torch.float8_e4m3fn)
    lp=torch.empty((rows,k//2),device=x.device,dtype=torch.uint8)
    rp=torch.empty_like(lp)
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=x.device)
    mids=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=x.device)
    evens=(torch.arange(8,device=x.device)%2==0)
    counters=torch.zeros(10,device=x.device,dtype=torch.int64)
    for start in range(0,rows,chunk):
        stop=min(start+chunk,rows)
        block=x[start:stop].float().reshape(-1,k//16,16)
        absmax=block.abs().amax(-1)
        ideal=(absmax/6).clamp_min(1e-12)
        scale=(ideal/g).clamp_min(torch.finfo(torch.float32).tiny).to(torch.float8_e4m3fn)
        sf[start:stop]=scale
        effective=scale.float()*g
        z=block/effective.unsqueeze(-1)
        # A zero scale necessarily reconstructs zero. Canonical zero codes keep
        # its hardware value zero, while counters expose all underflow cases.
        z=torch.where((effective==0).unsqueeze(-1),torch.zeros_like(z),z)
        distance=(z.abs().unsqueeze(-1)-levels).abs()
        nearest=distance==distance.amin(-1,keepdim=True)
        ties=nearest.sum(-1)>1
        low=nearest.to(torch.int8).argmax(-1)
        high=7-nearest.flip(-1).to(torch.int8).argmax(-1)
        legacy=torch.where(z<0,high,low)
        choose_even=nearest & evens
        rne=torch.where(choose_even.any(-1),choose_even.to(torch.int8).argmax(-1),low)
        sign=(z<0).to(torch.int64)*8
        lc=(legacy+sign).to(torch.uint8).reshape(stop-start,k)
        rc=(rne+sign).to(torch.uint8).reshape(stop-start,k)
        lp[start:stop]=lc[:,::2] | (lc[:,1::2]<<4)
        rp[start:stop]=rc[:,::2] | (rc[:,1::2]<<4)
        counters+=torch.stack([ties.sum(),(ties&(z<0)).sum(),(lc!=rc).sum(),
            ((z.abs().unsqueeze(-1)-mids).abs().amin(-1)<=1e-6).sum(),
            ((scale.float()>0)&(scale.float()<2**-6)).sum(),(scale.float()==0).sum(),
            (absmax==0).sum(),((absmax!=0)&(scale.float()==0)).sum(),
            (~z.isfinite()).sum(),(scale.float()==448).sum()])
    names=['exact_midpoints','negative_exact_midpoints','legacy_rne_code_changes',
           'near_midpoints_atol_1e-6_including_exact','nonzero_subnormal_scales','zero_scales',
           'all_zero_input_groups','nonzero_input_groups_with_zero_scale','nonfinite_normalized_values','scales_at_448']
    stats=dict(zip(names,map(int,counters.cpu().tolist())))
    stats.update(elements=x.numel(),groups=rows*k//16,global_scale=float(g),
                 midpoint_fraction=stats['exact_midpoints']/x.numel(),
                 changed_code_fraction=stats['legacy_rne_code_changes']/x.numel(),
                 scale_padding_excluded=True)
    return {'legacy':lp,'rne':rp,'scale':sf,'global':g.reshape(1),'stats':stats}


@torch.inference_mode()
def decode(packed,sf,g,*,dtype,include_global=True,chunk=128):
    """Independent nibble lookup and per-16 reconstruction (no encode logic)."""
    rows,k2=packed.shape
    out=torch.empty((rows,k2*2),dtype=dtype,device=packed.device)
    lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=packed.device)
    for start in range(0,rows,chunk):
        stop=min(rows,start+chunk)
        p=packed[start:stop]
        codes=torch.stack((p&15,p>>4),dim=-1).reshape(stop-start,k2*2)
        scales=sf[start:stop].float()
        if include_global:scales=scales*g
        decoded=lut[codes.long()]*scales.repeat_interleave(16,-1)
        out[start:stop]=decoded.to(dtype)
    return out


def by_modality(got,ref,idx):
    return {'all':summarize(got,ref),**{name:summarize(got,ref,indices.to(got.device)) for name,indices in idx.items()}}


def native(a,w,mode):
    recipe=[F.ScalingType.BlockWise1x16,F.ScalingType.TensorWise]
    return F.scaled_mm(a[mode].view(torch.float4_e2m1fn_x2),w[mode].view(torch.float4_e2m1fn_x2).t(),
        [swizzle(a['scale']),a['global']],recipe,[swizzle(w['scale']),w['global']],recipe,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16)


@torch.inference_mode()
def execute(args,report):
    torch.set_num_threads(6);torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False
    attn=importlib.import_module('diffsynth.core.attention.attention')
    if attn.ATTENTION_IMPLEMENTATION!='torch':raise RuntimeError('explicit torch SDPA required')
    sdpa_original=F.scaled_dot_product_attention
    sdpa_audit={'calls':0,'dtypes':set()}
    def audited_sdpa(q,k,v,*pos,**kwargs):
        if any(t.dtype!=torch.bfloat16 for t in (q,k,v)):raise RuntimeError('non-BF16 attention input')
        sdpa_audit['calls']+=1;sdpa_audit['dtypes'].add(tuple(str(t.dtype) for t in (q,k,v)))
        return sdpa_original(q,k,v,*pos,**kwargs)
    F.scaled_dot_product_attention=audited_sdpa
    paths=[Path(__file__),ROOT/'scripts/minimax_h3_svdquant_common.py',
           ROOT/'scripts/research/probe_h3_modality_pulse.py',ROOT/'scripts/research/smoke_nvfp4_systems_20261002.py',
           H3_DIT_PATH,args.state,*[args.cache/'p1'/f'sample_p1_s{s:02d}.pt' for s in args.steps],
           Path(importlib.import_module('diffsynth.models.minimax_h3_dit').__file__),Path(attn.__file__)]
    print('hashing immutable provenance',flush=True)
    report['files']={str(p):{'sha256':sha256(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths}
    report.update(torch=torch.__version__,torch_file=torch.__file__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
                  attention_implementation=attn.ATTENTION_IMPLEMENTATION,
                  environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','DIFFSYNTH_ATTENTION_IMPLEMENTATION','MINIMAX_H3_DIT_PATH','DIFFSYNTH_ROOT']})
    state=torch.load(args.state,map_location='cpu',weights_only=False)
    report['state_config']=state['config'];save(report,args.output)
    pipe=load_h3_pipeline(full=False,reserve_gib=35.)
    pipe.load_models_to_device(['dit']);pipe.dit.eval()
    block=pipe.dit.blocks[0]
    profiled=False
    for step in args.steps:
        path=args.cache/'p1'/f'sample_p1_s{step:02d}.pt'
        sample=torch.load(path,map_location='cpu',weights_only=False)
        inputs={}
        hidden,kw=raw_call_to_block0(pipe.dit,sample)
        handles=[]
        for suffix in args.layers:
            def capture(_module,inp,suffix=suffix):inputs[suffix]=inp[0].detach().cpu()
            handles.append(block.get_submodule(suffix).register_forward_pre_hook(capture))
        before=sdpa_audit['calls']
        try:block(hidden.cuda(),**tree_device(kw,'cuda'))
        finally:
            for h in handles:h.remove()
        if sdpa_audit['calls']<=before:raise RuntimeError('actual block SDPA not observed')
        idx,partdesc=partitions(sample,hidden.shape[0])
        del hidden,kw,sample
        print(f'step={step} captured {[ (k,list(v.shape)) for k,v in inputs.items() ]}',flush=True)
        for suffix in args.layers:
            x=inputs.pop(suffix).cuda()
            w,bias=block.get_submodule(suffix).load_from_disk(torch.bfloat16,'cuda',assign=False)
            if bias is not None:raise RuntimeError('expected bias-free qkv/fc2')
            baseline=F.linear(x,w)
            for recipe in ['plain','legacy_state_residual_corrected_branch']:
                start=time.time()
                branch=None
                if recipe=='plain':effective_x,effective_w=x,w
                else:
                    layer=state['layers']['blocks.0.'+suffix]
                    smooth,a,b=[layer[k].to(w) for k in ['smooth','final_a','final_b']]
                    effective_x=x/smooth
                    effective_w=w*smooth-b@a
                    branch=F.linear(F.linear(effective_x,a),b)
                    del smooth,a,b
                row={'prompt_id':1,'step':step,'layer':'blocks.0.'+suffix,'recipe':recipe,
                     'input_shape':list(effective_x.shape),'weight_shape':list(effective_w.shape),
                     'partitions':partdesc,'upstream':'full-shape BF16 block0 with torch SDPA; isolated linear perturbation',
                     'metrics':{}}
                qa=quantize_pack(effective_x,args.chunk_rows)
                qw=quantize_pack(effective_w,args.chunk_rows)
                row['activation_quant_stats']=qa['stats'];row['weight_quant_stats']=qw['stats']
                old_x=nvfp4_qdq(effective_x);old_w=nvfp4_qdq(effective_w)
                dx=decode(qa['legacy'],qa['scale'],qa['global'],dtype=torch.bfloat16)
                dw=decode(qw['legacy'],qw['scale'],qw['global'],dtype=torch.bfloat16)
                row['legacy_pack_roundtrip']={'activation':summarize(dx,old_x),'weight':summarize(dw,old_w)}
                if not torch.equal(dx,old_x) or not torch.equal(dw,old_w):raise RuntimeError(f'legacy roundtrip mismatch: {row["legacy_pack_roundtrip"]}')
                del dx,dw
                old_main=F.linear(old_x,old_w)
                old_full=old_main if branch is None else old_main+branch
                row['metrics']['legacy_qdq_vs_bf16_linear']=by_modality(old_full,baseline,idx)
                del old_x,old_w,old_full
                rne_x=decode(qa['rne'],qa['scale'],qa['global'],dtype=torch.bfloat16)
                rne_w=decode(qw['rne'],qw['scale'],qw['global'],dtype=torch.bfloat16)
                rne_main=F.linear(rne_x,rne_w)
                rne_full=rne_main if branch is None else rne_main+branch
                row['metrics']['rne_qdq_vs_bf16_linear']=by_modality(rne_full,baseline,idx)
                row['metrics']['rne_vs_legacy_qdq_main']=by_modality(rne_main,old_main,idx)
                del rne_x,rne_w,rne_full
                native_old=native(qa,qw,'legacy')
                native_rne=native(qa,qw,'rne')
                if not native_old.isfinite().all() or not native_rne.isfinite().all():raise RuntimeError('nonfinite native output')
                row['metrics']['native_legacy_vs_legacy_qdq_main']=by_modality(native_old,old_main,idx)
                row['metrics']['native_rne_vs_rne_qdq_main']=by_modality(native_rne,rne_main,idx)
                row['metrics']['native_rne_vs_native_legacy_main']=by_modality(native_rne,native_old,idx)
                native_full=native_rne if branch is None else native_rne+branch
                row['metrics']['native_rne_vs_bf16_linear']=by_modality(native_full,baseline,idx)
                del native_full,old_main,rne_main,native_old
                if not profiled:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        tmp=native(qa,qw,'rne');torch.cuda.synchronize()
                    names=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
                    report['native_profile_kernel_names']=names
                    if not any('sm120' in n and 'e2m1' in n for n in names):raise RuntimeError(f'no native SM120 FP4 kernel evidence: {names}')
                    profiled=True;del tmp
                # Independent real-valued packed-operand reference; global is
                # applied after FP32 GEMM, matching the intended native recipe.
                unscaled_x=decode(qa['rne'],qa['scale'],qa['global'],dtype=torch.float32,include_global=False)
                unscaled_w=decode(qw['rne'],qw['scale'],qw['global'],dtype=torch.float32,include_global=False)
                fp32_ref=F.linear(unscaled_x,unscaled_w)*(qa['global']*qw['global'])
                row['metrics']['native_rne_vs_fp32_unpacked_main']=by_modality(native_rne,fp32_ref,idx)
                if row['metrics']['native_rne_vs_fp32_unpacked_main']['all']['nmse']>5e-4:
                    raise RuntimeError('native FP32 reference mismatch exceeds packing safety gate')
                del unscaled_x,unscaled_w,fp32_ref,native_rne,qa,qw,effective_x,effective_w,branch
                row['elapsed_seconds_not_benchmark']=time.time()-start
                report['rows'].append(row)
                report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
                report['sdpa_audit']={'calls':sdpa_audit['calls'],'qkv_dtypes':[list(v) for v in sorted(sdpa_audit['dtypes'])]}
                save(report,args.output)
                print(json.dumps({'step':step,'layer':suffix,'recipe':recipe,'activation_stats':row['activation_quant_stats'],
                      'weight_stats':row['weight_quant_stats'],
                      'metric_all':{k:v['all']['nmse'] for k,v in row['metrics'].items()},
                      'peak_gpu_gib':report['peak_gpu_gib'],'seconds':row['elapsed_seconds_not_benchmark']}),flush=True)
                gc.collect();torch.cuda.empty_cache()
            del x,w,baseline
            gc.collect();torch.cuda.empty_cache()
    report['status']='complete';save(report,args.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,nargs='+',default=[0,19])
    p.add_argument('--layers',nargs='+',default=['attn.qkv_proj','mlp.fc2'])
    p.add_argument('--chunk-rows',type=int,default=128)
    p.add_argument('--cache',type=Path,default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s')
    p.add_argument('--state',type=Path,default=ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt')
    p.add_argument('--output',type=Path,default=ROOT/'results/research/E005_h3_native_contract.json')
    args=p.parse_args()
    report={'experiment':'E005','status':'partial','rows':[],
            'definition':'isolated linear numerical contract on BF16-upstream H3 activations, not full block parity',
            'limitations':['calibration prompt only','native GEMM consumes software-generated codes; no CUDA cvt audited',
                'legacy state residual uses corrected high-precision low-rank input','no native full-block rollout or video-quality claim',
                'no end-to-end performance claim','exact FP32 midpoint statistics depend on the stated division/scaling recipe']}
    try:execute(args,report)
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();save(report,args.output);raise

if __name__=='__main__':main()
