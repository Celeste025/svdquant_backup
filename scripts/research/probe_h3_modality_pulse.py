#!/usr/bin/env python3
"""E003: block0 modality-restricted error pulses, then 49 BF16 blocks.
No generation, calibration, token sampling, or native-FP4 speed claim.
"""
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from minimax_h3_svdquant_common import (
    H3_DIT_PATH, TARGET_SUFFIXES, load_h3_pipeline, tree_cpu, tree_device,
    nvfp4_qdq, install_runtime_hooks)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def summarize(got, ref, indices=None):
    if indices is not None:got,ref=got[indices],ref[indices]
    if not ref.numel():return {'err2':0.,'ref2':0.,'nmse':None,'max_abs':0.,'numel':0}
    # Chunk to avoid model-sized FP32/FP64 diagnostics intermediates.
    gf,rf=got.reshape(-1),ref.reshape(-1)
    err2=ref2=maxabs=0.
    for begin in range(0,rf.numel(),1024**2):
        g,r=gf[begin:begin+1024**2].float(),rf[begin:begin+1024**2].float()
        d=g-r
        err2+=d.square().sum(dtype=torch.float64).item()
        ref2+=r.square().sum(dtype=torch.float64).item()
        maxabs=max(maxabs,d.abs().max().item())
    return {'err2':err2,'ref2':ref2,'nmse':err2/max(ref2,1e-30),'max_abs':maxabs,'numel':ref.numel()}


def partitions(sample, tokens):
    kw=sample['input_kwargs']
    result={name:kw[key]['position_ids'].view(-1).long().cpu() for name,key in
            [('video','img_pos_info'),('audio','audio_pos_info'),('text','text_pos_info')]}
    used=torch.zeros(tokens,dtype=torch.bool)
    for name,idx in result.items():
        if idx.numel()!=idx.unique().numel() or (idx.numel() and (idx.min()<0 or idx.max()>=tokens)):
            raise RuntimeError(f'invalid {name} partition')
        if used[idx].any():raise RuntimeError(f'overlapping {name} partition')
        used[idx]=True
    result['pad_or_unassigned']=(~used).nonzero().flatten()
    tags=kw['token_tags'].view(-1).cpu()
    desc={}
    for name,idx in result.items():
        vals,counts=tags[idx].unique(return_counts=True)
        desc[name]={'tokens':idx.numel(),'indices_sha256':tensor_sha(idx),
                    'token_tag_counts':{str(int(v)):int(c) for v,c in zip(vals,counts)}}
    return result,desc


def save(report,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


@torch.inference_mode()
def execute(args,report):
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    print('Hashing model/state/inputs; read-only provenance',flush=True)
    input_paths=[args.cache/'p1'/f'sample_p1_s{step:02d}.pt' for step in args.steps]
    paths=[Path(__file__), ROOT/'scripts/minimax_h3_svdquant_common.py',args.state,H3_DIT_PATH,*input_paths]
    for module_name in ['diffsynth.models.minimax_h3_dit','diffsynth.models.minimax_h3_dit_comfy','diffsynth.core.attention.attention']:
        paths.append(Path(importlib.import_module(module_name).__file__))
    report['files']={str(p):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in paths}
    attn=importlib.import_module('diffsynth.core.attention.attention')
    report['attention_implementation']=attn.ATTENTION_IMPLEMENTATION
    if attn.ATTENTION_IMPLEMENTATION=='sage_attention':
        raise RuntimeError('E003 requires BF16 attention, got SageAttention; resolve environment before running')
    if attn.ATTENTION_IMPLEMENTATION!='torch':
        raise RuntimeError('E003 explicitly requires DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch')
    sdpa_original=torch.nn.functional.scaled_dot_product_attention
    sdpa_audit={'calls':0,'qkv_dtypes':set()}
    def audited_sdpa(q,k,v,*pos,**kwargs):
        sdpa_audit['calls']+=1
        sdpa_audit['qkv_dtypes'].add(tuple(str(z.dtype) for z in (q,k,v)))
        if any(z.dtype!=torch.bfloat16 for z in (q,k,v)):
            raise RuntimeError('non-BF16 Q/K/V observed in E003 attention')
        return sdpa_original(q,k,v,*pos,**kwargs)
    torch.nn.functional.scaled_dot_product_attention=audited_sdpa
    report['attention_contract']='Actual torch SDPA calls audited for BF16 Q/K/V; E001 default Sage backend differs, all E003 donors/references recomputed'
    state=torch.load(args.state,map_location='cpu',weights_only=False)
    report['state_config']=state['config']
    report['model_file']=str(H3_DIT_PATH)
    report['torch']=torch.__version__
    report['cuda']=torch.version.cuda
    report['device']=torch.cuda.get_device_name()
    report['environment']={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','DIFFSYNTH_ROOT','MINIMAX_H3_DIT_PATH','SVDQUANT_DATA_ROOT','DIFFSYNTH_ATTENTION_IMPLEMENTATION']}
    save(report,args.output)
    pipe=load_h3_pipeline(full=False,vram_limit_gib=args.vram_limit_gib)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    if len(pipe.dit.blocks)!=50:raise RuntimeError('expected 50 H3 blocks')
    block=pipe.dit.blocks[0]
    linears,weights,plain,residual={},{},{},{}
    for suffix in TARGET_SUFFIXES:
        old=block.get_submodule(suffix)
        w,bias=old.load_from_disk(torch.bfloat16,'cuda',assign=False)
        linear=nn.Linear(w.shape[1],w.shape[0],bias=bias is not None,device='cuda',dtype=torch.bfloat16)
        linear.weight.copy_(w)
        if bias is not None:linear.bias.copy_(bias)
        parent,attr=suffix.rsplit('.',1)
        setattr(block.get_submodule(parent),attr,linear)
        linears[suffix]=linear
        weights[suffix]=w.clone()
        plain[suffix]=nvfp4_qdq(w)
        s=state['layers']['blocks.0.'+suffix]
        smooth,a,b=[s[k].to(w) for k in ('smooth','final_a','final_b')]
        residual[suffix]=(nvfp4_qdq(w*smooth-b@a),smooth,a,b)
    del state,w,bias,s,smooth,a,b,old
    torch.cuda.empty_cache()

    def restore():
        for suffix,linear in linears.items():linear.weight.copy_(weights[suffix])

    for step,path in zip(args.steps,input_paths):
        case_start=time.time()
        sample=torch.load(path,map_location='cpu',weights_only=False)
        if sample['input_kwargs'].get('control_hints') is not None:
            raise RuntimeError('control_hints would confound block0 output pulse')
        call=tree_device(sample,'cuda')
        capture={}
        def pre_hook(_module,inp,kw):
            capture['hidden']=inp[0].detach().cpu()
            capture['kwargs']=tree_cpu(kw)
        def post_hook(_module,inp,out):
            capture['ref']=out.detach().cpu()
        restore()
        h1=block.register_forward_pre_hook(pre_hook,with_kwargs=True)
        h2=block.register_forward_hook(post_hook)
        teacher_sdpa_before=sdpa_audit['calls']
        print(f'step={step} teacher full forward',flush=True)
        try:teacher=tree_cpu(pipe.dit(*call['input_args'],**call['input_kwargs']))
        finally:h1.remove();h2.remove()
        if not isinstance(teacher,tuple) or len(teacher)!=2:raise RuntimeError('unexpected model output')
        ref=capture['ref']
        idx,desc=partitions(sample,ref.shape[0])
        case={'prompt_id':1,'step':step,'input':str(path),'tokens':ref.shape[0],
              'hidden_shape':list(ref.shape),'partitions':desc,
              'teacher_output_shapes':[list(v.shape) for v in teacher],
              'teacher_output_sha256':[tensor_sha(v) for v in teacher],
              'teacher_block0_sha256':tensor_sha(ref),'local':{},'arms':[],
              'teacher_sdpa_calls':sdpa_audit['calls']-teacher_sdpa_before,
              'qkv_dtypes':[list(v) for v in sorted(sdpa_audit['qkv_dtypes'])]}
        if case['teacher_sdpa_calls']<50:raise RuntimeError('full BF16 attention path not exercised')
        report['cases'].append(case)
        hidden=capture['hidden'].cuda()
        kw=tree_device(capture['kwargs'],'cuda')
        replay=block(hidden,**kw).cpu()
        case['block_bf16_replay']=summarize(replay,ref)
        if case['block_bf16_replay']['err2']!=0:raise RuntimeError('block BF16 mismatch')
        donors={}
        for mode in ['plain_w4a4','corrected_svdquant']:
            runtimes=[]
            try:
                for suffix,linear in linears.items():
                    if mode=='plain_w4a4':
                        linear.weight.copy_(plain[suffix])
                        runtimes.append(install_runtime_hooks(linear,None,None,None))
                    else:
                        q,smooth,a,b=residual[suffix]
                        linear.weight.copy_(q)
                        runtimes.append(install_runtime_hooks(linear,smooth,a,b))
                donors[mode]=block(hidden,**kw).cpu()
                if any(r.act.calls!=1 for r in runtimes):raise RuntimeError('unexpected quant hook call count')
            finally:
                for r in runtimes:r.remove()
                restore()
            case['local'][mode]={'all':summarize(donors[mode],ref),
                                 **{name:summarize(donors[mode],ref,ix) for name,ix in idx.items()}}
            case['local'][mode]['output_sha256']=tensor_sha(donors[mode])
            print(json.dumps({'step':step,'donor':mode,'local':case['local'][mode]}),flush=True)
        del hidden,kw,replay,capture
        gc.collect();torch.cuda.empty_cache()
        save(report,args.output)
        for arm,donor_name,subset in [
            ('bf16_zero_pulse',None,None),
            ('plain_full','plain_w4a4',None),
            ('corrected_full','corrected_svdquant',None),
            ('corrected_text_only','corrected_svdquant','text'),
            ('corrected_video_only','corrected_svdquant','video')]:
            # Exactly preserve teacher output outside intervention; avoid an
            # extra BF16 subtraction/addition while applying the same pulse.
            candidate=ref.clone()
            if donor_name:
                if subset is None:candidate=donors[donor_name].clone()
                else:candidate[idx[subset]]=donors[donor_name][idx[subset]]
            local=summarize(candidate,ref)
            candidate_gpu=candidate.cuda()
            ref_gpu=ref.cuda()
            hook_calls=[0]
            def inject(_module,inp,out):
                hook_calls[0]+=1
                if not torch.equal(out,ref_gpu):raise RuntimeError('upstream BF16 block changed before pulse')
                return candidate_gpu.clone()
            handle=block.register_forward_hook(inject)
            arm_start=time.time()
            sdpa_before=sdpa_audit['calls']
            print(f'step={step} arm={arm} full continuation',flush=True)
            try:output=tree_cpu(pipe.dit(*call['input_args'],**call['input_kwargs']))
            finally:handle.remove()
            if hook_calls!=[1]:raise RuntimeError('pulse was not applied exactly once')
            row={'arm':arm,'donor':donor_name,'subset':subset or 'all',
                 'pulse_local':local,'video':summarize(output[0],teacher[0]),
                 'audio':summarize(output[1],teacher[1]),
                 'seconds_including_disk_offload_not_benchmark':time.time()-arm_start,
                 'sdpa_calls':sdpa_audit['calls']-sdpa_before}
            if arm=='bf16_zero_pulse' and (row['video']['err2']!=0 or row['audio']['err2']!=0):
                raise RuntimeError(f'BF16 full replay mismatch: {row}')
            case['arms'].append(row)
            report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
            save(report,args.output)
            print(json.dumps({'step':step,**row}),flush=True)
            del output,candidate_gpu,ref_gpu,candidate
            gc.collect();torch.cuda.empty_cache()
        case['seconds_total']=time.time()-case_start
        del donors,ref,teacher,call,sample
        gc.collect();torch.cuda.empty_cache()
    report['status']='complete'
    save(report,args.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,nargs='+',default=[0,19])
    p.add_argument('--state',type=Path,default=ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt')
    p.add_argument('--cache',type=Path,default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s')
    p.add_argument('--output',type=Path,default=ROOT/'results/research/E003_h3_modality_pulse.json')
    p.add_argument('--vram-limit-gib',type=float,default=35.)
    args=p.parse_args()
    report={'experiment':'E003','status':'partial','native_fp4':False,
            'intervention':'replace selected block0 output rows by fixed quantized donor; rest BF16',
            'cases':[],'vram_limit_gib':args.vram_limit_gib,
            'limitations':['one calibration prompt, two captured timesteps','not free rollout or final video quality',
                           'fixed legacy-calibrated state with corrected runtime hook; not recalibrated',
                           'E001 default Sage attention differs; E003 forces actual BF16 torch SDPA and recomputes every donor',
                           'pulse amplitudes not energy normalized, so this is not intrinsic equal-energy sensitivity',
                           'old codebook tie rule retained','audio/pad pulses not independently tested',
                           'nonlinear full pulse is not sum of modality pulses','no novelty or performance claim']}
    try:execute(args,report)
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();save(report,args.output)
        raise

if __name__=='__main__':main()
