#!/usr/bin/env python3
"""E030: one donor forward, fixed rank16 right bases, six target attention calls.

Donor p30/s14 and target p36/s14 are existing block-mean free trajectories.
This is exploratory transfer, not a held-out quality or deployment evaluation.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import probe_h3_query_mean_k4 as common

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E030'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E030')
E017=ROOT/'results/research/E017/E017_denoise_block_mean.json'
MANIFEST=ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json'
BLOCKS=(0,24,48)
ARMS=('bf16_center','factorable_fp32')
H,D,R=56,128,16


def binding(report,budget):
    import torch
    import capture_h3_query_phase as cap
    import h3_native_fp4_attention as router_module
    import flashinfer.nvfp4_attention_sm120 as official
    from diffsynth.models.minimax_h3_dit import patchify_video,pack_audio
    manifest=json.loads(MANIFEST.read_text());old=json.loads(E017.read_text());assert old['status']=='complete'
    case=next(c for c in old['cases'] if c['prompt_id']==30);assert case['seed']==49771 and case['status']=='complete'
    prepared=common.load(case['prepared']);state={};noise={};steprefs={}
    for modality in ('video','audio'):
        row=next(s for s in case['steps'] if s['step']==14 and s['modality']==modality)
        ref={k:row[k] for k in ('file','bytes','sha256')};value=common.load(ref)
        state[modality]={k:value[k] for k in ('latents_before','timestep','sigma')}
        noise[modality]=value['noise_pred'];steprefs[modality]=ref
    value=dict(embedding=prepared['embedding'],text_token_tags=prepared['text_token_tags'],state=state)
    packed,call=cap.construct_call(value)
    contract=dict(expected_cu=packed['cu_seqlens'].tolist(),expected_refiner_cu=[0,prepared['embedding'].shape[0],prepared['embedding'].shape[0]])
    assert contract==dict(expected_cu=[0,22227,22272],expected_refiner_cu=[0,501,501])
    assert int(packed['img_pos'][0])==915 and packed['img_pos'].numel()==21312
    raw_ref=dict(video=patchify_video(-noise['video']),audio=pack_audio(-noise['audio']))
    report.update(sources={k:common.record(p) for k,p in dict(runner=__file__,common=common.__file__,capture_helper=cap.__file__,
        rollout_helper=cap.prior.__file__,loader_helper=cap.prior.old.__file__,router=router_module.__file__,official=official.__file__).items()},
        donor=dict(case=dict(prompt_id=30,seed=49771,step=14,arm='block_mean'),source_report=common.record(E017),
            prepared=case['prepared'],step_files=steprefs,attention_contract=contract,input_signature=cap.tree_signature(call),
            signature_matches_saved_call=cap.tree_signature(call)==case['dit_calls'][14]['actual_dit_inputs'],
            raw_reference_signature=cap.tree_signature(raw_ref),saved_raw_signature=case['dit_calls'][14]['raw_outputs']),
        manifest=common.record(MANIFEST),export_manifest=common.record(Path(manifest['export_dir'])/'manifest.json'))
    capture=json.loads((ROOT/'results/research/E018/capture_run.json').read_text())
    probe=json.loads((ROOT/'results/research/E018/probe_run.json').read_text())
    e029=json.loads((ROOT/'results/research/E029/run.json').read_text())
    assert capture['status']==probe['status']==e029['status']=='complete'
    report['target_reports']={name:common.record(ROOT/('results/research/'+name)) for name in ('E018/capture_run.json','E018/probe_run.json','E029/run.json')}
    report['targets']=[]
    for block in BLOCKS:
        budget();c=next(v for v in capture['cases'] if v['block']==block);p=next(v for v in probe['cases'] if v['block']==block)
        r=next(v for v in e029['layers'] if v['block']==block)
        refs={name:r['arms'][name]['output'] for name in ('original_block','original_global','euclidean')}
        refs['bf16']=next(v['output'] for v in p['outputs'] if v['mode']=='bf16' and v['shift']==0)
        raw=common.load(c['artifact']);kv=common.load(p['kv_packets'])
        assert raw['q'].shape==(22539,H,D) and kv['k_fp4'].shape==(1,H,22656,D//2)
        for ref in refs.values():assert common.load(ref['artifact']).shape==(22539,H,D)
        report['targets'].append(dict(block=block,capture=c['artifact'],kv_packets=p['kv_packets'],references=refs))
    return manifest,call,raw_ref


def capture_donor(manifest,call,raw_ref,report,budget):
    import torch
    import capture_h3_query_phase as cap
    import flashinfer.nvfp4_attention_sm120 as official
    old=cap.prior.old
    pipe=old.load_h3_pipeline(full=False,vram_limit_gib=30.)
    pipe.load_models_to_device(['dit']);pipe.dit.eval();old.make_h3_resident(pipe.dit)
    installed=old.install_native_h3(pipe.dit,Path(manifest['export_dir']),activation_packer=old.pack_activation_fast,chunk_rows=1024)
    assert installed['target_count']==installed['exact_roundtrip_count']==200
    old.memory_guard();router=cap.install_h3_fp4_attention(pipe.dit,mode='block_mean')
    original_dispatch=router.comfy._sdpa_varlen_attention;captured={}
    def observe(q,k,v,cu_seqlens,softmax_scale):
        block=router._active_main.get()
        if block in BLOCKS:
            assert block not in captured
            qp=official._pad_seq_len_to_128(q[:22227].transpose(0,1).unsqueeze(0).contiguous())
            captured[block]=dict(mu=qp.reshape(1,H,174,128,D).mean(3)[0].cpu(),global_mu=qp.mean(-2,keepdim=True)[0].cpu())
        return original_dispatch(q,k,v,cu_seqlens,softmax_scale)
    audit=old.RuntimeAudit();audit.phase='native'
    gpu_call=cap.tree_device(call,'cuda')
    try:
        router.comfy._sdpa_varlen_attention=observe;budget()
        with audit.installed(),old.collect_fastpack_checks() as checks,router.forward_context(**report['donor']['attention_contract'],diagnostics=False) as attn:
            assert report['complete_dit_calls']==0
            report['attempted_dit_calls']+=1
            outputs=pipe.dit(*gpu_call['args'],**gpu_call['kwargs']);torch.cuda.synchronize()
            report['complete_dit_calls']=1
    finally:
        router.comfy._sdpa_varlen_attention=original_dispatch;router.close()
    runtime=audit.row();assert {k:runtime[k] for k in ('sdpa_calls','scaled_mm_calls','disk_loads')}==dict(sdpa_calls=52,scaled_mm_calls=200,disk_loads=0)
    assert checks.summary['checked_calls']==200 and checks.summary['invalid_calls']==0
    assert attn.summary['fp4_calls']==50 and attn.summary['original_bf16_segments']==52
    assert sorted(captured)==list(BLOCKS)
    report['capture_attention_calls']=50
    report['donor'].update(runtime=runtime,packing=checks.summary,attention=attn.summary,raw_drift={})
    for modality,out in zip(('video','audio'),outputs,strict=True):
        cpu=out.detach().cpu();assert bool(cpu.isfinite().all())
        report['donor']['raw_drift'][modality]=dict(actual=common.trecord(cpu),
            versus_saved=common.metrics(cpu.reshape(1,-1),raw_ref[modality].reshape(1,-1)))
    report['bases']=[];weights=torch.full((174,),128.,dtype=torch.float64);weights[-1]=83
    for block in BLOCKS:
        item=captured[block];bases=[];stats=[]
        for h in range(H):
            delta=item['mu'][h].double()-item['global_mu'][h].double()
            _,s,vh=torch.linalg.svd(delta*weights.sqrt().unsqueeze(-1),full_matrices=False)
            b=vh[:R].contiguous();bases.append(b)
            energy=s.square();stats.append(dict(head=h,tail_fraction=float(energy[R:].sum()/energy.sum()) if float(energy.sum()) else None,
                orthogonality_max_abs=float((b@b.T-torch.eye(R,dtype=torch.float64)).abs().max())))
        path=DATA/f'block{block}_donor_basis.pt'
        torch.save(dict(**item,basis=torch.stack(bases),weights=weights,source_case=report['donor']['case']),path)
        report['bases'].append(dict(block=block,artifact=common.record(path),head_statistics=stats))
    old.memory_guard()
    # All resident model references and observer closures die on function return.


def probe_targets(report,budget,args):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    module=official.get_nvfp4_attention_sm120_module();report['layers']=[]
    for target in report['targets']:
        budget();raw=common.load(target['capture']);kv=common.load(target['kv_packets'])
        refs={name:common.load(ref['artifact']) for name,ref in target['references'].items()}
        basis_ref=next(b for b in report['bases'] if b['block']==target['block']);b64=common.load(basis_ref['artifact'])['basis']
        seg=tuple(raw[k].transpose(0,1).unsqueeze(0).contiguous().cuda() for k in ('q','k','v'))
        qpad=official._pad_seq_len_to_128(seg[0]);mu=qpad.reshape(1,H,177,128,D).mean(3);gmu=qpad.mean(-2,keepdim=True)
        # Only Kc is needed; no new correction or KV quantization.
        kc=official._pad_seq_len_to_128(seg[1]-seg[1].mean(-2,keepdim=True))
        delta64=mu[0].cpu().double()-gmu[0].cpu().double();b32=b64.float().cuda()
        gpu_kv={k:v.cuda() for k,v in kv.items()};layer=dict(block=target['block'],basis=basis_ref['artifact'],arms={})
        report['layers'].append(layer);factor_data=dict(basis=b64,mu=mu.cpu(),global_mu=gmu.cpu(),arms={})
        first_output=None
        for arm in ARMS:
            budget();roundoff=None
            if arm=='bf16_center':
                delta_r=(delta64@b64.transpose(-2,-1))@b64
                c32=gmu.float()+delta_r.unsqueeze(0).float().cuda();c=c32.to(torch.bfloat16)
                qcenter=(qpad.reshape(1,H,177,128,D)-c.unsqueeze(3)).reshape(1,H,22656,D).contiguous()
                correction=(c.float()@kc.transpose(-2,-1).float()).contiguous()
                factor_data['arms'][arm]=dict(center_fp32=c32.cpu(),center_bf16=c.cpu())
                del delta_r,c
            else:
                a=(mu.float()-gmu.float())@b32.transpose(-2,-1)
                c32=gmu.float()+a@b32
                qcenter=(qpad.float().reshape(1,H,177,128,D)-c32.unsqueeze(3)).reshape(1,H,22656,D).to(torch.bfloat16).contiguous()
                t=torch.cat((gmu.float(),b32.unsqueeze(0)),dim=2)@kc.transpose(-2,-1).float()
                correction=(t[:,:,:1]+a@t[:,:,1:]).contiguous()
                # Same mathematical center, different FP32 association: report, never byte-gate.
                direct=c32@kc.transpose(-2,-1).float()
                diff=correction-direct
                roundoff=dict(rms=float(diff.double().square().mean().sqrt()),max_abs=float(diff.abs().max()))
                factor_data['arms'][arm]=dict(a_fp32=a.cpu(),center_fp32=c32.cpu(),basis_fp32=b32.cpu())
                del a,t,direct,diff
            assert bool(qcenter.isfinite().all()) and bool(correction.isfinite().all())
            qcode=torch.empty((1,H,22656,64),device='cuda',dtype=torch.uint8)
            qsf=torch.empty((1,H,22656,8),device='cuda',dtype=torch.float8_e4m3fn)
            module.scaled_fp4_quant(qcenter,qcode,qsf,1)
            kwargs=dict(sm_scale=D**-.5,causal=False,per_block_mean=True,out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=22539)
            assert report['attention_probe_calls']<6;report['attention_probe_calls']+=1
            out=official.nvfp4_attention_sm120_fwd(qcode,gpu_kv['k_fp4'],gpu_kv['v_fp4_t'],qsf,gpu_kv['k_scale'],gpu_kv['v_scale_t'],correction,**kwargs)
            valid=out[0,:,:22539].transpose(0,1).contiguous().cpu();assert bool(valid.isfinite().all())
            path=DATA/f'block{target["block"]}_{arm}_output.pt';torch.save(valid,path)
            row=dict(output=dict(artifact=common.record(path),tensor=common.trecord(valid)),
                metrics={name:common.metrics(valid.transpose(0,1),ref.transpose(0,1)) for name,ref in refs.items()},
                correction_fp32_association_drift=roundoff,kernel_kwargs={k:str(v) if k=='out_dtype' else v for k,v in kwargs.items()})
            if first_output is None:first_output=valid
            else:row['versus_bf16_center']=common.metrics(valid.transpose(0,1),first_output.transpose(0,1))
            layer['arms'][arm]=row;common.save(args.output,report)
            del c32,qcenter,correction,qcode,qsf,out,valid
        path=DATA/f'block{target["block"]}_target_factors.pt';torch.save(factor_data,path);layer['factors']=common.record(path)
        del raw,kv,refs,b64,b32,seg,qpad,mu,gmu,kc,delta64,gpu_kv,factor_data,first_output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or RD/(args.phase+'.json');assert not args.output.exists()
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=600
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E030 original deadline')
    report=dict(experiment='E030',phase=args.phase,status='running',rank=R,attempted_dit_calls=0,complete_dit_calls=0,
        capture_attention_calls=0,attention_probe_calls=0,arms=list(ARMS),
        scope='Fixed donor p30/s14 Euclidean right basis transferred to p36/s14, same layers/heads. Exploratory transfer, not held-out quality. One BF16-center arm plus FP32 factorable T17+A arithmetic. Full correction materialized; no performance claim.')
    try:
        import torch
        torch.set_num_threads(6)
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==('' if args.phase=='check' else '0')
        manifest,call,raw_ref=binding(report,budget)
        if args.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            check=json.loads((RD/'check.json').read_text());assert check['status']=='complete'
            assert check['sources']==report['sources'] and check['donor']==report['donor'] and check['targets']==report['targets']
            assert not DATA.exists();DATA.mkdir(parents=True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.cuda.set_per_process_memory_fraction(min(1.,60*2**30/torch.cuda.get_device_properties(0).total_memory))
            with torch.inference_mode():
                capture_donor(manifest,call,raw_ref,report,budget)
                gc.collect();torch.cuda.empty_cache()
                report['allocated_after_model_release_gib']=torch.cuda.memory_allocated()/2**30
                common.save(args.output,report)
                probe_targets(report,budget,args)
            budget();assert report['complete_dit_calls']==1 and report['attention_probe_calls']==6
            report.update(status='complete',actual_attention_calls=report['capture_attention_calls']+report['attention_probe_calls'],
                cuda_initialized=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__':main()
