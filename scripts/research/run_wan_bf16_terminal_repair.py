#!/usr/bin/env python3
"""E046: native Wan trajectories with a matched-state original BF16 last step."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import run_vanilla_wan_native_v2 as deployed
import run_wan_terminal_intervention_v2 as terminal_helpers
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler

ref=deployed.ref
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'research_state/06_experiments/E046_wan_terminal_repair_manifest.json'
PLAN=ROOT/'research_state/06_experiments/E046_wan_terminal_repair_plan.md'
require,record,save_json=ref.require,ref.record,ref.save_json
tensor_record=terminal_helpers.tensor_record
signature=terminal_helpers.signature
snapshot_scheduler=terminal_helpers.snapshot_scheduler
restore_scheduler=terminal_helpers.restore_scheduler
difference=terminal_helpers.difference
save_tensor=terminal_helpers.save_tensor


def prerequisites(args,report):
    install_args=SimpleNamespace(manifest=args.manifest,arm='svdquant_nvfp4',prompt_ids=[args.prompt_id])
    m,cases,prepared=deployed.prerequisites(install_args,report)
    report['sources'].extend(record(p) for p in [Path(__file__),Path(terminal_helpers.__file__),PLAN])
    prior_dir=Path(m.get('native_reference_dir',ROOT/'results/research/E044'))
    prior={}
    for path in sorted(prior_dir.glob('worker_svdquant_nvfp4_*.json')):
        value=json.loads(path.read_text())
        if value.get('status')!='complete':continue
        for row in value['cases']:
            if row['prompt_id']==args.prompt_id:
                require(row['case_id'] not in prior,'Duplicate native reference case')
                prior[row['case_id']]=dict(worker=record(path),case=row)
    require(set(prior)=={c['case_id'] for c in cases},'Missing E044 native references')
    for case in cases:
        value=prior[case['case_id']]
        require(all(value['case'][k]==v for k,v in case.items()),'Native reference identity changed')
        prepared[case['case_id']]['native_reference']=value
    report['native_references']={k:dict(worker=v['worker'],final_latents=v['case']['final_latents']) for k,v in prior.items()}
    report['cpu_api']=dict(snapshot_without_callable_step=True,robust_tensor_record_source=record(Path(terminal_helpers.__file__)),
        transformer_forward_parameters=list(inspect.signature(WanTransformer3DModel.forward).parameters))
    return m,cases,prepared,install_args


@torch.inference_mode()
def run(args,m,cases,prepared,install_args,report):
    ref.budget(args);require(torch.cuda.is_available(),'CUDA unavailable');torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report['device']=dict(name=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    vae=AutoencoderKLWan.from_pretrained(m['model_dir'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe=WanPipeline.from_pretrained(m['model_dir'],vae=vae,text_encoder=None,tokenizer=None,
                                    torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config,flow_shift=8.)
    pipe.to('cuda');pipe.set_progress_bar_config(disable=True)
    require(pipe.vae.dtype==torch.float32 and pipe.text_encoder is None,'Pipeline component contract differs')
    require(not pipe.vae.use_tiling and not pipe.vae.use_slicing,'Unexpected VAE tiling/slicing')
    report['actual_configs']={key:ref.config_dict(getattr(pipe,key).config) for key in ('transformer','vae','scheduler')}
    deployed.install_model(pipe,install_args,m,report)
    require(not pipe.transformer._forward_pre_hooks,'Root DiT prehooks would precede input capture')
    report['capture_position']='Root forward entry before native_forward; root prehooks empty; all submodule quantization hooks run later'
    counts=report['actual_counts'];teacher=None;inside_dit=False;mode='native';active_row=None;capture={}
    native_forward,original_step=pipe.transformer.forward,pipe.scheduler.step
    original_mm,original_sdpa=F.scaled_mm,F.scaled_dot_product_attention

    def mm(*a,**kw):
        require(inside_dit and mode=='native','Unexpected native GEMM')
        counts['native_gemm']+=1
        return original_mm(*a,**kw)

    def sdpa(query,key,value,*a,**kw):
        if inside_dit:
            require(query.dtype==key.dtype==value.dtype==torch.bfloat16,'DiT SDPA dtype differs')
            counts['dit_sdpa']+=1
        return original_sdpa(query,key,value,*a,**kw)

    def native_dit(*a,**kw):
        nonlocal inside_dit
        ref.budget(args);counts['native_dit']+=1;before=counts.copy();inside_dit=True
        saved_call=None
        if pipe.scheduler.step_index==49:
            saved_call=dict(args=terminal_helpers.tree_copy(a),kwargs=terminal_helpers.tree_copy(kw))
        try:
            with deployed.fast.collect_fastpack_checks() as checks:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):out=native_forward(*a,**kw)
        finally:inside_dit=False
        counts['fastpack_checks']+=len(checks)
        call=dict(mode='native',native_gemm=counts['native_gemm']-before['native_gemm'],
                  dit_sdpa=counts['dit_sdpa']-before['dit_sdpa'],fastpack_checks=len(checks),invalid_flags=0)
        require(call==dict(mode='native',native_gemm=300,dit_sdpa=60,fastpack_checks=300,invalid_flags=0),'Native per-call contract differs')
        active_row['dit_calls'].append(call)
        if saved_call is not None:
            saved_call['output']=out[0].detach().clone()
            capture.setdefault('calls',[]).append(saved_call)
        return out

    def step(model_output,timestep,sample,*a,**kw):
        ref.budget(args)
        if pipe.scheduler.step_index==49:
            capture.update(sample=sample.detach().clone(),timestep=timestep.detach().clone(),
                cfg_output=model_output.detach().clone(),history=snapshot_scheduler(pipe.scheduler),
                scheduler_config=dict(pipe.scheduler.config))
            # Preserve the recoverable state before signatures or the last update.
            path=Path(m['data_dir'])/'native_full'/active_row['case_id']/'terminal_capture.pt'
            require(not path.exists(),'Keep earlier capture');torch.save(capture,path)
            active_row['terminal_capture']=record(path);save_json(report,args.output)
        counts['native_scheduler']+=1
        return original_step(model_output,timestep,sample,*a,**kw)

    def bf16_dit(call,branch):
        nonlocal inside_dit
        ref.budget(args);counts['bf16_dit']+=1;before=counts.copy();inside_dit=True
        try:
            with teacher.cache_context(branch),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out=teacher(*call['args'],**call['kwargs'])[0]
        finally:inside_dit=False
        require(counts['dit_sdpa']-before['dit_sdpa']==60 and counts['native_gemm']==before['native_gemm'],'Original BF16 call contract differs')
        active_row['dit_calls'].append(dict(mode='bf16',native_gemm=0,dit_sdpa=60,fastpack_checks=0,invalid_flags=0))
        return out

    pipe.transformer.forward=native_dit;pipe.scheduler.step=step
    F.scaled_mm,F.scaled_dot_product_attention=mm,sdpa
    try:
        for case in cases:
            ref.budget(args);entry=prepared[case['case_id']];before=counts.copy();capture={};mode='native'
            folder=Path(m['data_dir'])/'native_full'/case['case_id'];folder.mkdir(parents=True,exist_ok=False)
            row=dict(**case,status='running',steps=[],dit_calls=[],outputs=[],
                initial_noise=entry['old']['initial_noise'],embeddings=entry['embeddings_record'],
                reference_worker=entry['source'],native_reference=report['native_references'][case['case_id']])
            active_row=row;report['cases'].append(row)
            positive=entry['embeddings']['prompt_embeds'].cuda();negative=entry['embeddings']['negative_prompt_embeds'].cuda()

            def callback(pipeline,index,t,values):
                ref.budget(args);x=values['latents']
                require(x.dtype==torch.float32 and bool(x.isfinite().all()),'Invalid native sampler state')
                row['steps'].append(dict(index=index,timestep=float(t),sigma=float(pipeline.scheduler.sigmas[index]),
                    next_sigma=float(pipeline.scheduler.sigmas[index+1]),dtype=str(x.dtype),finite=True,
                    rms=float(x.square().mean().sqrt()),min=float(x.min()),max=float(x.max())))
                if (index+1)%10==0:print(f"{case['case_id']} native {index+1}/50",flush=True)
                return values

            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
            native_final=pipe(prompt_embeds=positive,negative_prompt_embeds=negative,latents=entry['initial'],
                height=480,width=832,num_frames=81,num_inference_steps=50,guidance_scale=6.,output_type='latent',
                callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'],max_sequence_length=512).frames
            torch.cuda.synchronize()
            row['native_phase']=dict(seconds_including_capture_stats_and_flag_checks=time.monotonic()-began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                independent_teacher_resident=teacher is not None)
            row['native_final']=save_tensor(folder/'final_latents.pt',native_final.detach().cpu())
            save_json(report,args.output)
            require(len(capture['calls'])==2 and capture['history']['_step_index']==49,'Terminal capture missing')
            require(capture['cfg_output'].dtype==torch.bfloat16 and capture['sample'].dtype==torch.float32,'Capture dtype differs')
            cond,uncond=[c['output'] for c in capture['calls']]
            require(torch.equal(uncond+6.*(cond-uncond),capture['cfg_output']),'Native CFG operation order differs')
            diag_started=time.monotonic()
            row['terminal_signatures']=signature({k:v for k,v in capture.items() if k!='history'})
            row['history_signatures']=signature(capture['history'])
            old_final=deployed.load_bound(entry['native_reference']['case']['final_latents']['artifact'])
            row['native_vs_e044']=difference(native_final,old_final)
            row['post_native_diagnostics_seconds']=time.monotonic()-diag_started
            row['schedule']=dict(timesteps=pipe.scheduler.timesteps.cpu().tolist(),sigmas=pipe.scheduler.sigmas.cpu().tolist())
            if teacher is None:
                ref.budget(args);torch.cuda.synchronize();allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
                began=time.monotonic()
                teacher=WanTransformer3DModel.from_pretrained(m['model_dir'],subfolder='transformer',
                    torch_dtype=torch.bfloat16,local_files_only=True).eval()
                loaded=time.monotonic();teacher.to('cuda');torch.cuda.synchronize();moved=time.monotonic()
                require(not any(isinstance(v,deployed.native.NativeWanLinear) for v in teacher.modules()),'Teacher contains native modules')
                report['independent_teacher']=dict(source='original BASE transformer; no PTQ state/hooks',
                    load_cpu_seconds=loaded-began,h2d_seconds=moved-loaded,load_and_h2d_seconds=moved-began,
                    parameter_bytes=sum(v.numel()*v.element_size() for v in teacher.parameters()),
                    buffer_bytes=sum(v.numel()*v.element_size() for v in teacher.buffers()),
                    allocated_before_bytes=allocated,reserved_before_bytes=reserved,
                    allocated_after_bytes=torch.cuda.memory_allocated(),reserved_after_bytes=torch.cuda.memory_reserved(),
                    additional_allocated_bytes=torch.cuda.memory_allocated()-allocated,
                    additional_reserved_bytes=torch.cuda.memory_reserved()-reserved)
            mode='bf16';torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
            bf16_cond=bf16_dit(capture['calls'][0],'cond');bf16_uncond=bf16_dit(capture['calls'][1],'uncond')
            bf16_cfg=bf16_uncond+6.*(bf16_cond-bf16_uncond)
            scheduler=restore_scheduler(capture['scheduler_config'],capture['history'])
            repaired=UniPCMultistepScheduler.step(scheduler,bf16_cfg,capture['timestep'],capture['sample'].clone(),return_dict=False)[0]
            counts['repair_scheduler']+=1;torch.cuda.synchronize()
            row['bf16_phase']=dict(seconds_including_two_forwards_and_final_update=time.monotonic()-began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            require(repaired.dtype==torch.float32 and bool(repaired.isfinite().all()),'Invalid repaired state')
            repair_folder=Path(m['data_dir'])/'bf16_last'/case['case_id'];repair_folder.mkdir(parents=True,exist_ok=False)
            row['repair_final']=save_tensor(repair_folder/'final_latents.pt',repaired.detach().cpu())
            values=dict(cond=bf16_cond,uncond=bf16_uncond,cfg=bf16_cfg)
            target=repair_folder/'bf16_outputs.pt';torch.save(values,target)
            row['bf16_outputs']=dict(artifact=record(target),tensors=signature(values));save_json(report,args.output)
            for variant,latent,latent_record,destination in [
                ('native_full',native_final,row['native_final'],folder),
                ('bf16_last',repaired,row['repair_final'],repair_folder)]:
                ref.budget(args);began=time.monotonic();torch.cuda.reset_peak_memory_stats()
                z=latent.to(pipe.vae.dtype)
                mean=torch.tensor(pipe.vae.config.latents_mean).view(1,pipe.vae.config.z_dim,1,1,1).to(z.device,z.dtype)
                inverse_std=1.0/torch.tensor(pipe.vae.config.latents_std).view(1,pipe.vae.config.z_dim,1,1,1).to(z.device,z.dtype)
                raw=pipe.vae.decode(z/inverse_std+mean,return_dict=False)[0];counts['public_vae_decode']+=1
                require(list(raw.shape)==[1,3,81,480,832] and bool(raw.isfinite().all()),'Invalid public decode')
                frames=pipe.video_processor.postprocess_video(raw,output_type='np')[0]
                media=dict(**case,variant=variant,status='complete',final_latents=latent_record,
                    raw_decoder=dict(shape=list(raw.shape),dtype=str(raw.dtype),finite=True,min=float(raw.min()),max=float(raw.max())))
                media.update(ref.write_media(frames,destination,16),seconds_decode_media=time.monotonic()-began,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                row['outputs'].append(media);save_json(report,args.output);del raw,frames,z
                print(f"{case['case_id']} decoded {variant}",flush=True)
            row['actual_counts']={k:v-before[k] for k,v in counts.items()}
            expected=dict(native_dit=100,bf16_dit=2,native_gemm=30000,dit_sdpa=6120,fastpack_checks=30000,
                native_scheduler=50,repair_scheduler=1,public_vae_decode=2,text_encoder=0)
            require(row['actual_counts']==expected and len(row['steps'])==50 and len(row['dit_calls'])==102,'Per-case count mismatch')
            row['status']='complete';save_json(report,args.output)
            del native_final,repaired,positive,negative,values,bf16_cond,bf16_uncond,bf16_cfg,old_final,cond,uncond,scheduler
        require(counts=={k:2*v for k,v in expected.items()},'Worker total counts differ')
        ref.budget(args);report['status']='complete'
    finally:F.scaled_mm,F.scaled_dot_product_attention=original_mm,original_sdpa


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt-id',type=int,choices=(161,192,269,316),required=True)
    p.add_argument('--manifest',type=Path,default=MANIFEST);p.add_argument('--output',type=Path)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--check-only',action='store_true')
    args=p.parse_args();args.output=args.output or ROOT/f'results/research/E046/{"check" if args.check_only else "worker"}_{args.prompt_id}.json'
    require(not args.output.exists(),'Keep earlier attempt')
    report=dict(experiment='E046',status='running',phase='check' if args.check_only else 'generate',prompt_id=args.prompt_id,cases=[],
        actual_counts=dict(native_dit=0,bf16_dit=0,native_gemm=0,dit_sdpa=0,fastpack_checks=0,
            native_scheduler=0,repair_scheduler=0,public_vae_decode=0,text_encoder=0))
    started=time.monotonic()
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        inputs=prerequisites(args,report)
        if args.check_only:
            require(not torch.cuda.is_available() and not torch.cuda.is_initialized(),'CPU check requires hidden CUDA')
            report.update(status='complete',cuda_available=False,model_weights_loaded=False,prepared_case_count=len(inputs[2]))
        else:run(args,*inputs,report)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds_total']=time.monotonic()-started;save_json(report,args.output)
        print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
