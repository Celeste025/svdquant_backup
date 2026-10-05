#!/usr/bin/env python3
"""E045 recovery v2: resume saved terminal history; never repeat the teacher trajectory."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import run_vanilla_wan_reference as ref
import run_vanilla_wan_native_v2 as deployed
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E045_wan_terminal_intervention_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E045_wan_terminal_intervention_plan.md'
AMENDMENT = ROOT/'research_state/06_experiments/E045_recovery_amendment.md'
require, record, save_json = ref.require, ref.record, ref.save_json


def tensor_record(value):
    # Even contiguous size-one tensors may retain zero strides. Fresh flat storage
    # makes byte views valid for scalar and expanded timestep inputs alike.
    x=value.detach().cpu()
    flat=torch.empty(x.numel(),dtype=x.dtype,device='cpu')
    flat.copy_(x.reshape(-1))
    return dict(shape=list(x.shape),dtype=str(x.dtype),finite=bool(flat.isfinite().all()),
                sha256=hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest())


def tree_copy(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: tree_copy(v) for k,v in value.items()}
    if isinstance(value, list):
        return [tree_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_copy(v) for v in value)
    return copy.deepcopy(value)


def signature(value):
    if torch.is_tensor(value):
        return {**tensor_record(value), 'device': str(value.device)}
    if isinstance(value, dict):
        return {k: signature(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [signature(v) for v in value]
    return value


def difference(x, y):
    require(x.shape == y.shape, 'Comparison shape mismatch')
    x,y=x.detach().cpu().reshape(-1),y.detach().cpu().reshape(-1)
    err=energy=maxabs=0.;changed=0
    for begin in range(0,x.numel(),1<<20):
        a,b=x[begin:begin+(1<<20)].double(),y[begin:begin+(1<<20)].double()
        d=a-b;err+=float(d.square().sum());energy+=float(b.square().sum())
        maxabs=max(maxabs,float(d.abs().max()));changed+=int((a!=b).sum())
    return dict(changed_elements=changed,max_abs=maxabs,nmse=err/max(energy,1e-300),equal=changed==0)


def save_tensor(path, value):
    torch.save(value, path)
    return dict(artifact=record(path), tensor=tensor_record(value))


def snapshot_scheduler(scheduler):
    # Exclude instance instrumentation closures: deepcopy(function) retains its closure.
    return copy.deepcopy({k:v for k,v in scheduler.__dict__.items() if not callable(v)})


def restore_scheduler(config, snapshot):
    scheduler=UniPCMultistepScheduler.from_config(config)
    scheduler.__dict__.update(copy.deepcopy(snapshot))
    return scheduler


def check(args, report):
    m=json.loads(args.manifest.read_text())
    case=next(c for c in m['cases'] if c['case_id']==args.case_id)
    require(case['prompt_id']==161,'Only the fixed clapping cases are supported')
    source_path=Path(m['reference_dir'])/'worker_161.json'
    source=json.loads(source_path.read_text())
    require(source['status']=='complete' and source['settings']==m['settings'],'E043 source/settings mismatch')
    old=next(c for c in source['cases'] if c['case_id']==args.case_id)
    require(old['status']=='complete' and all(old[k]==case[k] for k in case),'Reference case identity mismatch')
    initial=deployed.load_bound(old['initial_noise']['artifact'])
    require(tensor_record(initial)==old['initial_noise']['tensor'],'Initial noise mismatch')
    embeddings=deployed.load_bound(source['embeddings']['artifact'])
    for key in ('prompt_embeds','negative_prompt_embeds'):
        require(tensor_record(embeddings[key])==source['embeddings'][key],'Embedding mismatch')
    require(source['actual_configs']['vae'].get('patch_size') is None,'This decoder capture requires patch_size=None')
    require(initial.dtype==torch.float32 and list(initial.shape)==[1,16,21,60,104],'Unexpected initial state')
    identity=json.loads(Path(m['checkpoint_identity']).read_text())
    require(identity['status']=='complete','SVD identity incomplete')
    for item in identity['files']:
        path=Path(item['file']);stat=path.stat()
        require(str(path.resolve())==item['resolved'] and stat.st_size==item['bytes'] and
                stat.st_mtime_ns==item['mtime_ns'],'SVD checkpoint stat changed')
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    import deepcompressor.csrc.load as extension
    report.update(case=case,settings=m['settings'],manifest=record(args.manifest),plan=record(PLAN),amendment=record(AMENDMENT),
        reference_worker=record(source_path),initial_noise=old['initial_noise'],embeddings=source['embeddings'],
        checkpoint_identity=record(m['checkpoint_identity']),
        sources=[record(p) for p in [Path(__file__),Path(ref.__file__),Path(deployed.__file__),
            Path(deployed.native.__file__),Path(deployed.fast.__file__),
            Path(inspect.getsourcefile(WanPipeline)),Path(inspect.getsourcefile(UniPCMultistepScheduler)),
            Path(inspect.getsourcefile(AutoencoderKLWan))]],
        environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
            extension=extension._C.__file__,config_class=DiffusionPtqRunConfig.__name__))
    # Tiny CPU API fixture checks that a copied instance step closure is not restored.
    scheduler=UniPCMultistepScheduler.from_config(source['actual_configs']['scheduler'])
    scheduler.set_timesteps(50,device='cpu');scheduler.set_begin_index(0)
    scheduler.step=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('instrumentation must not be copied'))
    clone=restore_scheduler(scheduler.config,snapshot_scheduler(scheduler))
    require('step' not in clone.__dict__ and clone.step_index is None and clone.begin_index==0,'Snapshot closure exclusion failed')
    require(clone.sigmas.device.type=='cpu' and clone.timesteps.dtype==torch.int64,'Scheduler state API differs')
    report['cpu_snapshot_fixture']='noncallable state restored; instance step closure absent; CPU sigma retained'
    failed=json.loads(args.resume_worker.read_text())
    require(failed['status']=='failed_preserved' and failed['case']==case,'Wrong failed worker/case')
    expected=dict(teacher_dit=100,native_dit=0,native_gemm=0,dit_sdpa=6000,fastpack_checks=0,
        teacher_scheduler=50,shadow_scheduler=0,candidate_scheduler=0,public_vae_decode=0,decoder_chunks=0,text_encoder=0)
    require(failed['actual_counts']==expected and len(failed['steps'])==50,'Incomplete original teacher trajectory')
    terminal=deployed.load_bound(failed['terminal_capture'])
    require(terminal['history']['_step_index']==49 and len(terminal['calls'])==2,'Incomplete terminal snapshot')
    require(terminal['sample'].dtype==torch.float32 and terminal['cfg_output'].dtype==torch.bfloat16,'Snapshot dtypes differ')
    require(list(terminal['sample'].shape)==[1,16,21,60,104],'Snapshot sample shape differs')
    require(terminal['history']['sigmas'].device.type=='cpu','CPU snapshot inspection failed')
    # Real saved inputs, including the stride-zero Long that failed v1, are hashed.
    report['cpu_capture_signature']=signature(terminal)
    report['recovery_from']=dict(worker=record(args.resume_worker),capture=failed['terminal_capture'],
        failed_source=failed['sources'][0],original_final_tensor_saved=False)
    report['teacher_recovered_from_snapshot']=True
    report['inherited_counts']=expected
    report['steps']=failed['steps']
    report['inherited_steps_source']=record(args.resume_worker)
    return m,case,source,old,initial,embeddings,failed


@torch.inference_mode()
def run(args,m,case,source,old,initial,embeddings,failed,report):
    report['actual_counts']=copy.deepcopy(report['inherited_counts'])
    ref.budget(args);require(torch.cuda.is_available(),'CUDA unavailable');torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    destination=Path(m['data_dir'])/case['case_id']
    require(destination.is_dir() and {p.name for p in destination.iterdir()}=={'terminal_capture.pt'},'Recovery directory must contain only the saved capture')
    require(record(destination/'terminal_capture.pt')==failed['terminal_capture'],'Recovery capture path/binding differs')
    report['device']=dict(name=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    vae=AutoencoderKLWan.from_pretrained(m['model_dir'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe=WanPipeline.from_pretrained(m['model_dir'],vae=vae,text_encoder=None,tokenizer=None,
                                    torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config,flow_shift=8.)
    pipe.to('cuda');pipe.set_progress_bar_config(disable=True)
    require(pipe.vae.config.patch_size is None and not pipe.vae.use_tiling and not pipe.vae.use_slicing,'VAE contract differs')
    require(pipe.vae.dtype==torch.float32 and pipe.transformer.dtype==torch.bfloat16,'Model dtypes differ')
    report['actual_configs']={k:ref.config_dict(getattr(pipe,k).config) for k in ('transformer','vae','scheduler')}
    counts=report['actual_counts'];phase='native';inside_dit=False
    # No map_location override: preserve original CPU sigmas and CUDA:0 model state.
    terminal=torch.load(failed['terminal_capture']['file'],weights_only=False)
    require(terminal['history']['sigmas'].device.type=='cpu' and terminal['sample'].device.type=='cuda','Snapshot device placement differs')
    old_dit=pipe.transformer.forward
    old_mm,old_sdpa=F.scaled_mm,F.scaled_dot_product_attention

    def mm(*a,**kw):
        require(inside_dit and phase=='native','Unexpected native GEMM')
        counts['native_gemm']+=1
        return old_mm(*a,**kw)

    def sdpa(query,key,value,*a,**kw):
        if inside_dit:
            require(query.dtype==key.dtype==value.dtype==torch.bfloat16,'DiT attention dtype differs')
            counts['dit_sdpa']+=1
        return old_sdpa(query,key,value,*a,**kw)

    def dit(*a,**kw):
        nonlocal inside_dit
        ref.budget(args);counts['native_dit']+=1
        before=counts.copy();inside_dit=True
        try:
            with deployed.fast.collect_fastpack_checks() as checks:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):out=old_dit(*a,**kw)
            counts['fastpack_checks']+=len(checks)
            require(len(checks)==300 and counts['native_gemm']-before['native_gemm']==300,'Native coverage differs')
        finally:inside_dit=False
        require(counts['dit_sdpa']-before['dit_sdpa']==60,'Actual DiT SDPA count differs')
        return out

    pipe.transformer.forward=dit
    F.scaled_mm,F.scaled_dot_product_attention=mm,sdpa
    try:
        cond,uncond=[r['output'] for r in terminal['calls']]
        require(torch.equal(uncond+6.*(cond-uncond),terminal['cfg_output']),'Saved CFG order differs')
        report['terminal_capture']=failed['terminal_capture']
        report['terminal_signatures']=signature({k:v for k,v in terminal.items() if k!='history'})
        report['history_signatures']=signature(terminal['history'])
        clone=restore_scheduler(terminal['scheduler_config'],terminal['history'])
        teacher=UniPCMultistepScheduler.step(clone,terminal['cfg_output'],terminal['timestep'],terminal['sample'].clone(),return_dict=False)[0]
        counts['shadow_scheduler']+=1
        require(teacher.dtype==torch.float32 and bool(teacher.isfinite().all()),'Invalid reconstructed teacher final')
        report['shadow_vs_teacher']=dict(status='unavailable',reason='Original run final tensor was not saved; teacher is reconstructed from the pre-step snapshot')
        report['shadow_latent']=save_tensor(destination/'shadow_latent.pt',teacher.detach().cpu())
        previous=deployed.load_bound(old['final_latents']['artifact'])
        report['teacher_vs_e043']=difference(teacher,previous)
        report['teacher_reference_final']=old['final_latents']
        stats=dict(rms=float(teacher.square().mean().sqrt()),min=float(teacher.min()),max=float(teacher.max()))
        report['teacher_vs_original_last_callback']=dict(actual=stats,original=failed['steps'][-1],
            absolute_difference={k:abs(v-failed['steps'][-1][k]) for k,v in stats.items()})
        report['new_counts']={k:v-report['inherited_counts'][k] for k,v in counts.items()}
        save_json(report,args.output)

        native_args=SimpleNamespace(arm='svdquant_nvfp4',prompt_ids=[case['prompt_id']])
        local_manifest=dict(m,data_dir=str(destination))
        deployed.install_model(pipe,native_args,local_manifest,report)
        phase='native';native_outputs=[]
        for branch,call in zip(('cond','uncond'),terminal['calls'],strict=True):
            with pipe.transformer.cache_context(branch):
                native_outputs.append(pipe.transformer(*call['args'],**call['kwargs'])[0])
        native_cfg=native_outputs[1]+6.*(native_outputs[0]-native_outputs[1])
        clone=restore_scheduler(terminal['scheduler_config'],terminal['history'])
        candidate=UniPCMultistepScheduler.step(clone,native_cfg,terminal['timestep'],terminal['sample'].clone(),return_dict=False)[0]
        counts['candidate_scheduler']+=1
        outputs=destination/'native_outputs.pt'
        payload=dict(cond=native_outputs[0],uncond=native_outputs[1],cfg=native_cfg)
        torch.save(payload,outputs);report['native_outputs']=dict(artifact=record(outputs),tensors=signature(payload))
        require(candidate.dtype==torch.float32 and bool(candidate.isfinite().all()),'Invalid candidate state')
        first=teacher.clone();first[:,:,0].copy_(candidate[:,:,0])
        rest=teacher.clone();rest[:,:,1:].copy_(candidate[:,:,1:])
        corners=dict(teacher=teacher,native_terminal=candidate,first_only=first,rest_only=rest)
        report['corners']=[]
        for name,latent in corners.items():
            ref.budget(args);began=time.monotonic();torch.cuda.reset_peak_memory_stats()
            folder=destination/name;folder.mkdir(exist_ok=False)
            row=dict(corner=name,status='running',latent=save_tensor(folder/'latent.pt',latent.detach().cpu()))
            report['corners'].append(row);chunks=[]
            def capture_chunk(module,inputs,output):
                chunks.append(output.detach().cpu().clone())
                counts['decoder_chunks']+=1
            handle=pipe.vae.decoder.register_forward_hook(capture_chunk)
            try:
                # Exact public pipeline denormalization order, once per corner.
                z=latent.to(pipe.vae.dtype)
                mean=torch.tensor(pipe.vae.config.latents_mean).view(1,pipe.vae.config.z_dim,1,1,1).to(z.device,z.dtype)
                inverse_std=1.0/torch.tensor(pipe.vae.config.latents_std).view(1,pipe.vae.config.z_dim,1,1,1).to(z.device,z.dtype)
                z=z/inverse_std+mean
                decoded=pipe.vae.decode(z,return_dict=False)[0];counts['public_vae_decode']+=1
            finally:handle.remove()
            require(len(chunks)==21 and [v.shape[2] for v in chunks]==[1]+[4]*20,'Decoder temporal chunk contract differs')
            raw=torch.cat(chunks,2);del chunks
            require(list(raw.shape)==[1,3,81,480,832] and raw.dtype==torch.float32 and bool(raw.isfinite().all()),'Invalid preclamp pixels')
            row['preclamp_raw']=save_tensor(folder/'preclamp_raw.pt',raw)
            row['clamp_replay']=difference(raw.clamp(-1,1),decoded)
            row['public_raw']=tensor_record(decoded)
            row['clamped_raw_storage']='Reconstruct clamp(preclamp_raw,-1,1); actual public output comparison recorded'
            row['decoder_chunk_frames']=[1]+[4]*20
            frames=pipe.video_processor.postprocess_video(decoded,output_type='np')[0]
            row.update(ref.write_media(frames,folder,16),status='complete',seconds=time.monotonic()-began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            print(f"{case['case_id']} decoded {name}",flush=True);save_json(report,args.output)
            del raw,frames,decoded,z
        require(counts==dict(teacher_dit=100,native_dit=2,native_gemm=600,dit_sdpa=6120,fastpack_checks=600,
            teacher_scheduler=50,shadow_scheduler=1,candidate_scheduler=1,public_vae_decode=4,decoder_chunks=84,text_encoder=0),
            'Final actual call counts differ')
        report['new_counts']={k:v-report['inherited_counts'][k] for k,v in counts.items()}
        ref.budget(args);report['status']='complete'
    finally:
        F.scaled_mm,F.scaled_dot_product_attention=old_mm,old_sdpa


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-id',choices=('vbench0161_r0','vbench0161_r1'),required=True)
    p.add_argument('--manifest',type=Path,default=MANIFEST)
    p.add_argument('--resume-worker',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--deadline-unix',type=float)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();args.output=args.output or ROOT/f'results/research/E045/{"check" if args.check_only else "worker"}_{args.case_id}{"_v2" if args.check_only else ""}.json'
    require(not args.output.exists(),'Keep previous attempts')
    report=dict(experiment='E045',status='running',phase='check' if args.check_only else 'run',steps=[],
        actual_counts=dict(teacher_dit=0,native_dit=0,native_gemm=0,dit_sdpa=0,fastpack_checks=0,
            teacher_scheduler=0,shadow_scheduler=0,candidate_scheduler=0,public_vae_decode=0,decoder_chunks=0,text_encoder=0))
    started=time.monotonic()
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        inputs=check(args,report)
        if args.check_only:
            require(not torch.cuda.is_available() and not torch.cuda.is_initialized(),'CPU check requires hidden CUDA')
            report.update(status='complete',cuda_available=False,model_weights_loaded=False)
        else:run(args,*inputs,report)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        if not args.check_only and 'inherited_counts' in report:
            report['new_counts']={k:v-report['inherited_counts'][k] for k,v in report['actual_counts'].items()}
        report['seconds_total']=time.monotonic()-started;save_json(report,args.output)
        print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
