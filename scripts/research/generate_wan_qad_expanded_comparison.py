#!/usr/bin/env python3
"""E022 fixed-video comparison in baseline and selected-checkpoint phases.

CPU --check-only imports no torch. Baselines create the sixteen shared inputs
once; selected loads them verbatim after the frozen trainer selection completes.
Sampling, native conversion and VAE arithmetic follow the frozen E021 generator.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from generate_wan_qad_comparison import ROOT, DATA, BASE, RCM, SVD, sha256, save_json

RD = ROOT/'results/research/E022'
TRAIN = DATA/'research/20261003/E022/train'
ARTIFACTS = DATA/'research/20261003/E022/videos'
BASELINE_ARMS = ('bf16', 'plain_step0000', 'svd_lr')
SELECTED_ARM = 'qad_native_dev_selected'
SELECTION_RULE = 'min native development pooled NMSE; exact tie -> earlier step'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p))


def protocol(path):
    m = json.loads(path.read_text())
    require(m['experiment'] == 'E022_video_evaluation', 'Wrong video manifest')
    require(len(m['cases']) == 8 and len(m['trajectories']) == 16, 'Fixed 8x2 protocol required')
    settings = dict(height=480, width=832, frames=77, fps=16, steps=4, sigma_max=80., guidance=0.,
                    max_sequence_length=512, attention='BF16_FLASH_ATTENTION', latent_shape=[1,16,20,60,104],
                    actual_bf16_timesteps=[988.,932.,852.,608.], vae_dtype='BF16', vae_core=128, vae_halo=0)
    require(all(m['settings'][k] == v for k, v in settings.items()), 'Sampling settings changed')
    require([a['name'] for a in m['arms']] == ['bf16','plain','svd_lr',SELECTED_ARM], 'Arm protocol changed')
    require(m['model'] == dict(base_assets=str(BASE), explicit_rcm_transformer=str(RCM)), 'Model paths changed')
    cases = {c['prompt_id']: c for c in m['cases']}
    require(len(cases) == 8, 'Duplicate prompt IDs')
    trajectories = []
    for i, case in enumerate(m['cases']):
        require(case['index'] == i and case['seeds'] == [20261110+2*i,20261111+2*i], 'Seed formula changed')
        for replica in (0,1):
            expected = dict(trajectory_id=f'{case["prompt_id"]}_r{replica}', prompt_id=case['prompt_id'],
                            manifest_index=i, replica=replica, seed=case['seeds'][replica])
            require(m['trajectories'][2*i+replica] == expected, 'Trajectory order/identity changed')
            trajectories.append(expected | dict(prompt=case['prompt']))
    require(m['arms'][1]['weights'] == str(TRAIN/'packed_step0000.pt'), 'Plain checkpoint changed')
    return m, trajectories


def bind_plain(train_report, manifest):
    # This live report is read once for its completed step-zero export only.
    # Do not bind or hash the whole changing training report during baselines.
    train = json.loads(train_report.read_text())
    exports = [r for r in train['exports'] if r['step'] == 0]
    require(len(exports) == 1, 'Exactly one completed step-zero export is required')
    export = exports[0]
    require(export['file'] == manifest['arms'][1]['weights'] and export['module_count'] == 300
            and export['online_lowrank'] is False, 'Wrong step-zero export contract')
    artifact = file_record(export['file'])
    require(artifact['sha256'] == export['sha256'] and artifact['bytes'] == export['bytes'], 'Step-zero packed file changed')
    return dict(source_training_report=str(train_report), source_report_policy='Only the immutable exports[step=0] record is copied; no hash of the live training report.',
                export_record=export, packed_artifact=artifact)


def bind_selected(args, manifest):
    base = json.loads(args.baseline_report.read_text())
    require(base['status'] == 'complete' and base['manifest_reference']['sha256'] == sha256(args.manifest), 'Completed matching baselines required')
    require(base['artifact_dir'] == str(args.artifact_dir), 'Selected phase must reuse the baseline artifact directory')
    require(base['actual_totals'] == dict(dit_calls=192, native_mm_calls=38400, sdpa_calls=11520, videos=48), 'Baseline totals incomplete')
    train = json.loads(args.train_report.read_text())
    require(train['status'] == 'complete', 'Selected phase requires completed training')
    selection_path = Path(manifest['arms'][3]['selection_artifact'])
    selection = json.loads(selection_path.read_text())
    require(selection['status'] == 'complete' and selection['rule'] == SELECTION_RULE, 'Trainer selection incomplete or changed')
    exports = train['exports']
    require([r['step'] for r in exports] == [0,32,64,128], 'Expected all four fixed checkpoint candidates')
    require(all(math.isfinite(r['native_dev_pooled_nmse']) for r in exports), 'Nonfinite selection metric')
    winner = min(exports, key=lambda r: (r['native_dev_pooled_nmse'],r['step']))
    candidates = [{k:r[k] for k in ('step','file','sha256','native_dev_pooled_nmse')} for r in exports]
    require(selection['candidates'] == candidates, 'Selection candidates differ from final trainer exports')
    require(selection['selected_step'] == winner['step'] and selection['selected_packed_file'] == winner['file']
            and selection['selected_packed_sha256'] == winner['sha256'], 'Not the predeclared development-selected checkpoint')
    require(train['selection']['selected_step'] == winner['step'] and train['selection']['artifact']['sha256'] == sha256(selection_path), 'Final trainer selection reference differs')
    require(exports[0] == base['plain_step0000_binding']['export_record'], 'Step-zero export record changed after baseline binding')
    artifact = file_record(winner['file'])
    require(artifact['sha256'] == winner['sha256'] and artifact['bytes'] == winner['bytes'], 'Selected packed file changed')
    binding = dict(experiment='E022_video_evaluation', status='complete', rule=SELECTION_RULE,
        video_manifest=file_record(args.manifest), baseline_report=file_record(args.baseline_report),
        final_training_report=file_record(args.train_report), selection_artifact=file_record(selection_path),
        selected_step=winner['step'], selected_packed_artifact=artifact, same_checkpoint_as_plain=(winner['step']==0),
        policy='Selection from the frozen development rule only; no prompt/noise/checkpoint replacement after video inspection.')
    return binding, base


def execute(args, manifest, trajectories, report, binding, baselines):
    sys.path.insert(0,str(ROOT/'scripts'))
    sys.path.insert(0,str(ROOT/'third_party/deepcompressor'))
    os.environ.setdefault('SVDQUANT_DATA_ROOT',str(DATA))
    os.environ['RCM_RUNS_ROOT'] = str(args.artifact_dir/'svd_load_scratch')
    import torch
    import torch.nn.functional as F
    import diffusers
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from diffusers.utils import export_to_video
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from infer_rcm_wan_4step import decode_spatial_tiled, load_quantized_transformer
    import wan_mainweight_qad as qad
    import wan_native_nvfp4 as native
    from wan_nvfp4_fastpack import collect_fastpack_checks, pack_activation_fast, validate_quantizer_contract

    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Root must launch this phase on physical GPU0')
    require(torch.cuda.get_device_capability() == (12,0), 'SM120 required')
    require(torch.__version__ == '2.11.0+cu128' and diffusers.__version__ == '0.33.1', 'Use the established E021 environment')
    torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    device = torch.device('cuda'); started = time.time()

    def budget():
        if time.time() >= args.deadline_unix:
            raise TimeoutError('Caller deadline expired; keep partial artifacts')
        if torch.cuda.max_memory_allocated() > 60*1024**3:
            raise MemoryError('Fixed full-shape 60GiB allocated-memory guard exceeded')

    def checkpoint():
        report.update(elapsed_seconds_not_benchmark=time.time()-started,
                      peak_allocated_gib_not_benchmark=torch.cuda.max_memory_allocated()/1024**3)
        save_json(args.output,report); budget()

    def digest(value):
        return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

    def artifact(path,value):
        require(not path.exists(),f'Preserve existing artifact: {path}')
        path.parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
        return file_record(path)

    report.update(environment=dict(python=sys.executable,torch=torch.__version__,diffusers=diffusers.__version__,
                  cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']),
                  artifact_dir=str(args.artifact_dir),shared_inputs={},arms={})
    report['sources'] = {str(p):sha256(p) for p in [Path(__file__),Path(qad.__file__),Path(native.__file__),
        ROOT/'scripts/research/generate_wan_qad_comparison.py',ROOT/'scripts/research/wan_nvfp4_fastpack.py',
        ROOT/'scripts/infer_rcm_wan_4step.py',args.manifest,BASE/'vae/config.json',RCM/'config.json']}
    video_processor = VideoProcessor(vae_scale_factor=8)  # Same constructor/default spatial factor as E021's WanPipeline.
    with torch.inference_mode():
        if args.phase == 'baselines':
            args.artifact_dir.mkdir(parents=True,exist_ok=False)
            pipe = WanPipeline.from_pretrained(BASE,transformer=None,vae=None,torch_dtype=torch.bfloat16,local_files_only=True)
            pipe.text_encoder.to(device).eval();video_processor=pipe.video_processor
            angles=torch.tensor([math.atan(80.),1.5,1.4,1.,0.],dtype=torch.float64,device=device)
            t_steps=angles.sin()/(angles.cos()+angles.sin())
            ones=torch.ones(1,dtype=torch.float64,device=device)
            report['schedule']=dict(t_steps=t_steps.cpu().tolist(),timesteps_bf16=[(t.float()*ones*1000).to(torch.bfloat16).item() for t in t_steps[:-1]])
            for case in manifest['cases']:
                budget()
                embedding,_=pipe.encode_prompt(prompt=case['prompt'],do_classifier_free_guidance=False,max_sequence_length=512,device=device,dtype=torch.bfloat16)
                require(embedding.shape==(1,512,4096) and embedding.dtype==torch.bfloat16,'Embedding contract changed')
                for trajectory in trajectories[2*case['index']:2*case['index']+2]:
                    generator=torch.Generator(device=device).manual_seed(trajectory['seed'])
                    initial_noise=pipe.prepare_latents(batch_size=1,num_channels_latents=16,height=480,width=832,num_frames=77,dtype=torch.float32,device=device,generator=generator)
                    initial=initial_noise.to(torch.float64)*t_steps[0]
                    require(initial.shape==(1,16,20,60,104),'Latent shape changed')
                    noises=[torch.randn(initial.shape,dtype=torch.float32,device=device,generator=generator) for _ in range(4)]
                    payload=trajectory|dict(initial_latent_fp64=initial.cpu(),update_noises_fp32=[n.cpu() for n in noises],embedding_bf16=embedding.cpu(),t_steps_fp64=t_steps.cpu(),generator_state_after_draws=generator.get_state().cpu())
                    tid=trajectory['trajectory_id']
                    report['shared_inputs'][tid]=dict(artifact=artifact(args.artifact_dir/'shared_inputs'/(tid+'.pt'),payload),
                        initial_latent_sha256=digest(initial),initial_noise_sha256=digest(initial_noise),noise_sha256=[digest(n) for n in noises],embedding_sha256=digest(embedding))
                    del generator,initial_noise,initial,noises,payload
                    checkpoint()
                del embedding
            require(len({r['initial_latent_sha256'] for r in report['shared_inputs'].values()})==16,'Distinct trajectory seeds produced repeated initial tensors')
            del pipe;gc.collect();torch.cuda.empty_cache()
            arms=[dict(name='bf16',kind='bf16'),dict(name='plain_step0000',kind='packed',weights=binding['packed_artifact']['path']),dict(name='svd_lr',kind='svd')]
        else:
            report['shared_inputs']=baselines['shared_inputs']
            report['schedule']=baselines['schedule']
            t_steps=torch.tensor(report['schedule']['t_steps'],dtype=torch.float64,device=device)
            ones=torch.ones(1,dtype=torch.float64,device=device)
            arms=[dict(name=SELECTED_ARM,kind='packed',weights=binding['selected_packed_artifact']['path'])]
        require(report['schedule']['timesteps_bf16']==manifest['settings']['actual_bf16_timesteps'],'Actual schedule changed')
        checkpoint()

        original_sdpa,original_mm=F.scaled_dot_product_attention,F.scaled_mm
        counts=dict(dit_calls=0,native_mm_calls=0,sdpa_calls=0,videos=0)
        def sdpa(q,k,v,*pos,**kw):
            require(q.dtype==k.dtype==v.dtype==torch.bfloat16,'Attention must remain BF16')
            counts['sdpa_calls']+=1;return original_sdpa(q,k,v,*pos,**kw)
        def mm(a,b,*pos,**kw):
            require(a.dtype==b.dtype==torch.float4_e2m1fn_x2,'Expected actual FP4 scaled_mm inputs')
            counts['native_mm_calls']+=1;return original_mm(a,b,*pos,**kw)
        F.scaled_dot_product_attention,F.scaled_mm=sdpa,mm
        try:
            for arm in arms:
                budget();name=arm['name']
                model=WanTransformer3DModel.from_pretrained(RCM,torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
                installed={}
                if arm['kind']=='packed':
                    packed=torch.load(arm['weights'],map_location='cpu',weights_only=False)
                    installed=qad.install_packed(model,packed);del packed
                    require(not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),'Unexpected plain/QAD hooks')
                    require(not any('weight_master' in n for n,_ in model.named_parameters()),'Unexpected retained master weight')
                elif arm['kind']=='svd':
                    pipe=WanPipeline.from_pretrained(BASE,transformer=model,vae=None,text_encoder=None,tokenizer=None,torch_dtype=torch.bfloat16,local_files_only=True)
                    load_quantized_transformer(pipe,SVD,BASE);model=pipe.transformer.eval()
                    conversion=native.convert_wan_transformer_to_native(model,SVD,activation_packer='legacy',chunk_rows=1024)
                    require(conversion['target_count']==conversion['exact_roundtrip_count']==300,'Incomplete frozen SVD conversion')
                    for module in model.modules():
                        if isinstance(module,native.NativeWanLinear):
                            validate_quantizer_contract(module.activation_quantizer);module.activation_packer=pack_activation_fast
                    del module,pipe
                native_modules=[m for m in model.modules() if isinstance(m,native.NativeWanLinear)]
                expected=0 if arm['kind']=='bf16' else 300
                require(len(native_modules)==expected and all(m.execution_mode=='native' for m in native_modules),'Wrong native module count or hidden fallback')
                report['arms'][name]=dict(status='running',manifest_arm='plain' if name=='plain_step0000' else name,native_modules=len(native_modules),trajectories={})
                for trajectory in trajectories:
                    budget();tid=trajectory['trajectory_id'];shared=report['shared_inputs'][tid]
                    require(sha256(shared['artifact']['path'])==shared['artifact']['sha256'],'Saved shared inputs changed')
                    payload=torch.load(shared['artifact']['path'],map_location='cpu',weights_only=False)
                    require(all(payload[k]==v for k,v in trajectory.items()),'Shared trajectory identity changed')
                    latents=payload['initial_latent_fp64'].to(device).clone()
                    noises=[n.to(device) for n in payload['update_noises_fp32']];embedding=payload['embedding_bf16'].to(device)
                    require(digest(latents)==shared['initial_latent_sha256'] and [digest(n) for n in noises]==shared['noise_sha256'] and digest(embedding)==shared['embedding_sha256'],'Shared tensor identities differ')
                    require(torch.equal(payload['t_steps_fp64'],t_steps.cpu()),'Saved trajectory schedule differs')
                    row=trajectory|dict(status='running',shared_inputs=shared['artifact']['path'],initial_latent_sha256=shared['initial_latent_sha256'],noise_sha256=shared['noise_sha256'],embedding_sha256=shared['embedding_sha256'],steps=[])
                    report['arms'][name]['trajectories'][tid]=row
                    module_before=sum(m.native_calls for m in installed.values())
                    for step,(t_cur,t_next) in enumerate(zip(t_steps[:-1],t_steps[1:])):
                        budget();timestep=(t_cur.float()*ones*1000).to(torch.bfloat16);before=dict(counts)
                        with collect_fastpack_checks() if expected else nullcontext([]) as checks:
                            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                                velocity=model(hidden_states=latents.to(torch.bfloat16),timestep=timestep,encoder_hidden_states=embedding,return_dict=False)[0]
                        counts['dit_calls']+=1
                        actual_native=counts['native_mm_calls']-before['native_mm_calls'];actual_sdpa=counts['sdpa_calls']-before['sdpa_calls']
                        require(actual_native==expected and actual_sdpa==60 and len(checks)==expected,'Unexpected actual DiT execution counts')
                        require(velocity.dtype==torch.bfloat16 and bool(torch.isfinite(velocity).all()),'Invalid velocity')
                        latents=(1-t_next)*(latents-t_cur*velocity.to(torch.float64))+t_next*noises[step]
                        require(bool(torch.isfinite(latents).all()),'Nonfinite free-rollout latent')
                        row['steps'].append(dict(step=step,timestep=timestep.item(),actual_native_mm_calls=actual_native,actual_sdpa_calls=actual_sdpa,validated_fastpack_flags=len(checks),noise_sha256=shared['noise_sha256'][step],latent_after_sha256=digest(latents)))
                        report['actual_totals']=dict(counts);checkpoint()
                    row.update(actual_dit_calls=4,actual_native_mm_calls=sum(s['actual_native_mm_calls'] for s in row['steps']))
                    require(row['actual_native_mm_calls']==4*expected,'Wrong four-step native count')
                    require(not installed or sum(m.native_calls for m in installed.values())-module_before==1200,'Packed module call counter mismatch')
                    row['final_latent']=artifact(args.artifact_dir/name/tid/'final_latent.pt',latents.cpu())
                    row['final_latent_sha256']=digest(latents);row['status']='denoising_complete'
                    del payload,latents,noises,embedding,velocity
                    print(json.dumps(dict(phase=args.phase,arm=name,trajectory=tid,status=row['status'])),flush=True);checkpoint()
                report['arms'][name]['status']='denoising_complete'
                del model,installed,native_modules;gc.collect();torch.cuda.empty_cache()
        finally:
            F.scaled_dot_product_attention,F.scaled_mm=original_sdpa,original_mm

        budget()
        vae=AutoencoderKLWan.from_pretrained(BASE/'vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
        mean=torch.tensor(vae.config.latents_mean,device=device,dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1)
        reciprocal_std=1.0/torch.tensor(vae.config.latents_std,device=device,dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1)
        for arm in arms:
            for trajectory in trajectories:
                budget();row=report['arms'][arm['name']]['trajectories'][trajectory['trajectory_id']]
                latent=torch.load(row['final_latent']['path'],map_location='cpu',weights_only=False).to(device)
                normalized=latent.float().to(vae.dtype)/reciprocal_std+mean
                if hasattr(vae,'clear_cache'):vae.clear_cache()
                decoded=decode_spatial_tiled(vae,normalized,core=128,halo=0)
                require(decoded.shape==(1,3,77,480,832) and bool(torch.isfinite(decoded).all()),'Invalid decoded video')
                frames=video_processor.postprocess_video(decoded,output_type='np')[0]
                path=Path(row['final_latent']['path']).with_name('video.mp4');require(not path.exists(),f'Preserve {path}')
                export_to_video(frames,str(path),fps=16)
                row['video']=file_record(path)|dict(frames=len(frames),height=480,width=832,fps=16)
                row['status']='complete';counts['videos']+=1
                del latent,normalized,decoded,frames
                if hasattr(vae,'clear_cache'):vae.clear_cache()
                gc.collect();torch.cuda.empty_cache();report['actual_totals']=dict(counts);checkpoint()
            report['arms'][arm['name']]['status']='complete'
        expected_totals=dict(dit_calls=192,native_mm_calls=38400,sdpa_calls=11520,videos=48) if args.phase=='baselines' else dict(dit_calls=64,native_mm_calls=19200,sdpa_calls=3840,videos=16)
        require(counts==expected_totals,'Phase total execution counts differ')
        report['status']='complete';checkpoint()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('baselines','selected'),required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--manifest',type=Path,default=RD/'video_test_manifest.json')
    parser.add_argument('--train-report',type=Path,default=RD/'train_run.json')
    parser.add_argument('--baseline-report',type=Path,default=RD/'video_baselines.json')
    parser.add_argument('--artifact-dir',type=Path,default=ARTIFACTS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--deadline-unix',type=float)
    args=parser.parse_args()
    args.output=args.output or RD/(f'video_{args.phase}_cpucheck.json' if args.check_only else f'video_{args.phase}.json')
    require(not args.output.exists(),f'Preserve existing report: {args.output}')
    manifest,trajectories=protocol(args.manifest)
    if args.phase=='baselines':
        require(not args.artifact_dir.exists(),'Baselines create shared inputs exactly once; artifact directory must be new')
        binding=bind_plain(args.train_report,manifest);baselines=None
    else:
        require(not (args.artifact_dir/SELECTED_ARM).exists(),'Selected artifacts already exist')
        binding,baselines=bind_selected(args,manifest)
    report=dict(experiment='E022_video_evaluation',phase=args.phase,status='running',manifest_reference=file_record(args.manifest),
                manifest=manifest,deadline_unix=args.deadline_unix,limits=['Fixed local held-out text split, not globally unseen.',
                'SVD is a whole-recipe reference; no LR-only causal claim.','Diagnostic wall times are not deployment benchmarks.'])
    report['plain_step0000_binding' if args.phase=='baselines' else 'checkpoint_binding']=binding
    if args.check_only:
        compile(Path(__file__).read_text(),str(Path(__file__)),'exec')
        require('torch' not in sys.modules,'CPU check must not import torch')
        report.update(status='complete',scope='CPU schema, packed-file binding and compile check; no generation',torch_imported=False,cuda_initialized=False,
                      source=file_record(Path(__file__)),trajectories=len(trajectories))
        save_json(args.output,report)
        print(json.dumps(dict(status='complete',phase=args.phase,output=str(args.output),torch_imported=False)));return
    require(args.deadline_unix is not None and args.deadline_unix>time.time(),'Explicit future --deadline-unix required')
    if args.phase=='selected':
        path=Path(manifest['checkpoint_binding']['output'])
        if path.exists():require(json.loads(path.read_text())==binding,'Existing checkpoint binding differs')
        else:save_json(path,binding)
    try:
        execute(args,manifest,trajectories,report,binding,baselines)
    except BaseException:
        report.update(status='failed_partial_preserved',error=traceback.format_exc());save_json(args.output,report);raise
    print(json.dumps(dict(status=report['status'],output=str(args.output),actual_totals=report['actual_totals'])),flush=True)


if __name__=='__main__':main()
