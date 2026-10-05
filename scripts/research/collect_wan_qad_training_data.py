#!/usr/bin/env python3
"""E022: fixed prompt selection, then BF16 rCM teacher trajectories without VAE.

Prepare is CPU/stdlib only. Collect saves E020-compatible actual input/target
records and shared trajectory noise; it does not perform QAD or video scoring.
"""
from __future__ import annotations
import argparse,gc,json,math,os,random,time,traceback,unicodedata
from pathlib import Path
from generate_wan_qad_comparison import ROOT,BASE,RCM,sha256,save_json

RD=ROOT/'results/research/E022'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E022/data')
PLAN=ROOT/'research_state/06_experiments/E022_expanded_qad_plan.md'
OLD=ROOT/'results/research/E020/E020_cache_inventory.json'
E021=ROOT/'results/research/E021/generation_manifest.draft.json'
TOYS=[ROOT/f'research_state/06_experiments/{name}' for name in
      ('E012_h3_conditional_response_manifest.json','E013_h3_behavior_manifest.json')]
MANIFEST=RD/'data_manifest.json'
SHUFFLE_SEED=20261030
SEED_BASE=20261030


def file_record(path):
    p=Path(path).resolve()
    return dict(file=str(p),sha256=sha256(p),bytes=p.stat().st_size)


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def make_manifest():
    inventory=json.loads(OLD.read_text());assert inventory['status']=='complete'
    prompt_source=Path(inventory['later_generation_source']['manifest']['file'])
    source_record=inventory['later_generation_source']['manifest']
    assert sha256(prompt_source)==source_record['sha256']
    rows=json.loads(prompt_source.read_text())['all_cases']
    excluded=[]
    for group in inventory['distribution']['groups']:
        excluded.extend(dict(source='E020_known_calibration',id=p['prompt_id'],prompt=p['prompt']) for p in group['prompts'])
    assert len(excluded)==20
    excluded.extend(dict(source='E021',id=p['case_id'],prompt=p['prompt']) for p in json.loads(E021.read_text())['cases'])
    for path in TOYS:
        prior=json.loads(path.read_text())
        excluded.extend(dict(source=path.stem,id=str(p.get('id',p.get('prompt_id'))),prompt=p['prompt'])
                        for p in prior.get('conditions',prior.get('cases',[])))
    blocked={normalized(p['prompt']) for p in excluded}
    eligible=[];seen=set();rejected=[]
    for row in sorted(rows,key=lambda r:r['case_id']):
        key=normalized(row['prompt'])
        if key in blocked or key in seen:
            rejected.append(dict(case_id=row['case_id'],reason='excluded_text' if key in blocked else 'duplicate_text'))
            continue
        seen.add(key);eligible.append(row)
    random.Random(SHUFFLE_SEED).shuffle(eligible)
    assert len(eligible)>=36
    selected=[]
    for index,row in enumerate(eligible[:36]):
        selected.append(dict(index=index,prompt_id=row['case_id'],prompt=row['prompt'],dimensions=row['dimensions'],
            split='train' if index<32 else 'validation',seeds=[SEED_BASE+index*2+j for j in range(2)]))
    paths=[Path(__file__),PLAN,OLD,E021,*TOYS,prompt_source,Path(__file__).with_name('generate_wan_qad_comparison.py'),
           BASE/'model_index.json',BASE/'text_encoder/config.json',RCM/'config.json']
    assets=sorted((BASE/'text_encoder').glob('*.safetensors'))+sorted(RCM.glob('*.safetensors'))
    assert len(assets)==6
    return dict(experiment='E022',status='prepared_not_collected',cases=selected,
        selection=dict(source_cases=len(rows),eligible_unique_texts=len(eligible),shuffle_seed=SHUFFLE_SEED,
            rule='Sort source case_id, remove normalized exclusions/duplicates, Python Random(seed).shuffle, first32 train then4 validation',
            text_normalization='Unicode NFKC, casefold, retain alphanumeric characters only',
            rejected_source_cases=rejected,excluded_prompts=excluded,seed_rule='20261030 + 2*manifest_index + replica',
            limits='New E022 optimizer prompts, not globally unseen; no model-output/score-based selection.'),
        settings=dict(height=480,width=832,frames=77,steps=4,sigma_max=80.,guidance=0.,
            latent_shape=[1,16,20,60,104],patch_tokens=31200,max_sequence_length=512,
            attention='BF16_FLASH_ATTENTION',angles='[atan(80),1.5,1.4,1.0,0.0] FP64',
            actual_bf16_timesteps=[988.,932.,852.,608.],vae=False),
        counts=dict(train_prompts=32,validation_prompts=4,replicas=2,train_records=256,validation_records=32,teacher_calls=288),
        model=dict(base_assets=str(BASE),explicit_rcm_transformer=str(RCM)),
        sources={str(p.resolve()):file_record(p) for p in paths},
        asset_metadata={str(p.resolve()):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in assets},
        asset_policy='Bind local weight size/mtime before loading, plus hashed model/config/source files; no new full large-weight hash',
        data_dir=str(DATA),gpu=0,wall_seconds=1200,max_peak_allocated_gib=60)


def collect(args,manifest,report):
    import torch
    import torch.nn.functional as F
    import diffusers
    from diffusers import WanPipeline,WanTransformer3DModel
    from torch.nn.attention import sdpa_kernel,SDPBackend
    assert manifest==make_manifest(),'Manifest/source/asset metadata changed after CPU prepare'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert torch.cuda.get_device_capability()==(12,0)
    assert torch.__version__=='2.11.0+cu128' and diffusers.__version__=='0.33.1'
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    DATA.mkdir(parents=True,exist_ok=False)
    report.update(manifest=file_record(args.manifest),sources=manifest['sources'],asset_metadata=manifest['asset_metadata'],
        environment=dict(python=os.sys.executable,torch=torch.__version__,diffusers=diffusers.__version__,
                         device=torch.cuda.get_device_name(),cuda_visible_devices='0'),embeddings={},trajectories=[])
    def budget():
        if time.time()>=args.deadline_unix:raise TimeoutError('E022 data collection deadline expired')
        if torch.cuda.max_memory_allocated()>60*1024**3:raise MemoryError('E022 60GiB allocation guard')
    def digest(t):
        import hashlib
        return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    def artifact(path,value):
        assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(value,path);return file_record(path)
    def checkpoint():
        budget();save_json(args.output,report)
    inventory=dict(experiment='E022',status='collecting',manifest=report['manifest'],train=[],validation=[],
        teacher_target_policy='Actual unquantized BF16 rCM teacher output on its own trajectory; do not recompute teacher in trainer',
        common_tensor_contract=dict(hidden_shape=[1,16,20,60,104],embedding_shape=[1,512,4096],dtype='torch.bfloat16'),
        model=manifest['model'],sources=manifest['sources'])
    with torch.inference_mode():
        budget()
        pipe=WanPipeline.from_pretrained(BASE,transformer=None,vae=None,torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.text_encoder.cuda().eval()
        angles=torch.tensor([math.atan(80.),1.5,1.4,1.,0.],device='cuda',dtype=torch.float64)
        t_steps=angles.sin()/(angles.cos()+angles.sin());ones=torch.ones(1,device='cuda',dtype=torch.float64)
        report['schedule']=dict(t_steps=t_steps.cpu().tolist(),timesteps_bf16=[(t.float()*ones*1000).to(torch.bfloat16).item() for t in t_steps[:-1]])
        assert report['schedule']['timesteps_bf16']==manifest['settings']['actual_bf16_timesteps']
        # Save each embedding once separately. Records additionally include it
        # for compatibility with the existing trainer's direct torch.load path.
        for case in manifest['cases']:
            budget();prompt_id=case['prompt_id']
            embedding,_=pipe.encode_prompt(prompt=case['prompt'],do_classifier_free_guidance=False,
                max_sequence_length=512,device=torch.device('cuda'),dtype=torch.bfloat16)
            assert embedding.shape==(1,512,4096) and embedding.dtype==torch.bfloat16 and torch.isfinite(embedding).all()
            emb_ref=artifact(DATA/'embeddings'/f'{prompt_id}.pt',embedding.cpu())
            report['embeddings'][prompt_id]=dict(artifact=emb_ref,tensor_sha256=digest(embedding))
            for replica,seed in enumerate(case['seeds']):
                generator=torch.Generator(device='cuda').manual_seed(seed)
                initial_noise=pipe.prepare_latents(batch_size=1,num_channels_latents=16,height=480,width=832,
                    num_frames=77,dtype=torch.float32,device=torch.device('cuda'),generator=generator)
                initial=initial_noise.to(torch.float64)*t_steps[0]
                assert initial.shape==(1,16,20,60,104)
                noises=[torch.randn(initial.shape,device='cuda',dtype=torch.float32,generator=generator) for _ in range(4)]
                identity=f'{prompt_id}_r{replica}'
                shared=dict(prompt_id=prompt_id,prompt=case['prompt'],seed=seed,replica=replica,split=case['split'],
                    initial_latent_fp64=initial.cpu(),update_noises_fp32=[n.cpu() for n in noises],
                    embedding_ref=emb_ref,t_steps_fp64=t_steps.cpu(),generator_state_after_draws=generator.get_state().cpu())
                ref=artifact(DATA/'trajectories'/f'{identity}.pt',shared)
                report['trajectories'].append(dict(trajectory_id=identity,prompt_id=prompt_id,split=case['split'],seed=seed,replica=replica,
                    artifact=ref,initial_latent_sha256=digest(initial),initial_noise_sha256=digest(initial_noise),
                    noise_sha256=[digest(n) for n in noises],embedding_sha256=digest(embedding),steps=[]))
                del shared,initial_noise,initial,noises,generator
            del embedding
            checkpoint()
        del pipe
        gc.collect();torch.cuda.empty_cache();budget()
        model=WanTransformer3DModel.from_pretrained(RCM,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        old_sdpa,old_mm=F.scaled_dot_product_attention,F.scaled_mm
        actual=dict(sdpa=0,native_mm=0)
        def sdpa(q,k,v,*pos,**kw):
            assert q.dtype==k.dtype==v.dtype==torch.bfloat16
            actual['sdpa']+=1;return old_sdpa(q,k,v,*pos,**kw)
        def mm(*pos,**kw):
            actual['native_mm']+=1;raise RuntimeError('BF16 teacher unexpectedly called scaled_mm')
        F.scaled_dot_product_attention,F.scaled_mm=sdpa,mm
        try:
            for trajectory in report['trajectories']:
                budget();ref=trajectory['artifact'];assert sha256(ref['file'])==ref['sha256']
                shared=torch.load(ref['file'],map_location='cpu',weights_only=True)
                embedding=torch.load(shared['embedding_ref']['file'],map_location='cpu',weights_only=True).cuda()
                assert digest(embedding)==trajectory['embedding_sha256']
                latents=shared['initial_latent_fp64'].cuda();noises=[n.cuda() for n in shared['update_noises_fp32']]
                for step,(t_cur,t_next) in enumerate(zip(t_steps[:-1],t_steps[1:])):
                    budget();hidden=latents.to(torch.bfloat16);timestep=(t_cur.float()*ones*1000).to(torch.bfloat16)
                    before=dict(actual);report['teacher_calls_attempted']+=1
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        velocity=model(hidden_states=hidden,timestep=timestep,encoder_hidden_states=embedding,return_dict=False)[0]
                    assert actual['sdpa']-before['sdpa']==60 and actual['native_mm']==0
                    assert velocity.shape==hidden.shape and velocity.dtype==torch.bfloat16 and torch.isfinite(velocity).all()
                    report['complete_dit_calls']+=1
                    payload=dict(filename=trajectory['prompt_id'],prompt=shared['prompt'],seed=trajectory['seed'],replica=trajectory['replica'],
                        step=step,guidance=0,trajectory_ref=ref,input_args=(hidden.cpu(),),
                        input_kwargs=dict(timestep=timestep.cpu(),encoder_hidden_states=embedding.cpu(),return_dict=False),outputs=(velocity.cpu(),))
                    path=DATA/'records'/trajectory['split']/f'{trajectory["trajectory_id"]}_s{step}.pt'
                    record=dict(**artifact(path,payload),prompt_id=trajectory['prompt_id'],step=step,seed=trajectory['seed'],replica=trajectory['replica'],
                        split=trajectory['split'],trajectory_id=trajectory['trajectory_id'],input_sha256=digest(hidden),
                        embedding_sha256=trajectory['embedding_sha256'],teacher_output_sha256=digest(velocity),timestep=timestep.item())
                    inventory[trajectory['split']].append(record)
                    latents=(1-t_next)*(latents-t_cur*velocity.to(torch.float64))+t_next*noises[step]
                    assert torch.isfinite(latents).all()
                    trajectory['steps'].append(dict(**record,actual_bf16_sdpa=60,actual_native_mm=0,finite=True,
                        latent_after_sha256=digest(latents),update_noise_sha256=trajectory['noise_sha256'][step]))
                    del hidden,timestep,velocity,payload
                trajectory['final_latent']=artifact(DATA/'final_latents'/f'{trajectory["trajectory_id"]}.pt',latents.cpu())
                del shared,embedding,latents,noises
                checkpoint();print('teacher complete',trajectory['trajectory_id'],report['complete_dit_calls'],flush=True)
        finally:
            F.scaled_dot_product_attention,F.scaled_mm=old_sdpa,old_mm
            report['actual_total_calls']=actual
        assert len(inventory['train'])==256 and len(inventory['validation'])==32
        assert report['teacher_calls_attempted']==report['complete_dit_calls']==288 and actual==dict(sdpa=17280,native_mm=0)
        assert len({r['initial_noise_sha256'] for r in report['trajectories']})==72
        del model,p
        gc.collect();torch.cuda.empty_cache();budget()
    report.update(status='complete',inventory_file=str(args.inventory),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    inventory.update(status='complete',actual_total_calls=actual,complete_dit_calls=288)
    return inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','collect'),required=True)
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--output',type=Path,default=RD/'collect_run.json')
    parser.add_argument('--inventory',type=Path,default=RD/'data_inventory.json')
    parser.add_argument('--deadline-unix',type=float)
    args=parser.parse_args()
    if args.phase=='prepare':
        assert not args.manifest.exists()
        manifest=make_manifest();save_json(args.manifest,manifest)
        import ast
        ast.parse(Path(__file__).read_text())
        check=dict(status='complete',cuda_initialized=False,torch_imported=False,phase='CPU manifest selection/AST only',
            manifest=file_record(args.manifest),source=file_record(__file__),counts=manifest['counts'],
            selected_prompt_ids=[c['prompt_id'] for c in manifest['cases']])
        save_json(RD/'collect_cpu_check.json',check);print(json.dumps(check),flush=True);return
    assert not args.output.exists() and not args.inventory.exists()
    assert args.deadline_unix is not None
    started=time.time();args.deadline_unix=min(args.deadline_unix,started+1200)
    report=dict(experiment='E022_teacher',status='running',teacher_calls_attempted=0,complete_dit_calls=0,deadline_unix=args.deadline_unix)
    inventory=None
    try:
        manifest=json.loads(args.manifest.read_text());inventory=collect(args,manifest,report)
    except BaseException:report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds_total']=time.time()-started;save_json(args.output,report);print(report['status'],flush=True)
    if inventory is not None:
        inventory['collection_report']=file_record(args.output)
        save_json(args.inventory,inventory)


if __name__=='__main__':main()
