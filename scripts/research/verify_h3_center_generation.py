#!/usr/bin/env python3
"""Thin E038 CPU closure: shared inputs, schedules, observed calls, final states.

Reads 8 small prepared payloads and 24 final dual-modality tensors only. No
model imports, step-tensor reconstruction, quality metrics, or GPU execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('bf16', 'global_mean', 'coarse16')
IDENTITY = ('case_id', 'prompt_id', 'prompt', 'seed', 'replica')
SHAPES = {'video_latents': [1,24,37,36,64], 'audio_latents': [2,32,207]}


def require(value, message):
    if not value: raise RuntimeError(message)


def record(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return dict(file=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def tensor_record(value):
    x=value.detach().cpu().contiguous()
    return dict(shape=list(x.shape),dtype=str(x.dtype),
                sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def run(args):
    manifest=json.loads(args.manifest.read_text());mref=record(args.manifest)
    rd=args.report_dir or Path(manifest['report_dir'])
    paths={'prepare':rd/'prepare.json',**{a:rd/f'denoise_{a}.json' for a in ARMS}}
    reports={k:json.loads(p.read_text()) for k,p in paths.items()}
    # Establish terminal completion before loading any actual input or final tensor.
    require(all(r['status']=='complete' for r in reports.values()),'Wait for prepare and all three denoisers to complete')
    launcher_path=args.launcher or rd/'launcher_v2.json';launch=json.loads(launcher_path.read_text())
    require(launch['status']=='complete','Wait for the original launcher to complete')
    for a in ARMS:
        entries=[s for s in launch['stages'] if s['name']==f'denoise_{a}']
        require(len(entries)==1 and entries[0]['status']=='complete' and entries[0]['returncode']==0,'Worker completion')
        require(entries[0]['result_sha256']==record(paths[a])['sha256'],'Launcher denoiser result binding')
    require(manifest['arms']==list(ARMS) and len(manifest['cases'])==8,'Fixed allocation')
    cases=manifest['cases'];case_ids=[c['case_id'] for c in cases]
    require(all(r['manifest']==mref for r in reports.values()),'Same frozen input manifest')
    prep=reports['prepare'];pref=record(paths['prepare'])
    require(prep['actual_text_encoder_calls']==4 and prep['actual_dit_calls']==0,'Text-only shared preparation')
    require([c['case_id'] for c in prep['cases']]==case_ids,'Prepared case order')
    shared={}
    for case,p in zip(cases,prep['cases'],strict=True):
        require(all(case[k]==p[k] for k in IDENTITY),'Prepared identity')
        actual=record(p['file']);require(all(actual[k]==p[k] for k in ('file','bytes','sha256')),'Shared input file')
        payload=torch.load(p['file'],map_location='cpu',weights_only=True,mmap=True)
        require(all(payload[k]==case[k] for k in IDENTITY),'Prepared payload identity')
        require(tensor_record(payload['embedding'])['sha256']==p['embedding_sha256']
                and tensor_record(payload['text_token_tags'])['sha256']==p['text_token_tags_sha256'],'Text input SHA')
        noises={}
        for key,shape in SHAPES.items():
            x=payload[key];tr=tensor_record(x)
            require(tr['shape']==shape and x.dtype==torch.bfloat16 and bool(torch.isfinite(x).all()),'Initial noise contract')
            require(tr['sha256']==p['initial_noise_sha256'][key],'Shared initial noise SHA')
            noises[key]=tr
        shared[case['case_id']]=dict(prepared=actual,embedding_sha256=p['embedding_sha256'],
                initial_noise_sha256=p['initial_noise_sha256'],noise_tensors=noises,attention_contract=p['attention_contract'])
        del payload
    arms={};total_dit=0
    for arm in ARMS:
        r=reports[arm];require(r['arm']==arm and r['prepared_reference']==pref,'Prepared report binding')
        require([c['case_id'] for c in r['cases']]==case_ids and len(r['prepared_cases'])==8,'Case coverage')
        require(r['schedule_cpu']==prep['schedule_cpu'],'Same actual dual scheduler schedule')
        require(r['complete_dit_calls']==r['attempted_dit_calls']==r['allocated_dit_calls']==160,'No extra/missing DiT')
        expected={'sdpa_calls':102 if arm=='bf16' else 52,'scaled_mm_calls':200,'disk_loads':0}
        fp4=0 if arm=='bf16' else 50;counts={k:0 for k in expected};rows=[]
        for source,binding,c in zip(cases,r['prepared_cases'],r['cases'],strict=True):
            cid=source['case_id'];s=shared[cid]
            require(c['status']=='complete' and all(c[k]==source[k] for k in IDENTITY),'Completed identity')
            for key in ('prepared','initial_noise_sha256','embedding_sha256','attention_contract'):
                require(binding[key]==c[key]==s[key],'Identical bound prepared input across arms: '+key)
            require(c['schedule']==prep['schedule_cpu'] and len(c['steps'])==40 and len(c['dit_calls'])==20,'Sampler call shape')
            for i,step in enumerate(c['steps']):
                modality=('video','audio')[i%2];j=i//2
                require(step['step']==j and step['modality']==modality,'Original scheduler order')
                schedule=prep['schedule_cpu'][modality]
                require(step['timestep']==schedule['timesteps'][j] and step['sigma']==schedule['sigmas'][j],'Observed timestep/sigma')
                require(all(step[k]['finite'] for k in ('latent_before','velocity','latent_after')),'Finite step receipts')
            for j,d in enumerate(c['dit_calls']):
                require(d['step']==j and d['counts']==expected and d['finite'],'Actual per-forward execution')
                require(d['attention']['fp4_calls']==fp4 and d['attention']['finite_flag_count']==0,'Attention execution mode')
                require(d['zero_sf_checks']['checked_calls']==200 and d['zero_sf_checks']['invalid_calls']==0,'Native pack checks')
                for key in counts:counts[key]+=d['counts'][key]
            require(c['runtime_audit']=={k:v*20 for k,v in expected.items()},'Per-case observed counts')
            f=c['final_latents'];require(record(f['file'])==f,'Final artifact SHA')
            value=torch.load(f['file'],map_location='cpu',weights_only=True,mmap=True)
            require(set(value)==set(SHAPES),'Final modalities')
            tensors={}
            for key,shape in SHAPES.items():
                x=value[key];tr=tensor_record(x)
                require(tr==c['final_tensors'][key] and tr['shape']==shape and x.dtype==torch.bfloat16,'Actual final shape/dtype/SHA')
                require(bool(torch.isfinite(x).all()),'Actual final finite')
                tensors[key]={**tr,'finite':True}
            rows.append(dict(case_id=cid,final_latents=f,final_tensors=tensors,
                seconds_including_diagnostics=c['seconds_including_diagnostics'],
                synchronized_dit_seconds_sum=sum(d['synchronized_dit_seconds'] for d in c['dit_calls'])))
            del value
        require(all(r['runtime_audit'][k]==counts[k] for k in counts),'Arm observed totals')
        require(r['actual_fp4_attention_calls']==fp4*160,'Arm FP4 attention total')
        total_dit+=160
        arms[arm]=dict(dit_calls=160,runtime_counts=counts,fp4_attention_calls=fp4*160,cases=rows,
            seconds_total=r['seconds_total'],peak_allocated_gib=r['peak_allocated_bytes']/2**30,
            peak_reserved_gib=r['peak_reserved_bytes']/2**30,export_manifest=r['export_manifest'])
    require(len({a['export_manifest']['sha256'] for a in arms.values()})==1,'Same SVD checkpoint receipt')
    failures=[]
    for path in sorted((rd/'attempt01').glob('*.json')):
        prior=json.loads(path.read_text())
        if str(prior.get('status','')).startswith('failed'):
            failures.append(dict(report=record(path),status=prior['status'],attempted_dit_calls=prior.get('attempted_dit_calls'),
                complete_dit_calls=prior.get('complete_dit_calls'),error=prior.get('error')))
    require(not torch.cuda.is_initialized(),'CPU-only closure')
    return dict(experiment='E038_generation_closure',status='complete',cuda_initialized=False,
        shared_inputs=shared,schedule=prep['schedule_cpu'],arms=arms,actual_dit_calls=total_dit,
        final_artifacts=24,final_modality_tensors=48,observed_scheduler_calls=960,prior_failures_preserved=failures,
        references={k:record(p) for k,p in paths.items()},manifest=mref,launcher=record(launcher_path),source=record(__file__),
        scope='8 prepared payloads + 24 small final artifacts independently loaded; no intermediate tensor chain saved or reconstructed.',
        timing_limit='Generation times contain synchronized diagnostics and concurrent other-arm GPU execution. They are not a steady-state benchmark or a full-model sequence-parallel result.',
        semantics='All arms use native SVD linear weights; bf16 labels attention only. Shape/finite checks do not measure video/audio quality.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'research_state/06_experiments/E038_center_video_manifest.json')
    p.add_argument('--report-dir',type=Path);p.add_argument('--launcher',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA externally')
    output=args.output or (args.report_dir or ROOT/'results/research/E038')/'generation_summary.json'
    require(not output.exists(),'Preserve old output; use --output for another attempt')
    global torch
    import torch
    torch.set_num_threads(6);started=time.monotonic()
    result=dict(experiment='E038_generation_closure',status='failed_stop',source=record(__file__))
    try:result=run(args)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:
        result['seconds']=time.monotonic()-started;output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(output)
        print(json.dumps(dict(status=result['status'],output=str(output),sha256=record(output)['sha256'])))


if __name__=='__main__':main()
