#!/usr/bin/env python3
"""Lightweight CPU verification of a COMPLETE E022 video generation phase.

Verify saved input/output identities, per-step counters and fully decoded media.
No GPU replay, checkpoint-weight hashes or source-tree freeze. Contact sheets
use replica0 only, fixed frames, and separate baseline/selected stage folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E022'
BASELINE_ARMS=('bf16','plain_step0000','svd_lr')
SELECTED_ARM='qad_native_dev_selected'
FRAME_INDICES=(0,19,38,57,76)
FFMPEG=Path('/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def verify(args,generation,result):
    # Isolated CPU process; importing torch below does not initialize CUDA.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['IMAGEIO_FFMPEG_EXE']=str(args.ffmpeg)
    import torch
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    require(args.ffmpeg.is_file(),'Local FFmpeg executable missing')
    manifest=generation['manifest'];cases=manifest['cases'];trajectories=manifest['trajectories']
    ids=[t['trajectory_id'] for t in trajectories]
    prompts={c['prompt_id']:c for c in cases}
    arms=BASELINE_ARMS if args.phase=='baselines' else (SELECTED_ARM,)
    expected_totals=dict(dit_calls=192,native_mm_calls=38400,sdpa_calls=11520,videos=48) if args.phase=='baselines' else dict(dit_calls=64,native_mm_calls=19200,sdpa_calls=3840,videos=16)
    require(len(cases)==8 and len(ids)==len(set(ids))==16,'Expected fixed eight-prompt/two-replica protocol')
    require(set(generation['arms'])==set(arms) and set(generation['shared_inputs'])==set(ids),'Incomplete phase arms/shared inputs')
    require(generation['manifest_reference']['sha256']==sha(generation['manifest_reference']['path']),'Video manifest changed')
    result.update(manifest_reference=generation['manifest_reference'],files=[],shared_inputs={},arms={},contact_sheets=[],
        video_validator=dict(tool='Full CPU FFmpeg software decode via imageio-ffmpeg',executable=str(args.ffmpeg)),
        limitations=['Execution counts are independently summed from saved per-step records, not a second GPU trace.',
          'Initial FP64 latent, four FP32 noises and BF16 embedding are hashed from saved CPU tensors; raw pre-scaled initial FP32 noise was not saved separately.',
          'No source-tree or training-weight hashes are recomputed.',
          'Contact sheets show predetermined replica0 frames only and do not establish motion quality.'])
    stage_reports={a:generation for a in arms}
    if args.phase=='selected':
        base=json.loads(args.baselines.read_text())
        require(base['status']=='complete' and base['phase']=='baselines','Completed baseline report required')
        require(base['manifest_reference']==generation['manifest_reference'] and base['schedule']==generation['schedule'],'Selected phase changed manifest/schedule')
        require(base['shared_inputs']==generation['shared_inputs'],'Selected phase did not reuse the exact baseline shared-input records')
        require(generation['checkpoint_binding']['baseline_report']['sha256']==sha(args.baselines),'Selected phase bound a different baseline report')
        result['baseline_reference']=dict(path=str(args.baselines),sha256=sha(args.baselines))
        result['selected_shared_inputs_identical_to_baselines']=True
        for arm in BASELINE_ARMS:
            stage_reports[arm]=base
            require(set(base['arms'][arm]['trajectories'])==set(ids),'Missing baseline trajectories')
        display_arms=(*BASELINE_ARMS,SELECTED_ARM)
    else:display_arms=BASELINE_ARMS

    def file_check(record,category):
        p=Path(record['path']);actual=sha(p)
        require(actual==record['sha256'] and p.stat().st_size==record['bytes'],f'{category} file SHA/size differs: {p}')
        result['files'].append(dict(path=str(p),category=category,sha256=actual,bytes=p.stat().st_size,match=True))

    def tensor_sha(value):
        require(value.device.type=='cpu','Verifier must keep tensors on CPU')
        return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

    for trajectory in trajectories:
        tid=trajectory['trajectory_id'];shared=generation['shared_inputs'][tid]
        file_check(shared['artifact'],'shared_input')
        payload=torch.load(shared['artifact']['path'],map_location='cpu',weights_only=True)
        require(all(payload[k]==v for k,v in trajectory.items()),f'{tid}: shared identity differs')
        require(payload['prompt']==prompts[trajectory['prompt_id']]['prompt'],f'{tid}: shared prompt differs')
        initial=payload['initial_latent_fp64'];noises=payload['update_noises_fp32'];embedding=payload['embedding_bf16']
        require(initial.shape==(1,16,20,60,104) and initial.dtype==torch.float64,f'{tid}: initial contract')
        require(len(noises)==4 and all(n.shape==initial.shape and n.dtype==torch.float32 for n in noises),f'{tid}: noise contract')
        require(embedding.shape==(1,512,4096) and embedding.dtype==torch.bfloat16,f'{tid}: embedding contract')
        require(all(bool(torch.isfinite(t).all()) for t in [initial,embedding,*noises]),f'{tid}: nonfinite shared tensors')
        actual=dict(initial_latent_sha256=tensor_sha(initial),noise_sha256=[tensor_sha(n) for n in noises],embedding_sha256=tensor_sha(embedding))
        require(all(actual[k]==shared[k] for k in actual),f'{tid}: saved tensor hash differs')
        require(payload['t_steps_fp64'].tolist()==generation['schedule']['t_steps'],f'{tid}: saved schedule differs')
        for arm,source in stage_reports.items():
            row=source['arms'][arm]['trajectories'][tid]
            require(row['shared_inputs']==shared['artifact']['path'] and all(row[k]==actual[k] for k in actual),f'{tid}/{arm}: arm did not reference identical actual shared tensors')
        result['shared_inputs'][tid]=actual|dict(seed=trajectory['seed'],all_available_arms_match=True)
        del payload,initial,noises,embedding
    require(len({r['initial_latent_sha256'] for r in result['shared_inputs'].values()})==16,'Actual initial tensors are not distinct across sixteen trajectories')
    require(len({t['seed'] for t in trajectories})==16,'Trajectory seeds repeat')
    result['prompt_replica_independence']={}
    for case in cases:
        rows=[t for t in trajectories if t['prompt_id']==case['prompt_id']]
        require([t['replica'] for t in rows]==[0,1] and [t['seed'] for t in rows]==case['seeds'],f'{case["prompt_id"]}: replica/seed contract')
        hashes=[result['shared_inputs'][t['trajectory_id']]['initial_latent_sha256'] for t in rows]
        require(len(set(hashes))==2,f'{case["prompt_id"]}: two replicas share an initial tensor')
        result['prompt_replica_independence'][case['prompt_id']]=dict(seeds=case['seeds'],actual_initial_sha256=hashes,distinct=True)

    thumbs={}
    def decode(path,keep):
        frames=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24',input_params=['-threads','1'],output_params=['-threads','1'])
        metadata=next(frames);selected={};count=0
        for index,frame in enumerate(frames):
            count+=1
            if keep and index in FRAME_INDICES:
                selected[index]=Image.frombytes('RGB',tuple(metadata['size']),frame).resize((416,240),Image.Resampling.LANCZOS)
        require(count==77 and tuple(metadata['size'])==(832,480) and math.isclose(float(metadata['fps']),16.0,abs_tol=1e-8),f'Actual video contract differs: {path}')
        if keep:require(set(selected)==set(FRAME_INDICES),f'Sampled frames missing: {path}')
        return dict(decoded_frames=count,width=metadata['size'][0],height=metadata['size'][1],fps=metadata['fps'],duration=metadata.get('duration'),codec=metadata.get('codec')),selected

    counts=dict(dit_calls=0,native_mm_calls=0,sdpa_calls=0,videos=0)
    for arm in arms:
        source=generation['arms'][arm]
        require(source['status']=='complete' and set(source['trajectories'])==set(ids),f'{arm}: phase incomplete')
        expected=0 if arm=='bf16' else 300
        require(source['native_modules']==expected,f'{arm}: native module count differs')
        result['arms'][arm]={}
        for trajectory in trajectories:
            tid=trajectory['trajectory_id'];row=source['trajectories'][tid]
            require(row['status']=='complete' and len(row['steps'])==4 and row['actual_dit_calls']==4,f'{arm}/{tid}: four-step completion required')
            require(all(row[k]==v for k,v in trajectory.items()) and row['prompt']==prompts[trajectory['prompt_id']]['prompt'],f'{arm}/{tid}: identity differs')
            for i,step in enumerate(row['steps']):
                require(step['step']==i and step['timestep']==[988.,932.,852.,608.][i],f'{arm}/{tid}/{i}: timestep differs')
                require(step['actual_native_mm_calls']==expected and step['validated_fastpack_flags']==expected and step['actual_sdpa_calls']==60,f'{arm}/{tid}/{i}: execution count differs')
                require(step['noise_sha256']==result['shared_inputs'][tid]['noise_sha256'][i],f'{arm}/{tid}/{i}: noise differs')
                counts['dit_calls']+=1;counts['native_mm_calls']+=step['actual_native_mm_calls'];counts['sdpa_calls']+=step['actual_sdpa_calls']
            require(row['actual_native_mm_calls']==sum(s['actual_native_mm_calls'] for s in row['steps']),f'{arm}/{tid}: subtotal differs')
            file_check(row['final_latent'],'final_latent');file_check(row['video'],'video')
            latent=torch.load(row['final_latent']['path'],map_location='cpu',weights_only=True)
            require(latent.shape==(1,16,20,60,104) and latent.dtype==torch.float64 and bool(torch.isfinite(latent).all()),f'{arm}/{tid}: final tensor contract')
            latent_hash=tensor_sha(latent)
            require(latent_hash==row['final_latent_sha256']==row['steps'][-1]['latent_after_sha256'],f'{arm}/{tid}: final tensor SHA differs')
            del latent
            media,selected=decode(row['video']['path'],trajectory['replica']==0)
            require(all(row['video'][k]==v for k,v in dict(frames=77,width=832,height=480,fps=16).items()),f'{arm}/{tid}: recorded media metadata differs')
            if selected:thumbs[(trajectory['prompt_id'],arm)]=selected
            result['arms'][arm][tid]=dict(status='verified',actual_dit_calls=4,actual_native_mm_calls=row['actual_native_mm_calls'],final_latent_tensor_sha256=latent_hash,video_sha256=row['video']['sha256'],actual_video=media)
            counts['videos']+=1
        print(json.dumps(dict(phase=args.phase,arm=arm,verified_videos=16)),flush=True)
        result['recomputed_totals']=dict(counts);save(args.output,result)
    require(counts==expected_totals==generation['actual_totals'],'Recomputed phase totals differ')

    contact_dir=args.contact_dir or Path(generation['artifact_dir'])/'contact_sheets'/args.phase
    require(not contact_dir.exists(),f'Preserve previous stage contact sheets: {contact_dir}')
    contact_dir.mkdir(parents=True)
    try:
        fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        regular=ImageFont.truetype(fontpath,22);small=ImageFont.truetype(fontpath,18);title=ImageFont.truetype(fontpath,26)
    except OSError:
        regular=ImageFont.load_default(size=22);small=ImageFont.load_default(size=18);title=ImageFont.load_default(size=26)
    labels=dict(bf16='BF16',plain_step0000='Plain step 0000',svd_lr='SVD + LR (frozen)',qad_native_dev_selected='QAD dev-selected')
    for case in cases:
        cid=case['prompt_id'];tid=f'{cid}_r0'
        # In selected stage baseline artifacts were already verified previously;
        # read only replica0 videos needed for the combined fixed-frame display.
        for arm in display_arms:
            if (cid,arm) not in thumbs:
                row=stage_reports[arm]['arms'][arm]['trajectories'][tid]
                require(row['status']=='complete',f'{arm}/{tid}: cannot display incomplete baseline')
                file_check(row['video'],'baseline_replica0_contact_source')
                _,thumbs[(cid,arm)]=decode(row['video']['path'],True)
        left,top,gap,rowgap=238,135,10,20
        width=left+5*(416+gap)+12;height=top+len(display_arms)*(240+rowgap)+52
        sheet=Image.new('RGB',(width,height),(247,248,250));draw=ImageDraw.Draw(sheet)
        draw.text((18,14),f'E022 | {args.phase} | {cid} replica0 | seed {case["seeds"][0]}',font=title,fill=(18,25,35))
        draw.text((18,54),f'Prompt: {case["prompt"]}',font=regular,fill=(30,40,52))
        for col,index in enumerate(FRAME_INDICES):draw.text((left+col*(416+gap),100),f'frame {index} | {index/16:.4f} s',font=small,fill=(35,45,60))
        for row,arm in enumerate(display_arms):
            y=top+row*(240+rowgap)
            draw.text((18,y+100),labels[arm],font=small,fill=(20,28,40))
            for col,index in enumerate(FRAME_INDICES):sheet.paste(thumbs[(cid,arm)][index],(left+col*(416+gap),y))
        draw.text((18,height-38),'Fixed replica0 and frame indices for every prompt. Static samples do not establish motion quality.',font=small,fill=(65,70,82))
        path=contact_dir/(cid+'.png');sheet.save(path)
        result['contact_sheets'].append(dict(path=str(path),sha256=sha(path),prompt_id=cid,replica=0,seed=case['seeds'][0],row_order=list(display_arms),frame_indices=list(FRAME_INDICES),size=list(sheet.size)))
    require(not torch.cuda.is_initialized(),'Verifier unexpectedly initialized CUDA')
    result.update(status='complete',cuda_initialized=False,files_checked=len(result['files']),recomputed_totals=counts,
                  contact_sheet_count=len(result['contact_sheets']),contact_dir=str(contact_dir))
    save(args.output,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('baselines','selected'),required=True)
    parser.add_argument('--generation',type=Path)
    parser.add_argument('--baselines',type=Path,default=RD/'video_baselines.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--contact-dir',type=Path)
    parser.add_argument('--ffmpeg',type=Path,default=FFMPEG)
    args=parser.parse_args()
    args.generation=args.generation or RD/f'video_{args.phase}.json'
    args.output=args.output or RD/f'video_{args.phase}_validation.json'
    require(not args.output.exists(),f'Preserve previous validation: {args.output}')
    generation=json.loads(args.generation.read_text())
    require(generation['status']=='complete' and generation['phase']==args.phase,'Wait for matching generation phase COMPLETE')
    result=dict(experiment='E022_video_evaluation',phase=args.phase,status='running',scope='CPU persisted-artifact consistency and full software media decode; no quality scoring',
                generation=dict(path=str(args.generation),sha256=sha(args.generation)))
    try:verify(args,generation,result)
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());save(args.output,result);raise
    print(json.dumps(dict(status=result['status'],phase=args.phase,output=str(args.output),totals=result['recomputed_totals'],contact_dir=result['contact_dir'],cuda_initialized=False)),flush=True)


if __name__=='__main__':main()
