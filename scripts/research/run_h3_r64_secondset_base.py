#!/usr/bin/env python3
"""Reproducible 768p white-box BF16 generation, with per-case checkpoints."""
import argparse
import importlib
import json
from pathlib import Path
import random
import time
import traceback
import types

import av
import torch
import run_e121_h3_split_generation as e121
from minimax_h3_svdquant_common import read_prompt

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'results/research/E150'
DATA=Path('/data1/models/svdquant-wjq/research/20261101/E150_whitebox_bf16')
SOURCE=Path('/home/wjq/workspace/178866172854036')
MANIFEST=REPORT/'manifest.json'
ARM='bf16'

def prepare():
    assert not MANIFEST.exists()
    source=[json.loads(line) for line in SOURCE.open() if line.strip()]
    assert len(source)==124 and len({r['prompt_id'] for r in source})==124
    long=[r for r in source if int(r['num_frames'])==361]
    short=[r for r in source if int(r['num_frames'])!=361]
    assert len(long)==82 and len(short)==42
    # Stratify by dataset category: shuffle within category, pick the candidate
    # with least selected fraction, and ensure the historical race is retained.
    rng=random.Random(1502026)
    groups={}
    for r in long: groups.setdefault(r['category'],[]).append(r)
    for group in groups.values(): rng.shuffle(group)
    race=next(r for r in long if int(r['prompt_id'])==16)
    groups[race['category']].remove(race)
    chosen=[race]
    counts={c:int(c==race['category']) for c in groups}
    sizes={c:len(g)+counts[c] for c,g in groups.items()}
    while len(chosen)<41:
        eligible=[c for c,g in groups.items() if g]
        c=min(eligible,key=lambda k:(counts[k]/sizes[k],k))
        chosen.append(groups[c].pop());counts[c]+=1
    selected=short+chosen
    rows=[]
    for r in selected:
        n=int(r['num_frames']); aligned=((n-5+16)//17)*17+5
        rows.append({'prompt_id':int(r['prompt_id']),'seed':int(r['seed']),
                     'category':r['category'],'source_duration':r['duration'],
                     'source_frames':n,'frames':aligned})
    # Greedy balancing of predicted cost; deterministic tie breaking.
    groups_out=[{'gpu':i,'cases':[],'estimated_cost':0.0} for i in range(6)]
    for r in sorted(rows,key=lambda x:(-x['frames'],x['prompt_id'])):
        g=min(groups_out,key=lambda x:(x['estimated_cost'],x['gpu']))
        g['cases'].append(r['prompt_id']);g['estimated_cost']+=r['frames']**2
    result={'format':'e150-h3-whitebox-v1','prompt_source':str(SOURCE),'prompt_source_sha256':e121.base.sha256(SOURCE),
            'selection':'all non-361-frame prompts plus 41 of 82 361-frame prompts, stratified seeded 1502026 with race p16 forced',
            'settings':{'height':768,'width':1344,'steps':20,'cfg_scale':1.0,'flow_shift':12.0,'audio_flow_shift':3.0,'fps':24,'rand_device':'cpu'},
            'selected':sorted(rows,key=lambda x:x['prompt_id']),'groups':groups_out}
    REPORT.mkdir(parents=True,exist_ok=True)
    e121.base.save_json(MANIFEST,result)
    print(json.dumps({'selected':len(rows),'groups':groups_out},indent=2))

class DenoiseComplete(Exception): pass

@torch.inference_mode()
def denoise(pid):
    manifest=json.loads(MANIFEST.read_text())
    assert e121.base.sha256(SOURCE)==manifest['prompt_source_sha256']
    info=next(x for x in manifest['selected'] if x['prompt_id']==pid)
    prompt=read_prompt(pid); assert int(prompt['seed'])==info['seed']
    case=f'p{pid:03d}'; folder=REPORT/case; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'bf16_denoise.json'; stage=DATA/case/ARM; terminal=stage/'final_latents.pt'
    assert not path.exists() and not stage.exists()
    stage.mkdir(parents=True)
    report={'status':'running','prompt_id':pid,'seed':info['seed'],'prompt':prompt['prompt'],
            'source_frames':info['source_frames'],'frames':info['frames'],'settings':manifest['settings'],
            'manifest_sha256':e121.base.sha256(MANIFEST),'steps':[],'started_unix':time.time()}
    e121.base.save_json(path,report)
    router=None
    try:
        pipe=e121.base.load_h3_pipeline(full=True,vram_limit_gib=18.0)
        pipe.load_models_to_device(['dit']);pipe.dit.eval()
        report['linears']=e121.base.checked_linears(pipe.dit)
        router=e121.Router(pipe.dit,'bf16');report['attention_router']=router.manifest
        original_forward=pipe.dit.forward;original_step=pipe.step;original_loader=pipe.load_models_to_device
        final={};initial={}
        def forward(_self,*args,**kw):
            assert torch.cuda.mem_get_info()[0]>5*1024**3
            contract=e121.base.attention_contract(kw);torch.cuda.synchronize();started=time.monotonic()
            with router.forward_context(**contract,diagnostics=False) as log:
                output=original_forward(*args,**kw);torch.cuda.synchronize()
            video,audio=e121.base.split_output(output)
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            report['steps'].append({'step':len(report['steps']),'dit_seconds':time.monotonic()-started,
                'main_tokens':contract['expected_cu'][1],'attention_main_calls':log.summary['main_helper_calls']})
            e121.base.save_json(path,report)
            print(f'{case} step {len(report["steps"])}/20 {report["steps"][-1]["dit_seconds"]:.2f}s',flush=True)
            return output
        def step(_self,scheduler,latents,progress_id,noise_pred,**kw):
            name='video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
            assert name and progress_id==len(report['steps'])-1
            if progress_id==0:
                initial[name]=e121.base.tensor_sha256(latents)
                refroot=ROOT/('results/research/E124' if pid==16 else 'results/research/E150')
                reference=json.loads((refroot/f'p{pid:03d}/bf16_denoise.json').read_text())
                assert initial[name]==reference['initial_latent_sha256'][name], (pid,name,'initial latent mismatch')
            updated=original_step(scheduler,latents,progress_id,noise_pred=noise_pred,**kw)
            assert torch.isfinite(updated).all()
            if progress_id==19:final[name+'_latents']=updated.detach().cpu()
            return updated
        def stop(_self,names):
            if list(names)==['video_vae'] and len(report['steps'])==20:raise DenoiseComplete()
            return original_loader(names)
        pipe.dit.forward=types.MethodType(forward,pipe.dit);pipe.step=types.MethodType(step,pipe)
        pipe.load_models_to_device=types.MethodType(stop,pipe)
        try:
            pipe(prompt=prompt['prompt'],seed=info['seed'],height=manifest['settings']['height'],width=manifest['settings']['width'],num_frames=info['frames'],
                 num_inference_steps=20,cfg_scale=1.0,flow_shift=12.0,audio_flow_shift=3.0,rand_device='cpu',tiled=True)
        except DenoiseComplete:pass
        else:raise RuntimeError('VAE boundary not reached')
        assert len(report['steps'])==20 and set(final)=={'video_latents','audio_latents'} and set(initial)=={'video','audio'}
        with terminal.open('xb') as f:torch.save(final,f)
        report.update(status='denoise_complete',terminal_latents=str(terminal),terminal_sha256=e121.base.sha256(terminal),
                      initial_latent_sha256=initial,completed_unix=time.time())
    except BaseException:
        report.update(status='failed',error=traceback.format_exc(),failed_unix=time.time());raise
    finally:
        e121.base.save_json(path,report)
        if router is not None:router.close()

@torch.inference_mode()
def decode(pid):
    from diffsynth.utils.data.audio_video import write_video_audio
    case=f'p{pid:03d}';folder=REPORT/case;row=json.loads((folder/'bf16_denoise.json').read_text())
    terminal=Path(row['terminal_latents']);assert row['status']=='denoise_complete' and e121.base.sha256(terminal)==row['terminal_sha256']
    height=row['settings']['height'];width=row['settings']['width'];path=DATA/case/ARM/f'{case}_bf16_{width}x{height}_{row["frames"]}f_20steps.mp4';out=folder/'bf16_video.json'
    assert not path.exists() and not out.exists()
    pipe=e121.base.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda',model_configs=[
        e121.base.ModelConfig(path=e121.base.local_h3_files('FL2VA/video_vae/source/model.safetensors'),**e121.base.disk_config()),
        e121.base.ModelConfig(path=e121.base.local_h3_files('FL2VA/audio_vae/model.safetensors'),**e121.base.disk_config())],processor_config=None,vram_limit=28.0)
    latents=torch.load(terminal,map_location='cpu',weights_only=True)
    pipe.load_models_to_device(['video_vae'])
    decoded=pipe.video_vae.decode_video(latents['video_latents'].cuda(),dtype=torch.bfloat16,tiled=True,tile_size=256,tile_overlap=64)
    assert torch.isfinite(decoded).all();video=pipe.vae_output_to_video(decoded,min_value=0,max_value=1);del decoded
    pipe.load_models_to_device(['audio_vae'])
    audio=pipe.output_audio_format_check(pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(),dtype=torch.bfloat16))
    assert len(video)==row['frames'] and all(x.size==(width,height) for x in video) and torch.isfinite(audio).all()
    write_video_audio(video=video,audio=audio,output_path=str(path),fps=24,audio_sample_rate=pipe.audio_vae.sample_rate)
    with av.open(str(path)) as c:
        stream=c.streams.video[0];assert stream.width==width and stream.height==height and len(c.streams.audio)==1
        count=sum(1 for _ in c.decode(video=0));assert count==row['frames']
    e121.base.save_json(out,{'status':'complete','prompt_id':pid,'seed':row['seed'],'video':str(path),'sha256':e121.base.sha256(path),
                             'frames':count,'width':width,'height':height,'fps':24,'audio_streams':1,'bytes':path.stat().st_size})
    print(f'COMPLETE {case} {path}',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('prepare','denoise','decode'),required=True);ap.add_argument('--prompt-id',type=int)
    args=ap.parse_args();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
    if args.phase=='prepare':prepare()
    elif args.phase=='denoise':denoise(args.prompt_id)
    else:decode(args.prompt_id)

if __name__=='__main__':main()
