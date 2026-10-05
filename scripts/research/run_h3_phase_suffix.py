#!/usr/bin/env python3
"""E075: one cached, real attention-output intervention and original suffix."""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import capture_h3_query_phase as cap
import torch

base, old, prior = cap.base, cap.prior.old, cap.prior
ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E075'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E075')
PLAN = ROOT / 'research_state/06_experiments/E075_phase_suffix_plan.md'
LO, HI, N = 2379, 21387, 22539


def load(record):
    return torch.load(base.verify_file(record), map_location='cpu', weights_only=True, mmap=True)


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def inputs(report):
    manifest, value = cap.inputs_and_binding(None, report)
    cap.cpu_structure(value, report)
    source = json.loads((ROOT/'results/research/E017/E017_denoise_block_mean.json').read_text())
    case = next(c for c in source['cases'] if c['prompt_id'] == 36)
    probe = json.loads((ROOT/'results/research/E018/probe_run.json').read_text())
    row = next(c for c in probe['cases'] if c['block'] == 24)
    tensors = {}
    records = {}
    for arm, mode, shift in [('phase0','block_mean',0), ('phase64','block_mean',64),
                             ('phase128','block_mean',128), ('oracle','bf16',0)]:
        record = next(o['output']['artifact'] for o in row['outputs'] if o['mode']==mode and o['shift']==shift)
        tensors[arm] = load(record)
        records[arm] = record
    capture = json.loads((ROOT/'results/research/E018/capture_run.json').read_text())
    record = next(c['artifact'] for c in capture['cases'] if c['block']==24)
    qkv = load(record)
    assert equal(tensors['phase0'], qkv['router_output'])
    assert equal(tensors['phase0'][LO:HI], tensors['phase128'][LO:HI])
    energy = dict(phase0=0., phase64=0., phase_difference=0.)
    for start in range(LO, HI, 256):
        sl = slice(start, min(start+256, HI))
        a, b, ref = [tensors[k][sl].double() for k in ('phase0','phase64','oracle')]
        energy['phase0'] += float((a-ref).square().sum())
        energy['phase64'] += float((b-ref).square().sum())
        energy['phase_difference'] += float((b-a).square().sum())
    energy['relative_sse_change'] = energy['phase64']/energy['phase0']-1
    energy['difference_energy_over_original_error'] = energy['phase_difference']/energy['phase0']
    assert abs(energy['relative_sse_change']) < .01
    report.update(local_energy=energy, output_sources=records, qkv_source=record,
                  driver=base.file_record(__file__), plan=base.file_record(PLAN))
    return manifest, value, case, tensors, qkv


def guard(args, report):
    assert args.deadline_unix and time.time() < args.deadline_unix, 'deadline'
    assert report['calls'] <= 19
    assert torch.cuda.max_memory_allocated() < 60*1024**3


@torch.inference_mode()
def run(args, report):
    manifest, value, case, cached, qkv = inputs(report)
    check = json.loads((RD/'check.json').read_text())
    assert check['status']=='complete' and check['driver']==report['driver'] and check['plan']==report['plan']
    assert check['local_energy']==report['local_energy']
    guard(args, report)
    pipe = old.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    old.make_h3_resident(pipe.dit)
    installed = old.install_native_h3(pipe.dit, Path(manifest['export_dir']),
                                    activation_packer=old.pack_activation_fast, chunk_rows=1024)
    assert installed['target_count']==installed['exact_roundtrip_count']==200
    pipe.scheduler.set_timesteps(20, shift=12.)
    pipe.scheduler_audio.set_timesteps(20, shift=3.)
    schedules = {'video':pipe.scheduler, 'audio':pipe.scheduler_audio}
    assert {m:dict(timesteps=s.timesteps.tolist(),sigmas=s.sigmas.tolist()) for m,s in schedules.items()}==case['schedule']
    router = cap.install_h3_fp4_attention(pipe.dit, mode='block_mean')
    original = router.comfy._sdpa_varlen_attention
    gpu = base.tree_device(value,'cuda')
    cached_gpu = {k:v.to('cuda') for k,v in cached.items()}
    active = {}
    def helper(q,k,v,cu,scale):
        output = original(q,k,v,cu,scale)
        if active['step']==14 and router._active_main.get()==24:
            for name,tensor in zip(('q','k','v'),(q,k,v)):
                assert equal(tensor[:N].cpu(),qkv[name]), f'changed {name}'
            assert equal(output[:N],cached_gpu['phase0']), 'phase0 attention replay'
            output = output.clone()
            output[LO:HI] = cached_gpu[active['arm']][LO:HI]
            active['patches'] += 1
        return output
    router.comfy._sdpa_varlen_attention = helper
    DATA.mkdir(parents=True,exist_ok=True)
    report['arms'] = []
    try:
        for arm in ('phase0','phase128','phase64','oracle'):
            directory = DATA/arm
            directory.mkdir(exist_ok=False)
            state = base.tree_device(value['state'],'cuda')
            row = dict(arm=arm,status='running',steps=[])
            report['arms'].append(row)
            for step in range(14,15 if arm=='phase128' else 20):
                guard(args,report)
                assert report['calls'] < 19
                active.update(arm=arm,step=step,patches=0)
                for m,s in schedules.items():
                    state[m]['timestep'] = s.timesteps[step].to('cuda',torch.float32)
                    state[m]['sigma'] = s.sigmas[step]
                packed = base.make_packed(pipe,gpu['embedding'],gpu['text_token_tags'],state)
                audit=old.RuntimeAudit(); audit.phase='native'
                before = time.monotonic()
                with audit.installed(), old.collect_fastpack_checks() as checks, router.forward_context(
                        **report['attention_contract'],diagnostics=False) as attn:
                    report['calls'] += 1
                    output=base.model_fn_minimax_h3(dit=pipe.dit,
                        video_latents=state['video']['latents_before'],audio_latents=state['audio']['latents_before'],
                        packed=packed,prompt_embeds=gpu['embedding'],
                        timestep_video=state['video']['timestep'].reshape(1),timestep_audio=state['audio']['timestep'].reshape(1))
                    torch.cuda.synchronize()
                counts={k:audit.row()[k] for k in ('sdpa_calls','scaled_mm_calls','disk_loads')}
                assert counts==dict(sdpa_calls=52,scaled_mm_calls=200,disk_loads=0)
                assert attn.summary['fp4_calls']==50 and checks.summary['invalid_calls']==0
                assert active['patches']==int(step==14)
                payload={}
                for m,velocity in zip(('video','audio'),output):
                    previous=state[m]['latents_before']
                    updated=pipe.step(schedules[m],previous,step,noise_pred=velocity)
                    payload[m]=dict(latents_before=previous.cpu(),noise_pred=velocity.cpu(),latents_after=updated.cpu())
                    assert bool(torch.isfinite(updated).all())
                    if arm in ('phase0','phase128'):
                        saved=next(s for s in case['steps'] if s['step']==step and s['modality']==m)
                        reference=load({k:saved[k] for k in ('file','sha256','bytes')})
                        for key in payload[m]:
                            assert equal(payload[m][key],reference[key]), f'{arm}/{step}/{m}/{key} replay failed'
                    state[m]['latents_before']=updated
                path=directory/f's{step:02d}.pt';torch.save(payload,path)
                row['steps'].append(dict(step=step,artifact=base.file_record(path),counts=counts,
                                         patches=active['patches'],seconds=time.monotonic()-before))
                base.save(report,RD/'run.json')
                print(f'E075 {arm} step{step} complete',flush=True)
            final={m+'_latents':state[m]['latents_before'].cpu() for m in ('video','audio')}
            if arm=='phase0':
                ref=load(case['final_latents'])
                assert all(equal(v,ref[k]) for k,v in final.items()), 'baseline final mismatch'
            path=directory/'final_latents.pt';torch.save(final,path)
            row.update(status='complete',final_latents=base.file_record(path),patches=sum(s['patches'] for s in row['steps']))
            base.save(report,RD/'run.json')
    finally:
        router.comfy._sdpa_varlen_attention=original
        router.close()
    assert report['calls']==19
    report['status']='complete'


@torch.inference_mode()
def decode(args,report):
    import av
    from diffsynth.utils.data.audio_video import write_video_audio
    denoised=json.loads((RD/'run.json').read_text());assert denoised['status']=='complete'
    guard(args,report)
    pipe=old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda',
        model_configs=[old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'),**old.disk_config()),
                       old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/audio_vae/model.safetensors'),**old.disk_config())],
        processor_config=None,vram_limit=30.)
    pipe.video_vae.eval();pipe.audio_vae.eval()
    report['cases']=[]
    for arm in ('phase64','oracle'):
        guard(args,report)
        item=next(r for r in denoised['arms'] if r['arm']==arm);latents=load(item['final_latents'])
        pipe.load_models_to_device(['video_vae'])
        recon=pipe.video_vae.decode_video(latents['video_latents'].cuda(),dtype=torch.bfloat16,tiled=True,tile_size=256,tile_overlap=64)
        video=pipe.vae_output_to_video(recon,min_value=0,max_value=1);del recon
        assert len(video)==124 and all(f.size==(1024,576) for f in video)
        pipe.load_models_to_device(['audio_vae'])
        audio=pipe.output_audio_format_check(pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(),dtype=torch.bfloat16))
        path=DATA/f'{arm}.mp4';assert not path.exists()
        write_video_audio(video,audio,str(path),fps=24,audio_sample_rate=32000)
        with av.open(str(path)) as container: assert sum(1 for _ in container.decode(video=0))==124
        report['cases'].append(dict(arm=arm,video=base.file_record(path)))
        base.save(report,RD/'decode.json')
        del video,audio,latents;gc.collect();torch.cuda.empty_cache()
    report['status']='complete'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['check','run','decode'],required=True)
    parser.add_argument('--deadline-unix',type=float);args=parser.parse_args()
    RD.mkdir(parents=True,exist_ok=True)
    path=RD/f'{args.phase}.json';assert not path.exists()
    started=time.monotonic();report=dict(experiment='E075',phase=args.phase,status='running',calls=0)
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    try:
        if args.phase=='check':
            inputs(report);assert not torch.cuda.is_initialized();report['status']='complete'
        elif args.phase=='run':run(args,report)
        else:decode(args,report)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.monotonic()-started
        if torch.cuda.is_initialized():report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        base.save(report,path)


if __name__=='__main__':main()
