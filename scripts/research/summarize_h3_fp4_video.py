#!/usr/bin/env python3
"""Thin E017 CPU closure: inherited audits, original scheduler, raw media and answers.

No model/runner imports or GPU work. E010 media and numerical helpers and E015
original-scheduler loader are reused; weights and model assets are not rehashed.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import summarize_h3_paired_generation as e10
import summarize_h3_crossmodal_propagation as e15
import summarize_h3_full_fp4_attention as e16

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E017'
MANIFEST = ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json'
ARMS, MODS = ('block_mean', 'global_mean'), ('video', 'audio')
prior = e16.prior
require, rec, sig = prior.require, prior.tensor_record, prior.signature


def read_tensor(row, files):
    files.record(row)
    return prior.load_tensor_file(row['file'])


def expected_input(template, states):
    actual = copy.deepcopy(template)
    kw = actual['kwargs']; total = kw['x']['shape'][1]
    text = kw['prompt_embeds']['shape'][0]
    video, audio = prior.video_rows(states['video']['latents_before']), prior.audio_rows(states['audio']['latents_before'])
    ap = torch.arange(text, text+len(audio)); vp = torch.arange(text+len(audio), text+len(audio)+len(video))
    x = torch.zeros((1, total, 96), dtype=torch.bfloat16); ax = torch.zeros((1, total, 32), dtype=torch.bfloat16)
    x[0, vp] = video; ax[0, ap] = audio
    ts = torch.full((total,), 1.-float(states['video']['timestep'])/1000, dtype=torch.float32)
    ts[ap] = 1.-float(states['audio']['timestep'])/1000
    unique, inverse = torch.unique(ts, sorted=True, return_inverse=True)
    kw.update(x=rec(x), audio_x=rec(ax), unique_timesteps=rec(unique), inverse_indices=rec(inverse))
    return actual


def trajectory(row, arm, case, cached_info, checked_case, template, sched, files):
    cached = read_tensor(checked_case['prepared'], files)
    require(row['status'] == 'complete' and row['variant'] == arm and row['prompt_id'] == case['prompt_id']
            and row['seed'] == case['seed'] and row['prompt'] == case['prompt'], 'Wrong rollout case')
    require(row['embedding_sha256'] == cached_info['embedding_sha256'] == rec(cached['embedding'])['sha256']
            and row['initial_noise_sha256'] == cached_info['initial_noise_sha256'], 'Shared conditioning/noise changed')
    calls = row['dit_calls']; steps = {(r['step'], r['modality']): r for r in row['steps']}
    require([c['step'] for c in calls] == list(range(20)) and len(row['steps']) == len(steps) == 40, 'Wrong per-case call/step allocation')
    counts = dict(sdpa_calls=52, scaled_mm_calls=200, disk_loads=0)
    require(row['runtime_audit'] == {k: v*20 for k, v in counts.items()}, 'Case runtime count differs')
    contract = {**case, 'main_cu_seqlens': checked_case['attention_contract']['expected_cu'],
        'refiner_cu_seqlens': checked_case['attention_contract']['expected_refiner_cu']}
    contract['flashinfer_padded_length'] = ((contract['main_cu_seqlens'][1]+127)//128)*128
    last = {m: rec(cached[m+'_latents']) for m in MODS}
    require({m+'_latents': last[m]['sha256'] for m in MODS} == row['initial_noise_sha256'], 'Initial payload noise SHA differs')
    records = []
    for i, call in enumerate(calls):
        require(call['counts'] == counts and call['finite'] is True, 'Actual DiT count/finite mismatch')
        prior.check_flags(call['zero_sf_checks'], 'svd')
        e16.attention_check(call['attention'], contract, 'svd_'+arm, False)
        states = {}
        for m in MODS:
            s = steps[i, m]; payload = read_tensor(s, files); states[m] = payload
            require(set(payload) == {'latents_before','noise_pred','latents_after','timestep','sigma'}, 'Unexpected step payload')
            require(sig(payload) == s['tensors'] and s['finite'] is True
                    and s['scheduler_extra_signature'] == {'inpaint_mask':None,'input_latents':None}, 'Step tensor/mask contract differs')
            for tensor in payload.values(): e10.finite(tensor)
            require(rec(payload['latents_before']) == last[m], 'Broken initial/recursive chain')
            require(rec(e15.step(sched[m], payload, payload['noise_pred'], i)) == rec(payload['latents_after']), 'Original scheduler/own sample mismatch')
            require(rec(e10.raw_dit_rows(payload['noise_pred'], m)) == call['raw_outputs'][m], 'Raw DiT/velocity sign-layout mismatch')
            last[m] = rec(payload['latents_after'])
            require(last[m]['sha256'] == s['latent_sha256'], 'Updated latent SHA mismatch')
        require(call['actual_dit_inputs'] == expected_input(template, states), 'Actual packed DiT input differs from this recurrent state')
        records.append(dict(step=i, raw_outputs=call['raw_outputs'], after=copy.deepcopy(last),
            zero_sf_affected_calls=call['zero_sf_checks']['affected_call_indices_zero_based']))
    finals = read_tensor(row['final_latents'], files)
    require(sig(finals) == row['final_tensors'] == {m+'_latents': last[m] for m in MODS}, 'Final state differs from step19')
    return dict(prompt_id=case['prompt_id'], arm=arm, steps=records, final=row['final_latents'],
        final_tensors=row['final_tensors'], original_step_exact=40, actual_inputs_verified=20), finals


def answer_score(answers, weights):
    raw = [a['first_token_decoded'] for a in answers]; norm = [a.strip().casefold() for a in raw]
    invalid = [i for i,a in enumerate(norm) if a not in ('yes','no')]
    strict = statistics.mean(w*(1 if a == 'yes' else -1) for a,w in zip(raw, weights, strict=True))
    normalized = statistics.mean(w*(1 if a == 'yes' else -1) for a,w in zip(norm, weights, strict=True))
    return dict(strict=None if invalid else strict, normalized=None if invalid else normalized,
                invalid_indices=invalid, raw_strict_formula=strict)


def scores(report, previous, media, manifest):
    require(report['questions'] == previous['questions'] and report['weights'] == previous['weights']
            and len(report['questions']) == 29 and len(report['results']) == 8, 'Official question/weight/case coverage changed')
    require(set(report['model_files']) == set(previous['model_files']), 'VisionReward model inventory changed')
    for path, old_file in previous['model_files'].items():
        require(all(report['model_files'][path].get(k) == v for k,v in old_file.items()), 'VisionReward inherited model SHA/metadata changed')
    require(len({r['case_id'] for r in report['results']}) == 8, 'Colliding score labels')
    old = {(r['prompt_id'],r['variant']): r for r in previous['results']}
    rows, replay = {}, []
    for row in report['results']:
        key = row['prompt_id'], row['variant']; require(key not in rows and key in media, 'Duplicate/unexpected score identity')
        video = media[key]
        case = next(c for c in manifest['cases'] if c['prompt_id'] == key[0])
        require(row['video'] == video['video'] and row['video_sha256'] == video['video_sha256']
                and row['prompt'] == case['prompt'] and row['seed'] == case['seed'], 'Scored a different media/prompt/seed')
        require(len(row['answers']) == 29, 'Missing score questions')
        original_frame_contract = old[(key[0], 'native')]['answers'][0]['frames']
        for i,a in enumerate(row['answers']):
            require(a['question_index'] == i and a['question'] == report['questions'][i]
                    and a['expanded_query'] == report['questions'][i].replace('[[prompt]]', case['prompt'])
                    and a['weight'] == report['weights'][i], 'Question/query/weight changed')
            require(a['generated_token_ids'] and a['first_token_id'] == a['generated_token_ids'][0]
                    and a['normalized'] == a['first_token_decoded'].strip().casefold(), 'Raw answer contract mismatch')
            require(a['frames'] == original_frame_contract, 'Official frame selection/metadata changed')
        score = answer_score(row['answers'], report['weights'])
        for name, value in [('strict_official',score['strict']),('normalized_yes_no',score['normalized'])]:
            stored = row['scores'][name]
            require(stored['valid'] == (not score['invalid_indices']) and
                    ((stored['score'] is None and value is None) or (value is not None and math.isclose(stored['score'],value,abs_tol=1e-12))), 'Independent score differs')
        rows[key] = dict(prompt_id=key[0], variant=key[1], case_id=row['case_id'], **score)
        if key in old:
            changed = [{ 'index':i, 'old':old[key]['answers'][i]['first_token_decoded'], 'new':a['first_token_decoded']}
                for i,a in enumerate(row['answers']) if a['first_token_decoded'] != old[key]['answers'][i]['first_token_decoded']]
            replay.append(dict(prompt_id=key[0],variant=key[1],changed_answers=changed,
                token_or_full_text_changes=[i for i,a in enumerate(row['answers']) if any(a[k]!=old[key]['answers'][i][k]
                    for k in ('generated_token_ids','full_generated_decoded','input_token_ids'))]))
    require(set(rows) == set(media), 'Eight-media scoring coverage differs')
    changes = []
    answer_rows = {(r['prompt_id'],r['variant']):r for r in report['results']}
    for pid in (30,36):
        reference = answer_rows[pid,'native']
        for arm in ARMS:
            current = answer_rows[pid,arm]; a,b = rows[pid,arm], rows[pid,'native']
            changes.append(dict(prompt_id=pid,arm=arm,normalized_delta_vs_svd=None if a['normalized'] is None or b['normalized'] is None else a['normalized']-b['normalized'],
                all_answer_flips=[dict(index=i,question=report['questions'][i],weight=report['weights'][i],svd=x['first_token_decoded'],current=y['first_token_decoded'])
                    for i,(x,y) in enumerate(zip(reference['answers'],current['answers'],strict=True)) if x['first_token_decoded']!=y['first_token_decoded']]))
    return dict(scores=list(rows.values()),old_four_replay=replay,paired_changes=changes,
                scope='29 correlated auxiliary questions, not independent samples or an audio/quality-equivalence test')


def contact_sheet(images, prompt_id, path):
    from PIL import Image, ImageDraw
    arms = ('bf16','native','block_mean','global_mean')
    labels = ('BF16 (E010)','SVD + BF16 attention (E010)','SVD + FP4 block mean','SVD + FP4 global mean')
    width,height = 320,180
    canvas=Image.new('RGB',(6*width,36+4*(height+28)),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),f'E017 p{prompt_id}: same initial noise; all four free trajectories; six fixed frames',fill='black')
    for ri,(arm,label) in enumerate(zip(arms,labels,strict=True)):
        top=36+ri*(height+28)
        for ci,index in enumerate(e10.FRAME_INDICES):
            draw.text((ci*width+5,top+5),f'{label} | f{index} {index/24:.2f}s',fill='black')
            canvas.paste(images[prompt_id,arm][index].resize((width,height),Image.Resampling.LANCZOS),(ci*width,top+24))
    require(not path.exists(),'Preserve existing contact sheet');canvas.save(path)


def run(args, out):
    files = prior.Files(); manifest,_ = files.json(MANIFEST,complete=False); files.record(manifest['plan'])
    inherited = {k: files.json(manifest[k]['file'],complete=k!='e016_manifest')[0] for k in ('e010_summary','e010_decode','e010_visionreward','e016_check','e016_manifest','e016_summary')}
    for k in inherited: files.record(manifest[k])
    checked, _ = files.json(args.report_dir/'E017_check_block_mean.json')
    require(checked['cuda_initialized'] is False, 'Check initialized CUDA'); files.sources(checked['sources'])
    frozen, freeze_rec = files.json(args.report_dir/'frozen_contract.json')
    launcher, launcher_rec = files.json(args.report_dir/'launcher.json')
    require(launcher['freeze_sha256'] == freeze_rec['sha256'] and launcher['seconds_total'] <= 1800
            and launcher['deadline_epoch']-launcher['start_epoch'] == 1800, 'Shared deadline/freeze differs')
    require([s['name'] for s in launcher['stages']] == [f'denoise_{a}' for a in ARMS]+[f'decode_{a}' for a in ARMS]+['visionreward'], 'Stage allocation differs')
    require(sum(s.get('dit_calls',0) for s in launcher['stages']) == 80 and sum(s.get('videos',0) for s in launcher['stages']) == 4
            and launcher['stages'][-1]['evaluation_answers'] == 232, 'Call/media/question budget differs')
    pids, devices, end = set(), set(), launcher['start_epoch']
    for stage in launcher['stages']:
        files.verify(stage['report'],stage['report_sha256'])
        require(stage['status']=='complete' and stage['returncode']==0 and stage['pid'] not in pids
                and end <= stage['start_epoch'] < stage['end_epoch'] <= launcher['deadline_epoch']
                and stage['gpu_before']['index']==5, 'Process chain/GPU/deadline changed')
        pids.add(stage['pid']);devices.add(stage['gpu_before']['uuid']);end=stage['end_epoch']
    require(len(devices)==1,'GPU changed across stages')
    sched = e15.schedules(files); results, media, images, media_audits = [], {}, {}, {}
    old_media = {(r['prompt_id'],r['variant']):r for r in inherited['e010_decode']['cases']}
    old_audits = {c['prompt_id']:c['media'] for c in inherited['e010_summary']['cases']}
    for key,row in old_media.items():
        files.verify(row['video'],row['video_sha256']); require(old_audits[key[0]][key[1]]['video_sha256']==row['video_sha256'],'Inherited media mismatch')
        media[key]=row
        media_audits[str(key)],images[key]=e10.media_audit(row)
    for arm in ARMS:
        den,_=files.json(args.report_dir/f'E017_denoise_{arm}.json'); dec,_=files.json(args.report_dir/f'E017_decode_{arm}.json')
        require(den['attempted_dit_calls']==den['complete_dit_calls']==40 and den['sources']==dec['sources']==checked['sources'], 'Arm call/source contract differs')
        require(max(den['peak_allocated_bytes'],dec['peak_allocated_bytes'])<=60*1024**3
                and den['deadline_unix']==dec['deadline_unix']==launcher['deadline_epoch'], 'Arm resource/deadline differs')
        require(den['runtime_audit']['qkv_dtypes']==[['torch.bfloat16']*3], 'BF16 SDPA precision changed')
        files.record(dec['denoiser_reference'])
        for case,cached,cpu,row,video in zip(manifest['cases'],manifest['prepared_cases'],checked['cases'],den['cases'],dec['cases'],strict=True):
            oldcase=next(c for c in inherited['e016_manifest']['cases'] if c['prompt_id']==case['prompt_id'])
            template=read_tensor(oldcase['e014_artifacts']['bf16'],files)['actual_dit_inputs']
            result,finals=trajectory(row,arm,case,cached,cpu,template,sched,files)
            require(video['final_latents']==row['final_latents'] and video['variant']==arm and video['prompt_id']==case['prompt_id'], 'Decode/final identity mismatch')
            normalized={**video,'audio_pcm_file':video['audio_pcm_file']['file'],'audio_pcm_sha256':video['audio_pcm']['sha256']}
            files.record(video['audio_pcm_file']); result['media'],images[case['prompt_id'],arm]=e10.media_audit(normalized)
            media_audits[str((case['prompt_id'],arm))]=result['media']
            media[case['prompt_id'],arm]=video; results.append(result)
    vr,vrrec=files.json(args.report_dir/'E017_visionreward.json',complete=False)
    require(vr['status'] in ('complete','complete_with_invalid_answers'),'Incomplete auxiliary evaluation')
    for row in vr['results']:
        files.verify(row['case_json'],row['case_json_sha256'])
        require(json.loads(Path(row['case_json']).read_text())==media[row['prompt_id'],row['variant']], 'Scored sidecar differs from decoder record')
    args.artifact_dir.mkdir(parents=True,exist_ok=True); contacts=[]
    for pid in (30,36):
        path=args.artifact_dir/f'E017_p{pid:03d}_all4_contact.png';contact_sheet(images,pid,path);contacts.append(files.verify(path))
    out.update(trajectories=results,media_count=len(media),media_audits=media_audits,contact_sheets=contacts,visionreward=scores(vr,inherited['e010_visionreward'],media,manifest),
        launcher=launcher_rec,visionreward_report=vrrec,inherited_audits={k:manifest[k] for k in ('e010_summary','e016_summary')},files_verified=files.checked)
    for source in (__file__,e10.__file__,e15.__file__,e16.__file__,prior.__file__): files.verify(source)
    require(not torch.cuda.is_initialized(),'Summary initialized CUDA');out.update(status='complete',cuda_initialized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=RD);p.add_argument('--output',type=Path,default=RD/'independent_summary.json')
    p.add_argument('--artifact-dir',type=Path,default=Path('/data1/models/svdquant-wjq/research/20261002/E017/independent_summary'))
    a=p.parse_args();require(not a.output.exists(),'Preserve previous summary');torch.set_num_threads(6)
    out=dict(experiment='E017',status='running',source=dict(file=str(Path(__file__).resolve()),sha256=prior.file_sha(__file__)),
        scope='Independent recurrence/media/auxiliary-score closure for two previously inspected prompts; no perceptual or audio equivalence claim')
    start=time.monotonic()
    try:run(a,out)
    except Exception:out.update(status='failed',error=traceback.format_exc());raise
    finally:
        out['seconds_total']=time.monotonic()-start;a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps(dict(status=out['status'],output=str(a.output),seconds=out['seconds_total'])),flush=True)


if __name__=='__main__':main()
