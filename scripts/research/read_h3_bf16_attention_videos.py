#!/usr/bin/env python3
"""E083: original BF16 backbone, SDPA / Sage3 / Sage3+center128 videos."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

import evaluate_h3_baseline_distances as old

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/research/E083'
DATA = Path('/data1/models/svdquant-wjq/research/20261005/E083')
MANIFEST = ROOT / 'research_state/06_experiments/E083_bf16_attention_video_manifest.json'
ARMS = {'sage': 'bf16_sage3', 'center': 'bf16_sage3_center128'}
LABELS = {'bf16': 'BF16 + SDPA', 'sage': 'BF16 + SageAttention3', 'center': 'BF16 + Sage3 + V-center128'}
PAIRS = (('sage', 'bf16'), ('center', 'bf16'), ('center', 'sage'))


def record(path):
    path = Path(path)
    return dict(file=str(path), sha256=old.sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def triplets():
    import read_h3_sage3_videos as prior
    references = {r['case']: r for r in prior.triplets() if r['case'].startswith('vbench0161_')}
    manifest = read(MANIFEST)
    assert manifest['experiment'] == 'E083' and manifest['arms'] == list(ARMS.values())
    assert Path(manifest['data_dir']) == DATA and Path(manifest['report_dir']) == OUT
    for frozen in [manifest['runner'], *manifest['frozen_runtime_sources']]:
        assert record(frozen['file']) == frozen, 'Frozen source changed: ' + frozen['file']
    manifest_record = record(MANIFEST)
    validation_path = ROOT / 'results/research/E082/native_validation_v3.json'
    validation = read(validation_path)
    assert validation['status'] == 'complete' and validation['passed']
    expected_names = {f'blocks.{block}.{suffix}' for block in range(50)
                      for suffix in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')}
    rows = []
    for replica in (0, 1):
        cid = f'vbench0161_r{replica}'
        reference = references[cid]
        ref_path = ROOT / f'results/research/E073/denoise_full_bf16_r{replica}.json'
        ref_generation = read(ref_path)
        ref_case = next(c for c in ref_generation['cases'] if c['case_id'] == cid)
        launch_path = OUT / f'launcher_r{replica}.json'
        launcher = read(launch_path)
        assert launcher['status'] == 'complete'
        assert {(p['arm'], p['phase']) for p in launcher['phases']} == {
            (arm, phase) for arm in ARMS.values() for phase in ('denoise', 'decode')}
        assert len(launcher['phases']) == 4 and all(p['exit_code'] == 0 for p in launcher['phases'])
        row = dict(case=cid, label=reference['label'], seed=reference['seed'], settings=reference['settings'],
                   bf16=reference['bf16'], sha256={'bf16': reference['sha256']['bf16']},
                   initial_noise_sha256=ref_case['initial_noise_sha256'], embedding_sha256=ref_case['embedding_sha256'],
                   schedule=ref_case['schedule'], backbone='Original BF16 main linears in all three arms',
                   report_sources={'reference_denoise': record(ref_path), 'launcher': record(launch_path)})
        assert row['settings'] == manifest['settings'] and row['schedule'] == manifest['expected_schedule']
        for key, arm in ARMS.items():
            gp = OUT / f'denoise_{arm}_r{replica}.json'
            dp = OUT / f'decode_{arm}_r{replica}.json'
            gen, dec = read(gp), read(dp)
            assert gen['status'] == dec['status'] == 'complete'
            assert gen['arm'] == dec['arm'] == arm and gen['replica'] == dec['replica'] == replica
            assert gen['manifest'] == dec['manifest'] == manifest_record
            assert gen['settings'] == dec['settings'] == row['settings']
            assert gen['private_contract'] == dec['private_contract'] == validation['private_contract']
            assert gen['center_validation'] == dec['center_validation'] == record(validation_path)
            assert gen['sources'][Path(manifest['runner']['file']).name] == manifest['runner']
            assert dec['denoiser_reference'] == record(gp)
            assert gen['prepared_reference'] == dec['prepared_reference'] == ref_generation['prepared_reference']
            assert gen['native_installation'] is None
            linears = gen['bf16_main_linears']
            assert len(linears) == 200 and {x['name'] for x in linears} == expected_names
            assert all(x['module_type'] == 'Linear' and x['weight_dtype'] == 'torch.bfloat16'
                       and x['weight_device'].startswith('cuda') for x in linears)
            for field in ('torch', 'cuda', 'attention_implementation', 'sdpa_enabled'):
                assert gen['environment'][field] == ref_generation['environment'][field]
            assert gen['complete_dit_calls'] == gen['attempted_dit_calls'] == gen['allocated_dit_calls'] == 20
            assert gen['actual_fp4_attention_calls'] == 1000
            assert len(gen['cases']) == len(dec['cases']) == 1
            assert dec['actual_video_vae_calls'] == dec['actual_audio_vae_calls'] == 1
            g, d = gen['cases'][0], dec['cases'][0]
            assert g['status'] == d['status'] == 'complete'
            assert g['case_id'] == d['case_id'] == cid and g['seed'] == d['seed'] == row['seed']
            for field in ('prompt', 'prompt_sha256', 'initial_noise_sha256', 'embedding_sha256', 'prepared', 'attention_contract'):
                assert g[field] == ref_case[field], (cid, arm, field)
            assert d['settings'] == row['settings'] and d['prompt_sha256'] == ref_case['prompt_sha256']
            assert g['schedule'] == gen['schedule_cpu'] == row['schedule']
            assert len(g['dit_calls']) == 20 and [x['step'] for x in g['dit_calls']] == list(range(20))
            assert all(x['counts'] == dict(sdpa_calls=52, scaled_mm_calls=0, disk_loads=0)
                       and x['attention']['fp4_calls'] == 50 and x['finite'] for x in g['dit_calls'])
            assert g['runtime_audit'] == dict(sdpa_calls=1040, scaled_mm_calls=0, disk_loads=0)
            assert [x['block'] for x in g['first_dit_attention_routes']] == list(range(50))
            assert len(g['steps']) == 40
            for step in g['steps']:
                schedule = row['schedule'][step['modality']]
                assert step['timestep'] == schedule['timesteps'][step['step']]
                assert step['sigma'] == schedule['sigmas'][step['step']]
            assert d['final_latents'] == g['final_latents'] and d['final_tensors'] == g['final_tensors']
            assert d['media'] == dict(frames=124, width=1024, height=576, fps=24., audio_streams=1,
                                      audio_sample_rate=32000, audio_channels=2)
            expected_video = DATA / f'decode_{arm}_r{replica}' / f'{cid}.mp4'
            assert Path(d['video_path']) == expected_video and old.sha(expected_video) == d['video_sha256']
            row[key] = str(expected_video)
            row['sha256'][key] = d['video_sha256']
            row['report_sources'][key] = dict(denoise=record(gp), decode=record(dp))
        assert old.sha(row['bf16']) == row['sha256']['bf16']
        rows.append(row)
    return rows


def render(rows):
    dest = DATA / 'review'
    dest.mkdir(exist_ok=False)
    receipts = []
    for row in rows:
        frames = {arm: [Image.fromarray(f) for f in old.frames(row[arm])] for arm in LABELS}
        assert all(len(values) == 124 and all(f.size == (1024, 576) for f in values) for values in frames.values())
        indices = list(range(0, 124, 4)) + [123]
        assert len(indices) == 32
        pages = []
        for page in range(4):
            canvas = Image.new('RGB', (1536, 1424), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 4), row['case'] + ' | BF16 backbone: SDPA / Sage3 / Sage3+V-center128 | sampled frames', fill='black')
            for j, index in enumerate(indices[8*page:8*page+8]):
                for n, arm in enumerate(LABELS):
                    x, y = (j % 4)*384, 30+(j//4*3+n)*232
                    draw.text((x+3, y), f'{LABELS[arm]} frame {index}', fill='black')
                    canvas.paste(frames[arm][index].resize((384, 216)), (x, y+16))
            path = dest / f'{row["case"]}_page{page}.png'
            canvas.save(path)
            pages.append(str(path))
        target = dest / f'{row["case"]}_triplet.mp4'
        with av.open(str(target), 'w') as container:
            stream = container.add_stream('libx264', rate=24)
            stream.width, stream.height, stream.pix_fmt = 1536, 312, 'yuv420p'
            stream.options = {'crf': '20', 'preset': 'fast', 'threads': '4'}
            for i in range(124):
                frame = Image.new('RGB', (1536, 312), 'white')
                draw = ImageDraw.Draw(frame)
                for j, arm in enumerate(LABELS):
                    draw.text((j*512+5, 5), LABELS[arm], fill='black')
                    frame.paste(frames[arm][i].resize((512, 288)), (j*512, 24))
                for packet in stream.encode(av.VideoFrame.from_image(frame)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        receipts.append(dict(case=row['case'], pages=pages, frames=indices,
                             comparison_video=str(target), comparison_sha256=old.sha(target)))
        print('rendered', row['case'], flush=True)
    assert len(receipts) == 2 and sum(len(x['pages']) for x in receipts) == 8
    (OUT/'visual_manifest.json').write_text(json.dumps(dict(status='complete', rows=receipts,
        scope='BF16 backbone in all arms;32 sampled frames/clip; labeled contact sheets; full124-frame silent triplet at24fps, scaled512x288 panels'), indent=2))
    html = '<meta charset="utf-8"><title>H3 BF16 attention comparison</title><h1>BF16 + SDPA / BF16 + Sage3 / BF16 + Sage3 + V-center128</h1><p>Same prompt, actual noise, embeddings and settings. Silent comparison; original clips retain audio.</p>'
    for row in receipts:
        html += f'<h2>{row["case"]}</h2><video controls preload="metadata" width="1200" src="{Path(row["comparison_video"]).name}"></video>'
    (dest/'index.html').write_text(html)


@torch.inference_mode()
def metrics(rows):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    assert not (OUT/'metrics.json').exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True, reduction='none').cuda().eval()
    outputs = []
    for row in rows:
        for candidate, reference in PAIRS:
            comparison = candidate + '_vs_' + reference
            pair = dict(case=row['case']+'__'+comparison, label=row['label'], seed=row['seed'], cohort='E083',
                        candidate=row[candidate], reference=row[reference],
                        candidate_sha256=row['sha256'][candidate], reference_sha256=row['sha256'][reference])
            value = old.evaluate(pair, metric)
            value.update(candidate_arm=candidate, reference_arm=reference, comparison=comparison,
                         candidate_sha256=pair['candidate_sha256'], reference_sha256=pair['reference_sha256'])
            outputs.append(value)
            (OUT/(pair['case']+'.json')).write_text(json.dumps(value, indent=2))
            print(pair['case'], value['lpips_alex'], flush=True)
    summaries = {c+'_vs_'+r: {key: float(np.mean([v[key] for v in outputs if v['comparison'] == c+'_vs_'+r]))
                               for key in ('lpips_alex', 'l1_mae', 'l2_rmse')} for c, r in PAIRS}
    assert len(outputs) == 6
    (OUT/'metrics.json').write_text(json.dumps(dict(status='complete',
        definition='All124frames original1024x576 RGB[0,1]; LPIPS-Alex FP32, MAE/RMSE; BF16 backbone in all arms; paired distance is not quality;2clips equal-weight',
        sources=[record(Path(__file__)), record(Path(old.__file__)), record(OUT/'triplets.json')],
        summary=summaries, rows=outputs), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('render', 'metrics'), required=True)
    args = parser.parse_args()
    started = time.time()
    rows = triplets()
    receipt = OUT/'triplets.json'
    if receipt.exists():
        assert read(receipt) == rows, 'Preserve frozen video pairings'
    else:
        receipt.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (render if args.phase == 'render' else metrics)(rows)
    print('complete', args.phase, time.time()-started, flush=True)
