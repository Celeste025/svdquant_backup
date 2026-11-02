#!/usr/bin/env python3
"""Audit E159 paired rollout reports and export final latent NMSE comparison."""
import json
import csv
import hashlib

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research/E159'
rows=[]
missing=[]
for pid in (25,48,44,7):
    reference=json.loads((ROOT/f'results/research/E150/p{pid:03d}/bf16_denoise.json').read_text())
    pair={}
    for mode in ('proxy','reuse'):
        p=OUT/f'aligned_v2/{mode}/p{pid:03d}/bf16_denoise.json'
        d=json.loads(p.read_text()) if p.exists() else {}
        if d.get('status')!='denoise_complete' or 'comparison' not in d:
            missing.append({'prompt_id':pid,'mode':mode,'status':d.get('status','missing'),'steps':len(d.get('steps',[]))});continue
        assert d['settings']==reference['settings'] and d['frames']==reference['frames']
        assert d['seed']==reference['seed'] and d['prompt']==reference['prompt']
        assert d['initial_latent_sha256']==reference['initial_latent_sha256']
        assert d['comparison']['initial_noise_matches_bf16']
        assert len(d['steps'])==20 and all(s['attention_main_calls']==50 for s in d['steps'])
        cfg=d['attention_router']
        assert cfg['sparse_config']['query_block_size']==128 and cfg['sparse_config']['sparsity_ratio']==0
        assert cfg['quant_config']['nvfp4_ratio']==.9 and cfg['quant_config']['int8_ratio']==.1
        assert cfg['dense_first_steps']==1 and cfg['dense_first_layers']==1
        assert Path(d['terminal_latents']).exists()
        assert file_sha256(d['terminal_latents'])==d['terminal_sha256']
        assert file_sha256(reference['terminal_latents'])==reference['terminal_sha256']
        if mode=='reuse':
            assert cfg['cached_routes']==49 and cfg['cached_route_bytes']>0
        pair[mode]=d
    if len(pair)==2:
        a,b=[pair[m]['comparison']['video_latent_nmse'] for m in ('proxy','reuse')]
        rows.append(dict(prompt_id=pid,frames=reference['frames'],seed=reference['seed'],proxy_nmse=a,reuse_nmse=b,reuse_over_proxy=b/a,winner='reuse' if b<a else 'proxy',proxy_dit_seconds=sum(s['dit_seconds'] for s in pair['proxy']['steps']),reuse_dit_seconds=sum(s['dit_seconds'] for s in pair['reuse']['steps'])))
report={'status':'complete' if not missing else 'running','rows':rows,'pending':missing,'metric':'sum((video_latent - BF16)^2) / sum(BF16_video_latent^2)','policy':'Q128, skip0, NVFP4 .9 / INT8 .1, step0 dense INT8, layer0 BF16; BF16 linears; existing E150 baseline'}
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
if rows:
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
labels={25:'Cats playing Mahjong (5s)',48:'Kayaking (5s)',44:'Swimming (10s)',7:'Astronaut on Mars (10s)'}
lines=['# E159 — first-step exact-route reuse versus per-step DraftMap', '',
       'Status: '+report['status'], '',
       'Existing E150 BF16 final latents are reused. Prompt, seed, resolution, frames, scheduler settings and initial video/audio latent hashes are checked for each arm. Checkpoint hashes are verified. No BF16 regeneration.', '',
       'Both arms: BF16 linears, Q128, no skipped tiles, NVFP4 90% / INT8 10%; step 0 dense INT8 except layer 0 BF16. Layer 0 remains BF16 for all steps. Steps 1–19: proxy rebuilds DraftMap; reuse uses per-layer exact QK tile-max routes captured at step 0. Text prefix uses dense INT8; refiner and padding remain BF16.', '',
       '768 x 1344, 20 steps, CFG 1, flow shift 12, audio flow shift 3, CPU RNG. E150 frame alignment is preserved. GPUs 2 and 3 each process two cases, two arms per case.', '',
       '| Case | Frames | Seed | Proxy NMSE | Reuse NMSE | Reuse change |',
       '|---|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {labels[row['prompt_id']]} | {row['frames']} | {row['seed']} | {row['proxy_nmse']:.8f} | {row['reuse_nmse']:.8f} | {(row['reuse_over_proxy']-1)*100:+.2f}% |")
lines += ['', 'NMSE = sum((final video latent - BF16 latent)^2) / sum(BF16 latent^2). Lower is better. This measures final latent agreement, not perceptual video quality. Four cases and one seed each do not establish population-wide statistical superiority.', '',
          'Pending arms: '+str(len(missing)), '',
          'Artifacts: comparison.json, comparison.csv, aligned_v2/{proxy,reuse}/pXXX/bf16_denoise.json. The inherited report filename is retained; mode and attention_router identify the actual arm. Checkpoints are local under results/local/E159/aligned_v2.']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,indent=2))
