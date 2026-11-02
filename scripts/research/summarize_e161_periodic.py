#!/usr/bin/env python3
"""Audit periodic capture and compare the four matched BF16 rollouts."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E161'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[];pending=[]
for pid in (25,48,44,7):
 case=f'p{pid:03d}';ref=json.loads((ROOT/'results/research/E150'/case/'bf16_denoise.json').read_text())
 p=OUT/'refresh4/reuse'/case/'bf16_denoise.json';d=json.loads(p.read_text()) if p.exists() else {}
 e=OUT/f'{case}.json';ev=json.loads(e.read_text()) if e.exists() else {}
 if d.get('status')!='denoise_complete' or 'comparison' not in d or ev.get('status')!='complete':
  pending.append({'prompt_id':pid,'status':d.get('status','missing'),'steps':len(d.get('steps',[])),'video_status':ev.get('status','missing')});continue
 assert d['prompt']==ref['prompt'] and d['seed']==ref['seed'] and d['frames']==ref['frames'] and d['settings']==ref['settings'] and d['initial_latent_sha256']==ref['initial_latent_sha256']
 assert digest(d['terminal_latents'])==d['terminal_sha256'] and digest(ref['terminal_latents'])==ref['terminal_sha256']
 cfg=d['attention_router'];assert cfg['cached_routes']==49 and cfg['refresh_steps_1based']==[1,6,11,16]
 assert cfg['quant_config']['nvfp4_ratio']==.9 and cfg['quant_config']['int8_ratio']==.1 and cfg['sparse_config']['query_block_size']==128 and cfg['sparse_config']['sparsity_ratio']==0
 assert len(d['steps'])==20 and all(x['attention_main_calls']==50 for x in d['steps'])
 old=json.loads((ROOT/'results/research/E159/comparison.json').read_text());old=next(x for x in old['rows'] if x['prompt_id']==pid)
 oldvid=json.loads((ROOT/f'results/research/E160/{case}.json').read_text())
 assert ev['frames']==ref['frames'] and Path(ev['comparison_video']).exists()
 metrics=ev['metrics']['reuse'];assert len(metrics['lpips_per_frame'])==ref['frames'] and all(math.isfinite(metrics[k]) for k in ('lpips_alex','psnr_db'))
 rows.append({'prompt_id':pid,'frames':d['frames'],'seed':d['seed'],'proxy_nmse':old['proxy_nmse'],'first_only_nmse':old['reuse_nmse'],'periodic_nmse':d['comparison']['video_latent_nmse'],'proxy_lpips':oldvid['metrics']['proxy']['lpips_alex'],'first_only_lpips':oldvid['metrics']['reuse']['lpips_alex'],'periodic_lpips':metrics['lpips_alex'],'proxy_psnr':oldvid['metrics']['proxy']['psnr_db'],'first_only_psnr':oldvid['metrics']['reuse']['psnr_db'],'periodic_psnr':metrics['psnr_db'],'periodic_dit_seconds':sum(x['dit_seconds'] for x in d['steps']),'comparison_video':ev['comparison_video']})
report={'status':'complete' if not pending else 'running','rows':rows,'pending':pending,'refresh_steps_1based':[1,6,11,16],'refresh_step_policy':'dense INT8 attention and exact QK tile-max route capture; BF16 layer0','remaining_step_policy':'reuse cached route; 90% NVFP4 + 10% INT8; zero skip','metric':'video latent NMSE and full-resolution frame LPIPS Alex / PSNR vs E150 matched BF16'}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
if rows:
 with (OUT/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(report,indent=2))
