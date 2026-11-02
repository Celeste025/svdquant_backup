#!/usr/bin/env python3
"""Aggregate available results; safe to call from multiple GPU queues."""
import csv,fcntl,json,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/h3_r64_draft_secondset';out.mkdir(parents=True,exist_ok=True)
with (out/'summary.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);rows=[];pending=[]
 for variant in ('c1','c2','c3','c4','c5','c6','c7'):
  for pid in (16,11,45,89):
   p=out/variant/f'p{pid:03d}'/'metrics.json'
   if not p.exists():pending.append({'variant':variant,'prompt_id':pid});continue
   d=json.loads(p.read_text());assert d['status']=='complete' and len(d['steps'])==20
   rows.append({k:d[k] for k in ('variant','case','frames','rgb_nmse','lpips_alex','psnr_db','dit_seconds','video')})
 means={}
 for v in ('c1','c2','c3','c4','c5','c6','c7'):
  subset=[r for r in rows if r['variant']==v]
  if len(subset)==4:means[v]={k:statistics.mean(r[k] for r in subset) for k in ('rgb_nmse','lpips_alex','psnr_db','dit_seconds')}
 report={'status':'complete' if not pending else 'running','complete':len(rows),'pending':pending,'rows':rows,'four_case_means':means,'note':'DiT means combine two 124-frame and two 243-frame cases; includes weight swaps and first-use compilation.'}
 tmp=out/'summary.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,out/'summary.json')
 tmp=out/'summary.csv.tmp'
 with tmp.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['variant','case','frames','rgb_nmse','lpips_alex','psnr_db','dit_seconds','video']);writer.writeheader();writer.writerows(rows)
 os.replace(tmp,out/'summary.csv');print('summary',len(rows),'/28')
