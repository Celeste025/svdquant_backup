"""Stdlib reduction/source audit for E082 videos; no LPIPS inference rerun."""
from pathlib import Path
import json,math,hashlib,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E082'
def record(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return dict(file=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
start=time.monotonic();dest=OUT/'video_summary.json';assert not dest.exists()
new=json.loads((OUT/'metrics.json').read_text());old=json.loads((ROOT/'results/research/E079/metrics.json').read_text());triplets=json.loads((OUT/'triplets.json').read_text());assert new['status']==old['status']=='complete'
errors=[]
for row in new['rows']:
 frames=row['frame_metrics'];assert len(frames)==row['frames']==124
 assert [x['frame'] for x in frames]==list(range(124))
 n=124*1024*576*3;assert row['scalar_pixels']==n
 totals={k:sum(x[k] for x in frames) for k in ('abs_sum_uint8','squared_sum_uint8','reference_squared_sum_uint8')}
 for k,v in totals.items():assert v==row[k]
 computed=dict(lpips_alex=sum(x['lpips_alex'] for x in frames)/124,l1_mae=totals['abs_sum_uint8']/n/255,l2_rmse=math.sqrt(totals['squared_sum_uint8']/n)/255,l2_norm=math.sqrt(totals['squared_sum_uint8'])/255,relative_l2=math.sqrt(totals['squared_sum_uint8']/totals['reference_squared_sum_uint8']))
 for k,v in computed.items():
  error=abs(row[k]-v)/max(1,abs(v));assert error<1e-12;errors.append(error)
verified=[]
for row in triplets:
 for arm in ('bf16','sage','center'):
  rec=record(row[arm]);assert rec['sha256']==row['sha256'][arm];verified.append(rec)
rows=[]
for i in (0,1):
 cid=f'vbench0161_r{i}'
 before=next(r for r in old['rows'] if r['case']==cid+'__sage_vs_bf16')
 after=next(r for r in new['rows'] if r['case']==cid+'__center_vs_bf16')
 rows.append(dict(case_id=cid,before={k:before[k] for k in ('lpips_alex','l1_mae','l2_rmse')},after={k:after[k] for k in ('lpips_alex','l1_mae','l2_rmse')}))
report=dict(status='complete',definition='Full124frame 1024x576 paired distances to same BF16 reference; not quality percentages',rows=rows,verified_media=verified,checked_pairs=len(new['rows']),max_normalized_reduction_difference=max(errors),sources=[record(p) for p in (Path(__file__),OUT/'metrics.json',ROOT/'results/research/E079/metrics.json',OUT/'triplets.json')],scope='Independent stdlib aggregation of saved per-frame outputs and media SHA; no rerun of LPIPS network',seconds=time.monotonic()-start)
dest.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('sources','verified_media')},indent=2))
