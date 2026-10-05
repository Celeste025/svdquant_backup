"""E078 independent NumPy readout; no model/CUDA imports beyond tensor IO."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E078'
start=time.time();torch.set_num_threads(4);assert not torch.cuda.is_initialized()
p=json.loads((OUT/'run.json').read_text());assert p['status']=='complete'
maximum=0.;rows=[]
for c in p['comparisons']:
 key=c['key'];data={}
 for arm in ('B0','BA','S0','SA'):
  rec=next(r['artifact'] for r in p['cases'] if r['key']==key and r['arm']==arm)
  path=Path(rec['file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
  data[arm]=torch.load(path,map_location='cpu',weights_only=True)
 row={'key':key}
 for m in ('video','audio'):
  a={k:v[m].float().numpy().astype(np.float64) for k,v in data.items()}
  es=a['S0']-a['B0'];ea=a['BA']-a['B0'];both=a['SA']-a['B0'];i=both-es-ea
  energy=lambda x:float(np.square(x).sum())
  out=dict(svd_sse=energy(es),sage_sse=energy(ea),combined_sse=energy(both),additive_sse=energy(es+ea),net_amplification=energy(both)/energy(es+ea),combined_vs_svd=energy(both)/energy(es),attention_increment_ratio=energy(both-es)/energy(ea),interaction_sse=energy(i),cos_single_errors=float(np.sum(es*ea))/np.sqrt(energy(es)*energy(ea)))
  for k,v in out.items():
   err=abs(v-c[m][k])/max(1,abs(c[m][k]));maximum=max(maximum,err);assert err<1e-12
  row[m]=out
 rows.append(row)
result=dict(status='complete',cuda_initialized=torch.cuda.is_initialized(),seconds=time.time()-start,max_relative_difference=maximum,comparisons=rows)
assert not result['cuda_initialized']
with (OUT/'independent.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
