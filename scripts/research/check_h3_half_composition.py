"""Independent CPU readout of E077 raw tensors, no model imports."""
import hashlib,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research/E077'
start=time.time(); torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
p=json.loads((OUT/'run.json').read_text()); assert p['status']=='complete'
parent=json.loads((ROOT/'results/research/E065b/evaluate.json').read_text())
def read(rec):
 path=Path(rec['file'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
 return torch.load(path,map_location='cpu',weights_only=True)
rows=[]; greatest=0.
for group in p['comparisons']:
 k=group['key']; reference=read(parent['references'][k]['bf16']['artifact'])['raw_outputs']
 tensors={a:read(next(c['artifact'] for c in p['cases'] if c['key']==k and c['arm']==a)) for a in ('RR','CR','RC','CC')}
 for m in ('video','audio'):
  data={a:t[m].float().numpy().astype('float64') for a,t in tensors.items()}
  teacher=reference[m].float().numpy().astype('float64')
  calc={a:float(((v-teacher)**2).sum()) for a,v in data.items()}
  expected=group[m]['sse']
  for a in calc:
   d=abs(calc[a]-expected[a])/max(1,abs(expected[a]));greatest=max(greatest,d);assert d<1e-12
  nonlinear=data['CC']-data['CR']-data['RC']+data['RR']
  energy=float((nonlinear**2).sum());assert abs(energy/group[m]['interaction_energy']-1)<1e-12
  rows.append(dict(key=k,modality=m,sse=calc,interaction_energy=energy))
result=dict(status='complete',cuda_initialized=torch.cuda.is_initialized(),seconds=time.time()-start,max_relative_difference=greatest,rows=rows)
assert not result['cuda_initialized']
with (OUT/'independent.json').open('x') as f:json.dump(result,f,indent=2)
print({k:v for k,v in result.items() if k!='rows'})
