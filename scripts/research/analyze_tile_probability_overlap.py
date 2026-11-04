#!/usr/bin/env python3
"""Compare per-head top tiles from exact softmax means on captured H3 Q/K."""
import hashlib, json, math, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'third_party/anemoi-code'))
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
OUT=ROOT/'results/research/tile_probability_overlap_20261103'

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
 return h.hexdigest()

@torch.inference_mode()
def calculate(entry):
 path=Path(entry['path']); assert digest(path)==entry['sha256']
 d=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 q,k=(d[z].cuda() for z in ('q','k')); prefix=int(d['prefix_tokens']); grid=tuple(d['video_grid'])
 lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=128,enable_anchors=False)
 n=lay.counts.numel(); valid=lay.slot_valid; counts=lay.counts
 keyvalid=torch.cat((torch.ones(prefix,device=q.device,dtype=torch.bool),valid))
 result=torch.empty((q.shape[1],n,n),device=q.device)
 for begin in range(0,q.shape[1],4):
  end=min(begin+4,q.shape[1]); h=end-begin
  qv=q[0,begin:end,lay.indices+prefix].float().reshape(h,n,128,-1)
  keys=torch.cat((k[0,begin:end,:prefix].float(),k[0,begin:end,lay.indices+prefix].float()),dim=1)
  for row in range(n):
   nq=int(counts[row]); logits=torch.bmm(qv[:,row,:nq],keys.transpose(1,2))*q.shape[-1]**-.5
   logits.masked_fill_(~keyvalid.view(1,1,-1),-torch.inf)
   p=logits.softmax(-1)[:,:,prefix:].view(h,nq,n,128)
   result[begin:end,row]=p.sum((1,3))/(nq*counts)
 assert torch.isfinite(result).all()
 return result.cpu(), {'blocks':n,'prefix_tokens':prefix,'grid':grid,'valid_counts':counts.cpu().tolist()}

def main():
 torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32=False
 manifests=[json.loads((ROOT/x).read_text()) for x in ('results/research/E187/manifest.json','results/research/attention_step_similarity/manifest.json')]
 entries={(e['step'],e['layer']):e for m in manifests for e in m['captures']}
 maps={}; metadata=None
 for key,e in sorted(entries.items()):
  maps[key],meta=calculate(e)
  if metadata is not None: assert meta==metadata
  metadata=meta; print('calculated',key,flush=True)
 pairs=[]; n=metadata['blocks']**2
 for i,a in enumerate(sorted(maps)):
  for b in sorted(maps)[i+1:]:
   if a[0]!=b[0] and a[1]!=b[1]: continue
   row={'a':{'step':a[0],'layer':a[1]},'b':{'step':b[0],'layer':b[1]},'kind':'same_layer' if a[1]==b[1] else 'same_step','budgets':{}}
   for rate in (.1,.2,.3):
    count=math.floor(rate*n+.5)
    ia=torch.argsort(maps[a].flatten(1),dim=1,descending=True,stable=True)[:,:count]
    ib=torch.argsort(maps[b].flatten(1),dim=1,descending=True,stable=True)[:,:count]
    mask=torch.zeros((ia.shape[0],n),dtype=torch.bool).scatter_(1,ia,True)
    overlaps=mask.gather(1,ib).sum(1).float()/count
    row['budgets'][str(rate)]={'tiles':count,'overlap_mean':overlaps.mean().item(),'overlap_median':overlaps.median().item(),'overlap_min':overlaps.min().item(),'overlap_max':overlaps.max().item(),'overlap_by_head':overlaps.tolist()}
   pairs.append(row)
 out={'status':'complete','prompt_id':16,'seed':20026,'trajectory':'BF16 dense closed-loop, 480x832, 124 frames, 20 steps','capture_source':'E187 and attention_step_similarity; SHA256 verified','indices':'zero-based steps and layers','method':'Per head global ranking of video Q128 x K128 tiles by mean FP32 softmax probability over valid query-key pairs; denominator includes prefix and valid video keys; ragged padding excluded. Compare matching head and tile coordinates, overlap = intersection / k; k=round-half-up(fraction * number of video tiles).','metadata':metadata,'pairs':pairs}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'overlap.json').write_text(json.dumps(out,indent=2)+'\n')
 print('complete',flush=True)
if __name__=='__main__':main()
