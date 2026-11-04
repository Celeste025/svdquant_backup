#!/usr/bin/env python3
"""Compare native DraftMap and exact softmax-mean routes on identical QKV."""
import hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from run_e198_oracle_video import score
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs, _route_sm120_probability
from anemoi.layers.attention.mpa.routing import route_probability
from anemoi.layers.attention.mpa.backends.sm120_q64 import sm120_h3_draft_probability

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()

def unpack(plan, shape):
 ids=plan.block_ids.long();cols=shape[-1];index=torch.arange(cols,device=ids.device).view(1,1,1,cols)
 n4=plan.nvfp4_counts.unsqueeze(-1);n8=plan.fp8_counts.unsqueeze(-1)
 val=torch.where(index<n4,1,torch.where(index<n4+n8,2,0)).to(torch.uint8)
 out=torch.zeros((*shape[:-1],cols+1),dtype=torch.uint8,device=ids.device);out.scatter_(-1,torch.where(val==0,cols,ids),val);out=out[...,:cols]
 assert torch.equal((out==1).sum((-1,-2)),torch.full(shape[:2],plan.nvfp4_blocks_per_head,device=ids.device))
 assert torch.equal((out==2).sum((-1,-2)),torch.full(shape[:2],plan.fp8_blocks_per_head,device=ids.device))
 return out

@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 manifest=json.loads((ROOT/'results/research/E187/manifest.json').read_text());out=ROOT/'results/research/E199/map_overlap.json';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 result={'status':'running','source':'E187 BF16 trajectory; identical inputs for both map methods','prompt_id':16,'method':'Q128 native DraftMap maxpool_weight=.2 vs FP32 exact softmax probability mean; per-head global allocation, no anchors','samples':[]}
 for step,layer in ((0,25),(10,0),(10,25),(10,49)):
  entry=next(x for x in manifest['captures'] if x['step']==step and x['layer']==layer);path=Path(entry['path']);assert sha(path)==entry['sha256']
  data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);q,k,v=(data[z].cuda().contiguous() for z in ('q','k','v'));grid=tuple(data['video_grid']);prefix=int(data['prefix_tokens']);layout=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=128,enable_anchors=False)
  prep=_prepare_h3_sm120_inputs(q,k,v,video_token_indices=layout.indices,video_slot_valid=layout.slot_valid,video_valid_counts=layout.counts,prefix_tokens=prefix,query_block_size=128,phases=('int8',),has_prefix_query_int8=True,has_pool=True,has_maxpool=True,global_scales=None)
  proxy=sm120_h3_draft_probability(prep[0],prep[1],prep[-2],prep[-1],maxpool_weight=.2);del prep
  true=score(q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),prefix,grid,'prob_mean',0)
  row={'step_zero_based':step,'layer_zero_based':layer,'capture_sha256':entry['sha256'],'tile_count_per_head':layout.counts.numel()**2,'budgets':{}}
  for skip,nv in ((.75,.6),(0.,.9)):
   ratios=(nv,1-nv,0.,0.)
   native=_route_sm120_probability(proxy,None,None,anchor_count=0,sparsity_ratio=skip,ratios=ratios)
   exact=route_probability(true,None,anchor_count=0,prefix_blocks=0,sparsity_ratio=skip,nvfp4_ratio=nv,fp8_ratio=1-nv,fp16_ratio=0.)
   a,b=(unpack(plan,true.shape) for plan in (native,exact));int8=((a==2)&(b==2)).sum((-1,-2)).float()/native.fp8_blocks_per_head
   keep=((a>0)&(b>0)).sum((-1,-2)).float()/(native.fp8_blocks_per_head+native.nvfp4_blocks_per_head)
   same=(a==b).float().mean((-1,-2));matrix=[[int(((a==x)&(b==y)).sum()) for y in range(3)] for x in range(3)]
   row['budgets'][str(skip)]={'int8_overlap_mean':int8.mean().item(),'int8_overlap_by_head':int8[0].tolist(),'retained_overlap_mean':keep.mean().item(),'retained_overlap_by_head':keep[0].tolist(),'same_precision_fraction':same.mean().item(),'confusion_matrix_rows_draft_cols_softmax_skip_nvfp4_int8':matrix,'int8_tiles_per_head':native.fp8_blocks_per_head,'nvfp4_tiles_per_head':native.nvfp4_blocks_per_head}
  result['samples'].append(row);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row['budgets'],default=str)[:500],flush=True)
 result['status']='complete';out.write_text(json.dumps(result,indent=2)+'\n');print('E199_COMPLETE',flush=True)
if __name__=='__main__':main()
