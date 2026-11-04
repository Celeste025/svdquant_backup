#!/usr/bin/env python3
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'third_party/anemoi-code'))
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
@torch.inference_mode()
def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(6);out=[]
 for step in (0,10):
  d=np.load(ROOT/f'results/research/E217/step{step}_features.npz');p=torch.load(f'/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step{step}_block25.pt',map_location='cpu',weights_only=True,mmap=True);prefix=p['prefix_tokens'];grid=p['video_grid'];lay=materialize_ragged_2d_layout('cuda',frames=grid[0],height=grid[1],width=grid[2],logical_block=64,enable_anchors=False)
  for ih,h in enumerate((10,20,30)):
   column=int(d['selection_col_degree'][ih].argmax());count=int(lay.counts[column]);ids=lay.indices.view(-1,64)[column,:count]+prefix;q=p['q'][0,h].cuda().float();k=p['k'][0,h].cuda().float();perkey=torch.zeros(count,device='cuda');n=q.shape[0]-prefix
   for start in range(0,n,512):
    probs=(q[prefix+start:prefix+min(start+512,n)]@k.T/128**.5).softmax(-1);perkey+=probs[:,ids].sum(0)
   vals,indices=perkey.sort(descending=True);fraction=vals.cumsum(0)/vals.sum();out.append({'step':step,'head':h,'key_block':column,'tokens':count,'selection_query_fraction':float(d['selection_col_degree'][ih,column]),'video_query_average_attention_mass':float(vals.sum()/n),'top1_fraction_of_block_mass':float(fraction[0]),'top5_fraction_of_block_mass':float(fraction[min(4,count-1)]),'max_token_fraction':float(vals[0]/vals.sum()),'top_token_video_index':int(ids[indices[0]]-prefix),'top_token_spatial_yx':[int(((ids[indices[0]]-prefix)%(grid[1]*grid[2]))//grid[2]),int((ids[indices[0]]-prefix)%grid[2])],'top_token_k_norm':float(k[ids[indices[0]]].norm()),'block_median_k_norm':float(k[ids].norm(dim=-1).median())});print(out[-1],flush=True)
 (ROOT/'results/research/E217/top_tokens.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
