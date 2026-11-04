#!/usr/bin/env python3
"""Characterize high-attention video key columns and their 3D coordinates."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'third_party/anemoi-code'))
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
OUT=ROOT/'results/research/E217';TOP=ROOT/'results/research/tile_probability_overlap_q64_20261103';HEADS=(10,20,30)
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 for step in (0,10):
  p=torch.load(f'/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step{step}_block25.pt',map_location='cpu',weights_only=True,mmap=True)
  prefix=int(p['prefix_tokens']);grid=tuple(p['video_grid']);lay=materialize_ragged_2d_layout('cuda',frames=grid[0],height=grid[1],width=grid[2],logical_block=64,enable_anchors=False);counts=lay.counts;nb=counts.numel();idx=lay.indices.view(nb,64);valid=lay.slot_valid.view(nb,64)
  q=p['q'][0,list(HEADS)].cuda().float();k=p['k'][0,list(HEADS)].cuda().float();v=p['v'][0,list(HEADS)].cuda().float();qv=q[:,prefix+idx,:];kv=k[:,prefix+idx,:];vv=v[:,prefix+idx,:];factor=valid[None,:,:,None];cnt=counts[None,:,None]
  qm=(qv*factor).sum(2)/cnt;km=(kv*factor).sum(2)/cnt;vm=(vv*factor).sum(2)/cnt
  logits=qm@km.transpose(1,2)/128**.5
  qglobal=q[:,prefix:,:].mean(1);kmean_logit=(qglobal[:,None,:]*km).sum(-1)/128**.5
  # Full attention probabilities with prefix keys in denominator. Key columns follow original contiguous order.
  totalq=q.shape[1]-prefix;colmass=torch.zeros((len(HEADS),nb),device='cuda');prefixmass=torch.zeros(len(HEADS),device='cuda')
  for start in range(0,totalq,384):
   prob=torch.bmm(q[:,prefix+start:prefix+min(start+384,totalq)],k.transpose(1,2)).mul_(128**-.5).softmax(-1);prefixmass+=prob[:,:,:prefix].sum((1,2));vid=prob[:,:,prefix:];gather=vid[:,:,idx.reshape(-1)].view(len(HEADS),-1,nb,64);gather*=valid[None,None,:,:];colmass+=gather.sum((1,3))
  qpos=idx.cpu().numpy();mask=valid.cpu().numpy();pos=np.zeros((nb,4));
  for j in range(nb):
   indexes=qpos[j][mask[j]];frame=indexes//(grid[1]*grid[2]);assert np.unique(frame).size==1;sp=indexes%(grid[1]*grid[2]);yy=sp//grid[2];xx=sp%grid[2];pos[j]=[frame[0],j%lay.blocks_per_frame,yy.mean(),xx.mean()]
  with np.load(TOP/f'top_step_{step:02d}.npz') as d:top=d['true'][list(HEADS),:int(.1*nb*nb+.5)]
  sel=np.zeros((len(HEADS),nb*nb),bool);np.put_along_axis(sel,top,True,axis=1);sel=sel.reshape(len(HEADS),nb,nb)
  def tonp(x):return x.detach().cpu().numpy()
  np.savez_compressed(OUT/f'step{step}_features.npz',heads=HEADS,coords=pos,counts=tonp(counts),selection_col_degree=sel.mean(1),column_probability_mass=tonp(colmass)/totalq,column_probability_mean=(tonp(colmass)/totalq)/tonp(counts)[None,:],prefix_mass=tonp(prefixmass)/totalq,mean_logits=tonp(kmean_logit),pooled_qk=tonp(logits),k_mean_norm=tonp(km.norm(dim=-1)),k_rms=tonp(((kv.square()*factor).sum((2,3))/(cnt[...,0]*128)).sqrt()),v_rms=tonp(((vv.square()*factor).sum((2,3))/(cnt[...,0]*128)).sqrt()),q_mean_norm=tonp(qm.norm(dim=-1)))
  print('done',step,'blocks',nb,'per frame',lay.blocks_per_frame,flush=True)
if __name__=='__main__':main()
