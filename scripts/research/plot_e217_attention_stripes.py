#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/E217';a=[np.load(out/f'step{s}_features.npz') for s in (0,10)];fig,axs=plt.subplots(3,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained');summary=[]
for h,head in enumerate((10,20,30)):
 for t,step in enumerate((0,10)):
  d=a[t];deg=d['selection_col_degree'][h];coord=d['coords'];frame=coord[:,0].astype(int);reg=coord[:,1].astype(int);M=np.zeros((7,37));M[reg,frame]=deg
  ax=axs[h,t];im=ax.imshow(M,origin='lower',aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='magma');ax.set_ylabel(f'Head {head}\nSpatial block (0–6)')
  if h==0:ax.set_title(f'Step {step}')
  if h==2:ax.set_xlabel('Key latent time index')
  rho=spearmanr(deg,d['mean_logits'][h]).statistic;rhonk=spearmanr(deg,d['k_mean_norm'][h]).statistic
  summary.append({'head':head,'step':step,'rho_column_degree_vs_mean_logit':float(rho),'rho_column_degree_vs_k_block_mean_norm':float(rhonk),'mean_degree_by_time_mod5':[float(deg[frame%5==i].mean()) for i in range(5)],'fraction_cols_selected_by_at_least_half_queries':float((deg>=.5).mean()),'top_block':int(deg.argmax()),'top_block_coords':coord[deg.argmax()].tolist()})
fig.colorbar(im,ax=axs,label='Fraction of query blocks selecting this key block',shrink=.8);fig.suptitle('Strong key-block columns in true top-10% tile map | layer 25',fontsize=14);fig.savefig(out/'key_column_temporal_spatial.png',dpi=170);fig.savefig(out/'key_column_temporal_spatial.pdf');(out/'summary.json').write_text(json.dumps({'settings':'BF16 dense p16 seed20026, layer25, Q64 37x15x26 grid, steps0/10, heads10/20/30, top10% tiles, prefix included in softmax, video-only columns','metrics':'Spearman across 259 key blocks; query selection fraction of global per-head top10% tiles','results':summary},indent=2)+'\n')
