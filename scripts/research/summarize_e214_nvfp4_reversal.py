#!/usr/bin/env python3
"""Summarize strict paired NVFP4 error reversals across four H3 trajectories."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E214';PIDS=(25,7,48,44);labels=('Mahjong cats','Mars astronaut','Kayak','Swimmer')
nmse=np.full((4,20,50,56,3),np.nan);energy=np.full((4,20,50,56),np.nan);agg=np.full((4,20,50,3),np.nan)
for p,pid in enumerate(PIDS):
 folder=OUT/f'p{pid:03d}';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['cells']==1000
 seen=set()
 for line in (folder/'cells.jsonl').open():
  r=json.loads(line);s,l=r['step'],r['layer'];assert (s,l) not in seen and r['prompt_id']==pid;seen.add((s,l));energy[p,s,l]=r['reference_energy_by_head']
  for a in range(3):nmse[p,s,l,:,a]=r['arms'][a]['nmse_by_head'];agg[p,s,l,a]=r['arms'][a]['nmse_overall']
 assert len(seen)==1000
assert np.isfinite(nmse).all() and (nmse>=0).all() and np.isfinite(agg).all() and (energy>0).all()
# Fixed identical reference energy per head means NMSE and MSE have the same ordering.
d01=nmse[...,1]-nmse[...,0];d12=nmse[...,2]-nmse[...,1];d02=nmse[...,2]-nmse[...,0];flag=(d01>0)|(d12>0)|(d02>0)
# In this ordered triple d02 cannot be positive without at least one adjacent increase.
assert np.array_equal(flag,(d01>0)|(d12>0))
layerflag=(agg[...,1]>agg[...,0])|(agg[...,2]>agg[...,1])|(agg[...,2]>agg[...,0])
fig,axs=plt.subplots(2,2,figsize=(15,8),sharex=True,sharey=True,layout='constrained')
for i,ax in enumerate(axs.flat):
 image=ax.imshow(flag[i].mean(-1).T*100,origin='lower',aspect='auto',interpolation='nearest',extent=(-.5,19.5,-.5,49.5),cmap='magma',vmin=0,vmax=100)
 ax.set_title(f'p{PIDS[i]:03d} · {labels[i]}');ax.set_xlabel('Step (zero-based)');ax.set_ylabel('Layer (zero-based)')
fig.colorbar(image,ax=axs,label='Heads with any reversal (%)',shrink=.86);fig.suptitle('More NVFP4 increases head output NMSE: prevalence',fontsize=14)
fig.savefig(OUT/'reversal_by_step_layer.png',dpi=170);fig.savefig(OUT/'reversal_by_step_layer.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(13,6),layout='constrained');im=ax.imshow(flag.mean((0,1))*100,origin='lower',aspect='auto',interpolation='nearest',cmap='magma',vmin=0,vmax=100);ax.set(xlabel='Head (zero-based)',ylabel='Layer (zero-based)',title='Reversal frequency across 4 prompts × 20 steps');fig.colorbar(im,ax=ax,label='Prompt-step observations (%)');fig.savefig(OUT/'reversal_by_layer_head.png',dpi=170);fig.savefig(OUT/'reversal_by_layer_head.pdf');plt.close(fig)
with (OUT/'step_layer_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['prompt_id','step','layer','flagged_heads','head_fraction','overall_nmse_10','overall_nmse_40','overall_nmse_90','layer_reversal']);
 for p,pid in enumerate(PIDS):
  for s in range(20):
   for l in range(50):w.writerow([pid,s,l,int(flag[p,s,l].sum()),float(flag[p,s,l].mean()),*agg[p,s,l].tolist(),int(layerflag[p,s,l])])
with (OUT/'per_head_flags.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['prompt_id','step','layer','head','nmse_fp4_10','nmse_fp4_40','nmse_fp4_90','increase_10_40','increase_40_90','increase_10_90','any_increase'])
 for p,pid in enumerate(PIDS):
  for step in range(20):
   for layer in range(50):
    for head in range(56):
     x=nmse[p,step,layer,head];w.writerow([pid,step,layer,head,*x.tolist(),int(x[1]>x[0]),int(x[2]>x[1]),int(x[2]>x[0]),int(flag[p,step,layer,head])])
rows=[]
for l in range(50):
 for h in range(56):
  x=flag[:,:,l,h];rows.append({'layer':l,'head':h,'count':int(x.sum()),'frequency':float(x.mean()),'prompts_any':int(x.any(axis=1).sum()),'per_prompt_count':x.sum(axis=1).astype(int).tolist(),'median_max_increase_percentage_points':float(np.median(np.maximum.reduce([d01[:,:,l,h],d12[:,:,l,h],d02[:,:,l,h],np.zeros((4,20))]))*100)})
rows.sort(key=lambda x:(-x['count'],x['layer'],x['head']))
summary={'status':'complete','definition':'head flagged if any strict increase in video-query output NMSE between 10→40, 40→90, 10→90 percent NVFP4; fixed 10% INT8, same BF16 QKV and reference','prompts':list(PIDS),'shape':[4,20,50,56,3],'heads_observations':int(flag.size),'flagged_head_observations':int(flag.sum()),'flagged_head_fraction':float(flag.mean()),'flagged_layer_observations':int(layerflag.sum()),'layer_observations':int(layerflag.size),'layer_fraction':float(layerflag.mean()),'pair_fractions':{'10_to_40':float((d01>0).mean()),'40_to_90':float((d12>0).mean()),'10_to_90':float((d02>0).mean())},'prompt_head_fractions':{str(pid):float(flag[i].mean()) for i,pid in enumerate(PIDS)},'step_head_fractions':flag.mean((0,2,3)).tolist(),'layer_head_fractions':flag.mean((0,1,3)).tolist(),'counts_unique_layer_heads':{'ever':int(flag.any((0,1)).sum()),'in_all_four_prompts':int(flag.any(axis=1).all(axis=0).sum())},'effect_threshold_fractions':{str(pp):float((np.maximum.reduce([d01,d12,d02])>pp/100).mean()) for pp in (.001,.01,.1,1.)},'top_layer_heads':rows[:100],'all_layer_heads':rows,'interpretation':'descriptive across four selected prompts, not independent samples; Q64 DraftMap and INT8 prefix; no individual tile causality or video quality claim'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('all_layer_heads','top_layer_heads')},indent=2))
