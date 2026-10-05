#!/usr/bin/env python3
"""Summarize the paired teacher-state diagnostic; no quality claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
x=json.loads((ROOT/'results/research/E002_wan_temporal_errors.json').read_text())
rows=[]
for pid in sorted({s['prompt'] for s in x['series']}):
    ss={s['variant']:s for s in x['series'] if s['prompt']==pid}
    for branch in ['cond','cfg']:
        full=ss['w4a4_'+branch];w=ss['w4a16_'+branch];a=ss['activation_increment_'+branch]
        def total(s):return sum(s['raw']['gram'][i][i] for i in range(len(x['steps'])))
        ef,ew,ea=total(full),total(w),total(a)
        rows.append({'prompt':pid,'branch':branch,
                     'w4a4_adjacent_cosine':full['channel_centered']['adjacent_cosine_mean'],
                     'w4a16_adjacent_cosine':w['channel_centered']['adjacent_cosine_mean'],
                     'activation_increment_adjacent_cosine':a['channel_centered']['adjacent_cosine_mean'],
                     'w4a16_energy_over_full':ew/ef,
                     'activation_increment_energy_over_full':ea/ef,
                     'signed_cross_energy_over_full':(ef-ew-ea)/ef})
report={'status':'complete','rows':rows,'cache_replay_max_nmse':max(r['cache_replay_nmse'] for r in x['rows']),
        'decision':'stop full-W4A4 coherent temporal error hypothesis at CFG6; both centered adjacent cosines < 0.2',
        'limitations':['2 calibration prompts; 6 mid-trajectory steps','teacher-state QDQ, not native or closed-loop',
                       'energy decomposition has signed cross term, not independent variance fractions']}
(ROOT/'research_state/06_experiments/results/E002_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
for ax,pid in zip(axs, sorted({r['prompt'] for r in rows})):
    subset={r['branch']:r for r in rows if r['prompt']==pid}
    labels=['W4A4','W4A16','Activation increment']
    keys=['w4a4_adjacent_cosine','w4a16_adjacent_cosine','activation_increment_adjacent_cosine']
    pos=list(range(3))
    ax.bar([i-.17 for i in pos],[subset['cond'][k] for k in keys],width=.34,label='Conditional')
    ax.bar([i+.17 for i in pos],[subset['cfg'][k] for k in keys],width=.34,label='CFG6')
    ax.axhline(.2,color='black',ls='--',lw=1,label='Pre-set stop threshold')
    ax.set_xticks(pos,labels);ax.set_ylim(0,1);ax.set_title('Prompt '+pid)
    ax.set_ylabel('Adjacent error cosine after channel centering')
    ax.tick_params(axis='x',labelsize=8)
axs[0].legend(fontsize=8)
fig.suptitle('Wan teacher-state probe: 6 consecutive middle steps (exploratory)',fontsize=11)
fig.savefig(ROOT/'research_state/reports/figures/E002_temporal_correlation.png',dpi=180)
plt.close(fig)
