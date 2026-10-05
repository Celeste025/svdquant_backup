#!/usr/bin/env python3
"""Plot observed E007 paired drift; no causal attribution or quality claim."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
data=json.loads((ROOT/'research_state/06_experiments/results/E007_summary.json').read_text())
rows=data['block_rows'];x=[r['block'] for r in rows]
fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
ax[0].semilogy(x,[r['native_vs_qdq_nmse'] for r in rows],'-o',ms=3,color='#3568b0')
ax[0].set(xlabel='Block index',ylabel='NMSE (native vs legacy QDQ)',title='Complete upstream trajectories diverge')
ax[1].semilogy(x,[r['absolute_error_rms'] for r in rows],'-o',ms=3,label='Difference RMS',color='#c76a2c')
ax[1].semilogy(x,[r['qdq_reference_rms'] for r in rows],'-',label='QDQ reference RMS',color='#757575')
ax[1].set(xlabel='Block index',ylabel='RMS',title='Absolute difference and reference scale')
ax[1].legend(frameon=False,fontsize=9)
for a in ax:
 a.grid(alpha=.2);a.spines[['top','right']].set_visible(False);a.set_xticks([0,5,10,15,20,25,29])
fig.suptitle('rCM-Wan 1.3B · one calibration input · 31,200 tokens · same quantization recipe',fontsize=10)
p=ROOT/'research_state/reports/figures/E007_native_trajectory_drift.png';p.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(p,dpi=180)
print(p)
