#!/usr/bin/env python3
"""Aggregate and plot the scale-jitter audit; all NMSEs are ratios of sums."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(pattern):
    rows=[]
    for p in sorted(pattern):
        with p.open() as f:rows.extend(csv.DictReader(f))
    return rows


def aggregate(rows, keys, pairs):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    result=[]
    for k,rs in sorted(grouped.items()):
        row=dict(zip(keys,k));row['cases']=len(rs)
        for prefix,(err,ref) in pairs.items():
            use=[r for r in rs if r.get(err) and r.get(ref)]
            if use:
                row[prefix]=sum(float(r[err]) for r in use)/max(sum(float(r[ref]) for r in use),1e-30)
                row[prefix+'_median']=statistics.median(float(r[err])/max(float(r[ref]),1e-30) for r in use)
        for name in ['actual_relative_delta','scale_switch_fraction','switch_block_response_fraction']:
            if name in rs[0]:row[name+'_median']=statistics.median(float(r[name]) for r in rs)
        result.append(row)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();o=args.output
    local_pairs={'response_nmse':('response_err2','response_ref2'),'point_nmse':('point_err2','point_ref2'),
                 'linear_response_nmse':('linear_response_err2','linear_response_ref2')}
    functional_pairs={f'{k}_{m}_nmse':(f'{k}_{m}_err2',f'{k}_{m}_ref2') for k in ['final','block'] for m in ['response','point']}
    functional_pairs['activation_response_nmse']=('a_response_err2','a_response_ref2')
    rcm=read(o.glob('local_*.csv'));wan=read((o/'wan50').glob('local_*.csv'))
    fa=read(o.glob('functional_activation_*.csv'))
    fl=read([o/'functional_0000.csv',o/'functional_0005.csv'])
    fw=read([o/'wan50/functional_wan_endpoints.csv'])
    summary={
        'counts':{'rcm_local':len(rcm),'wan_local':len(wan),'rcm_activation_functional':len(fa),'rcm_latent_functional':len(fl),'wan_endpoint_functional':len(fw)},
        'rcm_local':aggregate(rcm,['axis','eps','mode'],local_pairs),
        'rcm_local_by_prompt':aggregate(rcm,['prompt','axis','eps','mode'],local_pairs),
        'wan_local':aggregate(wan,['axis','eps','mode'],local_pairs),
        'wan_natural_steps':aggregate([r for r in wan if r['axis']=='denoise' and float(r['eps'])==-1],['step','mode'],local_pairs),
        'rcm_activation_functional':aggregate(fa,['eps','layer','mode'],functional_pairs),
        'rcm_latent_functional':aggregate(fl,['eps','step','mode'],functional_pairs),
        'wan_endpoint_functional':aggregate(fw,['step','layer','mode'],functional_pairs),
        'dc_rcm_local':aggregate(read(o.glob('dc_local_*.csv')),['axis','eps','mode'],local_pairs),
        'dc_wan_natural_steps':aggregate([r for r in read((o/'wan50').glob('dc_local_*.csv')) if r['axis']=='denoise' and float(r['eps'])==-1],['step','mode'],local_pairs),
    }
    if (o/'wan50/dc_functional_wan_endpoints.csv').exists():
        summary['dc_wan_endpoint_functional']=aggregate(read([o/'wan50/dc_functional_wan_endpoints.csv']),['step','layer','mode'],functional_pairs)
    # Pairwise win counts describe cases; they are not independent statistical replicates.
    def paired_effect(rows,keys,err,baseline='dynamic_global'):
        groups=defaultdict(dict)
        for r in rows:groups[tuple(r[k] for k in keys)][r['mode']]=float(r[err])
        out=defaultdict(list)
        for v in groups.values():
            for mode,e in v.items():
                if mode!=baseline and baseline in v:out[mode].append(e/max(v[baseline],1e-30)-1)
        return {m:{'cases':len(x),'wins':sum(v<0 for v in x),'median_relative_change':statistics.median(x),
                   'min_relative_change':min(x),'max_relative_change':max(x)} for m,x in out.items()}
    summary['wan_mid_final_response_paired']=paired_effect([r for r in fw if r['step']=='24'],['prompt','layer'],'final_response_err2')
    summary['rcm_small_final_response_paired']=paired_effect([r for r in fa if float(r['eps'])==.01],['prompt','step','layer'],'final_response_err2')
    (o/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(15,4.2))
    modes=['frozen_block','hysteresis_mse5','continuous_scale_control']
    labels=['Frozen block scale','Hysteresis (5% MSE budget)','Continuous scale (control)']
    colors=['#c44e52','#4c72b0','#55a868']
    for mode,label,color in zip(modes,labels,colors):
        eps=[.001,.01,.05,.1];ys=[]
        for e in eps:
            r={z['mode']:z for z in summary['rcm_local'] if z['axis']=='denoise' and float(z['eps'])==e}
            ys.append(r[mode]['response_nmse']/r['dynamic_block']['response_nmse'])
        axs[0].plot(np.array(eps)*100,ys,'o-',label=label,color=color)
    axs[0].axhline(1,c='gray',ls='--');axs[0].set_xscale('log');axs[0].set_xlabel('Controlled activation change (% RMS)')
    axs[0].set_ylabel('Response error / dynamic block scale');axs[0].set_title('rCM: local Q/DQ response');axs[0].legend(fontsize=8)
    for j,(mode,label,color) in enumerate(zip(modes[:2],labels[:2],colors[:2])):
        vals=[]
        for s in ['0','24','48']:
            r={z['mode']:z for z in summary['wan_natural_steps'] if z['step']==s}
            vals.append(r[mode]['response_nmse']/r['dynamic_block']['response_nmse'])
        axs[1].bar(np.arange(3)+(j-.5)*.32,vals,.32,color=color,label=label)
    axs[1].axhline(1,c='gray',ls='--');axs[1].set_xticks(range(3),['1 -> 2','25 -> 26','49 -> 50'])
    axs[1].set_xlabel('Actual adjacent denoising steps (1-based)');axs[1].set_ylabel('Response error / dynamic block scale');axs[1].set_title('Wan-50: local Q/DQ response')
    for j,(mode,label,color) in enumerate(zip(modes[:2],labels[:2],colors[:2])):
        vals=[]
        for name in ['blocks.20.attn1.to_q','blocks.20.ffn.net.2']:
            r={z['mode']:z for z in summary['wan_endpoint_functional'] if z['step']=='24' and z['layer']==name}
            vals.append(100*(r[mode]['final_response_nmse']/r['dynamic_global']['final_response_nmse']-1))
        axs[2].bar(np.arange(2)+(j-.5)*.32,vals,.32,color=color,label=label)
    axs[2].axhline(0,c='gray',ls='--');axs[2].set_xticks(range(2),['Block 20: Q','Block 20: FFN down'])
    axs[2].set_ylabel('Final response error change (%)');axs[2].set_title('Wan-50: effect at denoiser output\n(actual steps 25 -> 26)')
    fig.tight_layout();fig.savefig(o/'scale_jitter_results.png',dpi=180);fig.savefig(o/'scale_jitter_results.pdf');plt.close(fig)
    print(json.dumps(summary['counts'],indent=2))


if __name__=='__main__':main()
