#!/usr/bin/env python3
"""Validate E142, export tables and per-input speed/error frontiers."""
import csv
import json
from pathlib import Path
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/home/wjq/workspace/svdquant-exp')
REPORT=ROOT/'results/research/E142'
SAMPLES=['race_s00_b00','race_s10_b25','clap_s10_b49']


def main():
    deadline=time.monotonic()+1800
    reports=[]
    for sample in SAMPLES:
        path=REPORT/f'{sample}.json'
        while True:
            try:
                row=json.loads(path.read_text())
                if row.get('status')=='complete':
                    reports.append(row)
                    break
            except (FileNotFoundError,json.JSONDecodeError):
                pass
            assert time.monotonic()<deadline, f'not completed: {sample}'
            time.sleep(10)
    names=list(reports[0]['summary'])
    assert len(names)==19 and all(set(r['summary'])==set(names) for r in reports)
    for r in reports:
        for name in names:
            assert len(r['rounds'][name])==2
            assert all(t['calls']==72 for t in r['rounds'][name])
            row=r['summary'][name]
            assert row['mean_ms']>0 and row['video_nmse']>=0 and np.isfinite(row['mean_ms']+row['video_nmse'])
    aggregate={}
    raw=[]
    for name in names:
        rows=[r['summary'][name] for r in reports]
        aggregate[name]={'median_ms':statistics.median(r['mean_ms'] for r in rows),
                         'ms_range':[min(r['mean_ms'] for r in rows),max(r['mean_ms'] for r in rows)],
                         'median_video_nmse':statistics.median(r['video_nmse'] for r in rows),
                         'video_nmse_range':[min(r['video_nmse'] for r in rows),max(r['video_nmse'] for r in rows)],
                         'speedup_over_sage3':[r['summary']['sage3_optimized']['mean_ms']/r['summary'][name]['mean_ms'] for r in reports],
                         'spec':reports[0]['specs'][name]}
    for report in reports:
        summaries=report['summary']
        for name,row in summaries.items():
            dominated=any(other['mean_ms']<=row['mean_ms'] and other['video_nmse']<=row['video_nmse']
                          and (other['mean_ms']<row['mean_ms'] or other['video_nmse']<row['video_nmse'])
                          for other in summaries.values())
            raw.append({'sample':report['sample'],'arm':name,'ms':row['mean_ms'],
                        'speedup_bf16':row['speedup_over_bf16'],
                        'nmse_all_percent':row['output_nmse']*100,
                        'nmse_video_percent':row['video_nmse']*100,
                        'cv_batch_percent':row['cv_batch_percent'],'pareto':not dominated})
    with (REPORT/'tradeoff.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    recommendations={}
    for r in reports:
        limit=r['summary']['sage3_optimized']['mean_ms']
        valid=[(name,row) for name,row in r['summary'].items()
               if row['family']=='draft' and row['mean_ms']<=limit]
        best=min(valid,key=lambda item:item[1]['video_nmse']) if valid else None
        recommendations[r['sample']]={'sage3_latency_limit_ms':limit,
                                      'lowest_error_draft_within_sage3_latency':best}
    summary={'status':'complete','samples':SAMPLES,'arms':19,'complete_output_not_query_sampling':True,
             'aggregates':aggregate,'per_input_budget_choices':recommendations,
             'scope':'three calls only; selection on this set is exploratory, not held-out validation; no video-quality claim',
             'timing_warning':'three identical-model GPUs, one per sample; normalized paired ratios preferred over cross-GPU absolute averages'}
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for log in [False,True]:
        fig,axes=plt.subplots(1,3,figsize=(18,5.5))
        for ax,r in zip(axes,reports):
            points=r['summary']
            colors={.25:'#4477AA',.5:'#EE6677',1.:'#228833'}
            for fraction,color in colors.items():
                group=[(n,v) for n,v in points.items() if v['family']=='draft' and v['retained_int8_fraction']==fraction]
                group.sort(key=lambda item:item[1]['skip_fraction'])
                xs=[v['mean_ms'] for _,v in group]
                ys=[v['video_nmse']*100 for _,v in group]
                ax.plot(xs,ys,'o-',color=color,label=f'INT8 {fraction:.0%} of retained',lw=1,ms=5)
                for _,v in group:
                    ax.annotate(f"{v['skip_fraction']:.0%}",(v['mean_ms'],v['video_nmse']*100),
                                xytext=(4,4),textcoords='offset points',fontsize=8)
            for name,label,marker,color in [('sage2pp','Sage2++','s','#AA3377'),
                                             ('sage3_optimized','Sage3 optimized','D','#000000'),
                                             ('draft_skip70_i033','70/20/10','*','#CCBB44')]:
                p=points[name]
                ax.scatter(p['mean_ms'],p['video_nmse']*100,marker=marker,s=90,color=color,label=label,zorder=5)
            ax.axvline(points['bf16_sdpa']['mean_ms'],color='gray',linestyle='--',label='BF16 latency')
            ax.set_title(f"{r['case']} | step {r['step']} | block {r['layer']}\nN={r['shape_hnd'][2]:,}, H=56, D=128")
            ax.set_xlabel('Full attention API latency (ms)')
            ax.set_ylabel('Video-output NMSE (%)'+(' — log axis' if log else ''))
            if log:
                ax.set_yscale('log')
            ax.grid(True,alpha=.25)
        axes[0].legend(fontsize=8)
        fig.suptitle('H3 real-input attention tradeoffs — labels indicate skipped video-block fraction',fontsize=14)
        fig.tight_layout()
        fig.savefig(REPORT/('tradeoff_log.png' if log else 'tradeoff.png'),dpi=160)
        plt.close(fig)
    print(json.dumps(recommendations,indent=2),flush=True)


if __name__=='__main__':
    main()
