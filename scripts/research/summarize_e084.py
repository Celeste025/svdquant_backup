#!/usr/bin/env python3
"""CPU verification and paired aggregation of E084, including existing RGB videos."""
import csv
import json
from pathlib import Path
import time
from itertools import zip_longest
import numpy as np
import torch
from generate_wan_qad_comparison import ROOT, save_json, sha256

RD = ROOT/'results/research/E084'
OLD = ROOT/'results/research/E022'


def main():
    import av
    start = time.time()
    torch.set_num_threads(4)
    run = json.loads((RD/'run.json').read_text())
    assert run['status'] == 'complete'
    max_relative = 0.0
    checks = 0

    def close(x, y):
        nonlocal max_relative, checks
        if x is None or y is None:
            assert x is y
            return
        diff = abs(x-y)/max(1, abs(x), abs(y))
        max_relative = max(max_relative, diff)
        checks += 1
        assert diff < 1e-11, (x, y)

    def arr(x):
        return x.double().numpy()

    def metric(x, r):
        e = x-r
        sse, ref2 = float(np.sum(e*e)), float(np.sum(r*r))
        return dict(sse=sse, ref2=ref2, nmse=sse/ref2)

    def cross(a, b):
        a2, b2, dot = float(np.sum(a*a)), float(np.sum(b*b)), float(np.sum(a*b))
        return dict(a2=a2, b2=b2, twice_dot=2*dot,
                    cosine=dot/(a2*b2)**0.5 if a2*b2 else None,
                    net_interaction_fraction=2*dot/(a2+b2) if a2+b2 else None)

    def load_trace(arm, tid):
        record = run['arms'][arm][tid]['trace']
        assert sha256(record['path']) == record['sha256']
        return torch.load(record['path'], map_location='cpu', weights_only=True)

    rows = []
    for tid in run['arms']['bf16']:
        teacher = load_trace('bf16', tid)
        for arm in ('plain', 'qad'):
            trace = load_trace(arm, tid)
            previous = None
            for s, (b, q) in enumerate(zip(teacher, trace)):
                old = run['arms'][arm][tid]['steps'][s]
                vb, vq, vs = arr(b['velocity']), arr(q['velocity']), arr(q['same_state_velocity'])
                xb, xq = arr(b['before']), arr(q['before'])
                yb, yq = arr(b['after']), arr(q['after'])
                metrics = dict(same_state=metric(vs, vb), rollout_velocity=metric(vq, vb),
                               latent_before=metric(xq, xb), latent_after=metric(yq, yb))
                for name, m in metrics.items():
                    for k, v in m.items(): close(v, old[name][k])
                a, drift = vs-vb, vq-vs
                dec = cross(a, drift)
                for k, v in dec.items(): close(v, old['prediction_decomposition'][k])
                close(metrics['rollout_velocity']['sse'], dec['a2']+dec['b2']+dec['twice_dot'])
                c, d = (1-old['t_next'])*(xq-xb), -(1-old['t_next'])*old['t']*(vq-vb)
                upd = cross(c, d)
                for k, v in upd.items(): close(v, old['update_decomposition'][k])
                close(metrics['latent_after']['sse'], upd['a2']+upd['b2']+upd['twice_dot'])
                closure = float(np.max(np.abs((yq-yb)-c-d)))
                close(closure, old['update_closure_maxabs'])
                if previous is not None:
                    ac = a-np.mean(a, axis=(2,3,4), keepdims=True)
                    pc = previous-np.mean(previous, axis=(2,3,4), keepdims=True)
                    for name, values in [('raw', cross(previous,a)), ('channel_centered',cross(pc,ac))]:
                        for k,v in values.items():close(v,old['adjacent_same_state_error'][name][k])
                previous = a
                rows.append(dict(arm=arm, trajectory=tid, prompt_id=run['arms'][arm][tid]['prompt_id'],
                                 step=s, timestep=old['timestep'], **metrics,
                                 prediction_decomposition=dec, update_decomposition=upd,
                                 adjacent=old.get('adjacent_same_state_error')))
        print('verified', tid, flush=True)

    def aggregate(rr, field):
        v = [r[field] for r in rr]
        return dict(pooled_nmse=sum(x['sse'] for x in v)/sum(x['ref2'] for x in v),
                    mean_nmse=float(np.mean([x['nmse'] for x in v])),
                    sse=sum(x['sse'] for x in v), ref2=sum(x['ref2'] for x in v), n=len(v))

    fields = ('same_state','rollout_velocity','latent_after')
    by_step = []
    for step in range(4):
        item = dict(step=step,timestep=run['schedule']['timesteps_bf16'][step], arms={})
        for arm in ('plain','qad'):
            rr = [r for r in rows if r['arm']==arm and r['step']==step]
            item['arms'][arm] = {f:aggregate(rr,f) for f in fields}
            for field in ('prediction_decomposition','update_decomposition'):
                sums = {k:sum(r[field][k] for r in rr) for k in ('a2','b2','twice_dot')}
                total = sums['a2']+sums['b2']+sums['twice_dot']
                item['arms'][arm][field] = sums | dict(cross_fraction_of_component_energy=
                    sums['twice_dot']/(sums['a2']+sums['b2']) if total else None)
            if step:
                item['arms'][arm]['adjacent_mean_cosine'] = {k:float(np.mean([r['adjacent'][k]['cosine'] for r in rr])) for k in ('raw','channel_centered')}
        item['qad_vs_plain'] = {}
        for f in fields:
            pp = {r['trajectory']:r[f]['nmse'] for r in rows if r['arm']=='plain' and r['step']==step}
            qq = {r['trajectory']:r[f]['nmse'] for r in rows if r['arm']=='qad' and r['step']==step}
            item['qad_vs_plain'][f] = dict(improved=sum(qq[k]<pp[k] for k in pp),
                pooled_reduction=1-item['arms']['qad'][f]['pooled_nmse']/item['arms']['plain'][f]['pooled_nmse'],
                mean_reduction=1-item['arms']['qad'][f]['mean_nmse']/item['arms']['plain'][f]['mean_nmse'])
        by_step.append(item)
    overall = {a:{f:aggregate([r for r in rows if r['arm']==a],f) for f in fields} for a in ('plain','qad')}
    dev = json.loads((OLD/'finalsummary.json').read_text())
    dev_rows = [v for v in dev['validation'] if v['arm']=='native' and v['optimizer_step'] in (0,64)]
    dev_steps = {}
    for v in dev_rows:
        dev_steps[str(v['optimizer_step'])] = []
        for s in range(4):
            rr = [r for r in v['rows'] if r['step']==s]
            dev_steps[str(v['optimizer_step'])].append(dict(step=s, pooled_nmse=sum(r['err2'] for r in rr)/sum(r['ref2'] for r in rr)))
    summary = dict(status='complete', run_sha256=sha256(RD/'run.json'), rows=rows,
                   by_step=by_step, overall=overall, development_by_step=dev_steps,
                   independent_verification=dict(checks=checks,max_relative=max_relative,cuda_initialized=torch.cuda.is_initialized()))
    save_json(RD/'timestep_summary.json',summary)
    with (RD/'per_trajectory_step.csv').open('w') as f:
        cols = ['arm','trajectory','prompt_id','step','timestep']+[x+'_nmse' for x in fields]
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in cols[:5]}|{x+'_nmse':r[x]['nmse'] for x in fields})

    baseline=json.loads((OLD/'video_baselines.json').read_text())['arms']
    selected=json.loads((OLD/'video_selected.json').read_text())['arms']['qad_native_dev_selected']['trajectories']
    endpoints=[]
    def frames(path):
        with av.open(path) as container:
            for frame in container.decode(video=0):yield frame.to_ndarray(format='rgb24')
    for tid,ref in baseline['bf16']['trajectories'].items():
        cases={'bf16':ref,'plain':baseline['plain_step0000']['trajectories'][tid],'qad':selected[tid]}
        for case in cases.values():
            for key in ('initial_latent_sha256','noise_sha256','embedding_sha256'):assert case[key]==ref[key]
            for key in ('video','final_latent'):assert sha256(case[key]['path'])==case[key]['sha256']
        tensors={k:arr(torch.load(v['final_latent']['path'],map_location='cpu',weights_only=True)) for k,v in cases.items()}
        row=dict(trajectory=tid,prompt_id=ref['prompt_id'],latent={},rgb={})
        for arm in ('plain','qad'):
            row['latent'][arm]=metric(tensors[arm],tensors['bf16'])
            target=next(r for r in rows if r['trajectory']==tid and r['arm']==arm and r['step']==3)['latent_after']
            for k,v in row['latent'][arm].items():close(v,target[k])
        totals={'ref2':0,'plain':0,'qad':0};n=0
        for triple in zip_longest(*(frames(cases[k]['video']['path']) for k in ('bf16','plain','qad'))):
            assert all(x is not None for x in triple)
            r,p,q=[x.astype(np.int32) for x in triple]
            assert r.shape==p.shape==q.shape==(480,832,3)
            totals['ref2']+=int(np.sum(r*r,dtype=np.int64))
            totals['plain']+=int(np.sum((p-r)**2,dtype=np.int64))
            totals['qad']+=int(np.sum((q-r)**2,dtype=np.int64));n+=1
        assert n==77
        for arm in ('plain','qad'):
            row['rgb'][arm]=dict(sse=totals[arm],ref2=totals['ref2'],nmse=totals[arm]/totals['ref2'])
        endpoints.append(row)
    endpoint_summary={}
    for field in ('latent','rgb'):
        val={}
        for arm in ('plain','qad'):
            vv=[r[field][arm] for r in endpoints]
            val[arm]=dict(pooled_nmse=sum(v['sse'] for v in vv)/sum(v['ref2'] for v in vv),mean_nmse=float(np.mean([v['nmse'] for v in vv])))
        val['improved']=sum(r[field]['qad']['nmse']<r[field]['plain']['nmse'] for r in endpoints)
        val['reduction']={k:1-val['qad'][k]/val['plain'][k] for k in ('pooled_nmse','mean_nmse')}
        endpoint_summary[field]=val
    save_json(RD/'endpoint_summary.json',dict(status='complete',rows=endpoints,summary=endpoint_summary,
              definition='sum((x-ref)^2)/sum(ref^2); RGB from saved MP4 all 77 frames, no alignment or crop',
              checks=checks,max_relative=max_relative,cuda_initialized=torch.cuda.is_initialized(),seconds=time.time()-start))

    print(json.dumps(dict(by_step=by_step,overall=overall,endpoints=endpoint_summary,dev=dev_steps),indent=2))


def plot():
    summary=json.loads((RD/'timestep_summary.json').read_text())
    by_step=summary['by_step']
    fields=('same_state','rollout_velocity','latent_after')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,field,title in zip(axes,fields,['Same BF16 input: DiT NMSE','Free trajectories: DiT NMSE','After each update: latent NMSE']):
        for arm,color in [('plain','#777777'),('qad','#2266bb')]:
            ax.plot(range(1,5),[s['arms'][arm][field]['pooled_nmse'] for s in by_step],marker='o',label=arm,color=color)
        ax.set(title=title,xlabel='Denoising step (1 to 4)',xticks=[1,2,3,4],ylabel='Pooled NMSE vs BF16')
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(RD/'timestep_nmse.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    import sys
    if '--plot-only' in sys.argv:plot()
    else:main()
