#!/usr/bin/env python3
"""E058: exact second-update arithmetic decomposition of saved H3 corners."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from summarize_h3_temporal_four_corner import load, sha, measures
from analyze_h3_adjacent_error_v2 import same

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E058'
PLAN = ROOT/'research_state/06_experiments/E058_h3_update_rounding_plan.md'
FLOW = Path('/home/wjq/workspace/DiffSynth-Studio/diffsynth/diffusion/flow_match.py')


def interaction(values):
    return values['QQ']-values['QB']-values['BQ']+values['BB']


def attribution(observed, model, rounding, centered):
    if centered:
        axes = tuple(range(2, observed.ndim))
        observed, model, rounding = [x-x.mean(axes,keepdim=True) for x in (observed, model, rounding)]
    error = float((observed-model-rounding).abs().max())
    assert error < 1e-12
    energies = [float(x.square().sum()) for x in (observed, model, rounding)]
    cross = 2*float((model*rounding).sum())
    assert abs(energies[0]-energies[1]-energies[2]-cross) < 1e-9*max(1,energies[0])
    return dict(actual_interaction_energy=energies[0], model_interaction_energy=energies[1],
                update_arithmetic_interaction_energy=energies[2], twice_model_dot_arithmetic=cross,
                model_energy_over_actual=energies[1]/energies[0],
                arithmetic_energy_over_actual=energies[2]/energies[0],
                signed_cross_over_actual=cross/energies[0],
                identity_max_abs=error)


def main():
    out=RD/'summary.json'
    if out.exists(): raise FileExistsError(out)
    started=time.monotonic();torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('original_h3_flow',FLOW)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    schedulers={}
    for modality,shift in [('video',12.),('audio',3.)]:
        sched=module.FlowMatchScheduler('MiniMax-H3');sched.set_timesteps(20,shift=shift)
        schedulers[modality]=sched
    source_path=ROOT/'results/research/E057/summary.json'
    source=json.loads(source_path.read_text());assert source['status']=='complete'
    rows=[];prepared_records=[]
    for row in source['sources']:
        payloads={k:load(rec) for k,rec in row['time_axis_corners'].items()}
        prepared={k:load(p['prepared_artifact']) for k,p in payloads.items()}
        prepared_records.extend(p['prepared_artifact'] for p in payloads.values())
        assert same(prepared['BB']['state'],prepared['BQ']['state'])
        assert same(prepared['QB']['state'],prepared['QQ']['state'])
        step=payloads['QB']['next_step'];modalities={}
        for modality,sched in schedulers.items():
            h=sched.sigmas[step+1]-sched.sigmas[step]
            ys,us,vs,rs={},{},{},{}
            for key,p in payloads.items():
                state=prepared[key]['state'][modality]
                z,v,y=state['latents_before'],p['velocities'][modality],p['next_endpoints'][modality]
                assert z.dtype==v.dtype==y.dtype==torch.bfloat16
                assert torch.equal(state['sigma'],sched.sigmas[step])
                assert torch.equal(state['timestep'],sched.timesteps[step])
                replay=sched.step(v,state['timestep'],z)
                assert torch.equal(replay,y), (row['case_id'],row['source_arm'],key,modality)
                assert torch.isfinite(z).all() and torch.isfinite(v).all() and torch.isfinite(y).all()
                ys[key]=y.double();vs[key]=v.double()
                us[key]=z.double()+h.double()*vs[key];rs[key]=ys[key]-us[key]
            observed=interaction(ys);model=h.double()*interaction(vs);rounding=interaction(rs)
            assert float((interaction(us)-model).abs().max())<1e-12
            modalities[modality]=dict(step=step,h_fp32=float(h),bf16_cpu_replay_exact=True,
                actual={k:measures(ys,c) for k,c in [('raw',False),('channel_centered',True)]},
                exact_arithmetic_fixed_inputs={k:measures(us,c) for k,c in [('raw',False),('channel_centered',True)]},
                interaction_attribution={k:attribution(observed,model,rounding,c)
                                         for k,c in [('raw',False),('channel_centered',True)]})
        rows.append(dict(case_id=row['case_id'],source_arm=row['source_arm'],modalities=modalities))
    assert not torch.cuda.is_initialized()
    result=dict(experiment='E058',status='complete',seconds=time.monotonic()-started,
                cuda_initialized=False,new_model_forwards=0,cases=rows,
                plan_sha256=sha(PLAN),script_sha256=sha(__file__),scheduler_sha256=sha(FLOW),
                e057_summary_sha256=sha(source_path),prepared_artifacts=prepared_records,
                limitation='Second-update arithmetic only; first-update rounded inputs and model predictions fixed')
    RD.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',seconds=result['seconds'],rows=[
        dict(case=r['case_id'],arm=r['source_arm'],modality=m,
             interaction_ratio_actual=v['actual']['raw']['interaction_norm_over_additive_norm'],
             interaction_ratio_exact=v['exact_arithmetic_fixed_inputs']['raw']['interaction_norm_over_additive_norm'],
             attribution=v['interaction_attribution']['raw']) for r in rows for m,v in r['modalities'].items()]),indent=2))


if __name__=='__main__':main()
