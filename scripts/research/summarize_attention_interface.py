#!/usr/bin/env python3
"""Evaluate frozen E006 resource-allocation gates, without selecting cases."""
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'results/research/E006_h3_attention_interface.json'
data=json.loads(source.read_text())
expected={(p,s,b) for p in (1,20) for s in (0,19) for b in (0,24,48)}
assert data['status']=='complete',data['status']
assert {(c['prompt_id'],c['step'],c['block']) for c in data['cases']}==expected
assert len(data['cases'])==12
rows=[]
for c in data['cases']:
 assert c['status']=='complete'
 assert c['bf16_zero_replay']['err2']==0
 for segment in c['segments']:
  assert all(v==0 for counts in segment['own_scale_byte_mismatches'].values() for v in counts.values())
  assert segment['fixed_correction_vs_T1_byte_mismatches']==0
 e=c['attention_factorial']['video']['energy']
 assert all(math.isfinite(v) for v in e.values())
 assert all(e[k]>0 for k in ('joint','additive','interaction','reference'))
 rows.append(dict(prompt_id=c['prompt_id'],step=c['step'],block=c['block'],
  interaction_over_joint=e['interaction']/e['joint'],
  joint_over_additive=e['joint']/e['additive'],
  fixed_over_joint=e['fixed_joint']/e['joint'],
  fixed_interaction_over_interaction=e['fixed_interaction']/e['interaction'],
  interaction_energy=e['interaction'],fixed_interaction_energy=e['fixed_interaction'],
  projection_nmse=e['projection']/e['reference'],attention_nmse=e['attention']/e['reference'],
  joint_nmse=e['joint']/e['reference'],fixed_joint_nmse=e['fixed_joint']/e['reference'],
  signed_additive_interaction_cross_over_joint=e['additive_interaction_cross']/e['joint']))
med=lambda key:statistics.median(r[key] for r in rows)
gates={
 'interaction_median_at_least_point1':med('interaction_over_joint')>=.1,
 'harm_median_at_least_1point1':med('joint_over_additive')>=1.1,
 'harm_at_least_8_of_12':sum(r['joint_over_additive']>1 for r in rows)>=8,
 'fixed_median_ratio_at_most_point9':med('fixed_over_joint')<=.9,
 'fixed_improves_at_least_8_of_12':sum(r['fixed_over_joint']<1 for r in rows)>=8,
 'fixed_interaction_energy_median_lower':med('fixed_interaction_energy')<med('interaction_energy')}
passed=all(gates.values())
summary={
 'experiment':'E006','status':'complete','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'primary':'video attention rows; each of 12 frozen cases once; padding excluded',
 'medians':{k:med(k) for k in rows[0] if k not in ('prompt_id','step','block')},
 'counts':{'joint_worse_than_additive':sum(r['joint_over_additive']>1 for r in rows),
           'fixed_better_than_joint':sum(r['fixed_over_joint']<1 for r in rows)},
 'gates':gates,'continue_to_fixed_block24':passed,
 'decision':'continue only to preregistered block24 four-case BF16 continuation' if passed else 'stop current harmful QKV-scale-switching route; no continuation or parameter search',
 'rows':rows,
 'limitations':['exploratory resource gates, not significance tests','correlated calibration cases, not heldout',
  'qkv projection only; other linear layers BF16','oracle frozen scales are not a deployment method','P scales remain dynamic','local output, not generated video quality']}
out=ROOT/'research_state/06_experiments/results/E006_summary.json'
out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
print('prompt step block I/joint joint/additive fixed/joint')
for r in rows:
 print(r['prompt_id'],r['step'],r['block'],*[round(r[k],4) for k in ('interaction_over_joint','joint_over_additive','fixed_over_joint')])
