#!/usr/bin/env python3
"""Small, dependency-free summary for the targeted timestep smoothing probe."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from statistics import mean

p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--config', type=Path)
a = p.parse_args()
rows = list(csv.DictReader(a.input.open()))
for r in rows:
    for k in ('local_alpha','local_validation_nmse','global_validation_nmse','validation_relative_gain','smooth_log_ratio_median_abs','smooth_log_ratio_p90_abs','smooth_cosine_to_global'):
        r[k] = float(r[k])
by = defaultdict(list)
for r in rows: by[r['layer']].append(r)
summary = []
for layer, x in by.items():
    local, global_ = sum(r['local_validation_nmse'] for r in x), sum(r['global_validation_nmse'] for r in x)
    summary.append({'layer': layer, 'timesteps': len(x), 'distinct_local_alphas': sorted({r['local_alpha'] for r in x}),
                    'local_vs_global_pooled_relative_gain': (global_ - local) / global_,
                    'mean_per_timestep_relative_gain': mean(r['validation_relative_gain'] for r in x),
                    'median_abs_log_s_ratio_mean': mean(r['smooth_log_ratio_median_abs'] for r in x),
                    'p90_abs_log_s_ratio_mean': mean(r['smooth_log_ratio_p90_abs'] for r in x),
                    'cosine_to_global_min': min(r['smooth_cosine_to_global'] for r in x)})
a.output.parent.mkdir(parents=True, exist_ok=True)
(a.output.with_suffix('.json')).write_text(json.dumps({'summary': summary, 'records': rows}, indent=2))
with a.output.with_suffix('.md').open('w') as f:
    f.write('# E087 timestep-aware smoothing probe\n\n')
    if a.config:
        cfg = json.loads(a.config.read_text())
        f.write(f"Exploratory configuration: {len(cfg['fit_prompt_ids'])} fitting prompts and {len(cfg['validation_prompt_ids'])} disjoint validation prompts; {len(cfg['capture_steps'])} timesteps; {len(cfg['probe_layers'])} selected linears; smoothing grid={cfg.get('smoothing_grid_size', 10)}. Objective: parent DiT block endpoint NMSE for a single W4A4+smoothing candidate.\n\n")
        f.write(f"Scope note: {cfg.get('study_note', '')}\n\n")
    else:
        f.write('Objective: parent DiT block endpoint NMSE for a single W4A4+smoothing candidate.\n\n')
    f.write('|layer|distinct best α|local vs global gain|mean |log(s_local/s_global)||min cosine|\n|---|---|---:|---:|---:|\n')
    for s in summary:
        f.write(f"|{s['layer']}|{s['distinct_local_alphas']}|{s['local_vs_global_pooled_relative_gain']:.3%}|{s['median_abs_log_s_ratio_mean']:.3f}|{s['cosine_to_global_min']:.5f}|\n")
