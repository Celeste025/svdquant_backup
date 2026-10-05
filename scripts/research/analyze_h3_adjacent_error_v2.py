#!/usr/bin/env python3
"""E056: CPU-only adjacent teacher-state native H3 errors; no quality claim."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'research_state/06_experiments/E056_h3_adjacent_error_plan.md'
MANIFEST = ROOT/'research_state/06_experiments/E015_h3_crossmodal_propagation_manifest.json'
ARMS = ('bf16', 'plain', 'svd')


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_record(rec):
    path = Path(rec['file'])
    assert path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256'], path
    return torch.load(path, map_location='cpu', weights_only=False)


def project(error, reference):
    # Preserve channel axes; video reduces T,H,W; stereo audio only time.
    axes = tuple(range(2, error.ndim))
    centered = error - error.mean(axes, keepdim=True)
    ref = reference - reference.mean(axes, keepdim=True)
    denom = ref.square().sum(axes, keepdim=True)
    alpha = (centered * ref).sum(axes, keepdim=True) / denom.clamp_min(1e-100)
    residual = centered - alpha * ref
    return {'raw': error, 'channel_centered': centered, 'output_proportional_removed': residual}


def pair_stats(a, b):
    energy_a, energy_b = a.square().sum(), b.square().sum()
    cosine = (a*b).sum() / (energy_a * energy_b).sqrt().clamp_min(1e-100)
    axes = tuple(range(2, a.ndim))
    per_channel = ((a*b).sum(axes) /
                   (a.square().sum(axes)*b.square().sum(axes)).sqrt().clamp_min(1e-100))
    return dict(cosine=float(cosine), mse=[float(a.square().mean()), float(b.square().mean())],
                per_channel_cosine=per_channel.tolist(),
                equal_weight_sum_energy_over_diagonal=float((a+b).square().sum()/(energy_a+energy_b)),
                equal_weight_sum_is_not_solver_endpoint=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest = json.loads(MANIFEST.read_text())
    reports = {a: json.loads((ROOT/f'results/research/E015/evaluate_{a}.json').read_text()) for a in ARMS}
    assert all(r['status'] == 'complete' for r in reports.values())
    assert manifest['settings']['cfg_scale'] == 1.0
    result = dict(experiment='E056', status='running', model='MiniMax-H3 pruned',
                  settings=manifest['settings'], script_sha256=sha(__file__), plan_sha256=sha(PLAN),
                  source_manifest_sha256=sha(MANIFEST), cases=[], sources=[])
    for case in manifest['cases']:
        steps = []
        for index in (0, 1):
            payloads = {}
            for arm in ARMS:
                rec = case['e014_outputs'][arm] if index == 0 else next(
                    c['artifact'] for c in reports[arm]['cases']
                    if c['id'] == case['id'] and c['corner'] == 'BB')
                payload = load_record(rec)
                assert payload['arm'] == arm
                if index == 1:
                    assert payload['corner'] == 'BB' and payload['next_step'] == case['next_step']
                payloads[arm] = payload
                result['sources'].append(rec)
            assert all(same(payloads[a]['actual_dit_inputs'], payloads['bf16']['actual_dit_inputs']) for a in ARMS)
            steps.append(payloads)
        row = dict(case_id=case['id'], steps=[case['source_step'], case['next_step']], arms={})
        for arm in ('plain', 'svd'):
            modalities = {}
            for modality, shape in [('video', (1,24,37,36,64)), ('audio', (2,32,207))]:
                projected = []
                for payloads in steps:
                    ref, quant = (payloads[a]['velocities'][modality] for a in ('bf16', arm))
                    assert ref.shape == quant.shape == shape
                    assert ref.dtype == quant.dtype == torch.bfloat16
                    assert torch.isfinite(ref).all() and torch.isfinite(quant).all()
                    projected.append(project(quant.double()-ref.double(), ref.double()))
                stats = {key: pair_stats(projected[0][key], projected[1][key]) for key in projected[0]}
                raw_mse = stats['raw']['mse']
                for value in stats.values():
                    value['retained_energy_fraction'] = [x/y for x,y in zip(value['mse'], raw_mse)]
                modalities[modality] = stats
            row['arms'][arm] = modalities
        result['cases'].append(row)
    assert not torch.cuda.is_initialized()
    result.update(status='complete', seconds=time.monotonic()-started,
                  cuda_initialized=False, new_model_forwards=0,
                  limitation='Two adjacent teacher windows, not propagated errors or population covariance')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'seconds': result['seconds'], 'rows': [
        {'case': row['case_id'], 'arm': arm, 'modality': modality,
         'cosine': {k:v['cosine'] for k,v in stats.items()},
         'remaining_energy': stats['output_proportional_removed']['retained_energy_fraction']}
        for row in result['cases'] for arm, mods in row['arms'].items() for modality,stats in mods.items()]}, indent=2))


if __name__ == '__main__':
    main()
