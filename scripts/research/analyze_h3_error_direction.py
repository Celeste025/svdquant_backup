#!/usr/bin/env python3
"""E067: frozen saved-error geometry, CPU only; no causal or quality claim."""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import signal
import statistics
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
import torch
import check_h3_swiglu_interaction as io

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'results/research/E067/analysis.json'
PLAN = ROOT/'research_state/06_experiments/E067_h3_error_direction_plan.md'
KINDS = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')
DOMAINS = ('video', 'joint')
MODES = ('raw', 'centered')
SHAPES = ('full_m', 'm512')


def record(path):
    return dict(file=str(Path(path).resolve()), bytes=Path(path).stat().st_size, sha256=io.sha(path))


def guard(started):
    assert time.monotonic()-started < 600, 'CPU wall budget exceeded'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 8*1024**3, 'CPU RSS budget exceeded'
    assert not torch.cuda.is_initialized()


def close(a, b):
    difference = abs(a-b)/max(abs(a), abs(b), 1e-300)
    assert difference < 1e-10, (a, b, difference)
    return difference


def energy(value):
    return float(np.einsum('ij,ij->', value, value, dtype=np.float64, optimize=False))


def geometry(er, ec):
    assert er.shape == ec.shape and er.ndim == 2
    assert np.isfinite(er).all() and np.isfinite(ec).all()
    rr, cc = energy(er), energy(ec)
    dot = float(np.einsum('ij,ij->', er, ec, dtype=np.float64, optimize=False))
    delta = ec-er
    dd = energy(delta)
    result = dict(restart_sse=rr, carry_sse=cc, dot=dot, delta_sse=dd,
        carry_over_restart=cc/rr if rr else None,
        cosine=dot/(rr*cc)**.5 if rr and cc else None,
        delta_over_restart=dd/rr if rr else None, elements=er.size)
    close(dd, rr+cc-2*dot)
    if rr == 0:
        result.update(rho=None, orthogonal_sse=None, radial_delta_sse=None,
            orthogonal_over_carry=None, orthogonal_over_delta=None, rho_is_shrink=None,
            closure_relative_error=None, orthogonality_over_restart=None)
        return result
    rho = dot/rr
    orthogonal = ec-rho*er
    oo = energy(orthogonal)
    radial = (rho-1)**2*rr
    residual_dot = float(np.einsum('ij,ij->', orthogonal, er, dtype=np.float64, optimize=False))
    closure = max(close(cc, rho*rho*rr+oo), close(dd, radial+oo))
    assert abs(residual_dot)/rr < 1e-10
    result.update(rho=rho, orthogonal_sse=oo, radial_delta_sse=radial,
        orthogonal_over_carry=oo/cc if cc else None,
        orthogonal_over_delta=oo/dd if dd else None,
        rho_is_shrink=bool(0 <= rho <= 1), closure_relative_error=closure,
        orthogonality_over_restart=residual_dot/rr)
    return result


def describe(values):
    finite = [value for value in values if value is not None]
    return dict(count=len(values), defined=len(finite), minimum=min(finite) if finite else None,
                median=statistics.median(finite) if finite else None, maximum=max(finite) if finite else None)


def aggregate(layers):
    summary = {}
    for kind in ('all', *KINDS):
        rows = [row for row in layers if kind == 'all' or row['layer'].endswith(kind)]
        assert rows
        groups = {}
        for shape in SHAPES:
            groups[shape] = {}
            for mode in MODES:
                groups[shape][mode] = {}
                for domain in DOMAINS:
                    metrics = [row[shape][mode][domain] for row in rows]
                    groups[shape][mode][domain] = dict(
                        **{field: describe([row[field] for row in metrics]) for field in
                           ('cosine', 'rho', 'carry_over_restart', 'delta_over_restart',
                            'orthogonal_over_carry', 'orthogonal_over_delta')},
                        shrinking_rho_count=sum(row['rho_is_shrink'] is True for row in metrics),
                        orthogonal_majority_of_change_count=sum(row['orthogonal_over_delta'] is not None and
                            row['orthogonal_over_delta'] > .5 for row in metrics))
        groups['shape_control'] = {mode: {domain: describe([
            row['shape_control'][mode][domain]['difference_over_full_delta'] for row in rows])
            for domain in DOMAINS} for mode in MODES}
        summary[kind] = groups
    return summary


def main():
    assert not DEST.exists()
    started = time.monotonic()
    torch.set_num_threads(2)
    report = dict(experiment='E067', status='running', sources={}, files={}, cases=[],
                  full_model_cases=[], new_model_forwards=0, cuda_initialized=False)
    def timeout(_signum, _frame):
        raise TimeoutError('E067 600-second CPU deadline reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        sources = dict(plan=PLAN, runner=Path(__file__), io_helper=Path(io.__file__),
            local_evaluation=ROOT/'results/research/E066/evaluate.json',
            local_independent=ROOT/'results/research/E066/independent.json',
            full_evaluation=ROOT/'results/research/E065b/evaluate.json',
            full_independent=ROOT/'results/research/E065b/independent_outputs.json')
        report['sources'] = {name: record(path) for name, path in sources.items()}
        read = lambda name: json.loads(sources[name].read_text())
        local, checked, full, full_checked = [read(name) for name in
            ('local_evaluation', 'local_independent', 'full_evaluation', 'full_independent')]
        assert all(value['status'] == 'complete' for value in (local, checked, full, full_checked))
        assert checked['evaluation_sha256'] == report['sources']['local_evaluation']['sha256']
        assert full_checked['evaluation_sha256'] == report['sources']['full_evaluation']['sha256']
        assert local['source_evaluation'] == report['sources']['full_evaluation']
        assert local['source_checks']['outputs'] == report['sources']['full_independent']
        assert checked['cuda_initialized'] is full_checked['cuda_initialized'] is False
        verified = {row['key']: row for row in checked['cases']}
        for case in local['cases']:
            layers = []
            checked_layers = {row['layer']: row for row in verified[case['key']]['layers']}
            for name, row in case['layers'].items():
                guard(started)
                payload = io.load(row['artifact'], report['files'])
                assert payload['layer'] == name and payload['case_id'] == case['case_id']
                selected = payload['sample_indices'].numpy()
                masks = {domain: np.flatnonzero(np.isin(selected, payload['modality_indices'][domain].numpy()))
                         for domain in DOMAINS}
                target = io.array(payload['target'])
                tensors = dict(full_m=payload['outputs'], m512=payload['subset_outputs'])
                errors = {shape: {arm: io.array(tensors[shape][arm])-target for arm in ('restart', 'carry')}
                          for shape in SHAPES}
                item = dict(layer=name, sample_counts={domain: len(mask) for domain, mask in masks.items()},
                            full_m={}, m512={}, shape_control={})
                for mode in MODES:
                    for shape in SHAPES:
                        item[shape][mode] = {}
                    item['shape_control'][mode] = {}
                    for domain, mask in masks.items():
                        arrays = {shape: {arm: values[mask] for arm, values in errors[shape].items()}
                                  for shape in SHAPES}
                        if mode == 'centered':
                            arrays = {shape: {arm: value-value.mean(axis=0, keepdims=True)
                                     for arm, value in values.items()} for shape, values in arrays.items()}
                        deltas = {}
                        for shape in SHAPES:
                            er, ec = arrays[shape]['restart'], arrays[shape]['carry']
                            stats = geometry(er, ec)
                            item[shape][mode][domain] = stats
                            deltas[shape] = ec-er
                            if mode == 'raw':
                                expected = checked_layers[name]['sample' if shape == 'full_m' else 'subset'][domain]
                                for field in ('restart_sse', 'carry_sse'):
                                    close(stats[field], expected[field])
                        difference = energy(deltas['full_m']-deltas['m512'])
                        denominator = item['full_m'][mode][domain]['delta_sse']
                        item['shape_control'][mode][domain] = dict(difference_sse=difference,
                            full_delta_sse=denominator, difference_over_full_delta=difference/denominator if denominator else None)
                        del arrays, deltas, er, ec
                layers.append(item)
                del payload, tensors, target, errors
                if len(layers) % 50 == 0:
                    print(case['key'], len(layers), '/200', flush=True)
            report['cases'].append(dict(key=case['key'], layers=layers, summary=aggregate(layers)))
        index = {(row['case_id'], row['position'], row['arm']): row for row in full['cases']}
        stats_index = {row['case_id']+'/'+row['position']: row for row in full_checked['cases']}
        for case in local['cases']:
            guard(started)
            k = case['key']
            teacher_rec = full['references'][k]['bf16']
            teacher = io.load(teacher_rec['artifact'], report['files'])
            records = {arm: index[(case['case_id'], case['position'], arm)] for arm in ('restart', 'carry')}
            values = {arm: io.load(row['artifact'], report['files']) for arm, row in records.items()}
            assert io.byte_equal(values['restart']['actual_dit_inputs'], values['carry']['actual_dit_inputs'])
            for arm in ('restart', 'carry'):
                assert io.signature(values[arm]['velocities']) == records[arm]['velocities']
            assert io.signature(teacher['velocities']) == teacher_rec['velocities']
            row = dict(key=k, raw={}, centered={})
            for modality in ('video', 'audio'):
                target = io.array(teacher['velocities'][modality])
                errors = {}
                for arm in ('restart', 'carry'):
                    error = io.array(values[arm]['velocities'][modality])-target
                    b, c = error.shape[:2]
                    errors[arm] = error.reshape(b, c, -1).transpose(2, 0, 1).reshape(-1, b*c)
                for mode in MODES:
                    arrays = errors if mode == 'raw' else {arm: value-value.mean(axis=0, keepdims=True)
                                                          for arm, value in errors.items()}
                    stats = geometry(arrays['restart'], arrays['carry'])
                    row[mode][modality] = stats
                    field = 'sse' if mode == 'raw' else 'channel_centered_sse'
                    for arm in ('restart', 'carry'):
                        close(stats[arm+'_sse'], stats_index[k]['stats'][arm][modality][field])
            report['full_model_cases'].append(row)
        guard(started)
        assert len(report['cases']) == len(report['full_model_cases']) == 4
        report.update(status='complete', seconds=time.monotonic()-started,
            verified_pt_files=len(report['files']), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            scope='Geometry of saved teacher-input local samples and complete saved DiT velocities; no native or model execution',
            limitations=['New local direction statistics cover only the saved 512 joint samples, not all tokens.',
                'Direction changes, if present, do not establish the cause of full-model error reversal.',
                'Pure radial local shrinkage can also reduce cancellation after composition.',
                'No video quality, independent generalization, NVFP4-specific mechanism, or novelty claim.'])
    except Exception as exc:
        report.update(status='failed_stop', seconds=time.monotonic()-started, error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, indent=2, allow_nan=False)+'\n'
        assert len(text.encode()) <= 20*1024**2
        with DEST.open('x') as stream:
            stream.write(text)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('files', 'cases', 'sources', 'full_model_cases')}, indent=2))


if __name__ == '__main__':
    main()
