#!/usr/bin/env python3
"""Independent CPU/NumPy checks of E081 metrics and the 2x2 decomposition."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import verify_e080_pv as independent


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/research/E081/probe.json'
OLD_SOURCE = ROOT / 'results/research/E080/v2/pv_probe_v2.json'
DEST = ROOT / 'results/research/E081/independent_verification.json'
ARMS = ('base', 'center32', 'center128', 'residual32', 'residual128', 'massfix32', 'massfix128')
REFERENCES = ('vs_pv_oracle', 'vs_full_attention')
GROUPS = (32, 128)


def energy(value):
    return float(np.sum(value * value, dtype=np.float64))


def contrast(value):
    return value[:, 1::2] - value[:, 0::2]


def dot(x, y):
    return float(np.sum(x * y, dtype=np.float64))


def attribution(base, residual, massfix, center, reference):
    e = base - reference
    dv = residual - base
    dm = massfix - base
    closure = center - base - dv - dm
    terms = dict(base_sse=energy(e), v_change_energy=energy(dv), mass_change_energy=energy(dm),
                 twice_base_error_dot_v=2. * dot(e, dv), twice_base_error_dot_mass=2. * dot(e, dm),
                 twice_v_dot_mass=2. * dot(dv, dm))
    expanded = sum(terms.values())
    rounding = 2. * dot(e + dv + dm, closure) + energy(closure)
    actual = energy(center - reference)
    assert np.isclose(expanded + rounding, actual, rtol=2e-12, atol=1e-8)
    return dict(**terms, sum_without_rounding_residual=expanded,
                rounding_residual_energy=energy(closure), rounding_contribution_to_center_sse=rounding,
                reconstructed_center_sse=expanded + rounding, actual_center_sse=actual,
                note='SSE effects include the explicit 2<dV,dM> interaction and are not additive percentages.')


def classify(arms, group, reference):
    base = arms['base'][reference]['sse']
    center = arms[f'center{group}'][reference]['sse']
    residual = arms[f'residual{group}'][reference]['sse']
    massfix = arms[f'massfix{group}'][reference]['sse']
    gain = base - center
    positive = gain > 0.
    mass_fraction = (base - massfix) / gain if positive else None
    removal_fraction = (residual - center) / gain if positive else None
    if not positive:
        label = 'center_not_beneficial_here'
    elif mass_fraction >= .8:
        label = 'mass_correction_explains_most_local_gain'
    elif removal_fraction >= .2:
        label = 'both_factors_warrant_attention'
    else:
        label = 'v_representation_dominant'
    return dict(group=group, reference=reference, center_positive=positive,
                output_sse_change_pct={name: 100. * (arms[name][reference]['sse'] / base - 1.)
                                       for name in (f'center{group}', f'residual{group}', f'massfix{group}')},
                center_absolute_sse_improvement=gain, massfix_fraction_of_center_gain=mass_fraction,
                removing_massfix_fraction_of_center_gain=removal_fraction,
                massfix_at_least_80pct_center_gain=bool(positive and mass_fraction >= .8),
                removing_massfix_loses_at_least_20pct_center_gain=bool(positive and removal_fraction >= .2),
                classification=label)


def main():
    start = time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not DEST.exists(), 'Preserve executed verification receipts'
    torch.set_num_threads(2)
    producer = json.loads(SOURCE.read_text())
    old_report = json.loads(OLD_SOURCE.read_text())
    assert producer['status'] == old_report['status'] == 'complete' and len(producer['rows']) == 4
    source_records = {key: independent.record(row['file']) for key, row in producer['sources'].items()}
    assert source_records == producer['sources']
    errors = dict(comparisons=0, max_absolute_difference=0., max_normalized_difference=0.)
    report = dict(experiment='E081', status='running', source_report=independent.record(SOURCE),
                  E080_report=independent.record(OLD_SOURCE), verified_executed_sources=source_records,
                  independent_sources={Path(path).name: independent.record(path)
                                       for path in (__file__, independent.__file__)},
                  environment=dict(python=sys.executable, numpy=np.__version__, torch=torch.__version__,
                                   CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),
                  method='CPU NumPy FP64 arithmetic; torch only loads saved CPU tensors', rows=[])
    for row in producer['rows']:
        output = independent.record(row['outputs']['file'])
        assert output == row['outputs']
        data = torch.load(output['file'], map_location='cpu', weights_only=True)
        old_row = next(x for x in old_report['rows'] if (x['case_id'], x['block']) == (row['case_id'], row['block']))
        old_output = independent.record(old_row['outputs']['file'])
        assert old_output == old_row['outputs']
        old = torch.load(old_output['file'], map_location='cpu', weights_only=True)
        assert data['query_indices'].tolist() == row['query_indices'] == old['query_indices'].tolist()
        exact = independent.numpy_tensor(data['exact_attention'])
        native = independent.numpy_tensor(data['native_capture'])
        for name in ('exact_attention', 'native_capture'):
            assert np.array_equal(independent.numpy_tensor(data[name]).view(np.uint8),
                                  independent.numpy_tensor(old[name]).view(np.uint8))
        checked = dict(case_id=row['case_id'], block=row['block'], step=row['step'],
                       verified_outputs=output, verified_E080_outputs=old_output, conditions=[])
        for previous in row['conditions']:
            condition = previous['condition']
            values = data[condition]
            oracle = independent.numpy_tensor(values['oracle'])
            assert np.array_equal(oracle.view(np.uint8), independent.numpy_tensor(old[condition]['oracle']).view(np.uint8))
            item = dict(condition=condition, oracle_vs_full=independent.metrics(oracle, exact), arms={},
                        E080_replay_exact={}, groups=[])
            independent.compare(item['oracle_vs_full'], previous['oracle_vs_full'], errors)
            for arm in ARMS:
                value = independent.numpy_tensor(values[arm])
                item['arms'][arm] = dict(vs_pv_oracle=independent.metrics(value, oracle),
                                          vs_full_attention=independent.metrics(value, exact))
                for reference in REFERENCES:
                    independent.compare(item['arms'][arm][reference], previous['arms'][arm][reference], errors)
                if arm in ('base', 'center32', 'center128'):
                    item['E080_replay_exact'][arm] = bool(np.array_equal(value.view(np.uint8),
                                                                        independent.numpy_tensor(old[condition][arm]).view(np.uint8)))
                    assert item['E080_replay_exact'][arm]
            simulated = independent.bf16_round_numpy(independent.numpy_tensor(values['base']))
            item['base_simulator_vs_native'] = independent.metrics(simulated, native)
            item['native_vs_full'] = independent.metrics(native, exact)
            for name in ('base_simulator_vs_native', 'native_vs_full'):
                independent.compare(item[name], previous[name], errors)
            item['native_gap_energy_ratio'] = item['base_simulator_vs_native']['sse'] / max(item['native_vs_full']['sse'], 1e-30)
            independent.compare({'native_gap_energy_ratio': item['native_gap_energy_ratio']},
                                {'native_gap_energy_ratio': previous['native_gap_energy_ratio']}, errors)
            for group in GROUPS:
                raw = {name: independent.numpy_tensor(values[name]) for name in
                       ('base', f'residual{group}', f'massfix{group}', f'center{group}', f'correction{group}')}
                b, residual, massfix, center, correction = [raw[name].astype(np.float64) for name in
                       ('base', f'residual{group}', f'massfix{group}', f'center{group}', f'correction{group}')]
                dv, dm = residual - b, massfix - b
                closure = energy(center - b - dv - dm) / max(energy(center - b), 1e-30)
                # Match producer's FP32 delta for checking its recorded scalar;
                # also measure the independent FP64 subtraction of saved values.
                delta32 = (raw[f'center{group}'] - raw[f'residual{group}']).astype(np.float64)
                correction_gap = energy(delta32 - correction) / max(energy(correction), 1e-30)
                independent.compare({'closure': closure}, {'closure': previous['factorial_closure_relative_energy'][str(group)]}, errors)
                independent.compare({'correction_gap': correction_gap},
                                    {'correction_gap': previous['correction_identity_relative_energy'][str(group)]}, errors)
                assert closure < 1e-8 and correction_gap < 1e-8
                group_record = dict(group=group, factorial_closure_relative_energy=closure,
                                    correction_identity_relative_energy=correction_gap,
                                    correction_identity_fp64_subtraction_relative_energy=energy(center-residual-correction)/max(energy(correction),1e-30),
                                    massfix_identity_relative_energy=energy(dm-correction)/max(energy(correction),1e-30),
                                    attributions={}, classifications={})
                for reference_name, reference_array in (('vs_pv_oracle', oracle), ('vs_full_attention', exact)):
                    reference64 = reference_array.astype(np.float64)
                    group_record['attributions'][reference_name] = dict(
                        output=attribution(b, residual, massfix, center, reference64),
                        neighbor_difference=attribution(contrast(b), contrast(residual), contrast(massfix), contrast(center), contrast(reference64)))
                    group_record['classifications'][reference_name] = classify(item['arms'], group, reference_name)
                item['groups'].append(group_record)
            checked['conditions'].append(item)
        report['rows'].append(checked)
    primary = []
    for condition in ('native_qk', 'exact_qk'):
        for group in GROUPS:
            for reference in REFERENCES:
                cells = []
                for row in report['rows']:
                    if row['block'] != 24:
                        continue
                    entry = next(c for c in row['conditions'] if c['condition'] == condition)
                    classified = next(g for g in entry['groups'] if g['group'] == group)['classifications'][reference]
                    cells.append(dict(case_id=row['case_id'], **classified))
                assert len(cells) == 2
                if not all(c['center_positive'] for c in cells):
                    label = 'center_not_consistently_beneficial'
                elif all(c['massfix_at_least_80pct_center_gain'] for c in cells):
                    label = 'mass_correction_explains_most_local_gain'
                elif all(c['removing_massfix_loses_at_least_20pct_center_gain'] for c in cells):
                    label = 'both_factors_warrant_attention'
                elif not any(c['removing_massfix_loses_at_least_20pct_center_gain'] for c in cells):
                    label = 'v_representation_dominant'
                else:
                    label = 'seed_dependent_do_not_force_single_mechanism'
                primary.append(dict(condition=condition, group=group, reference=reference, classification=label, cells=cells))
    assert not torch.cuda.is_initialized()
    report.update(status='complete', primary_block24_classification=primary, numeric_verification=errors,
                  cuda_initialized=False, seconds=time.perf_counter()-start)
    DEST.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='complete', verification=errors, seconds=report['seconds'], output=str(DEST))))
    for entry in primary:
        print(json.dumps(dict(condition=entry['condition'], group=entry['group'], reference=entry['reference'],
                              classification=entry['classification'],
                              massfix_gain_fractions=[c['massfix_fraction_of_center_gain'] for c in entry['cells']],
                              removal_gain_fractions=[c['removing_massfix_fraction_of_center_gain'] for c in entry['cells']])))


if __name__ == '__main__':
    main()
