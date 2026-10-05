#!/usr/bin/env python3
"""Independent CPU/NumPy readout of frozen E080 PV outputs; no model calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/research/E080/v2/pv_probe_v2.json'
DEST = ROOT / 'results/research/E080/v2/independent_verification.json'
ARMS = ('base', 'center32', 'center128', 'p_only', 'pair', 'random_pair', 'shuffle')
CONTROLS = ('center32', 'center128', 'p_only', 'random_pair', 'shuffle')


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'file': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def numpy_tensor(tensor):
    assert isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
    assert tensor.dtype == torch.float32 and tuple(tensor.shape) == (56, 72, 128)
    out = tensor.numpy()
    assert np.isfinite(out).all()
    return out


def metrics(value, reference):
    value = np.asarray(value, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = value - reference
    contrast = np.diff(value.reshape(56, 36, 2, 128), axis=2).reshape(56, 36, 128)
    reference_contrast = np.diff(reference.reshape(56, 36, 2, 128), axis=2).reshape(56, 36, 128)
    error_contrast = contrast - reference_contrast
    reference_energy = np.sum(reference_contrast * reference_contrast, dtype=np.float64)
    dot = np.sum(contrast * reference_contrast, dtype=np.float64)
    return {
        'sse': float(np.sum(error * error, dtype=np.float64)),
        'reference_energy': float(np.sum(reference * reference, dtype=np.float64)),
        'contrast_sse': float(np.sum(error_contrast * error_contrast, dtype=np.float64)),
        'contrast_reference_energy': float(reference_energy),
        'contrast_gain': float(dot / reference_energy),
        'contrast_cosine': float(dot / max(np.sqrt(np.sum(contrast * contrast, dtype=np.float64))
                                         * np.sqrt(reference_energy), 1e-30)),
    }


def bf16_round_numpy(value):
    """Finite FP32 -> BF16 nearest-even -> FP32, independently via IEEE bits."""
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xffff0000)).view(np.float32)


def compare(actual, expected, errors):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        reference = expected[name]
        absolute = abs(value - reference)
        normalized = absolute / max(abs(reference), 1.)
        errors['comparisons'] += 1
        errors['max_absolute_difference'] = max(errors['max_absolute_difference'], absolute)
        errors['max_normalized_difference'] = max(errors['max_normalized_difference'], normalized)
        assert np.isclose(value, reference, rtol=2e-12, atol=1e-9), (name, value, reference)


def cell_gate(block, arms):
    scores = {name: arms[name]['vs_pv_oracle']['contrast_sse'] for name in ARMS}
    pair = arms['pair']['vs_pv_oracle']
    base = arms['base']['vs_pv_oracle']
    best = min(CONTROLS, key=scores.get)
    result = {
        'pair_contrast_change_pct': 100. * (scores['pair'] / scores['base'] - 1.),
        'pair_output_sse_change_pct': 100. * (pair['sse'] / base['sse'] - 1.),
        'best_control': best,
        'best_control_contrast_change_pct': 100. * (scores[best] / scores['base'] - 1.),
        'pair_vs_best_control_change_pct': 100. * (scores['pair'] / scores[best] - 1.),
        'output_sse_increase_at_most_5pct': bool(pair['sse'] <= 1.05 * base['sse']),
    }
    if block == 24:
        result.update(primary_contrast_improvement_at_least_10pct=bool(scores['pair'] <= .90 * scores['base']),
                      primary_beats_best_control_at_least_5pct=bool(scores['pair'] <= .95 * scores[best]))
        result['cell_pass'] = bool(result['primary_contrast_improvement_at_least_10pct']
                                   and result['primary_beats_best_control_at_least_5pct']
                                   and result['output_sse_increase_at_most_5pct'])
    else:
        assert block == 0
        result['secondary_contrast_increase_at_most_5pct'] = bool(scores['pair'] <= 1.05 * scores['base'])
        result['cell_pass'] = bool(result['secondary_contrast_increase_at_most_5pct']
                                   and result['output_sse_increase_at_most_5pct'])
    return result


def main():
    start = time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not DEST.exists(), 'Preserve prior independent receipt'
    torch.set_num_threads(2)
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'complete' and len(source['rows']) == 4
    sources = {name: record(row['file']) for name, row in source['sources'].items()}
    assert sources == source['sources'], 'Executed source identity changed'
    errors = {'comparisons': 0, 'max_absolute_difference': 0., 'max_normalized_difference': 0.}
    report = dict(experiment='E080', status='running', method='CPU NumPy FP64 metrics; torch used only for PT loading',
                  driver=record(__file__), source_report=record(SOURCE), verified_executed_sources=sources,
                  environment={'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
                               'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}, rows=[])
    for row in source['rows']:
        output = record(row['outputs']['file'])
        assert output == row['outputs'], 'Saved output SHA/size mismatch'
        data = torch.load(output['file'], map_location='cpu', weights_only=True)
        assert data['query_indices'].tolist() == row['query_indices']
        exact = numpy_tensor(data['exact_attention'])
        native = numpy_tensor(data['native_capture'])
        checked = dict(case_id=row['case_id'], block=row['block'], step=row['step'],
                       verified_outputs=output, conditions=[])
        for previous in row['conditions']:
            condition = previous['condition']
            values = data[condition]
            oracle = numpy_tensor(values['oracle'])
            item = dict(condition=condition, oracle_vs_full=metrics(oracle, exact), arms={})
            compare(item['oracle_vs_full'], previous['oracle_vs_full'], errors)
            for arm in ARMS:
                value = numpy_tensor(values[arm])
                item['arms'][arm] = dict(vs_pv_oracle=metrics(value, oracle),
                                          vs_full_attention=metrics(value, exact))
                for reference in ('vs_pv_oracle', 'vs_full_attention'):
                    compare(item['arms'][arm][reference], previous['arms'][arm][reference], errors)
            simulated = bf16_round_numpy(numpy_tensor(values['base']))
            item['base_simulator_vs_native'] = metrics(simulated, native)
            item['native_vs_full'] = metrics(native, exact)
            for name in ('base_simulator_vs_native', 'native_vs_full'):
                compare(item[name], previous[name], errors)
            item['native_gap_energy_ratio'] = item['base_simulator_vs_native']['sse'] / max(item['native_vs_full']['sse'], 1e-30)
            compare({'native_gap_energy_ratio': item['native_gap_energy_ratio']},
                    {'native_gap_energy_ratio': previous['native_gap_energy_ratio']}, errors)
            item['prespecified_gate'] = cell_gate(row['block'], item['arms'])
            checked['conditions'].append(item)
        direction = [np.sign(item['prespecified_gate']['pair_contrast_change_pct']) for item in checked['conditions']]
        checked['qk_conditions_direction_compatible'] = bool(direction[0] == direction[1])
        report['rows'].append(checked)
    cells = [condition['prespecified_gate']['cell_pass'] for row in report['rows'] for condition in row['conditions']]
    direction_pass = all(row['qk_conditions_direction_compatible'] for row in report['rows'])
    all_pass = all(cells) and direction_pass
    report['decision'] = dict(all_cells_pass=all(cells), qk_direction_compatible=direction_pass,
                              proceed_to_video=all_pass,
                              conclusion='Advance to bounded video test' if all_pass else 'Stop current C007 representation; no video or kernel escalation')
    assert not torch.cuda.is_initialized()
    report.update(status='complete', numeric_verification=errors, cuda_initialized=False,
                  seconds=time.perf_counter() - start)
    DEST.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], decision=report['decision'],
                          verification=errors, seconds=report['seconds'], output=str(DEST)), indent=2))
    for row in report['rows']:
        for item in row['conditions']:
            print(json.dumps(dict(case_id=row['case_id'], block=row['block'], condition=item['condition'],
                                  gate=item['prespecified_gate'], native_gap=item['native_gap_energy_ratio'])))


if __name__ == '__main__':
    main()
