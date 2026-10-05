#!/usr/bin/env python3
"""CPU NumPy readout of saved E082 native center128 acceptance outputs."""
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
SOURCE = ROOT / 'results/research/E082/native_validation_v3.json'
PRIOR = ROOT / 'results/research/E081/probe.json'
DEST = ROOT / 'results/research/E082/native_independent_verification.json'


def main():
    start = time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not DEST.exists(), 'Preserve prior verification receipts'
    torch.set_num_threads(2)
    source = json.loads(SOURCE.read_text())
    prior = json.loads(PRIOR.read_text())
    assert source['status'] == prior['status'] == 'complete' and source['passed'] and len(source['rows']) == 4
    sources = {name: independent.record(entry['file']) for name, entry in source['sources'].items()}
    assert sources == source['sources']
    private = source['private_contract']['private']
    private_sources = {name: independent.record(entry['file']) for name, entry in private['sources'].items()}
    assert private_sources == private['sources']
    assert independent.record(private['loader']['file']) == private['loader']
    assert independent.record(private['binary']['file']) == private['binary']
    errors = dict(comparisons=0, max_absolute_difference=0., max_normalized_difference=0.)
    report = dict(experiment='E082', phase='independent_native_acceptance', status='running',
                  source_report=independent.record(SOURCE), prior_report=independent.record(PRIOR),
                  verified_sources=sources, verified_private_sources=private_sources,
                  verified_loader=private['loader'], verified_binary=private['binary'],
                  independent_sources={Path(path).name: independent.record(path) for path in (__file__, independent.__file__)},
                  environment=dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                                   CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),
                  method='Saved CPU tensors, NumPy FP64 metrics, no model/extension execution', rows=[])
    for row in source['rows']:
        output = independent.record(row['outputs']['file'])
        assert output == row['outputs']
        payload = torch.load(output['file'], map_location='cpu', weights_only=True)
        old_row = next(x for x in prior['rows'] if (x['case_id'], x['block']) == (row['case_id'], row['block']))
        old_output = independent.record(old_row['outputs']['file'])
        assert old_output == old_row['outputs']
        old = torch.load(old_output['file'], map_location='cpu', weights_only=True)
        assert payload['query_indices'].tolist() == old['query_indices'].tolist()
        tensors = {name: independent.numpy_tensor(payload[name]) for name in ('candidate', 'simulator', 'oracle', 'full')}
        for name, old_value in (('simulator', old['native_qk']['center128']),
                                ('oracle', old['native_qk']['oracle']), ('full', old['exact_attention'])):
            assert np.array_equal(tensors[name].view(np.uint8), independent.numpy_tensor(old_value).view(np.uint8))
        gap = independent.metrics(tensors['candidate'], tensors['simulator'])['sse']
        energy = independent.metrics(tensors['simulator'], tensors['oracle'])['sse']
        ratio = gap / max(energy, 1e-30)
        independent.compare(dict(simulator_gap_energy=gap, center_error_energy=energy, gap_ratio=ratio),
                            {name: row[name] for name in ('simulator_gap_energy', 'center_error_energy', 'gap_ratio')}, errors)
        assert ratio <= .01
        native = independent.numpy_tensor(old['native_capture'])
        native_stats = independent.metrics(native, tensors['full'])
        center_stats = independent.metrics(tensors['candidate'], tensors['full'])
        report['rows'].append(dict(case_id=row['case_id'], block=row['block'], verified_outputs=output,
                                  verified_E081_outputs=old_output, simulator_gap_energy=gap,
                                  center_simulator_error_energy=energy, gap_ratio=ratio,
                                  center_vs_pv_oracle=independent.metrics(tensors['candidate'], tensors['oracle']),
                                  center_vs_full_attention=center_stats, native_base_vs_full_attention=native_stats,
                                  center_vs_native_base_full_sse_change_pct=100.*(center_stats['sse']/native_stats['sse']-1.),
                                  center_vs_native_base_full_contrast_sse_change_pct=100.*(center_stats['contrast_sse']/native_stats['contrast_sse']-1.),
                                  original_replay_producer_receipts={name: row[name] for name in
                                      ('zero_mean_byte_exact', 'original_capture_byte_exact')},
                                  baseline_seconds=row['baseline_seconds'], center_seconds=row['center_seconds'],
                                  time_ratio=row['time_ratio']))
    assert not torch.cuda.is_initialized()
    report.update(status='complete', acceptance_gap_pass=True, numeric_verification=errors,
                  cuda_initialized=False, seconds=time.perf_counter()-start,
                  constant_v_max_abs_producer_receipt=source['constant_v_max_abs'],
                  scope='Independent SHA/metric validation of saved selected-query outputs; full no-op/capture replay, constant-V and timing are retained producer receipts, not independently rerun.')
    DEST.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='complete', seconds=report['seconds'], verification=errors, output=str(DEST))))
    for row in report['rows']:
        print(json.dumps({key: row[key] for key in ('case_id', 'block', 'gap_ratio',
                         'center_vs_native_base_full_sse_change_pct', 'center_vs_native_base_full_contrast_sse_change_pct')}))


if __name__ == '__main__':
    main()
