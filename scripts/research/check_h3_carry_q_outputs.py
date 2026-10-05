#!/usr/bin/env python3
"""E065 independent NumPy complete-DiT output readout, zero CUDA/forwards.

This checks saved input/output tensors and full-model errors. It does not
reexecute candidate SVD, calibration kernels, or the original checkpoint.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
import torch
import check_h3_swiglu_interaction as io

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E065'
CASES = ('e010_p030_s05', 'e010_p036_s14')
POSITIONS = ('source_teacher', 'next_teacher')
ARMS = ('restart', 'carry')


def metrics(output, teacher):
    result = {}
    for modality in ('video', 'audio'):
        target = io.array(teacher[modality])
        error = io.array(output[modality])-target
        assert np.isfinite(error).all()
        centered = error-error.mean(axis=tuple(range(2, error.ndim)), keepdims=True)
        sse = float(np.sum(error*error, dtype=np.float64))
        energy = float(np.sum(target*target, dtype=np.float64))
        result[modality] = dict(sse=sse, reference_energy=energy,
            nmse=sse/max(energy, 1e-30), elements=error.size,
            channel_centered_sse=float(np.sum(centered*centered, dtype=np.float64)))
    return result


def check_stats(actual, recorded):
    maximum = 0.
    for m in ('video', 'audio'):
        assert actual[m]['elements'] == recorded[m]['elements']
        for field in ('sse', 'reference_energy', 'nmse', 'channel_centered_sse'):
            a, b = actual[m][field], recorded[m][field]
            difference = abs(a-b)/max(abs(a), abs(b), 1e-300)
            assert difference < 1e-10, (m, field, a, b)
            maximum = max(maximum, difference)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=REPORTS)
    args = parser.parse_args()
    reports = args.reports
    started = time.monotonic()
    torch.set_num_threads(2)
    destination = reports/'independent_outputs.json'
    assert not destination.exists()
    ep = reports/'evaluate.json'
    result = json.loads(ep.read_text())
    assert result['status'] == 'complete'
    if result['experiment'] == 'E065b':
        assert result['complete_dit_calls'] == result['attempted_dit_calls'] == 9
        assert result['calibration_bf16_calls'] == 0 and result['reused_calibration_bf16_calls'] == 8
        source = result['source_report']
        assert io.sha(source['file']) == source['sha256']
        previous_report = json.loads(Path(source['file']).read_text())
        assert previous_report['complete_dit_calls'] == 9 and previous_report['calibration_bf16_calls'] == 8
    else:
        assert result['experiment'] == 'E065'
        assert result['complete_dit_calls'] == result['attempted_dit_calls'] == 17
        assert result['calibration_bf16_calls'] == 8
    for path, record in result['sources'].items():
        assert Path(path).stat().st_size == record['bytes'] and io.sha(path) == record['sha256'], path
    rows = result['cases']
    index = {(r['case_id'], r['position'], r['arm']): r for r in rows}
    assert len(index) == len(rows) == 9
    replays = [r for r in rows if r['arm'] == 'legacy_selected']
    assert len(replays) == 1
    files, loaded = {}, {}

    def load(record):
        path = record['file']
        if path not in loaded:
            loaded[path] = io.load(record, files)
        return loaded[path]

    def validate(record, expected_actual):
        payload = load(record['artifact'])
        actual = payload['actual_dit_inputs']
        if record.get('experiment') != 'E014':
            actual = io.signature(actual)
        assert actual == expected_actual
        for field in ('raw_outputs', 'velocities'):
            assert io.signature(payload[field]) == record[field]
        return payload

    maximum, summaries = 0., []
    for cid in CASES:
        for position in POSITIONS:
            key = cid+'/'+position
            records = result['references'][key]
            expected = records['legacy_selected']['actual_input_signature']
            old = {arm: validate(rec, expected) for arm, rec in records.items() if rec is not None}
            assert set(old) == {'bf16', 'plain', 'legacy_selected'}
            new = {}
            for arm in ARMS:
                row = index[(cid, position, arm)]
                assert row['status'] == 'complete' and row['actual_dit_input_signature'] == expected
                new[arm] = load(row['artifact'])
                assert io.byte_equal(new[arm]['actual_dit_inputs'], old['legacy_selected']['actual_dit_inputs'])
                for field in ('raw_outputs', 'velocities'):
                    assert io.signature(new[arm][field]) == row[field]
                audit = row['runtime_audit']
                assert audit['scaled_mm_calls'] == 200 and audit['sdpa_calls'] == 102 and audit['disk_loads'] == 0
                assert row['fastpack_checks']['checked_calls'] == 200 and row['fastpack_checks']['invalid_calls'] == 0
            stats = {arm: metrics(payload['velocities'], old['bf16']['velocities'])
                for arm, payload in {**old, **new}.items() if arm != 'bf16'}
            for arm in ARMS:
                maximum = max(maximum, check_stats(stats[arm], index[(cid, position, arm)]['metrics']))
            ratios = {}
            for modality in ('video', 'audio'):
                ratios[modality] = {field: {baseline: stats['carry'][modality][field]/stats[baseline][modality][field]
                    for baseline in ('restart', 'legacy_selected', 'plain')}
                    for field in ('sse', 'channel_centered_sse')}
            summaries.append(dict(case_id=cid, position=position, stats=stats, carry_ratios=ratios))
    replay = replays[0]
    k = replay['case_id']+'/'+replay['position']
    previous = load(result['references'][k]['legacy_selected']['artifact'])
    fresh = load(replay['artifact'])
    for field in ('actual_dit_inputs', 'raw_outputs', 'velocities'):
        assert io.byte_equal(fresh[field], previous[field]), field
    assert replay['legacy_replay_exact']
    assert not torch.cuda.is_initialized()
    final = dict(experiment=result['experiment'], status='complete', seconds=time.monotonic()-started,
        cuda_initialized=False, new_model_forwards=0, script_sha256=io.sha(__file__),
        evaluation_sha256=io.sha(ep), cases=summaries, files=files,
        verified_pt_files=len(files), maximum_metric_relative_difference=maximum,
        legacy_replay_byte_exact=True,
        carry_improves_video_sse_all_four=all(r['carry_ratios']['video']['sse']['restart'] < 1 for r in summaries),
        scope='Independent NumPy full-model output statistics and saved input/output identity; no candidate SVD/kernel rerun',
        limitation='Four repeatedly studied diagnostics; no video quality or generalization claim')
    with destination.open('x') as stream:
        json.dump(final, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in final.items() if k not in ('cases', 'files')}, indent=2))
    print(json.dumps([dict(case=r['case_id'], position=r['position'], video=r['carry_ratios']['video'])
        for r in summaries], indent=2))


if __name__ == '__main__':
    main()
