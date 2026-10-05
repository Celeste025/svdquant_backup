#!/usr/bin/env python3
"""Describe independently checked E066 and E065b results; no model execution."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E066'


def record(path):
    return dict(file=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    started = time.monotonic()
    destination = REPORTS/'summary.json'
    assert not destination.exists()
    paths = dict(evaluation=REPORTS/'evaluate.json', independent=REPORTS/'independent.json',
                 process_exit=REPORTS/'process_exit.json',
                 parent_summary=ROOT/'results/research/E065b/summary.json')
    sources = {name: record(path) for name, path in paths.items()}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    result, check, parent = data['evaluation'], data['independent'], data['parent_summary']
    assert result['status'] == check['status'] == parent['status'] == 'complete'
    assert data['process_exit']['exit_code'] == 0
    assert check['evaluation_sha256'] == sources['evaluation']['sha256']
    assert check['cuda_initialized'] is False and check['new_model_forwards'] == 0
    assert result['source_evaluation'] == parent['sources']['evaluate.json']
    assert result['source_checks']['outputs'] == parent['sources']['independent_outputs.json']
    for source in parent['sources'].values():
        assert record(Path(source['file'])) == source
    cases, audio_reversals = [], []
    full = {row['case_id']+'/'+row['position']: row for row in parent['full_model_cases']}
    assert set(full) == {row['key'] for row in check['cases']}
    video_ratios = []
    for case in check['cases']:
        assert len(case['layers']) == 200
        layers = case['layers']
        assert all(row['full'][domain]['carry_sse'] < row['full'][domain]['restart_sse']
                   for row in layers for domain in ('video', 'joint'))
        shape_change = [abs(row['sample']['video']['carry_over_restart']/
                            row['subset']['video']['carry_over_restart']-1) for row in layers]
        flips = lambda a, b: sum((row[a]['video']['carry_over_restart'] < 1) !=
                                 (row[b]['video']['carry_over_restart'] < 1) for row in layers)
        video_ratios.extend(row['full']['video']['carry_over_restart'] for row in layers)
        audio_reversals.extend(dict(key=case['key'], layer=row['layer'],
            carry_over_restart=row['full']['audio']['carry_over_restart']) for row in layers
            if row['full']['audio']['carry_sse'] > row['full']['audio']['restart_sse'])
        cases.append(dict(key=case['key'], full_by_domain=case['full_by_domain'],
            sample_by_domain=case['sample_by_domain'], subset_by_domain=case['subset_by_domain'],
            video_rank_flips_M512_to_fullM_same_sample=flips('subset', 'sample'),
            video_rank_flips_fullM_sample_to_all_rows=flips('sample', 'full'),
            maximum_relative_change_of_video_carry_restart_ratio_M512_to_fullM_same_sample=max(shape_change),
            full_model=full[case['key']]))
    summary = dict(experiment='E066', status='complete', seconds=time.monotonic()-started,
        sources=sources, script=record(Path(__file__)), new_model_forwards=0,
        torch_imported='torch' in sys.modules, cases=cases,
        full_video_layer_state_count=len(video_ratios),
        full_video_carry_better_count=sum(value < 1 for value in video_ratios),
        full_video_ratio_range=[min(video_ratios), max(video_ratios)],
        full_audio_reversals=audio_reversals,
        runtime={name: result[name] for name in ('elapsed_seconds', 'complete_dit_calls',
            'local_native_calls', 'full_native_calls', 'subset_native_calls', 'activation_packs',
            'peak_allocated_bytes', 'data_bytes')},
        independent_seconds=check['seconds'],
        mathematical_boundary='For these fixed candidates and teacher inputs, any fixed nonnegative sum of the observed video or joint per-layer SSE values still favors carry whenever at least one weight is positive; this does not rule out training another candidate with a weighted objective.',
        limitations=['Related layer/state cells are not independent generation samples.',
            'The shape control includes both native main and BF16 low-rank GEMMs.',
            'Local teacher-input SSE ordering does not identify the causal source of the whole-model reversal.',
            'No new video, quality, or method contribution is established.'])
    with destination.open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
