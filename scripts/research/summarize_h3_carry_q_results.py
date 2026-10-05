#!/usr/bin/env python3
"""Descriptive E065b readout from independently checked saved results; CPU only."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E065b'


def record(path):
    path = Path(path).resolve()
    return dict(file=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def distribution(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), below_one=sum(v < 1 for v in values))


def main():
    start = time.monotonic()
    destination = RD / 'summary.json'
    assert not destination.exists()
    sources = {n: record(RD / n) for n in
               ('evaluate.json', 'independent_outputs.json', 'independent_selection.json', 'process_exit.json')}
    evaluation, outputs, selection, termination = [json.loads(Path(sources[n]['file']).read_text())
        for n in sources]
    assert all(r['status'] == 'complete' for r in (evaluation, outputs, selection))
    assert termination['exit_code'] == 0 and not termination['timeout_exit']
    assert outputs['evaluation_sha256'] == selection['evaluation_sha256'] == sources['evaluate.json']['sha256']
    checked_rows = {r['layer']: r for r in selection['layers']}
    iterations = {arm: collections.Counter() for arm in ('restart', 'carry')}
    monotonic = collections.Counter()
    per_case = collections.defaultdict(list)
    per_type = collections.defaultdict(list)
    metadata_sources = []
    for row in evaluation['search_layers']:
        bound = row['metadata']
        actual = record(bound['file'])
        assert actual == bound
        assert selection['json_files'][bound['file']] == bound
        metadata_sources.append(actual)
        data = json.loads(Path(bound['file']).read_text())
        arms = data['arms']
        for arm in iterations:
            history = arms[arm]
            iterations[arm][history['selected_k']] += 1
            scores = [c['score'] for c in history['candidates']]
            monotonic[arm] += all(b <= a for a, b in zip(scores, scores[1:]))
            assert history['selected_score'] == checked_rows[row['layer']]['selections'][arm]['score']
        carry = arms['carry']['candidates'][arms['carry']['selected_k']]
        restart = arms['restart']['candidates'][arms['restart']['selected_k']]
        assert carry['case_ids'] == restart['case_ids']
        for cid, a, b in zip(carry['case_ids'], carry['per_case_sse'], restart['per_case_sse'], strict=True):
            per_case[cid].append(a / b)
        per_type[row['layer'].split('.', 2)[2]].append(checked_rows[row['layer']]['carry_over_restart'])
    final_cases = []
    for row in outputs['cases']:
        stats = row['stats']
        final_cases.append(dict(case_id=row['case_id'], position=row['position'],
            carry_over_restart={m: row['carry_ratios'][m]['sse']['restart'] for m in ('video', 'audio')},
            centered_video_carry_over_restart=row['carry_ratios']['video']['channel_centered_sse']['restart'],
            video_carry_over_legacy=row['carry_ratios']['video']['sse']['legacy_selected'],
            video_carry_over_plain=row['carry_ratios']['video']['sse']['plain'],
            video_restart_over_legacy=stats['restart']['video']['sse']/stats['legacy_selected']['video']['sse'],
            video_restart_over_plain=stats['restart']['video']['sse']/stats['plain']['video']['sse']))
    result = dict(experiment='E065b', status='complete', seconds=time.monotonic()-start,
        new_model_forwards=0, torch_imported=False, script=record(__file__), sources=sources,
        metadata_sources=metadata_sources,
        local_layer_ratios=distribution([r['carry_over_restart'] for r in selection['layers']]),
        local_cell_ratios=distribution([v for values in per_case.values() for v in values]),
        local_by_case={k: distribution(v) for k, v in per_case.items()},
        local_by_type={k: distribution(v) for k, v in per_type.items()},
        selected_iterations=iterations, recorded_monotonic_nonincreasing_layers=monotonic,
        full_model_cases=final_cases,
        scope='Descriptive statistics of independently checked selected SSEs and recorded candidate curves; no new experiment',
        limitations=['Unselected candidate scores were not independently recomputed.',
                    'Layer/state cells are correlated and are not independent generation samples.',
                    'No video-quality, causal-mechanism, convergence, or novelty claim.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'metadata_sources')}, indent=2))


if __name__ == '__main__':
    main()
