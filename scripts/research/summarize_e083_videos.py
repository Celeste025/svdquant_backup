#!/usr/bin/env python3
"""Independent stdlib reduction and six-media SHA audit for E083 videos."""
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research/E083'
COMPARISONS = ('sage_vs_bf16', 'center_vs_bf16', 'center_vs_sage')
SCALARS = ('lpips_alex', 'l1_mae', 'l2_rmse')


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return dict(file=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def main():
    started = time.monotonic()
    dest = OUT/'video_summary.json'
    assert not dest.exists()
    metrics = json.loads((OUT/'metrics.json').read_text())
    triplets = json.loads((OUT/'triplets.json').read_text())
    assert metrics['status'] == 'complete' and len(metrics['rows']) == 6 and len(triplets) == 2
    assert {r['case'] for r in metrics['rows']} == {
        f'vbench0161_r{replica}__{comparison}' for replica in (0, 1) for comparison in COMPARISONS}
    assert {r['case'] for r in triplets} == {'vbench0161_r0', 'vbench0161_r1'}
    for source in metrics['sources']:
        assert record(source['file']) == source
    pairings = {r['case']: r for r in triplets}
    errors = []
    for row in metrics['rows']:
        cid, comparison = row['case'].split('__')
        candidate, reference = comparison.split('_vs_')
        assert row['comparison'] == comparison and row['candidate_arm'] == candidate and row['reference_arm'] == reference
        assert row['seed'] == pairings[cid]['seed']
        assert row['candidate_sha256'] == pairings[cid]['sha256'][candidate]
        assert row['reference_sha256'] == pairings[cid]['sha256'][reference]
        frames = row['frame_metrics']
        assert len(frames) == row['frames'] == 124
        assert [frame['frame'] for frame in frames] == list(range(124))
        n = 124*1024*576*3
        assert row['scalar_pixels'] == n
        totals = {key: sum(frame[key] for frame in frames) for key in
                  ('abs_sum_uint8', 'squared_sum_uint8', 'reference_squared_sum_uint8')}
        assert all(value == row[key] for key, value in totals.items())
        assert all(math.isfinite(frame['lpips_alex']) for frame in frames)
        computed = dict(lpips_alex=sum(frame['lpips_alex'] for frame in frames)/124,
                        l1_mae=totals['abs_sum_uint8']/n/255,
                        l2_rmse=math.sqrt(totals['squared_sum_uint8']/n)/255,
                        l2_norm=math.sqrt(totals['squared_sum_uint8'])/255,
                        relative_l2=math.sqrt(totals['squared_sum_uint8']/totals['reference_squared_sum_uint8']))
        for key, value in computed.items():
            error = abs(row[key]-value)/max(1., abs(value))
            assert error < 1e-12
            errors.append(error)
    for comparison in COMPARISONS:
        selected = [row for row in metrics['rows'] if row['comparison'] == comparison]
        assert len(selected) == 2
        for key in SCALARS:
            expected = sum(row[key] for row in selected)/2
            error = abs(metrics['summary'][comparison][key]-expected)/max(1., abs(expected))
            assert error < 1e-12
            errors.append(error)
    verified_media = []
    verified_reports = {}
    for row in triplets:
        for arm in ('bf16', 'sage', 'center'):
            media = record(row[arm])
            assert media['sha256'] == row['sha256'][arm]
            verified_media.append(dict(case=row['case'], arm=arm, **media))
        for key, source in row['report_sources'].items():
            sources = source.values() if key in ('sage', 'center') else (source,)
            for item in sources:
                checked = record(item['file'])
                assert checked == item
                verified_reports[checked['file']] = checked
    assert len(verified_media) == len({r['file'] for r in verified_media}) == 6
    rows = []
    for replica in (0, 1):
        cid = f'vbench0161_r{replica}'
        rows.append(dict(case_id=cid, **{
            comparison: {key: next(r for r in metrics['rows'] if r['case'] == cid+'__'+comparison)[key]
                         for key in SCALARS} for comparison in COMPARISONS}))
    report = dict(status='complete',
                  definition='BF16 backbone in every arm; full124frame 1024x576 paired distances, not quality percentages',
                  rows=rows, summary=metrics['summary'], verified_media=verified_media,
                  verified_generation_reports=list(verified_reports.values()), checked_pairs=6,
                  max_normalized_reduction_difference=max(errors),
                  sources=[record(p) for p in (Path(__file__), OUT/'metrics.json', OUT/'triplets.json')],
                  scope='Independent stdlib aggregation of saved per-frame outputs, pairing/report hashes and six original media SHA; no rerun of LPIPS network',
                  seconds=time.monotonic()-started)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('sources', 'verified_media', 'verified_generation_reports')}, indent=2))


if __name__ == '__main__':
    main()
