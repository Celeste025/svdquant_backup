#!/usr/bin/env python3
"""CPU-only E034 reduction: fixed outputs, coarse geometry, and full-path receipts.

No GPU work, whole-Q re-reduction, source-tree freeze, or cross-run timing claims.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import probe_h3_query_mean_k4 as common

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E034'
N, NP, H, D, G, K = 22539, 22656, 56, 128, 177, 16
ARMS = ('global', 'fullblock', 'codebook', 'coarse16')
BLOCKS = (0, 24, 48)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-20), (a, b)


def load_output(ref):
    import torch
    value = common.load(ref['artifact'])
    assert value.shape == (N, H, D) and value.dtype == torch.bfloat16
    actual = common.trecord(value)
    assert actual['finite']
    for key, expected in ref['tensor'].items():
        assert actual[key] == expected, (ref['artifact']['file'], key)
    return value


def metrics(value, reference):
    return common.metrics(value.transpose(0, 1), reference.transpose(0, 1))


def expected_counts(arm, n):
    private = arm in ('codebook', 'coarse16')
    return dict(preprocess=n, k_mean=n,
        q_global_mean=n if arm in ('global', 'codebook') else 0,
        q_block_mean=n if arm in ('fullblock', 'codebook') else 0,
        q_coarse_mean=K*n if arm == 'coarse16' else 0,
        q_coarse_stack=n if arm == 'coarse16' else 0,
        construct=n if arm == 'codebook' else 0,
        q_pack=n, k_pack=n, v_pack=n, correction_gemm=n,
        official_attention=0 if private else n,
        private_attention=n if private else 0, attention=n)


def timing_receipt(entry):
    assert entry['status'] == 'complete' and entry['warmup_calls'] == 3
    assert len(entry['repeats']) == 10
    assert entry['actual_counts'] == expected_counts(entry['arm'], 13)
    for index, row in enumerate(entry['repeats']):
        assert row['index'] == index
        assert row['actual_counts'] == expected_counts(entry['arm'], 1)
        for key in ('cuda_ms', 'synchronized_wall_ms'):
            assert math.isfinite(row[key]) and row[key] > 0
        before, after = row['before'], row['after']
        assert all(isinstance(v, int) and v >= 0 for v in (*before.values(), *after.values()))
        for category in ('allocated', 'reserved'):
            assert after['peak_'+category] >= max(before[category], after[category])
            assert row['incremental_peak_'+category] == after['peak_'+category]-before[category]
    def stats(key):
        values = [r[key] for r in entry['repeats']]
        return dict(values=values, median=statistics.median(values), min=min(values), max=max(values))
    cuda, wall = stats('cuda_ms'), stats('synchronized_wall_ms')
    close(cuda['median'], entry['median_cuda_ms'])
    close(wall['median'], entry['median_synchronized_wall_ms'])
    peaks = {k: max(r['after']['peak_'+k] for r in entry['repeats']) for k in ('allocated', 'reserved')}
    increments = {k: max(r['incremental_peak_'+k] for r in entry['repeats']) for k in ('allocated', 'reserved')}
    for key in peaks:
        assert peaks[key] == entry['peak_'+key]
        assert increments[key] == entry['max_incremental_peak_'+key]
    return dict(cuda_ms=cuda, synchronized_wall_ms=wall,
        resident_baseline=entry['resident_baseline'], after_warmup_baseline=entry['after_warmup_baseline'],
        peak_bytes=peaks, max_incremental_peak_bytes=increments,
        repeats=entry['repeats'], actual_counts=entry['actual_counts'])


def check_coarse(entry, geometry):
    import torch
    payload = common.load(entry['centers']['artifact'])
    assert set(payload) == {'centers', 'ids', 'boundaries'}
    centers, ids, boundaries = (payload[k] for k in ('centers', 'ids', 'boundaries'))
    assert centers.shape == (H, K, D) and centers.dtype == torch.bfloat16
    assert bool(centers.isfinite().all())
    assert ids.shape == (1, H, G) and ids.dtype == torch.int32
    assert boundaries.shape == (K+1,) and boundaries.dtype == torch.int32
    expected = [j*G//K for j in range(K+1)]
    assert boundaries.tolist() == geometry['boundaries'] == expected
    intervals = []
    for j, (lo, hi) in enumerate(zip(expected[:-1], expected[1:])):
        assert hi > lo and bool((ids[:, :, lo:hi] == j).all())
        valid = max(0, min(N, D*hi)-D*lo)
        intervals.append(dict(center=j, group_start=lo, group_end=hi,
            token_start=D*lo, padded_token_end=D*hi, valid_tokens=valid,
            mean_denominator=D*(hi-lo), padding_tokens=D*(hi-lo)-valid))
    assert geometry['intervals'] == intervals
    assert expected[0] == 0 and expected[-1] == G
    assert sum(r['valid_tokens'] for r in intervals) == N
    assert sum(r['mean_denominator'] for r in intervals) == NP
    assert sum(r['padding_tokens'] for r in intervals) == 117
    assert intervals[-1]['valid_tokens'] == 1419 and intervals[-1]['mean_denominator'] == 1536
    actual = {key:common.trecord(value) for key,value in payload.items()}
    assert actual == entry['centers']['tensors']
    assert actual['ids'] == geometry['ids']
    return dict(artifact=entry['centers']['artifact'], tensors=actual, intervals=intervals,
        provenance=entry['centers']['provenance'],
        verification_boundary='Saved C/id/boundaries, finite values, coverage and mean-denominator geometry checked independently. Padding-zero construction and direct BF16 padded-Q mean follow the frozen runner source; padded Q is not saved and this reducer does not re-reduce full Q or claim an independent numeric check of C.')


def compare_heads(arms):
    rows = []
    for head in range(H):
        nmse = {arm:arms[arm]['metrics']['bf16']['per_head'][head]['nmse'] for arm in ARMS}
        gap = nmse['global']-nmse['fullblock']
        rows.append(dict(head=head, nmse=nmse, coarse_minus_codebook_nmse=nmse['coarse16']-nmse['codebook'],
            retained_global_to_block_gap={arm:(nmse['global']-nmse[arm])/gap if gap > 0 else None
                                         for arm in ('codebook', 'coarse16')}))
    def comparison(a, b):
        return {label:sum(test(row['nmse'][a], row['nmse'][b]) for row in rows)
                for label,test in [('better',lambda x,y:x<y),('equal',lambda x,y:x==y),('worse',lambda x,y:x>y)]}
    pooled = {arm:arms[arm]['metrics']['bf16']['pooled']['nmse'] for arm in ARMS}
    gap = pooled['global']-pooled['fullblock']
    return dict(per_head=rows, coarse_versus_codebook=comparison('coarse16','codebook'),
        coarse_versus_global=comparison('coarse16','global'), coarse_versus_block=comparison('coarse16','fullblock'),
        pooled_nmse=pooled, pooled_coarse_minus_codebook_nmse=pooled['coarse16']-pooled['codebook'],
        pooled_coarse_relative_nmse_increase_vs_codebook=pooled['coarse16']/pooled['codebook']-1,
        pooled_fraction_of_codebook_improvement_over_global=(pooled['global']-pooled['coarse16'])/(pooled['global']-pooled['codebook'])
            if pooled['global'] > pooled['codebook'] else None,
        pooled_retained_global_to_block_gap={arm:(pooled['global']-pooled[arm])/gap if gap > 0 else None
                                           for arm in ('codebook','coarse16')})


def frontier(arms):
    points = {arm:[arms[arm]['metrics']['bf16']['pooled']['nmse'],
                   arms[arm]['timing']['cuda_ms']['median'], arms[arm]['timing']['peak_bytes']['allocated']]
              for arm in ARMS}
    dominated_by = {arm:[] for arm in ARMS}
    relations = []
    for a in ARMS:
        for b in ARMS:
            if a != b and all(x <= y for x,y in zip(points[a],points[b])) and any(x < y for x,y in zip(points[a],points[b])):
                dominated_by[b].append(a)
                relations.append(dict(dominates=a, dominated=b,
                    cuda_observed_ranges_strictly_separated=arms[a]['timing']['cuda_ms']['max'] < arms[b]['timing']['cuda_ms']['min']))
    return dict(dimensions=['BF16_reference_pooled_NMSE','median_fullpath_cuda_ms','peak_allocated_bytes'],
        measured_points=points, dominated_by=dominated_by, relations=relations,
        nondominated=[a for a in ARMS if not dominated_by[a]],
        scope='Same-run four-arm observed point estimates only. Ten-repeat ranges are descriptive, not confidence intervals. No E033 chunk timing, population-level speed, or video-quality claim.')


def reduce(args, result):
    import torch
    run = json.loads(args.input.read_text())
    assert run['status'] == 'complete' and run['actual_dit_calls'] == 0
    assert run['attention_calls'] == run['expected_attention_calls'] == 156
    assert run['arms'] == list(ARMS) and run['warmups'] == 3 and run['repeats'] == 10
    prior_path = Path(run['reference_report']['file'])
    assert common.record(prior_path) == run['reference_report']
    prior = json.loads(prior_path.read_text())
    assert prior['status'] == 'complete'
    assert len(run['benchmark']) == 12 and len(run['inputs']) == 3
    assert {(r['block'],r['arm']) for r in run['benchmark']} == {(b,a) for b in BLOCKS for a in ARMS}
    assert sorted(r['block'] for r in run['inputs']) == list(BLOCKS)
    result.update(source_run=common.record(args.input), reference_report=run['reference_report'],
        recorded_sources=run['sources'], source_verification='Output/reference artifacts checked; official/generated source tree and model/capture weights are not rehashed.',
        actual_dit_calls=0, native_calls=156, measured_repeats=120, warmup_calls=36,
        timing_scope=run['timing_scope'], memory_scope=run['memory_scope'], layers=[])
    counts = {key:0 for key in expected_counts('global',1)}
    for source in run['inputs']:
        oldsource = next(r for r in prior['inputs'] if r['block'] == source['block'])
        assert source['capture'] == oldsource['capture']
        assert source['outputs']['bf16'] == oldsource['outputs']['bf16']
        bf16 = load_output(source['outputs']['bf16'])
        layer = dict(block=source['block'], references=source['outputs'], arms={})
        for arm in ARMS:
            entry = next(r for r in run['benchmark'] if r['block'] == source['block'] and r['arm'] == arm)
            timing = timing_receipt(entry)
            value = load_output(entry['output'])
            errors = dict(bf16=metrics(value,bf16))
            close(errors['bf16']['pooled']['nmse'],entry['versus_bf16']['pooled_nmse'])
            if arm != 'coarse16':
                old = next(r for r in prior['benchmark'] if r['block'] == source['block'] and r['arm'] == arm)
                assert source['outputs'][arm] == old['output']
                ref = load_output(source['outputs'][arm])
                errors['same_arm_e033'] = metrics(value,ref)
                close(errors['same_arm_e033']['pooled']['nmse'],entry['versus_e033'][arm]['pooled_nmse'])
                del ref
            for key in counts: counts[key] += timing['actual_counts'][key]
            layer['arms'][arm] = dict(output=entry['output'], timing=timing, metrics=errors)
            if arm == 'coarse16':layer['coarse_contract'] = check_coarse(entry,run['coarse_geometry'])
            del value
        layer['head_comparison'] = compare_heads(layer['arms'])
        layer['observed_frontier'] = frontier(layer['arms'])
        result['layers'].append(layer)
        common.save(args.output,result)
        print(json.dumps(dict(block=source['block'],coarse_vs_codebook=layer['head_comparison']['coarse_versus_codebook'],
            nondominated=layer['observed_frontier']['nondominated'])),flush=True)
        del bf16
    assert counts == run['actual_counts']
    assert counts['attention'] == 156 and counts['official_attention'] == counts['private_attention'] == 78
    assert counts['construct'] == 39 and counts['q_coarse_mean'] == 624
    result['all_heads'] = {name:{label:sum(layer['head_comparison'][name][label] for layer in result['layers'])
        for label in ('better','equal','worse')} for name in ('coarse_versus_codebook','coarse_versus_global','coarse_versus_block')}
    assert all(sum(row.values()) == 168 for row in result['all_heads'].values())
    assert not torch.cuda.is_initialized()
    result.update(status='complete',actual_counts=counts,cuda_initialized=False,
        independent_comparisons=dict(outputs_vs_bf16=12,existing_arms_vs_e033=9),
        interpretation='All three fixed cases and 168 heads retained. Error refers to saved-packet BF16 attention output, not video quality. Timing includes necessary preparation plus consumption; no sums of separate timings or comparisons with unsampled chunk arms.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=RD/'run.json')
    parser.add_argument('--output',type=Path,default=RD/'summary.json')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not args.output.exists(), 'Preserve existing summary/failed attempt'
    started=time.time()
    result=dict(experiment='E034',status='running',reducer=common.record(__file__))
    try:
        import torch
        torch.set_num_threads(6)
        reduce(args,result)
    except BaseException:
        result.update(status='failed_preserved',error=traceback.format_exc())
        raise
    finally:
        result['seconds']=time.time()-started
        common.save(args.output,result)
        print(json.dumps(dict(status=result['status'],output=str(args.output))),flush=True)


if __name__ == '__main__':main()
