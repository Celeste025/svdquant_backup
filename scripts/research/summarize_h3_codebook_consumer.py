#!/usr/bin/env python3
"""CPU-only E033 reducer: saved outputs, ten controls, and full-path receipts.

No GPU work, reruns, source-tree freeze, or claim of video quality. Dominance is
reported only for the measured NMSE/time/allocated-memory point estimates.
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
RD = ROOT / 'results/research/E033'
N, H, D = 22539, 56, 128
ARMS = ('global', 'fullblock', 'chunk4096', 'chunk8192', 'codebook')
CALLS = {'global': 1, 'fullblock': 1, 'chunk4096': 6, 'chunk8192': 3, 'codebook': 1}


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-20), (a, b)


def load_output(ref):
    import torch
    value = common.load(ref['artifact'])
    assert value.shape == (N, H, D) and value.dtype == torch.bfloat16
    receipt = common.trecord(value)
    assert receipt['finite']
    # Older BF16 references omit a finite field; every recorded field must match.
    for key, expected in ref['tensor'].items():
        assert receipt[key] == expected, (ref['artifact']['file'], key)
    return value


def metrics(value, reference):
    return common.metrics(value.transpose(0, 1), reference.transpose(0, 1))


def ulp_control(value, reference):
    """Independent signed-int BF16 ordering; no benchmark numeric helper import."""
    import torch
    rows = []
    for head in range(H):
        a, b = value[:, head].contiguous(), reference[:, head].contiguous()
        assert bool(a.isfinite().all()) and bool(b.isfinite().all())
        def monotone(t):
            signed = t.view(torch.int16).to(torch.int32)
            # Negative BF16 code s maps to -32768-s; -0 and +0 both become 0.
            return torch.where(signed < 0, -32768-signed, signed)
        distance = (monotone(a)-monotone(b)).abs()
        rows.append(dict(head=head, max_ulp=int(distance.max()),
            count_gt1_ulp=int((distance > 1).sum()), count_nonzero_ulp=int((distance != 0).sum()),
            max_abs=float((a.double()-b.double()).abs().max())))
    return dict(max_ulp=max(r['max_ulp'] for r in rows),
        count_gt1_ulp=sum(r['count_gt1_ulp'] for r in rows),
        count_nonzero_ulp=sum(r['count_nonzero_ulp'] for r in rows),
        max_abs=max(r['max_abs'] for r in rows), per_head=rows,
        finite_bf16_engineering_limit=1, signed_zeros_merged=True)


def expected_counts(arm, repeats):
    return dict(preprocess=repeats, k_mean=repeats,
        q_block_mean=0 if arm == 'global' else repeats,
        q_global_mean=repeats if arm in ('global', 'codebook') else 0,
        construct=repeats if arm == 'codebook' else 0,
        q_pack=repeats, k_pack=repeats, v_pack=repeats,
        correction_gemm=repeats*CALLS[arm], full_correction_gather=0,
        official_attention=0 if arm == 'codebook' else repeats*CALLS[arm],
        private_attention=repeats if arm == 'codebook' else 0,
        attention=repeats*CALLS[arm])


def timing_receipt(entry):
    arm = entry['arm']
    assert entry['status'] == 'complete' and entry['warmup_calls'] == 3
    assert len(entry['repeats']) == 10
    assert entry['actual_counts'] == expected_counts(arm, 13)
    for index, row in enumerate(entry['repeats']):
        assert row['index'] == index and row['actual_counts'] == expected_counts(arm, 1)
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


def observed_frontier(arms):
    points = {name: [value['metrics']['bf16']['pooled']['nmse'],
                     value['timing']['cuda_ms']['median'], value['timing']['peak_bytes']['allocated']]
              for name, value in arms.items()}
    dominated_by = {name: [] for name in arms}
    comparisons = []
    for better, a in points.items():
        for worse, b in points.items():
            if better != worse and all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b)):
                dominated_by[worse].append(better)
                comparisons.append(dict(dominates=better, dominated=worse,
                    cuda_observed_ranges_strictly_separated=(arms[better]['timing']['cuda_ms']['max']
                                                            < arms[worse]['timing']['cuda_ms']['min'])))
    return dict(dimensions=['BF16_reference_pooled_NMSE', 'median_fullpath_cuda_ms', 'peak_allocated_bytes'],
        measured_points=points, dominated_by=dominated_by,
        nondominated=[name for name in ARMS if not dominated_by[name]], relations=comparisons,
        scope='Exact observed point comparisons only; ten-repeat ranges are descriptive, not confidence intervals. No cross-model, video-quality, or population-level claim.')


def head_comparison(arms):
    code = arms['codebook']['metrics']['bf16']
    block = arms['fullblock']['metrics']['bf16']
    glob = arms['global']['metrics']['bf16']
    rows = []
    for c, b, g in zip(code['per_head'], block['per_head'], glob['per_head']):
        assert c['head'] == b['head'] == g['head']
        denominator = g['nmse']-b['nmse']
        rows.append(dict(head=c['head'], codebook_nmse=c['nmse'], block_nmse=b['nmse'], global_nmse=g['nmse'],
            retained_global_to_block_gap=(g['nmse']-c['nmse'])/denominator if denominator > 0 else None))
    def counts(reference):
        return {name: sum(test(r['codebook_nmse'], r[reference]) for r in rows)
                for name, test in [('better', lambda a,b:a<b), ('equal', lambda a,b:a==b), ('worse', lambda a,b:a>b)]}
    denominator = glob['pooled']['nmse']-block['pooled']['nmse']
    return dict(per_head=rows, versus_global=counts('global_nmse'), versus_block=counts('block_nmse'),
        pooled_retained_global_to_block_gap=(glob['pooled']['nmse']-code['pooled']['nmse'])/denominator if denominator > 0 else None)


def reduce(args, result):
    import torch
    run = json.loads(args.input.read_text())
    assert run['status'] == 'complete', 'Only reduce a completed E033 run; do not hide a failed/stopped control.'
    assert run['actual_dit_calls'] == 0
    assert run['validation_attention_calls'] == 10 and run['benchmark_attention_calls'] == 468
    assert run['actual_counts']['attention'] == 478 and run['consumer_control']['passed']
    assert run['arms'] == list(ARMS) and run['warmups'] == 3 and run['repeats'] == 10
    result.update(source_run=common.record(args.input), recorded_sources=run['sources'],
                  source_verification='Run/output artifacts checked; no rehash of official/generated source tree.',
                  actual_dit_calls=0, native_calls=dict(validation=10, benchmark=468, total=478),
                  counter_limitation='The frozen runner executes adaptive.means_and_key once for each of three validation cases but omits those three calls from its preprocess/k_mean/q_block_mean/q_global_mean counters. Recorded validation/total preparation counters are incomplete and retained as reported. Ten validation native calls and all benchmark counters are unaffected; no rerun or retroactive source change.',
                  validation=[])
    expected_pairs = {(0,'codebook'), (24,'codebook'), (48,'codebook'), (0,'identity'), (0,'zero')}
    assert len(run['validation']) == 5
    assert {(r['block'],r['mode']) for r in run['validation']} == expected_pairs
    for entry in run['validation']:
        assert entry['status'] == 'complete' and entry['actual_counts']['attention'] == 2
        assert entry['actual_counts']['private_attention'] == entry['actual_counts']['official_attention'] == 1
        assert len(entry['inputs']) == 6 and entry['ids']['dtype'] == 'torch.int32'
        expected_rows = {'codebook':16, 'identity':177, 'zero':1}[entry['mode']]
        assert entry['table']['shape'] == [1,H,expected_rows,22656]
        value, ref = load_output(entry['outputs']['private']), load_output(entry['outputs']['old'])
        control, errors = ulp_control(value, ref), metrics(value, ref)
        prior = entry['private_vs_same_input_old']
        for key in ('max_ulp', 'count_gt1_ulp', 'count_nonzero_ulp'):
            assert control[key] == prior[key]
        close(control['max_abs'], prior['max_abs']); close(errors['pooled']['nmse'], prior['pooled_nmse'])
        assert control['count_gt1_ulp'] == 0
        result['validation'].append(dict(block=entry['block'], mode=entry['mode'], outputs=entry['outputs'],
            same_input_receipts=dict(packets=entry['inputs'], table=entry['table'], ids=entry['ids']),
            provenance_boundary='The frozen runner used one shared packet/table set for the pair. Numeric outputs are independently recomputed; GPU packet bytes are not regenerated by this CPU reducer.',
            ulp=control, metrics=errors))
        del value, ref
    assert len(run['benchmark']) == 15
    assert {(r['block'],r['arm']) for r in run['benchmark']} == {(b,a) for b in (0,24,48) for a in ARMS}
    result['layers'] = []
    counts = {key:0 for key in expected_counts('global',1)}
    for source in run['inputs']:
        refs = {name:load_output(ref) for name,ref in source['outputs'].items()}
        assert set(refs) == {'bf16','original_block','original_global','codebook'}
        layer = dict(block=source['block'], references=source['outputs'], arms={})
        for arm in ARMS:
            entry = next(r for r in run['benchmark'] if r['block'] == source['block'] and r['arm'] == arm)
            receipt = timing_receipt(entry)
            output = load_output(entry['output'])
            errors = {name:metrics(output, reference) for name,reference in refs.items()}
            close(errors['bf16']['pooled']['nmse'], entry['versus_bf16']['pooled_nmse'])
            for key in counts:
                counts[key] += receipt['actual_counts'][key]
            layer['arms'][arm] = dict(output=entry['output'], timing=receipt, metrics=errors)
            del output
        layer['observed_frontier'] = observed_frontier(layer['arms'])
        layer['codebook_head_comparison'] = head_comparison(layer['arms'])
        result['layers'].append(layer)
        common.save(args.output,result)
        print(json.dumps(dict(block=source['block'], nondominated=layer['observed_frontier']['nondominated'],
                              codebook_heads=layer['codebook_head_comparison']['versus_global'])),flush=True)
        del refs
    assert counts == run['benchmark_counts']
    assert counts['attention'] == 468 and counts['private_attention'] == 39
    assert counts['official_attention'] == 429 and counts['construct'] == 39
    for key in counts:
        assert run['actual_counts'][key] == run['validation_counts'][key]+counts[key]
    result.update(status='complete', benchmark_counts=counts, validation_counts=run['validation_counts'],
        actual_counts=run['actual_counts'], cuda_initialized=False,
        interpretation='All three cases and 168 heads retained. Quality metric is fixed-packet operator error against saved BF16, not generated-video quality. Preparation and consumer time are included together; no separate-stage timing sums are used.')
    assert not torch.cuda.is_initialized()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=RD/'run.json')
    p.add_argument('--output',type=Path,default=RD/'summary.json')
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not args.output.exists(), 'Preserve existing summary/failed attempt'
    started=time.time()
    result=dict(experiment='E033',status='running',reducer=common.record(__file__))
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


if __name__=='__main__':main()
