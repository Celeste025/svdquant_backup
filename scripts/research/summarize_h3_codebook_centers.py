#!/usr/bin/env python3
"""E032 independent CPU output, cluster-distortion and preparation-cost reduction."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
import summarize_h3_transferred_centers as cpu

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E032'


def timing(row):
    assert row['warmup'] == 3 and row['repeats'] == 10
    out = dict(warmup=3, repeats=10, products=row['products'], memory_bytes=row['memory_bytes'], lifetime=row['lifetime'])
    for field in ('wall_ms', 'cuda_event_ms'):
        xs = row[field]['repeats']; assert len(xs) == 10 and all(math.isfinite(x) and x > 0 for x in xs)
        out[field] = dict(repeats=xs, median=statistics.median(xs), min=min(xs), max=max(xs))
        for key in ('median', 'min', 'max'):
            cpu.close(out[field][key], row[field][key])
    for field in ('allocated', 'reserved'):
        m = row['memory_bytes']; assert m['peak'][field] >= m['baseline'][field]
        assert m['peak'][field] - m['baseline'][field] == m['increment'][field]
    for p in row['products'].values():
        width = {'torch.float32': 4, 'torch.int32': 4, 'torch.bfloat16': 2}[p['dtype']]
        assert math.prod(p['shape']) * width == p['bytes']
    out['returned_product_bytes'] = sum(p['bytes'] for p in row['products'].values())
    return out


def distortion(layer):
    small = cpu.load(layer['centers'])
    specs = dict(centers=((56, 16, 128), torch.bfloat16), ids=((56, 177), torch.int32),
                 mu=((56, 177, 128), torch.bfloat16), global_mu=((56, 1, 128), torch.bfloat16))
    for key, (shape, dtype) in specs.items():
        assert small[key].shape == shape and small[key].dtype == dtype and small[key].isfinite().all()
    ids = small['ids'].long(); assert ids.min().item() >= 0 and ids.max().item() < 16
    mu, global_mu, centers = [small[k].double() for k in ('mu', 'global_mu', 'centers')]
    weights = torch.tensor([128.] * 176 + [11.], dtype=torch.float64)
    rows = []
    for h in range(56):
        counts = torch.bincount(ids[h], minlength=16)
        weighted = torch.bincount(ids[h], weights=weights, minlength=16)
        assert counts.sum().item() == 177 and weighted.sum().item() == 22539
        loss_from_padding = counts.double() * 128 - weighted
        expected_loss = torch.zeros(16, dtype=torch.float64); expected_loss[ids[h, -1]] = 117
        assert torch.equal(loss_from_padding, expected_loss)
        total = ((mu[h] - global_mu[h]).square() * weights[:, None]).sum().item()
        error = ((mu[h] - centers[h, ids[h]]).square() * weights[:, None]).sum().item()
        row = dict(head=h, global_residual_energy=total, postcast_residual_energy=error,
            residual_fraction=error / total if total else None, groups_per_center=counts.tolist(),
            valid_queries_per_center=weighted.tolist(), empty_centers=(counts == 0).sum().item(),
            tail_group_id=ids[h, -1].item(), unique_bf16_center_rows=torch.unique(centers[h], dim=0).shape[0])
        old = layer['distortion']['per_head'][h]
        for key, expected in old.items():
            if isinstance(expected, list):
                assert row[key] == expected
            else:
                cpu.close(row[key], expected)
        rows.append(row)
    total = math.fsum(r['global_residual_energy'] for r in rows)
    pooled = math.fsum(r['postcast_residual_energy'] for r in rows) / total
    cpu.close(total, layer['distortion']['global_residual_energy'])
    cpu.close(pooled, layer['distortion']['energy_weighted_residual_fraction'])
    return dict(artifact=layer['centers'], global_residual_energy=total,
        energy_weighted_residual_fraction=pooled, per_head=rows,
        head_residual_distribution=dict(min=min(r['residual_fraction'] for r in rows),
            median=statistics.median(r['residual_fraction'] for r in rows), max=max(r['residual_fraction'] for r in rows)),
        total_empty_centers=sum(r['empty_centers'] for r in rows), heads_with_empty_centers=sum(r['empty_centers'] > 0 for r in rows),
        unique_center_row_range=[min(r['unique_bf16_center_rows'] for r in rows), max(r['unique_bf16_center_rows'] for r in rows)],
        tail_weight=11, valid_queries=22539, id_range=[ids.min().item(), ids.max().item()],
        boundary='Actual saved BF16 mu/global/C and int32 ids used directly for independent FP64 distortion/counts. No clustering rerun or FP64-nearest reassignment; production assignment remains FP32 norm+bmm.')


def effects(values):
    b, g, c = [values[k] for k in ('original_block', 'original_global', 'codebook')]
    gap = g - b
    return dict(nmse=values, baseline_global_minus_block=gap, positive_baseline_gap=gap > 0,
        retained_gap={key: (g - value) / gap if gap else None for key, value in values.items()},
        codebook_minus_reference={key: c - value for key, value in values.items() if key != 'codebook'},
        codebook_relative_to_reference={key: c / value - 1 if value else None for key, value in values.items() if key != 'codebook'})


def summarize(report):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(6)
    source = json.loads((RD / 'run.json').read_text()); launch = json.loads((RD / 'launcher.json').read_text())
    check = json.loads((RD / 'check.json').read_text())
    assert source['status'] == launch['status'] == check['status'] == 'complete' and launch['returncode'] == 0
    assert cpu.record(RD / 'run.json')['sha256'] == launch['result_sha256']
    assert source['attention_calls'] == 3 and source['actual_dit_calls'] == 0
    assert source['clusters'] == 16 and source['lloyd_iterations'] == 6
    for key in ('sources', 'references', 'inputs'):
        assert source[key] == check[key]
    for ref in source['sources'].values():
        assert cpu.record(ref['file']) == ref
    old = {}
    for name in ('oracle', 'transfer', 'adaptive'):
        ref = source['references'][name]; assert cpu.record(ref['file']) == ref
        old[name] = json.loads(Path(ref['file']).read_text())
    report.update(inputs={name: cpu.record(RD / (name + '.json')) for name in ('run', 'launcher', 'check')},
        inherited_source_bindings=source['sources'], environment=source['environment'], geometry=source['geometry'],
        timing_boundary=source['timing_boundary'], layers=[], compact_accuracy=[], compact_timing=[])
    total_scopes = 0
    for layer, inputs in zip(source['layers'], source['inputs'], strict=True):
        assert layer['block'] == inputs['block']
        r29 = next(l['arms'] for l in old['oracle']['layers'] if l['block'] == layer['block'])
        r30 = next(l['arms'] for l in old['transfer']['layers'] if l['block'] == layer['block'])
        r31 = next(l['outputs'] for l in old['adaptive']['layers'] if l['block'] == layer['block'])
        prior = {key: r29[key] for key in ('original_block', 'original_global', 'euclidean')}
        prior.update(transferred_factorable=r30['factorable_fp32'], adaptive_range1=r31['range1'])
        for key, row in prior.items():
            assert inputs['outputs'][key] == row['output']
        refs = {key: cpu.output_load(ref) for key, ref in inputs['outputs'].items()}
        value = cpu.output_load(layer['output']); actual = cpu.metrics(value, refs)
        for key in refs:
            cpu.compare_report(actual[key], layer['metrics'][key])
        out = dict(block=layer['block'], output=layer['output'], references=inputs['outputs'], metrics=actual,
                   distortion=distortion(layer), reference_vs_bf16={}, timings={})
        report['layers'].append(out)
        for key, row in prior.items():
            m = cpu.metrics(refs[key], {'bf16': refs['bf16']})['bf16']
            cpu.compare_report(m, row['metrics']['bf16']); out['reference_vs_bf16'][key] = m
        allmetrics = dict(out['reference_vs_bf16'], codebook=actual['bf16'])
        out['pooled_effect'] = effects({key: row['pooled']['nmse'] for key, row in allmetrics.items()})
        heads = [dict(head=h, **effects({key: row['per_head'][h]['nmse'] for key, row in allmetrics.items()})) for h in range(56)]
        out['per_head_effect'] = heads
        out['head_counts'] = {ref: dict(improved=sum(h['codebook_minus_reference'][ref] < 0 for h in heads),
            worsened=sum(h['codebook_minus_reference'][ref] > 0 for h in heads), equal=sum(h['codebook_minus_reference'][ref] == 0 for h in heads)) for ref in prior}
        for method, scopes in layer['timings'].items():
            assert set(scopes) == ({'resident_means', 'resident_raw_qk'} if method == 'codebook' else {'resident_raw_qk'})
            out['timings'][method] = {scope: timing(row) for scope, row in scopes.items()}
            total_scopes += len(scopes)
            for scope, row in out['timings'][method].items():
                report['compact_timing'].append(dict(block=layer['block'], method=method, scope=scope,
                    cuda_ms=row['cuda_event_ms']['median'], cuda_min_ms=row['cuda_event_ms']['min'], cuda_max_ms=row['cuda_event_ms']['max'],
                    wall_ms=row['wall_ms']['median'], memory_bytes=row['memory_bytes'], returned_product_bytes=row['returned_product_bytes']))
        report['compact_accuracy'].append(dict(block=layer['block'], **out['pooled_effect'], head_counts=out['head_counts']))
        del refs, value
    assert total_scopes == 12 and [l['block'] for l in report['layers']] == [0, 24, 48]
    report['counts'] = dict(actual_native_attention=3, actual_dit=0, completed_timing_scopes=12,
        measured_invocations=120, warmup_invocations=36, independent_clustering_runs=0)
    report['preserved_cpu_fixture_history'] = {name: cpu.record(RD / name) for name in
        ('source_before_fixture_tie_fix.py', 'check_before_fixture_tie_fix.json', 'fixture_tie_diagnostic.json')}
    report['limitations'] = [
        'Current accuracy path gathers full G-row correction for the old kernel. A C/id/T16 consumer is not implemented or benchmarked.',
        'Candidate returns C/id/Qcenter/T16; global/block return Qcenter plus their correction. Preparation times have different output products and cannot be subtracted to predict end-to-end speedup; two scopes are not additive.',
        'Memory baseline includes resident fixtures; allocated increments are peak-minus-baseline and reserved reflects warm allocator history. Product bytes are not peaks.',
        'Clustering was not rerun. Independent FP64 distortion uses the actual saved centers and assignment; an FP64 nearest-center requirement would be a different numerical contract.',
        'Single prompt/step and three layers, fixed K16/six Lloyd iterations. Tensor NMSE gap retention is not video quality or generalization.',
        'Preserved pre-GPU CPU fixture failure arose from norm+bmm versus direct-distance ordering of tied farthest points. Only fixture input changed; production algorithm was unchanged.']
    assert not torch.cuda.is_initialized()
    report.update(status='complete', cuda_initialized=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RD / 'independent_summary.json')
    args = p.parse_args(); assert not args.output.exists(), 'Preserve prior attempt'
    report = dict(experiment='E032', status='running', reducer=cpu.record(__file__), cpu_metrics_helper=cpu.record(cpu.__file__))
    start = time.time()
    try:
        summarize(report)
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc()); raise
    finally:
        report['cpu_seconds'] = time.time() - start
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(dict(status=report['status'], output=str(args.output), sha256=cpu.record(args.output)['sha256'])))


if __name__ == '__main__':
    cpu.DEADLINE = time.monotonic() + 300
    import torch
    cpu.torch = torch
    main()
