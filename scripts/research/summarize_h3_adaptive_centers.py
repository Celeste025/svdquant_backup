#!/usr/bin/env python3
"""E031 thin CPU output/timing reduction, reusing verified E030 FP64 helpers."""
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
RD = ROOT / 'results/research/E031'
METHODS = ('full_svd', 'range0', 'range1')


def timing(row):
    assert row['warmup'] == 3 and row['repeats'] == 10
    out = dict(warmup=3, repeats=10, memory_bytes=row['memory_bytes'], products=row['products'], lifetime=row['lifetime'])
    for field in ('wall_ms', 'cuda_event_ms'):
        xs = row[field]['repeats']
        assert len(xs) == 10 and all(math.isfinite(x) and x > 0 for x in xs)
        out[field] = dict(repeats=xs, median=statistics.median(xs), min=min(xs), max=max(xs))
        for key in ('median', 'min', 'max'):
            cpu.close(out[field][key], row[field][key])
    for field in ('allocated', 'reserved'):
        m = row['memory_bytes']
        assert m['peak'][field] >= m['baseline'][field]
        assert m['peak'][field] - m['baseline'][field] == m['increment'][field]
    for p in row['products'].values():
        width = {'torch.float32': 4, 'torch.bfloat16': 2}[p['dtype']]
        assert math.prod(p['shape']) * width == p['bytes']
    out['returned_product_bytes'] = sum(p['bytes'] for p in row['products'].values())
    return out


def factors_and_projection(row):
    tensors = cpu.load(row['factors'])
    assert set(tensors) == {'a', 'b', 'center32'}
    for key, shape in dict(a=(56, 177, 16), b=(56, 16, 128), center32=(56, 177, 128)).items():
        assert tensors[key].shape == shape and tensors[key].dtype == torch.float32 and tensors[key].isfinite().all()
    b = tensors['b'].double()
    orth = (b @ b.transpose(-2, -1) - torch.eye(16, dtype=torch.float64)).abs().amax((1, 2))
    p = row['projection']; heads = p['per_head']
    assert [h['head'] for h in heads] == list(range(56))
    for h, v in enumerate(heads):
        cpu.close(orth[h].item(), v['basis_orthogonality_max_abs'])
        cpu.close(v['projection_energy'] / v['total_energy'], v['projection_fraction'])
        cpu.close(v['center32_residual_energy'] / v['total_energy'], v['center32_residual_fraction'])
    total = math.fsum(v['total_energy'] for v in heads)
    proj = math.fsum(v['projection_energy'] for v in heads) / total
    center = math.fsum(v['center32_residual_energy'] for v in heads) / total
    cpu.close(total, p['total_energy'])
    cpu.close(proj, p['energy_weighted_projection_fraction'])
    cpu.close(center, p['energy_weighted_center32_residual_fraction'])
    return dict(factors=row['factors'], shapes_dtypes_finite_valid=True,
        orthogonality_max_abs=orth.max().item(), orthogonality_per_head=orth.tolist(),
        total_energy=total, energy_weighted_projection_fraction=proj,
        energy_weighted_center32_residual_fraction=center,
        per_head_scalar_projection=heads,
        boundary='Basis orthogonality recomputed from actual saved FP32 factors in FP64; projection energy/fraction and aggregate scalar consistency checked. This run did not save GPU mu/global; no claim of recomputing projection energies from unstored means.')


def effects(nmse):
    block, global_ = nmse['original_block'], nmse['original_global']
    gap = global_ - block
    return dict(nmse=nmse, baseline_global_minus_block=gap,
        retained_gap={key: (global_ - value) / gap if gap else None for key, value in nmse.items()},
        positive_baseline_gap=gap > 0)


def summarize(report):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(6)
    source = json.loads((RD / 'run.json').read_text())
    launcher = json.loads((RD / 'launcher.json').read_text())
    check = json.loads((RD / 'check.json').read_text())
    assert source['status'] == launcher['status'] == check['status'] == 'complete'
    assert launcher['returncode'] == 0 and cpu.record(RD / 'run.json')['sha256'] == launcher['result_sha256']
    assert source['attention_calls'] == 9 and source['actual_dit_calls'] == 0
    assert source['sources'] == check['sources'] and source['inputs'] == check['inputs']
    assert cpu.record(source['sources']['runner']['file']) == source['sources']['runner']
    assert {k: v for k, v in source['omega'].items() if k != 'artifact'} == check['omega']
    omega = cpu.load(source['omega']['artifact'])
    assert omega.shape == (56, 177, 16) and omega.dtype == torch.float32 and omega.isfinite().all()
    assert cpu.tensor_sha(omega) == source['omega']['tensor']['sha256']
    assert source['omega']['seed'] == 20261031
    # Validate the frozen draw without sampling another random matrix or fitting a basis.
    r29ref = source['references']['E029/run.json']; assert cpu.record(r29ref['file']) == r29ref
    r29 = json.loads(Path(r29ref['file']).read_text())
    r30ref = source['references']['E030/run.json']; assert cpu.record(r30ref['file']) == r30ref
    r30 = json.loads(Path(r30ref['file']).read_text())
    report.update(inputs={p: cpu.record(RD / (p + '.json')) for p in ('run', 'launcher', 'check')},
        inherited_source_bindings=source['sources'], environment=source['environment'], geometry=source['geometry'],
        omega=source['omega'], frozen_omega_matches_check=True,
        timing_boundary=source['timing_boundary'], layers=[], compact_accuracy=[], compact_timing=[])
    scope_count = 0
    for layer, inputs in zip(source['layers'], source['inputs'], strict=True):
        assert layer['block'] == inputs['block'] and tuple(layer['outputs']) == METHODS
        refs = {key: cpu.output_load(value) for key, value in inputs['outputs'].items()}
        old29 = next(l['arms'] for l in r29['layers'] if l['block'] == layer['block'])
        old30 = next(l['arms'] for l in r30['layers'] if l['block'] == layer['block'])
        for key in ('original_block', 'original_global', 'euclidean'):
            assert inputs['outputs'][key] == old29[key]['output']
        assert inputs['outputs']['transferred_factorable'] == old30['factorable_fp32']['output']
        out = dict(block=layer['block'], references=inputs['outputs'], timings={}, outputs={}, reference_vs_bf16={})
        report['layers'].append(out)
        for method, scopes in layer['timings'].items():
            assert set(scopes) == ({'resident_means', 'resident_raw_qk'} if method in METHODS else {'resident_raw_qk'})
            out['timings'][method] = {scope: timing(value) for scope, value in scopes.items()}
            scope_count += len(scopes)
            for scope, value in out['timings'][method].items():
                report['compact_timing'].append(dict(block=layer['block'], method=method, scope=scope,
                    cuda_ms=value['cuda_event_ms']['median'], cuda_min_ms=value['cuda_event_ms']['min'], cuda_max_ms=value['cuda_event_ms']['max'],
                    wall_ms=value['wall_ms']['median'], returned_product_bytes=value['returned_product_bytes'], memory_bytes=value['memory_bytes']))
        for name in ('original_block', 'original_global', 'euclidean', 'transferred_factorable'):
            m = cpu.metrics(refs[name], {'bf16': refs['bf16']})['bf16']
            old = old30['factorable_fp32']['metrics']['bf16'] if name == 'transferred_factorable' else old29[name]['metrics']['bf16']
            cpu.compare_report(m, old); out['reference_vs_bf16'][name] = m
        for method in METHODS:
            row = layer['outputs'][method]; value = cpu.output_load(row['output'])
            m = cpu.metrics(value, refs)
            for key in refs:
                cpu.compare_report(m[key], row['metrics'][key])
            out['outputs'][method] = dict(output=row['output'], metrics=m, projection=factors_and_projection(row))
            del value
        bf16_metrics = dict(out['reference_vs_bf16'], **{k: v['metrics']['bf16'] for k, v in out['outputs'].items()})
        out['pooled_effect'] = effects({k: v['pooled']['nmse'] for k, v in bf16_metrics.items()})
        heads = [dict(head=h, **effects({k: v['per_head'][h]['nmse'] for k, v in bf16_metrics.items()})) for h in range(56)]
        out['per_head_effect'] = heads
        out['head_comparison_counts'] = {method: {reference: dict(
            improved=sum(h['nmse'][method] < h['nmse'][reference] for h in heads),
            worsened=sum(h['nmse'][method] > h['nmse'][reference] for h in heads),
            equal=sum(h['nmse'][method] == h['nmse'][reference] for h in heads))
            for reference in ('original_block', 'original_global', 'euclidean', 'transferred_factorable', 'full_svd')}
            for method in METHODS}
        out['method_order'] = sorted(METHODS, key=lambda name: bf16_metrics[name]['pooled']['nmse'])
        report['compact_accuracy'].append(dict(block=layer['block'], **out['pooled_effect'], method_order=out['method_order'], head_counts=out['head_comparison_counts']))
        del refs
    assert [l['block'] for l in report['layers']] == [0, 24, 48] and scope_count == 24
    report['counts'] = dict(actual_native_attention=9, actual_dit=0, measured_scopes=24,
        measured_invocations=240, warmup_invocations=72,
        note='24 completed timing records each contain 3 warmups and 10 raw measurements; 9 separate accuracy forwards. No extra summary GPU calls.')
    report['limitations'] = [
        'Resident-means and resident-raw-QK scopes are separate complete measurements, never additive. Neither includes FP4 packing or attention.',
        'Candidate raw-QK output is A/B/c32/Qcenter/T17; global/block return Qcenter plus their actual correction. Different products prohibit predicting whole-consumer speedup by subtracting these timings.',
        'Returned product bytes are exact tensor sizes, not peak memory. Peak/baseline includes resident fixtures; allocated increments are arithmetic peak-minus-baseline, and reserved is warm allocator history.',
        'All outputs independently read and reduced in CPU FP64. Projection scalar consistency and basis orthogonality are checked without recomputing SVD or unstored GPU mean tensors.',
        'Single input, three layers, one frozen random draw. Torch SVD/QR preparation costs constrain this implementation; they are not lower bounds for all optimized kernels.',
        'Retained gap is a tensor NMSE-difference ratio, not video quality. E029 Euclidean used FP64 fitting and rounded BF16 centers; comparing it with these FP32 factorable centers also changes numerical contract.']
    assert not torch.cuda.is_initialized()
    report.update(status='complete', cuda_initialized=False, independent_svd_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RD / 'independent_summary.json')
    args = p.parse_args(); assert not args.output.exists(), 'Preserve prior attempt'
    report = dict(experiment='E031', status='running', reducer=cpu.record(__file__), cpu_metrics_helper=cpu.record(cpu.__file__))
    started = time.time()
    try:
        summarize(report)
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc()); raise
    finally:
        report['cpu_seconds'] = time.time() - started
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(dict(status=report['status'], output=str(args.output), sha256=cpu.record(args.output)['sha256'])))


if __name__ == '__main__':
    # The frozen E030 helper imports torch only in its CLI. Initialize its CPU
    # dependency and shared deadline explicitly; its source remains unchanged.
    cpu.DEADLINE = time.monotonic() + 300
    import torch
    cpu.torch = torch
    main()
