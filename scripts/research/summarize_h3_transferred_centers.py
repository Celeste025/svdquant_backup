#!/usr/bin/env python3
"""E030 saved-output CPU reduction; no donor fitting, SVD or GPU execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E030'
ARMS = ('bf16_center', 'factorable_fp32')


def budget():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError('300-second CPU reducer budget')


def record(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(file=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def load(ref):
    budget(); assert record(ref['file']) == ref
    return torch.load(ref['file'], map_location='cpu', weights_only=True, mmap=True)


def output_load(ref):
    value = load(ref['artifact'])
    assert value.shape == (22539, 56, 128) and value.dtype == torch.bfloat16
    assert tensor_sha(value) == ref['tensor']['sha256']
    return value


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def metrics(value, references):
    sums = {name: torch.zeros((5, 56), dtype=torch.float64) for name in references}
    maxima = {name: torch.zeros(56, dtype=torch.float64) for name in references}
    for start in range(0, len(value), 256):
        budget(); a = value[start:start + 256].double(); assert a.isfinite().all()
        for name, ref in references.items():
            b = ref[start:start + 256].double(); assert b.isfinite().all()
            d = a - b
            sums[name] += torch.stack((d.square().sum((0, 2)), b.square().sum((0, 2)),
                a.square().sum((0, 2)), (a * b).sum((0, 2)),
                torch.full((56,), a.shape[0] * 128, dtype=torch.float64)))
            maxima[name] = torch.maximum(maxima[name], d.abs().amax((0, 2)))
    def finish(x, maximum):
        e, r, v, dot, count = x.tolist()
        return dict(error_energy=e, reference_energy=r, value_energy=v, dot=dot, elements=count,
                    nmse=e / r if r else None, cosine=dot / math.sqrt(v * r) if v * r else None,
                    rms_error=math.sqrt(e / count), max_abs=maximum)
    return {name: dict(pooled=finish(x.sum(1), maxima[name].max().item()),
        per_head=[dict(head=h, **finish(x[:, h], maxima[name][h].item())) for h in range(56)]) for name, x in sums.items()}


def compare_report(got, expected):
    for scope in ('pooled', 'per_head'):
        pairs = [(got[scope], expected[scope])] if scope == 'pooled' else zip(got[scope], expected[scope], strict=True)
        for actual, prior in pairs:
            for key, value in prior.items():
                close(actual[key], value)


def effect(values):
    b, g, e, t, f = [values[k] for k in ('original_block', 'original_global', 'euclidean', *ARMS)]
    gap = g - b
    return dict(nmse=values, baseline_global_minus_block=gap,
        retained_gap={k: (g - v) / gap if gap else None for k, v in values.items()},
        transfer_cost_nmse=t - e, transfer_cost_relative_to_same_sample=t / e - 1 if e else None,
        fp32_contract_cost_nmse=f - t, fp32_contract_cost_relative_to_bf16_center=f / t - 1 if t else None,
        positive_baseline_gap=gap > 0)


def small_artifacts(source, layer):
    basis_row = next(b for b in source['bases'] if b['block'] == layer['block'])
    donor = load(basis_row['artifact']); target = load(layer['factors'])
    assert donor['source_case'] == dict(prompt_id=30, seed=49771, step=14, arm='block_mean')
    assert donor['basis'].shape == (56, 16, 128) and donor['basis'].dtype == torch.float64
    assert donor['mu'].shape == (56, 174, 128) and donor['global_mu'].shape == (56, 1, 128)
    assert torch.equal(donor['weights'], torch.tensor([128.] * 173 + [83.], dtype=torch.float64))
    assert int(donor['weights'].sum()) == 22227
    assert torch.equal(target['basis'], donor['basis'])
    assert target['mu'].shape == (1, 56, 177, 128) and target['global_mu'].shape == (1, 56, 1, 128)
    assert all(t.isfinite().all() for t in (donor['basis'], donor['mu'], donor['global_mu'], target['mu'], target['global_mu']))
    orth = (donor['basis'] @ donor['basis'].transpose(-2, -1) - torch.eye(16, dtype=torch.float64)).abs().amax((1, 2))
    assert orth.max().item() < 1e-10
    for h, row in enumerate(basis_row['head_statistics']):
        assert row['head'] == h
        close(orth[h].item(), row['orthogonality_max_abs'])
    c = target['arms']['bf16_center']; f = target['arms']['factorable_fp32']
    assert c['center_fp32'].shape == c['center_bf16'].shape == f['center_fp32'].shape == (1, 56, 177, 128)
    assert c['center_fp32'].dtype == f['center_fp32'].dtype == torch.float32 and c['center_bf16'].dtype == torch.bfloat16
    assert f['a_fp32'].shape == (1, 56, 177, 16) and f['a_fp32'].dtype == torch.float32
    assert f['basis_fp32'].shape == (56, 16, 128) and f['basis_fp32'].dtype == torch.float32
    assert torch.equal(f['basis_fp32'], donor['basis'].float())
    assert torch.equal(c['center_bf16'], c['center_fp32'].to(torch.bfloat16))
    assert all(t.isfinite().all() for arm in target['arms'].values() for t in arm.values())
    return dict(donor_artifact=basis_row['artifact'], target_artifact=layer['factors'],
        donor_valid_tokens=22227, donor_padded_tokens=22272, donor_groups=174, donor_tail_weight=83,
        target_valid_tokens=22539, target_groups=177, target_tail_weight=11,
        orthogonality_max_abs=orth.max().item(), orthogonality_by_head=orth.tolist(),
        same_saved_basis_exact=True, factor_shapes_valid=True, center_bf16_cast_exact=True,
        svd_recomputed=False)


def summarize(report):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(6)
    source = json.loads((RD / 'run.json').read_text())
    launcher = json.loads((RD / 'launcher.json').read_text())
    check = json.loads((RD / 'check.json').read_text())
    assert source['status'] == launcher['status'] == check['status'] == 'complete'
    assert launcher['returncode'] == 0 and record(RD / 'run.json')['sha256'] == launcher['result_sha256']
    assert source['sources'] == check['sources'] and source['targets'] == check['targets']
    assert record(source['sources']['runner']['file']) == source['sources']['runner']
    assert source['attempted_dit_calls'] == source['complete_dit_calls'] == 1
    assert source['capture_attention_calls'] == 50 and source['attention_probe_calls'] == 6 and source['actual_attention_calls'] == 56
    assert source['donor']['attention_contract'] == dict(expected_cu=[0, 22227, 22272], expected_refiner_cu=[0, 501, 501])
    assert source['donor']['signature_matches_saved_call'] is True
    assert source['donor']['raw_reference_signature'] == source['donor']['saved_raw_signature']
    runtime = source['donor']['runtime']; packing = source['donor']['packing']
    assert runtime['sdpa_calls'] == 52 and runtime['scaled_mm_calls'] == 200 and runtime['disk_loads'] == 0
    assert packing['checked_calls'] == 200 and packing['invalid_calls'] == 0
    e029ref = source['target_reports']['E029/run.json']
    assert record(e029ref['file']) == e029ref
    e029 = json.loads(Path(e029ref['file']).read_text())
    report.update(inputs={p: record(RD / (p + '.json')) for p in ('run', 'launcher', 'check')},
        inherited_source_bindings=source['sources'], donor_case=source['donor']['case'],
        donor_runtime=runtime, donor_packing=packing, donor_raw_drift_reported=source['donor']['raw_drift'],
        donor_raw_sha_exact={m: r['actual']['sha256'] == source['donor']['saved_raw_signature'][m]['sha256'] for m, r in source['donor']['raw_drift'].items()},
        actual_calls=dict(dit=1, capture_fp4_attention=50, probe_fp4_attention=6, capture_bf16_sdpa=52),
        allocated_after_model_release_gib_reported=source['allocated_after_model_release_gib'],
        peak_allocated_gib_reported=source['peak_allocated_gib'], layers=[], compact_table=[])
    for layer, target in zip(source['layers'], source['targets'], strict=True):
        assert layer['block'] == target['block'] and list(layer['arms']) == list(ARMS)
        prior = next(l for l in e029['layers'] if l['block'] == layer['block'])
        for name in ('original_block', 'original_global', 'euclidean'):
            assert target['references'][name] == prior['arms'][name]['output']
        refs = {name: output_load(ref) for name, ref in target['references'].items()}
        out = dict(block=layer['block'], small_artifacts=small_artifacts(source, layer), references=target['references'], arms={})
        report['layers'].append(out)
        for name in ('original_block', 'original_global', 'euclidean'):
            m = metrics(refs[name], {'bf16': refs['bf16']})['bf16']
            compare_report(m, prior['arms'][name]['metrics']['bf16'])
            out['arms'][name] = dict(versus_bf16=m)
        new = {name: output_load(layer['arms'][name]['output']) for name in ARMS}
        for name in ARMS:
            m = metrics(new[name], refs)
            for key in refs:
                compare_report(m[key], layer['arms'][name]['metrics'][key])
            out['arms'][name] = dict(output=layer['arms'][name]['output'], versus_bf16=m['bf16'], metrics=m,
                correction_association_drift_reported=layer['arms'][name]['correction_fp32_association_drift'])
        cross = metrics(new['factorable_fp32'], {'bf16_center': new['bf16_center']})['bf16_center']
        compare_report(cross, layer['arms']['factorable_fp32']['versus_bf16_center'])
        out['between_new_arms'] = cross
        out['pooled_effect'] = effect({n: a['versus_bf16']['pooled']['nmse'] for n, a in out['arms'].items()})
        heads = [dict(head=h, **effect({n: a['versus_bf16']['per_head'][h]['nmse'] for n, a in out['arms'].items()})) for h in range(56)]
        out['per_head_effect'] = heads
        out['head_counts'] = {cost: dict(improved=sum(h[cost] < 0 for h in heads), worsened=sum(h[cost] > 0 for h in heads),
            equal=sum(h[cost] == 0 for h in heads)) for cost in ('transfer_cost_nmse', 'fp32_contract_cost_nmse')}
        out['head_counts']['positive_global_minus_block_gap'] = sum(h['positive_baseline_gap'] for h in heads)
        out['new_arm_vs_baseline_head_counts'] = {name: {baseline: dict(
            improved=sum(h['nmse'][name] < h['nmse'][baseline] for h in heads),
            worsened=sum(h['nmse'][name] > h['nmse'][baseline] for h in heads),
            equal=sum(h['nmse'][name] == h['nmse'][baseline] for h in heads))
            for baseline in ('original_block', 'original_global')} for name in ARMS}
        report['compact_table'].append(dict(block=layer['block'], **out['pooled_effect'], head_counts=out['head_counts']))
        del refs, new
    assert [l['block'] for l in report['layers']] == [0, 24, 48]
    report['limitations'] = [
        'Transfer cost compares transferred BF16-center output with E029 same-sample Euclidean rank16 at the same target. Factorable-FP32 cost compares the two new arms and includes coefficient precision, subtraction, center rounding and multiplication association changes.',
        'Retained gap is (NMSE_global-NMSE_candidate)/(NMSE_global-NMSE_block), a tensor-error ratio, not video quality. Head signs are retained; no mean head ratio or arbitrary gate.',
        'One donor p30 and one target p36, same timestep/resolution; both are historical research examples. No held-out generalization or deployable basis claim.',
        'FP32 factorable arm still materializes full correction. No measured 17-row consumer, speed or memory savings.',
        'Saved outputs, references and small factors independently read; no SVD or model rerun. Donor forward drift, correction-association drift and GPU memory are report-bound observations, not independently regenerated tensors.']
    assert not torch.cuda.is_initialized()
    report.update(status='complete', cuda_initialized=False, independent_svd_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RD / 'independent_summary.json')
    args = p.parse_args(); assert not args.output.exists(), 'Preserve prior attempt'
    report = dict(experiment='E030', status='running', reducer=record(__file__)); started = time.time()
    try:
        summarize(report)
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        report['cpu_seconds'] = time.time() - started
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(dict(status=report['status'], output=str(args.output), sha256=record(args.output)['sha256'])))


if __name__ == '__main__':
    DEADLINE = time.monotonic() + 300
    import torch
    main()
