#!/usr/bin/env python3
"""E029 independent saved-output CPU reduction; no SVD and no GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E029'
GEOMETRIES = ('euclidean', 'kc_score', 'k4_error_score')
ARMS = ('original_block', 'original_global', *GEOMETRIES)


def record(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(file=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def load(ref):
    budget()
    assert record(ref['file']) == ref
    return torch.load(ref['file'], map_location='cpu', mmap=True, weights_only=True)


def output_load(row):
    value = load(row['artifact'])
    assert value.shape == (22539, 56, 128) and value.dtype == torch.bfloat16
    assert tensor_sha(value) == row['tensor']['sha256']
    return value


def metrics(value, references):
    sums = {name: torch.zeros((5, 56), dtype=torch.float64) for name in references}
    maxabs = {name: torch.zeros(56, dtype=torch.float64) for name in references}
    for start in range(0, len(value), 256):
        budget(); a = value[start:start + 256].double()
        assert a.isfinite().all()
        for name, ref in references.items():
            b = ref[start:start + 256].double()
            assert b.isfinite().all()
            d = a - b
            sums[name] += torch.stack((d.square().sum((0, 2)), b.square().sum((0, 2)),
                a.square().sum((0, 2)), (a * b).sum((0, 2)),
                torch.full((56,), a.shape[0] * 128, dtype=torch.float64)))
            maxabs[name] = torch.maximum(maxabs[name], d.abs().amax((0, 2)))
    def finish(x, maximum):
        e, r, v, dot, count = x.tolist()
        return dict(error_energy=e, reference_energy=r, value_energy=v, dot=dot, elements=count,
                    nmse=e / r if r else None, cosine=dot / math.sqrt(v * r) if v * r else None,
                    rms_error=math.sqrt(e / count), max_abs=maximum)
    return {name: dict(pooled=finish(x.sum(1), maxabs[name].max().item()),
                      per_head=[dict(head=h, **finish(x[:, h], maxabs[name][h].item())) for h in range(56)])
            for name, x in sums.items()}


def projections(layer):
    c = load(layer['centers'])
    shapes = dict(block_mean=(56, 177, 128), global_mean=(56, 1, 128),
                  k_mean=(56, 1, 128), weights=(177,))
    for key, shape in shapes.items():
        assert c[key].shape == shape and c[key].isfinite().all()
    assert torch.equal(c['weights'], torch.tensor([128.] * 176 + [11.], dtype=torch.float64))
    assert tensor_sha(c['block_mean'].unsqueeze(0)) == layer['actual_gpu_mu']['sha256']
    assert tensor_sha(c['global_mean'].unsqueeze(0)) == layer['actual_gpu_global_mu']['sha256']
    result = {}
    for name in GEOMETRIES:
        small = c['restricted'][name]
        expected = dict(a=((56, 177, 16), torch.float64), b=((56, 16, 128), torch.float64),
                        center_fp32=((56, 177, 128), torch.float32), center_bf16=((56, 177, 128), torch.bfloat16))
        for key, (shape, dtype) in expected.items():
            assert small[key].shape == shape and small[key].dtype == dtype and small[key].isfinite().all()
        assert torch.equal(small['center_fp32'].to(torch.bfloat16), small['center_bf16'])
        rows = layer['arms'][name]['projection']
        assert [r['head'] for r in rows] == list(range(56))
        for h in rows:
            close(h['theoretical_tail_energy'] / h['theoretical_total_energy'], h['theoretical_tail_fraction'])
            close(h['theoretical_tail_energy'], h['ideal_projection']['error_energy'])
            for key in ('ideal_projection', 'fp32_reconstruction', 'post_bf16_center'):
                r = h[key]
                close(r['total_energy'], h['theoretical_total_energy'])
                close(r['fraction'], r['error_energy'] / r['total_energy'])
            actual_rounding = (small['center_bf16'][h['head']].float() - small['center_fp32'][h['head']]).abs().max().item()
            close(actual_rounding, h['bf16_rounding_max_abs'])
        total = math.fsum(r['theoretical_total_energy'] for r in rows)
        reduced = dict(total_energy=total,
            theoretical_tail_fraction=math.fsum(r['theoretical_tail_energy'] for r in rows) / total,
            **{key: math.fsum(r[key]['error_energy'] for r in rows) / total
               for key in ('ideal_projection', 'fp32_reconstruction', 'post_bf16_center')})
        for key, value in reduced.items():
            close(value, layer['arms'][name]['energy_weighted_projection'][key])
        result[name] = dict(**reduced, factors_shapes_and_dtypes_valid=True, center_cast_exact=True,
            verification_scope='Saved scalar tail/projection consistency and factor shapes/cast; no covariance, SVD, or new projection recomputation.')
    return result


def comparison(candidate, block, global_):
    advantage = global_ - block
    return dict(candidate_nmse=candidate, block_nmse=block, global_nmse=global_,
        candidate_minus_block=candidate - block, candidate_minus_global=candidate - global_,
        candidate_over_block=candidate / block if block else None,
        retained_global_minus_block_advantage=(global_ - candidate) / advantage if advantage else None,
        baseline_global_minus_block=advantage,
        baseline_advantage_positive=advantage > 0)


def summarize(report):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(6)
    source = json.loads((RD / 'run.json').read_text())
    launcher = json.loads((RD / 'launcher.json').read_text())
    check = json.loads((RD / 'check.json').read_text())
    assert source['status'] == launcher['status'] == check['status'] == 'complete'
    assert launcher['returncode'] == 0 and record(RD / 'run.json')['sha256'] == launcher['result_sha256']
    assert source['attention_calls'] == 15 and source['actual_dit_calls'] == 0 and source['rank'] == 16
    assert source['sources'] == check['sources'] and source['inputs'] == check['inputs']
    assert record(source['sources']['runner']['file']) == source['sources']['runner']
    report.update(inputs={p: record(RD / (p + '.json')) for p in ('run', 'launcher', 'check')},
        inherited_source_bindings=source['sources'], geometry=source['geometry'], scope=source['scope'],
        actual_attention_calls=15, actual_dit_calls=0, layers=[], compact_table=[])
    for layer, input_row in zip(source['layers'], source['inputs'], strict=True):
        assert layer['block'] == input_row['block'] and list(layer['arms']) == list(ARMS)
        refs = {key: output_load(r) for key, r in input_row['references'].items()}
        out = dict(block=layer['block'], references=input_row['references'], arms={},
                   center_artifact=layer['centers'], projection=projections(layer))
        report['layers'].append(out)
        for name in ARMS:
            row = layer['arms'][name]; value = output_load(row['output'])
            computed = metrics(value, refs)
            for refname, computed_row in computed.items():
                old = row['metrics'][refname]
                for scope in ('pooled', 'per_head'):
                    pairs = [(computed_row[scope], old[scope])] if scope == 'pooled' else zip(computed_row[scope], old[scope], strict=True)
                    for got, expected in pairs:
                        for key, val in expected.items():
                            close(got[key], val)
            out['arms'][name] = dict(output=row['output'], metrics=computed)
            if name in ('original_block', 'original_global'):
                refname = 'block_mean' if name == 'original_block' else 'global_mean'
                out['arms'][name]['historical_endpoint_drift'] = dict(
                    tensor_sha_exact=row['output']['tensor']['sha256'] == input_row['references'][refname]['tensor']['sha256'],
                    metrics=computed[refname])
            del value
        bf = {n: out['arms'][n]['metrics']['bf16'] for n in ARMS}
        out['candidate_comparison'] = {}
        for name in GEOMETRIES:
            pooled = comparison(bf[name]['pooled']['nmse'], bf['original_block']['pooled']['nmse'], bf['original_global']['pooled']['nmse'])
            heads = [dict(head=h, **comparison(bf[name]['per_head'][h]['nmse'],
                bf['original_block']['per_head'][h]['nmse'], bf['original_global']['per_head'][h]['nmse'])) for h in range(56)]
            counts = {endpoint: dict(improved=sum(r['candidate_minus_' + endpoint] < 0 for r in heads),
                worsened=sum(r['candidate_minus_' + endpoint] > 0 for r in heads),
                equal=sum(r['candidate_minus_' + endpoint] == 0 for r in heads)) for endpoint in ('block', 'global')}
            out['candidate_comparison'][name] = dict(pooled=pooled, per_head=heads,
                head_counts=counts, positive_baseline_advantage_heads=sum(r['baseline_advantage_positive'] for r in heads))
            report['compact_table'].append(dict(block=layer['block'], geometry=name, **pooled, head_counts=counts))
        out['pooled_candidate_order'] = sorted(GEOMETRIES, key=lambda n: bf[n]['pooled']['nmse'])
        out['per_head_candidate_order'] = [dict(head=h, order=sorted(GEOMETRIES, key=lambda n: bf[n]['per_head'][h]['nmse'])) for h in range(56)]
        out['per_head_winner_counts'] = {n: sum(r['order'][0] == n for r in out['per_head_candidate_order']) for n in GEOMETRIES}
        del refs
    assert [r['block'] for r in report['layers']] == [0, 24, 48]
    orders = [r['pooled_candidate_order'] for r in report['layers']]
    report['ranking'] = dict(layer_orders=orders, same_winner_all_layers=len({o[0] for o in orders}) == 1,
                             same_order_all_layers=all(o == orders[0] for o in orders))
    report['limitations'] = [
        'Advantage retention=(NMSE_global-NMSE_candidate)/(NMSE_global-NMSE_block), using contemporaneous endpoints and the same BF16 reference. It is a tensor-error ratio, not quality; ratios are not clipped.',
        'Per-head ratios whose global-minus-block denominator is negative retain that sign and must not be interpreted as retained positive advantage; no mean of head ratios is used.',
        'Three geometries fit different current-sample rank16 oracle centers on one prompt/step. No deployed basis, generalization, media quality, or speed conclusion follows.',
        'Rounded BF16 centers are reused in both Q-c and cKc, but BF16 rounding can exceed exact rank16. Full correction is materialized.',
        'All 15 saved tensors independently reduced with CPU FP64. Center verification uses stored factors/shapes, cast and scalar residual consistency, without repeating SVD.']
    assert not torch.cuda.is_initialized()
    report.update(status='complete', cuda_initialized=False, independent_svd_calls=0)


def budget():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError('300-second CPU reducer budget')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RD / 'independent_summary.json')
    args = p.parse_args(); assert not args.output.exists(), 'Preserve prior attempt'
    report = dict(experiment='E029', status='running', reducer=record(__file__))
    started = time.time()
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
