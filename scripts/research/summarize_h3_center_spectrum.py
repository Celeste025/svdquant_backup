#!/usr/bin/env python3
"""E028 scalar-only independent reduction; no torch import, SVD, or GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E028'
RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def record(path):
    p = Path(path).resolve()
    return dict(file=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-15), (a, b)


def quantiles(values):
    xs = sorted(values)
    result = {}
    for label, q in [('min', 0), ('p25', .25), ('median', .5), ('p90', .9), ('max', 1)]:
        pos = (len(xs) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        result[label] = xs[lo] * (hi - pos) + xs[hi] * (pos - lo) if lo != hi else xs[lo]
    return result


def reduce(report):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    src = json.loads((RD / 'run.json').read_text())
    launch = json.loads((RD / 'launch.json').read_text())
    check = json.loads((RD / 'check.json').read_text())
    assert src['status'] == launch['status'] == check['status'] == 'complete'
    assert launch['returncode'] == 0 and launch['elapsed_s'] < 300
    assert src['actual_gpu_calls'] == 0 and src['cuda_initialized'] is False
    assert src['sources'] == check['sources'] and src['inputs'] == check['inputs']
    assert [r['block'] for r in src['layers']] == [0, 24, 48]
    assert src['completed_layers'] == 3 and src['completed_head_geometries'] == 504
    assert src['geometry']['tail_query_weight'] == 11 and src['geometry']['ranks'] == list(RANKS)
    assert record(src['sources']['runner']['file']) == src['sources']['runner']
    report.update(inputs={p: record(RD / (p + '.json')) for p in ('run', 'launch', 'check')},
        plan=record(ROOT / 'research_state/06_experiments/E028_center_spectrum_plan.md'),
        source_bindings_inherited=src['sources'], geometry=src['geometry'],
        computation_seconds=src['seconds'], layers=[], compact_residual_table_percent=[])
    checked = 0
    for layer in src['layers']:
        out = dict(block=layer['block'], geometries={})
        report['layers'].append(out)
        assert set(layer['geometries']) == {'euclidean', 'kc_score', 'k4_error_score'}
        for name, geometry in layer['geometries'].items():
            assert [h['head'] for h in geometry['heads']] == list(range(56))
            heads = []
            for h in geometry['heads']:
                spectrum = h['squared_singular_values']
                assert len(spectrum) == 128 and all(math.isfinite(x) and x >= 0 for x in spectrum)
                assert all(a >= b for a, b in zip(spectrum, spectrum[1:]))
                total = math.fsum(spectrum)
                assert total > 0
                close(total, h['total_energy'])
                assert h['rank0_residual_fraction'] == 1
                tails = {str(r): math.fsum(spectrum[r:]) for r in RANKS}
                fractions = {r: e / total for r, e in tails.items()}
                for rank in tails:
                    close(tails[rank], h['residual_energy'][rank])
                    close(fractions[rank], h['residual_fraction'][rank])
                if name != 'euclidean':
                    close(total / 128, h['softmax_scaled_total_energy'])
                    assert h['psd']['min_eigenvalue'] >= -h['psd']['negative_tolerance']
                heads.append(dict(head=h['head'], total_energy=total,
                                  residual_energy=tails, residual_fraction=fractions))
                checked += 1
            energy = math.fsum(h['total_energy'] for h in heads)
            close(energy, geometry['aggregate']['total_energy'])
            assert geometry['aggregate']['nonzero_energy_heads'] == 56
            curves = {}
            for rank in map(str, RANKS):
                pooled = math.fsum(h['residual_energy'][rank] for h in heads) / energy
                q = quantiles([h['residual_fraction'][rank] for h in heads])
                old = geometry['aggregate']['rank_curves'][rank]
                close(pooled, old['energy_weighted_residual_fraction'])
                for key, value in q.items():
                    close(value, old['head_quantiles'][key])
                curves[rank] = dict(energy_weighted_residual_fraction=pooled,
                    head_quantiles=q,
                    min_residual_head=min(heads, key=lambda h: h['residual_fraction'][rank])['head'],
                    max_residual_head=max(heads, key=lambda h: h['residual_fraction'][rank])['head'])
            energy_order = sorted(heads, key=lambda h: h['total_energy'], reverse=True)
            out['geometries'][name] = dict(total_energy=energy, rank_curves=curves, heads=heads,
                head_energy_concentration={str(n): dict(
                    heads=[h['head'] for h in energy_order[:n]],
                    energy_fraction=math.fsum(h['total_energy'] for h in energy_order[:n]) / energy)
                    for n in (1, 4, 8)},
                max_reported_spectral_frobenius_relative_difference=max(
                    h['spectral_vs_frobenius_relative_difference'] for h in geometry['heads']),
                covariance_negative_eigenvalue_count=sum(
                    h['psd']['negative_count'] for h in geometry['heads'] if h['psd']))
            report['compact_residual_table_percent'].append(dict(block=layer['block'], geometry=name,
                **{'r' + str(r): 100 * curves[str(r)]['energy_weighted_residual_fraction'] for r in (4, 8, 16, 32)}))
    assert checked == 504
    report.update(status='complete', spectra_checked=checked, scalar_energies_checked=504 * 128,
        actual_gpu_calls=0, svd_calls=0, torch_imported=False,
        arithmetic='Python float64 math.fsum of all saved squared singular values; independent linear-interpolation head quantiles. No covariance/SVD recomputation.',
        limitations=[
            'Per-head, per-geometry same-input oracle spectra; different geometries optimize different bases. Three layers share one p36/step14 input.',
            'Energy weighting is sum of residual energies divided by sum of total energies, not the mean head fraction; head quantiles and energy concentration retained separately.',
            'CPU BF16 means precede FP64 spectra. GPU reduction byte parity and deployment basis generalization are not established.',
            'K4 representation-error geometry captures only the center-error term, omitting the changed-Q quantization term. No native output accuracy, quality, performance or all-method lower bound follows.',
            'This scalar reducer inherits tensor/decoder validation from the bound completed run; it does not rerun large inputs, covariance, eigendecomposition, or SVD.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RD / 'independent_summary.json')
    args = p.parse_args()
    assert not args.output.exists(), 'Preserve prior attempt'
    result = dict(experiment='E028', status='running', reducer=record(__file__))
    started = time.time()
    try:
        reduce(result)
    except BaseException:
        result.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        result['reducer_seconds'] = time.time() - started
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output), sha256=record(args.output)['sha256'])))


if __name__ == '__main__':
    main()
