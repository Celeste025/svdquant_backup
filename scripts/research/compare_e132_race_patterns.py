#!/usr/bin/env python3
"""Compare W4A4 and 10% token-recompute DiT output-error hot spots."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / 'results/research/E132'


def corr(x, y):
    a, b = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def top_ids(x, fraction=.01):
    n = max(1, round(x.size * fraction))
    return set(np.argpartition(x.ravel(), -n)[-n:].tolist())


def main():
    a = np.load(ROOT / 'w4a4_token_maps.npz')['token_error_energy']
    b = np.load(ROOT / 'recompute10_token_maps.npz')['token_error_energy']
    assert a.shape == b.shape == (20, 37, 18, 32)
    base = json.loads((ROOT / 'w4a4_summary.json').read_text())
    hybrid = json.loads((ROOT / 'recompute10_summary.json').read_text())
    rows = []
    for step in range(20):
        x, y = a[step], b[step]
        ta, tb = top_ids(x), top_ids(y)
        common = ta & tb
        rows.append({'step': step, 'nmse_w4a4': base['step_nmse'][step],
                     'nmse_recompute10': hybrid['step_nmse'][step],
                     'relative_energy_reduction': 1 - float(y.sum() / x.sum()),
                     'same_step_log_energy_corr': corr(np.log(np.maximum(x, 1e-12)),
                                                       np.log(np.maximum(y, 1e-12))),
                     'top1pct_overlap_fraction': len(common) / len(ta),
                     'top1pct_jaccard': len(common) / len(ta | tb),
                     'w4a4_top1pct_energy_share': float(x.ravel()[list(ta)].sum() / x.sum()),
                     'recompute_top1pct_energy_share': float(y.ravel()[list(tb)].sum() / y.sum()),
                     'energy_reduction_at_original_hotspots': 1 - float(y.ravel()[list(ta)].sum() /
                                                                         x.ravel()[list(ta)].sum())})
    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'step'}
    result = {'format': 'e132-pattern-comparison-v1', 'case': 'race_p016',
              'input': 'identical cached BF16 teacher calls, not free-rollout comparison',
              'token_grid_thw': [37, 18, 32], 'hotspot': 'top 1% video tokens by squared output-error energy',
              'means': means, 'w4a4_adjacent': {k: base[k] for k in base if k.startswith('mean_adjacent')},
              'recompute_adjacent': {k: hybrid[k] for k in hybrid if k.startswith('mean_adjacent')},
              'steps': rows}
    (ROOT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'means': means, 'w4a4_adjacent': result['w4a4_adjacent'],
                      'recompute_adjacent': result['recompute_adjacent']}, indent=2))


if __name__ == '__main__':
    main()
