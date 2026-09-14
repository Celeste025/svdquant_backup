#!/usr/bin/env python3
"""Plot local/error-horizon relationships from the completed rCM audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.input.open()))
    actual = {(r['prompt_id'], r['step'], r['layer']): r for r in rows if r['mode'] == 'actual'}
    normalized = {(r['prompt_id'], r['step'], r['layer']): r for r in rows if r['mode'] == 'normalized'}
    keys = sorted(actual.keys())
    if actual.keys() != normalized.keys(): raise RuntimeError('actual/normalized records are not paired')
    prompts = sorted({k[0] for k in keys}); colors = dict(zip(prompts, ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728')))
    local = np.array([float(actual[k]['local_nmse']) for k in keys]); final_a = np.array([float(actual[k]['final_latent_nmse']) for k in keys])
    scale = np.array([float(normalized[k]['scale']) for k in keys]); final_n = np.array([float(normalized[k]['final_latent_nmse']) for k in keys])
    predicted_n = final_a * scale**2
    ratio = final_n / predicted_n
    metrics = {
        'actual_local_vs_final_spearman': float(spearmanr(local, final_a).statistic),
        'normalized_local_vs_final_spearman': float(spearmanr([float(normalized[k]['local_nmse']) for k in keys], final_n).statistic),
        'scaling_prediction_log_spearman': float(spearmanr(np.log(predicted_n), np.log(final_n)).statistic),
        'norm_over_actual_times_s2_median': float(np.median(ratio)),
        'norm_over_actual_times_s2_p10_p90': [float(x) for x in np.percentile(ratio, [10, 90])],
        'records': len(keys),
    }
    (args.output_dir / 'actual_normalized_relationship_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for prompt in prompts:
        idx = np.array([k[0] == prompt for k in keys])
        ax.scatter(local[idx], final_a[idx], s=27, alpha=.75, color=colors[prompt], label=f'prompt {prompt}')
    ax.set(xscale='log', yscale='log', xlabel='actual local Linear NMSE', ylabel='actual final-latent NMSE',
           title=f'Local error poorly predicts final impact\nSpearman = {metrics["actual_local_vs_final_spearman"]:.3f}')
    ax.legend(frameon=False); ax.grid(alpha=.2, which='both')

    ax = axes[1]
    for prompt in prompts:
        idx = np.array([k[0] == prompt for k in keys])
        ax.scatter(predicted_n[idx], final_n[idx], s=27, alpha=.75, color=colors[prompt], label=f'prompt {prompt}')
    lo, hi = min(predicted_n.min(), final_n.min()), max(predicted_n.max(), final_n.max())
    ax.plot([lo, hi], [lo, hi], '--', color='black', lw=1, label='exact $s^2$ scaling')
    ax.set(xscale='log', yscale='log', xlabel=r'predicted normalized final NMSE: actual $×s^2$', ylabel='measured normalized final-latent NMSE',
           title=f'Approximate, not exact, $s^2$ scaling\nlog-Spearman = {metrics["scaling_prediction_log_spearman"]:.3f}; median ratio = {metrics["norm_over_actual_times_s2_median"]:.2f}')
    ax.legend(frameon=False); ax.grid(alpha=.2, which='both')
    fig.tight_layout(); fig.savefig(args.output_dir / 'actual_vs_normalized_final_nmse.png', dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(ratio), bins=24, color='#9467bd', alpha=.85)
    ax.axvline(0, color='black', ls='--', lw=1, label='exact $s^2$ scaling')
    ax.set(xlabel=r'$log_{10}(mathrm{measured norm}/(mathrm{actual}×s^2))$', ylabel='intervention records', title='Deviation from the local-linear scaling prediction')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(args.output_dir / 'actual_vs_normalized_scaling_residual.png', dpi=220); plt.close(fig)


if __name__ == '__main__': main()
