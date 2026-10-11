#!/usr/bin/env python3
"""Render the compact E087 fast four-prompt smoothing probe figure."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    args = p.parse_args()

    rows = list(csv.DictReader(args.input.open()))
    by_layer: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for key in ('timestep', 'local_alpha', 'global_alpha', 'local_validation_nmse',
                    'global_validation_nmse', 'validation_relative_gain'):
            row[key] = float(row[key])
        by_layer[row['layer']].append(row)
    for values in by_layer.values():
        values.sort(key=lambda r: r['timestep'])

    labels = {
        'blocks.3.attn.qkv_proj': 'Block 3 · QKV projection',
        'blocks.3.attn.out_proj': 'Block 3 · attention output projection',
        'blocks.13.attn.qkv_proj': 'Block 13 · QKV projection',
        'blocks.13.attn.out_proj': 'Block 13 · attention output projection',
    }
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.8), constrained_layout=True)
    fig.suptitle('MiniMax-H3: timestep-aware smoothing, held-out block-end NMSE', fontsize=16, fontweight='bold')

    for row_id, layer in enumerate(sorted(by_layer)):
        values = by_layer[layer]
        t = np.array([r['timestep'] for r in values])
        local = 1e5 * np.array([r['local_validation_nmse'] for r in values])
        global_ = 1e5 * np.array([r['global_validation_nmse'] for r in values])
        gain = 100 * np.array([r['validation_relative_gain'] for r in values])
        alpha = np.array([r['local_alpha'] for r in values])
        global_alpha = values[0]['global_alpha']

        ax = axes[row_id, 0]
        ax.plot(t, global_, color='#4c78a8', marker='o', ms=3.6, lw=1.8, label='Global smoothing')
        ax.plot(t, local, color='#f58518', marker='o', ms=3.6, lw=1.8, label='Per-timestep smoothing')
        ax.set_ylabel('Held-out NMSE (×10⁻⁵)')
        ax.set_title(labels.get(layer, layer), loc='left', fontsize=11, fontweight='bold')
        ax.set_xlim(-0.4, 19.4); ax.set_xticks(range(0, 20, 2)); ax.grid(axis='y', alpha=.22)
        if row_id == 0: ax.legend(frameon=False, ncol=2, fontsize=9, loc='upper left')

        ax = axes[row_id, 1]
        colors = np.where(gain >= 0, '#54a24b', '#e45756')
        ax.bar(t, gain, color=colors, width=.72)
        ax.axhline(0, color='#555555', lw=.9)
        ax.set_ylabel('Local vs global gain (%)')
        ax.set_xlim(-0.6, 19.6); ax.set_xticks(range(0, 20, 2)); ax.grid(axis='y', alpha=.22)
        pooled = 100 * (global_.sum() - local.sum()) / global_.sum()
        ax.text(.02, .93, f'Pooled: {pooled:+.2f}%', transform=ax.transAxes,
                va='top', fontsize=10, color='#24693d' if pooled >= 0 else '#a83838')

        ax = axes[row_id, 2]
        ax.plot(t, alpha, color='#7a5195', marker='o', ms=4, lw=1.8, label='Local α')
        ax.axhline(global_alpha, color='#4c78a8', lw=1.5, ls='--', label=f'Global α = {global_alpha:.1f}')
        ax.set_ylim(.05, .95); ax.set_yticks(np.arange(.1, 1.0, .1))
        ax.set_ylabel('Selected smoothing α')
        ax.set_xlim(-0.4, 19.4); ax.set_xticks(range(0, 20, 2)); ax.grid(axis='y', alpha=.22)
        if row_id == 0: ax.legend(frameon=False, fontsize=9, loc='upper right')

    for ax in axes[-1]: ax.set_xlabel('Denoising timestep')
    if args.config:
        cfg = json.loads(args.config.read_text())
        note = (f"Fit prompts: {', '.join(map(str, cfg['fit_prompt_ids']))}   ·   "
                f"Held-out prompts: {', '.join(map(str, cfg['validation_prompt_ids']))}   ·   "
                f"{cfg.get('smoothing_grid_size', 10)}-point α grid   ·   Positive gain = lower held-out NMSE")
    else:
        note = 'Positive gain = lower held-out NMSE'
    fig.text(.5, -.01, note,
             ha='center', fontsize=10, color='#444444')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
