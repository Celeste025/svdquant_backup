#!/usr/bin/env python3
"""Render source-backed 3D FC2 activation comparisons for E090."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch


SAMPLES = {
    1: Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p1_s00.pt'),
    11: Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p11_s00_calib.pt'),
}
OUTPUT = Path('results/research/E090/figures')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    return p.parse_args()


def read_sample(prompt_id, path):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != 'h3-native-bench-sample-v1' or payload.get('target_block') != 25:
        raise ValueError(f'Unexpected sample contract: {path}')
    x = payload['linear_inputs']['mlp.fc2']
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != 14336:
        raise ValueError(f'Unexpected FC2 input: {x.shape}, {x.dtype}')
    cache = torch.load(payload['cache_call'], map_location='cpu', weights_only=True)
    meta = cache['meta']
    if meta['prompt_id'] != prompt_id or meta['step'] != 0:
        raise ValueError('Cached-call prompt/step mismatch')
    absolute = x.abs()
    peak_value, peak_flat = absolute.reshape(-1).max(dim=0)
    row, channel = divmod(int(peak_flat), x.shape[1])
    # Both complete token lengths happen to divide into exactly 64 row bins;
    # no tensor cells are dropped or interpolated in this max pooling.
    if x.shape[0] % 64 or x.shape[1] % 64:
        raise ValueError('Overview pooling would omit tensor cells')
    pooled = absolute.reshape(64, x.shape[0] // 64, 64, x.shape[1] // 64)
    pooled = pooled.amax(dim=(1, 3)).float().numpy()
    pooled_quantiles = np.quantile(pooled, [0.5, 0.9, 0.99, 0.999])
    return dict(prompt_id=prompt_id, path=str(path), cache_call=str(payload['cache_call']),
                seed=int(meta['seed']), step=int(meta['step']), x=x,
                shape=list(x.shape), peak_row=row, peak_channel=channel,
                signed_peak_value=float(x[row, channel]), absmax=float(peak_value),
                rms=float(x.float().square().mean().sqrt()),
                count_abs_gt_1000=int((absolute > 1000).sum()),
                count_abs_gt_2512=int((absolute > 2512).sum()),
                pooled=pooled, pooled_quantiles=pooled_quantiles.tolist())


def style_3d(ax):
    ax.view_init(elev=26, azim=-62)
    ax.set_box_aspect((1.4, 1.0, 0.62))
    ax.xaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1))
    ax.yaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1))
    ax.zaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1))
    ax.tick_params(labelsize=8, pad=1)


def overview(data, output):
    fig = plt.figure(figsize=(15, 7.3), dpi=175)
    norm = Normalize(vmin=0, vmax=5500)
    for panel, d in enumerate(data, start=1):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        row_step = d['shape'][0] // 64
        col_step = d['shape'][1] // 64
        rows = (np.arange(64) + 0.5) * row_step
        cols = (np.arange(64) + 0.5) * col_step
        channels, tokens = np.meshgrid(cols, rows)
        z = d['pooled']
        ax.plot_surface(channels, tokens, z, rstride=1, cstride=1,
                        facecolors=plt.cm.viridis(norm(z)), linewidth=0,
                        antialiased=False, shade=False)
        ax.scatter([d['peak_channel']], [d['peak_row']], [d['absmax']],
                   color='#e4572e', s=34, depthshade=False)
        ax.set(xlim=(0, 14336), ylim=(0, d['shape'][0]), zlim=(0, 5500),
               xlabel='Input channel (K)', ylabel='Token row', zlabel='Max |activation|')
        ax.set_title(f'p{d["prompt_id"]}  •  seed {d["seed"]}  •  max {d["absmax"]:,.0f}',
                     fontsize=12, pad=18)
        style_3d(ax)
    fig.suptitle('H3 block 25 FC2 input • step 0 • full-tensor amplitude overview',
                 fontsize=15, y=0.98)
    fig.text(0.5, 0.055,
             'Each surface cell is the exact max |x| in one token-row × channel bin '
             '(p1: 350×224 values; p11: 351×224). Same linear z-axis; orange dot = global peak.',
             ha='center', va='center', fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.87, wspace=0.05)
    fig.savefig(output, dpi=175, facecolor='white')
    plt.close(fig)


def peak_windows(data, output):
    fig = plt.figure(figsize=(15, 7.3), dpi=175)
    for panel, d in enumerate(data, start=1):
        r, c = d['peak_row'], d['peak_channel']
        rs = max(0, r - 32)
        re = min(d['shape'][0], r + 33)
        cs = max(0, c - 48)
        ce = min(d['shape'][1], c + 49)
        patch = d['x'][rs:re, cs:ce].float().numpy()
        channels, tokens = np.meshgrid(np.arange(cs, ce), np.arange(rs, re))
        limit = 5500 if d['prompt_id'] == 1 else 2700
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        ax.plot_surface(channels, tokens, patch, rstride=1, cstride=1,
                        facecolors=plt.cm.coolwarm(norm(patch)), linewidth=0,
                        antialiased=False, shade=False, alpha=0.86)
        ax.scatter([c], [r], [d['signed_peak_value']], color='#101820',
                   s=38, depthshade=False)
        ax.plot([c, c], [r, r], [0, d['signed_peak_value']],
                color='#334e8d', linewidth=2.4)
        ax.set(xlim=(cs, ce - 1), ylim=(rs, re - 1), zlim=(-limit, limit),
               xlabel='Input channel (K)', ylabel='Token row', zlabel='Activation value')
        ax.set_title(f'p{d["prompt_id"]} local peak: x[{r}, {c}] = '
                     f'{d["signed_peak_value"]:,.0f}', fontsize=12, pad=18)
        style_3d(ax)
        # Both largest-magnitude cells are negative. Looking from below the
        # near-zero surface keeps the negative trough visible rather than
        # hiding it behind the mesh.
        ax.view_init(elev=-25, azim=-62)
    fig.suptitle('Raw signed activation values around each prompt’s own largest-magnitude cell',
                 fontsize=15, y=0.98)
    fig.text(0.5, 0.055,
             'Unpooled 65×97 tensor-coordinate windows; linear z-axes with individually labeled ranges. '
             'Blue = negative, red = positive. Windows are centered at different coordinates.',
             ha='center', va='center', fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.87, wspace=0.05)
    fig.savefig(output, dpi=175, facecolor='white')
    plt.close(fig)


def main():
    a = parse_args()
    data = [read_sample(i, path) for i, path in SAMPLES.items()]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    overview(data, a.output_dir / 'h3_fc2_p1_p11_activation_3d_overview.png')
    peak_windows(data, a.output_dir / 'h3_fc2_p1_p11_activation_3d_peaks.png')
    peak_channels = sorted({d['peak_channel'] for d in data})
    cross_prompt_channel_maxima = {
        str(channel): {str(d['prompt_id']): float(d['x'][:, channel].abs().max())
                       for d in data}
        for channel in peak_channels
    }
    metadata = dict(format='h3-fc2-activation-3d-figure-v1',
                    figure_note='Overview is exact per-bin max abs; peak windows are raw signed values.',
                    cross_prompt_peak_channel_maxima=cross_prompt_channel_maxima,
                    panel_data=[{k: v for k, v in d.items() if k not in ('x', 'pooled')}
                                for d in data])
    (a.output_dir / 'h3_fc2_p1_p11_activation_3d_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    for d in data:
        print(f'p{d["prompt_id"]}: shape={d["shape"]}, seed={d["seed"]}, '
              f'peak={d["signed_peak_value"]} at ({d["peak_row"]}, '
              f'{d["peak_channel"]}), RMS={d["rms"]:.4f}')


if __name__ == '__main__':
    main()
