#!/usr/bin/env python3
"""Paired decoded-frame RGB NMSE and LPIPS for E133 versus BF16/W4A4."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path

import av
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[2]


def frames(path):
    with av.open(str(path)) as container:
        yield from (frame.to_ndarray(format='rgb24') for frame in container.decode(video=0))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('p016', 'vbench_000'), required=True)
    args = parser.parse_args()
    case = args.case
    specs = {'bf16': ('E124' if case == 'p016' else 'E126', 'bf16'),
             'w4a4': ('E128', 'w4a4_bf16_attention'),
             'random10': ('E133', 'random10'),
             'cycle10': ('E133', 'cycle10')}
    paths, initial = {}, []
    for name, (experiment, arm) in specs.items():
        folder = ROOT / 'results/research' / experiment / case
        media = json.loads((folder / f'{arm}_video.json').read_text())
        denoise = json.loads((folder / f'{arm}_denoise.json').read_text())
        paths[name] = Path(media['video'])
        initial.append(denoise['initial_latent_sha256'])
    assert all(x == initial[0] for x in initial)
    expected = (576, 1024, 3) if case == 'p016' else (768, 1344, 3)
    torch.set_num_threads(4)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True,
                                                    reduction='none').cuda().eval()
    pairs = [(arm, 'bf16') for arm in ('w4a4', 'random10', 'cycle10')]
    accum = {f'{a}_vs_{b}': {'sse': 0, 'energy': 0, 'lpips': []} for a, b in pairs}
    count = 0
    for count, values in enumerate(zip_longest(*(frames(p) for p in paths.values())), 1):
        assert count <= 124 and all(x is not None and x.shape == expected for x in values)
        images = dict(zip(paths, values, strict=True))
        for a, b in pairs:
            row = accum[f'{a}_vs_{b}']
            x, y = images[a], images[b]
            diff = x.astype(np.int32) - y.astype(np.int32)
            row['sse'] += int(np.square(diff).sum(dtype=np.int64))
            row['energy'] += int(np.square(y.astype(np.int32)).sum(dtype=np.int64))
            tx = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32).div_(255)
            ty = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32).div_(255)
            row['lpips'].append(float(metric(tx, ty).item()))
            metric.reset()
    assert count == 124
    summary = {k: {'rgb_nmse': v['sse']/v['energy'], 'lpips_alex': float(np.mean(v['lpips']))}
               for k, v in accum.items()}
    result = {'format': 'e133-video-metrics-v1', 'case': case, 'frames': count,
              'initial_latent_sha256': initial[0], 'sources': {k: str(v) for k, v in paths.items()},
              'summary': summary}
    out = ROOT / 'results/research/E133' / f'{case}_video_metrics.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
