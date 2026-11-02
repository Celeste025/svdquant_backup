#!/usr/bin/env python3
"""E136 paired decoded-video metrics for timestep precision windows."""
from __future__ import annotations

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
    case = parser.parse_args().case
    specs = {
        'bf16': ('E124' if case == 'p016' else 'E126', 'bf16'),
        'svdquant': ('E129', 'svdquant_bf16_attention'),
        'bf16_first10pct': ('E134', 'bf16_first10pct'),
        'bf16_last10pct': ('E134', 'bf16_last10pct'),
        'bf16_both10pct': ('E134', 'bf16_both10pct'),
        'int8_both10pct': ('E136', 'ends_int8_g128'),
        'bf16_both10pct_sage3_middle': ('E136', 'ends_bf16_svd_sage3'),
    }
    paths, initial = {}, []
    for label, (experiment, arm) in specs.items():
        folder = ROOT / 'results/research' / experiment / case
        media = json.loads((folder / f'{arm}_video.json').read_text())
        denoise = json.loads((folder / f'{arm}_denoise.json').read_text())
        paths[label] = Path(media['video'])
        assert paths[label].is_file(), paths[label]
        initial.append(denoise['initial_latent_sha256'])
    assert all(value == initial[0] for value in initial), 'initial video/audio latents differ'
    expected = (576, 1024, 3) if case == 'p016' else (768, 1344, 3)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True,
                                                    reduction='none').cuda().eval()
    accum = {arm: {'sse': 0, 'energy': 0, 'lpips': []} for arm in paths if arm != 'bf16'}
    count = 0
    for count, values in enumerate(zip_longest(*(frames(path) for path in paths.values())), 1):
        assert count <= 124 and all(value is not None and value.shape == expected for value in values)
        images = dict(zip(paths, values, strict=True))
        reference = images['bf16']
        ty = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32).div_(255)
        energy = int(np.square(reference.astype(np.int32)).sum(dtype=np.int64))
        for arm, row in accum.items():
            image = images[arm]
            delta = image.astype(np.int32) - reference.astype(np.int32)
            row['sse'] += int(np.square(delta).sum(dtype=np.int64))
            row['energy'] += energy
            tx = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32).div_(255)
            row['lpips'].append(float(metric(tx, ty).item()))
            metric.reset()
    assert count == 124
    result = {'format': 'e136-video-metrics-v1', 'case': case, 'frames': count,
              'initial_latent_sha256': initial[0],
              'sources': {label: str(path) for label, path in paths.items()},
              'summary': {arm: {'rgb_nmse': row['sse']/row['energy'],
                                'lpips_alex': float(np.mean(row['lpips']))}
                          for arm, row in accum.items()}}
    output = ROOT / 'results/research/E136' / f'{case}_video_metrics.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
