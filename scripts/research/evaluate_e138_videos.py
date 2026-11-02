#!/usr/bin/env python3
"""Validate paired E138 outputs and measure full-frame NMSE/LPIPS."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from evaluate_e136_videos import frames

ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=['race', 'clap0'], required=True)
    ap.add_argument('--experiment', choices=['E138', 'E139', 'E140', 'E141'], default='E138')
    ap.add_argument('--reference-experiment', choices=['E121','E125','E139'])
    ap.add_argument('--arms', nargs=2, default=['nearest', 'activation_sr'])
    args = ap.parse_args()
    case, arms = args.case, args.arms
    folder = ROOT/'results/research'/args.experiment/case
    reference_folder = ROOT/'results/research'/('E121' if case == 'race' else 'E125')/case
    if args.reference_experiment:
        reference_folder = ROOT/'results/research'/args.reference_experiment/case
    if (folder/'bf16_video.json').is_file():
        reference_folder = folder
    reference = json.loads((reference_folder/'bf16_video.json').read_text())
    reference_report = json.loads((reference_folder/'bf16_denoise.json').read_text())
    paths = {'bf16': Path(reference['video'])}
    reports = []
    for arm in arms:
        report = json.loads((folder/f'{arm}_denoise.json').read_text())
        assert report['status'] == 'denoise_complete' and len(report['steps']) == 20
        reports.append(report)
        paths[arm] = Path(json.loads((folder/f'{arm}_video.json').read_text())['video'])
    assert reports[0]['initial_latent_sha256'] == reports[1]['initial_latent_sha256']
    assert reports[0]['initial_latent_sha256'] == reference_report['initial_latent_sha256']
    if 'state_sha256' in reports[0]['linears']:
        assert reports[0]['linears']['state_sha256'] == reports[1]['linears']['state_sha256']
    else:
        assert reports[0]['linears'] == reports[1]['linears']
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True, reduction='none').cuda().eval()
    accum = {arm: {'sse': 0, 'energy': 0, 'lpips': []} for arm in arms}
    count = 0
    for count, values in enumerate(zip_longest(*(frames(p) for p in paths.values())), 1):
        assert count <= 124 and all(v is not None and v.shape == (576, 1024, 3) for v in values)
        images = dict(zip(paths, values))
        ref = images['bf16']
        ty = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).cuda().float()/255
        for arm, row in accum.items():
            delta = images[arm].astype(np.int32)-ref.astype(np.int32)
            row['sse'] += int(np.square(delta).sum(dtype=np.int64))
            row['energy'] += int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
            tx = torch.from_numpy(images[arm]).permute(2,0,1).unsqueeze(0).cuda().float()/255
            row['lpips'].append(float(metric(tx, ty)))
            metric.reset()
    assert count == 124
    result = {'case': case, 'frames': count, 'initial_latent_sha256': reports[0]['initial_latent_sha256'],
              'experiment': args.experiment,
              'dit_seconds': {arm: {'sum': float(sum(s['dit_seconds'] for s in r['steps'])),
                                   'per_step': [s['dit_seconds'] for s in r['steps']]}
                              for arm, r in zip(arms, reports)},
              'sources': {k: str(v) for k,v in paths.items()},
              'metric_scope': 'decoded RGB divergence and LPIPS to matched BF16; inspect media as well',
              'summary': {k: {'rgb_nmse': v['sse']/v['energy'], 'lpips_alex': float(np.mean(v['lpips']))} for k,v in accum.items()}}
    (folder/'video_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
