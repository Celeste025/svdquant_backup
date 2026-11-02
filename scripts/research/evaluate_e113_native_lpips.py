#!/usr/bin/env python3
"""Full-resolution paired LPIPS-Alex for E113 attention video controls."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from summarize_e113_native_videos import ARMS, frames, sha256

ROOT=Path(__file__).resolve().parents[2]
PAIRS=(('sage3','bf16'),('native_fp4','bf16'),('native_mixed','bf16'),
       ('native_mixed','native_fp4'))


def model_hash(metric):
    digest=hashlib.sha256()
    for name,value in sorted(metric.net.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(case):
    directory=ROOT/'results/research/E113'/case
    source_path=directory/'four_arm_attention_summary.json'
    source=json.loads(source_path.read_text())
    if source['format']!='e113-four-arm-attention-video-summary-v1' or source['frames']!=124:
        raise RuntimeError('invalid source summary')
    videos={arm:Path(source['sources'][arm]) for arm in ARMS}
    for arm,path in videos.items():
        if sha256(path)!=source['sha256'][arm]:
            raise RuntimeError(f'video hash mismatch: {arm}')
    metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,
                                                 reduction='none').cuda().eval()
    weights=model_hash(metric)
    scores={f'{a}_vs_{b}':[] for a,b in PAIRS}
    count=0
    for count,frame_set in enumerate(zip_longest(*(frames(videos[arm]) for arm in ARMS)),start=1):
        if count>124 or any(frame is None or frame.shape!=(576,1024,3) for frame in frame_set):
            raise RuntimeError(f'frame mismatch at {count-1}')
        images=dict(zip(ARMS,frame_set,strict=True))
        candidates=np.stack([images[a] for a,b in PAIRS])
        references=np.stack([images[b] for a,b in PAIRS])
        c=torch.from_numpy(candidates).permute(0,3,1,2).cuda().float()/255
        r=torch.from_numpy(references).permute(0,3,1,2).cuda().float()/255
        values=metric(c,r).detach().cpu().flatten().tolist()
        metric.reset()
        if len(values)!=len(PAIRS) or not all(math.isfinite(x) and x>=0 for x in values):
            raise RuntimeError(f'invalid LPIPS values on frame {count-1}')
        for pair,value in zip(PAIRS,values,strict=True):
            scores[f'{pair[0]}_vs_{pair[1]}'].append(float(value))
        if count%20==0:
            print(f'{case}: LPIPS {count}/124',flush=True)
    if count!=124:
        raise RuntimeError(f'incomplete video frames: {count}')
    means={key:float(np.mean(values)) for key,values in scores.items()}
    report={'format':'e113-four-arm-lpips-v1','case':case,'frames':count,
            'definition':'TorchMetrics pretrained AlexNet, FP32, normalize=True, paired full-resolution decoded RGB8 frames; lower means closer to named reference, not necessarily better quality',
            'pairs':means,'per_frame':scores,'metric_state_sha256':weights,
            'source_summary':str(source_path.resolve()),
            'source_summary_sha256':sha256(source_path),
            'environment':{'torch':torch.__version__,'torchmetrics':torchmetrics.__version__,
                           'gpu':torch.cuda.get_device_name()}}
    output=directory/'four_arm_attention_lpips.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(means,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('race','clap0'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    evaluate(args.case)


if __name__=='__main__':
    main()
