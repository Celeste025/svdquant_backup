#!/usr/bin/env python3
"""Full-frame distances for the existing H3 SVDQuant/BF16 paired videos."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import av
import numpy as np
import torch
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research/E074'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text())


def build_pairs():
    pairs = []
    decoded = read('results/research/E010_h3_decode.json')['cases']
    dn = {c['prompt_id']: c for c in read('results/research/E010_h3_denoise_native.json')['cases']}
    dr = {c['prompt_id']: c for c in read('results/research/E010_h3_denoise_bf16.json')['cases']}
    for pid, label in ((30, '中文文字'), (36, '产品展示')):
        c = next(c for c in decoded if c['prompt_id'] == pid and c['variant'] == 'native')
        r = next(c for c in decoded if c['prompt_id'] == pid and c['variant'] == 'bf16')
        pairs.append((f'p{pid:03d}', label, 'E010', c, r, dn[pid], dr[pid]))
    candidates = read('results/research/E038/decode_bf16.json')['cases']
    native_denoise = {c['case_id']: c for c in read('results/research/E038/denoise_bf16.json')['cases']}
    labels = {161: '拍手', 192: '叠衣服', 269: '汽车转弯', 316: '大象喷水'}
    refs, ref_denoise = {}, {}
    for replica in (0, 1):
        for c in read(f'results/research/E073/decode_full_bf16_r{replica}.json')['cases']:
            refs[c['case_id']] = c
        for c in read(f'results/research/E073/denoise_full_bf16_r{replica}.json')['cases']:
            ref_denoise[c['case_id']] = c
    for c in sorted(candidates, key=lambda c: c['case_id']):
        cid = c['case_id']
        pairs.append((cid, labels[c['prompt_id']], 'E038_E073', c, refs[cid], native_denoise[cid], ref_denoise[cid]))
    manifest = []
    for cid, label, cohort, c, r, dc, dr in pairs:
        assert c['status'] == r['status'] == dc['status'] == dr['status'] == 'complete'
        for key in ('seed', 'prompt_sha256', 'settings'):
            assert c[key] == r[key], (cid, key)
        for key in ('initial_noise_sha256', 'embedding_sha256'):
            assert dc[key] == dr[key], (cid, key)
        if cohort == 'E010':
            assert dc['video_timesteps'] == dr['video_timesteps']
            assert dc['audio_timesteps'] == dr['audio_timesteps']
        else:
            assert dc['schedule'] == dr['schedule']
        def path(case):
            return case['video_path'] if 'video_path' in case else case['video']
        for case in (c, r):
            assert sha(path(case)) == case['video_sha256'], cid
        manifest.append(dict(case=cid, label=label, cohort=cohort, seed=c['seed'],
                             candidate=path(c), reference=path(r),
                             candidate_sha256=c['video_sha256'], reference_sha256=r['video_sha256'],
                             initial_noise_sha256=dc['initial_noise_sha256'],
                             embedding_sha256=dc['embedding_sha256'], settings=c['settings']))
    assert len(manifest) == 10
    return manifest


def frames(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 24
        for frame in container.decode(stream):
            yield frame.to_ndarray(format='rgb24')


@torch.inference_mode()
def evaluate(pair, metric):
    absolute_sum = square_sum = reference_square_sum = 0
    traces = []
    for i, (c, r) in enumerate(itertools.zip_longest(frames(pair['candidate']), frames(pair['reference']))):
        assert c is not None and r is not None, 'frame count mismatch'
        assert c.shape == r.shape == (576, 1024, 3)
        diff = c.astype(np.int32) - r.astype(np.int32)
        a = int(np.abs(diff).sum(dtype=np.int64))
        s = int(np.square(diff).sum(dtype=np.int64))
        p = int(np.square(r.astype(np.int32)).sum(dtype=np.int64))
        absolute_sum += a
        square_sum += s
        reference_square_sum += p
        ct = torch.from_numpy(c.copy()).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32).div_(255)
        rt = torch.from_numpy(r.copy()).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32).div_(255)
        score = float(metric(ct, rt).item())
        metric.reset()
        assert math.isfinite(score)
        if i == 0:
            identical = float(metric(rt, rt).item())
            metric.reset()
            assert abs(identical) < 1e-7, identical
        traces.append(dict(frame=i, lpips_alex=score, abs_sum_uint8=a,
                           squared_sum_uint8=s, reference_squared_sum_uint8=p))
    assert len(traces) == 124
    count = 124 * 576 * 1024 * 3
    return dict(case=pair['case'], label=pair['label'], cohort=pair['cohort'], seed=pair['seed'],
                frames=124, scalar_pixels=count,
                lpips_alex=float(np.mean([r['lpips_alex'] for r in traces])),
                l1_mae=absolute_sum / (255 * count),
                l2_norm=math.sqrt(square_sum) / 255,
                l2_rmse=math.sqrt(square_sum / count) / 255,
                relative_l2=math.sqrt(square_sum / reference_square_sum),
                abs_sum_uint8=absolute_sum, squared_sum_uint8=square_sum,
                reference_squared_sum_uint8=reference_square_sum, frame_metrics=traces)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'metrics.json').exists(), 'refuse to overwrite completed results'
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pairs = build_pairs()
    (OUT / 'pairs.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + '\n')
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True, reduction='none').cuda().eval()
    model_hash = hashlib.sha256()
    for name, value in sorted(metric.net.state_dict().items()):
        model_hash.update(name.encode())
        model_hash.update(value.cpu().contiguous().numpy().tobytes())
    rows = []
    for pair in pairs:
        row = evaluate(pair, metric)
        rows.append(row)
        (OUT / f"{pair['case']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'frame_metrics'}, ensure_ascii=False), flush=True)
    scalar_keys = ('lpips_alex', 'l1_mae', 'l2_norm', 'l2_rmse', 'relative_l2')
    groups = {}
    for label in ['all_10', 'action_8', 'E010_2'] + list(dict.fromkeys(r['label'] for r in rows)):
        group = [r for r in rows if label == 'all_10' or
                 (label == 'action_8' and r['cohort'] == 'E038_E073') or
                 (label == 'E010_2' and r['cohort'] == 'E010') or r['label'] == label]
        groups[label] = dict(cases=len(group), **{k: float(np.mean([r[k] for r in group])) for k in scalar_keys})
    result = dict(status='complete', definition={
        'baseline': 'Existing native NVFP4 W4A4 + BF16 rank32 SVDQuant baseline; attention BF16',
        'reference': 'Original full BF16 H3; paired actual noise/embedding/schedule',
        'media': 'Decoded MP4 RGB uint8, full 1024x576, all 124 frames, no resizing/cropping/alignment; audio excluded',
        'lpips': 'TorchMetrics pretrained LPIPS AlexNet, FP32, normalize=True for [0,1] input; per-frame then per-case mean',
        'l1_mae': 'sum(abs(candidate-reference)) / (255 * T * H * W * 3)',
        'l2_norm': 'sqrt(sum((candidate-reference)^2)) / 255',
        'l2_rmse': 'l2_norm / sqrt(T * H * W * 3)',
        'relative_l2': 'L2(candidate-reference) / L2(reference)',
        'aggregation': 'Every group reports arithmetic means over videos; no seed/case exclusions',
    }, environment=dict(torch=torch.__version__, torchmetrics=torchmetrics.__version__, av=av.__version__,
                        gpu=torch.cuda.get_device_name(), metric_state_sha256=model_hash.hexdigest()),
                  script_sha256=sha(__file__), seconds=time.time()-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), rows=rows, summary=groups)
    (OUT / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'per_case.csv').open('w', newline='') as f:
        keys = ('case', 'label', 'seed', 'frames') + scalar_keys
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in rows)
    print(json.dumps(dict(status='complete', seconds=result['seconds'], summary=groups), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
