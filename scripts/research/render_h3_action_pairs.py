#!/usr/bin/env python3
"""E073 fixed-frame, anonymous paired readout; no model or quality scorer."""
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import time

import av
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'results/research/E073'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E073')
FRAMES = list(range(0, 124, 2)) + [123]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    manifest = json.loads((ROOT / 'research_state/06_experiments/E073_behavior_manifest.json').read_text())
    reference, receipts = {}, []
    for replica in (0, 1):
        launch = json.loads((REPORTS / f'launcher_r{replica}.json').read_text())
        assert launch['status'] == 'complete'
        p = REPORTS / f'decode_full_bf16_r{replica}.json'
        decoded = json.loads(p.read_text())
        denoised = json.loads((REPORTS / f'denoise_full_bf16_r{replica}.json').read_text())
        assert decoded['status'] == denoised['status'] == 'complete'
        assert denoised['complete_dit_calls'] == denoised['attempted_dit_calls'] == 80
        assert len(decoded['cases']) == 4
        for case in denoised['cases']:
            assert len(case['steps']) == 40 and len(case['dit_calls']) == 20
            assert case['schedule'] == manifest['expected_schedule']
            assert all(row['counts'] == dict(sdpa_calls=102, scaled_mm_calls=0, disk_loads=0)
                       for row in case['dit_calls'])
        reference.update({c['case_id']: c for c in decoded['cases']})
        receipts.append(dict(replica=replica, dit_calls=80, updates=160,
                             decode_sha256=digest(p), cases=4))
    q = json.loads((ROOT / 'results/research/E038/decode_bf16.json').read_text())
    assert q['status'] == 'complete'
    quantized = {c['case_id']: c for c in q['cases']}
    public, private = DATA / 'paired_readout', DATA / 'private_readout'
    public.mkdir(exist_ok=False)
    private.mkdir(mode=0o700, exist_ok=False)
    records, keys = [], []
    for n, case in enumerate(manifest['cases'], start=1):
        name = f'case{n:02d}'
        arms = [('original_bf16', reference[case['case_id']]),
                ('native_svd_bf16_attention', quantized[case['case_id']])]
        if secrets.randbelow(2):
            arms.reverse()
        frames, clips = {}, []
        for label, (method, item) in zip(('A', 'B'), arms):
            src = Path(item['video']['file'])
            assert digest(src) == item['video']['sha256']
            assert item['settings'] == manifest['settings']
            target = public / f'{name}_{label}.mp4'
            shutil.copyfile(src, target)
            with av.open(str(target)) as container:
                decoded_frames = [f.to_image() for f in container.decode(video=0)]
            assert len(decoded_frames) == 124 and all(im.size == (1024, 576) for im in decoded_frames)
            frames[label] = {i: decoded_frames[i].resize((384, 216)) for i in FRAMES}
            clips.append(dict(label=label, file=str(target), sha256=digest(target)))
            keys.append(dict(case=name, label=label, method=method, source=str(src)))
        pages = []
        for page, start in enumerate(range(0, len(FRAMES), 16)):
            indices = FRAMES[start:start + 16]
            canvas = Image.new('RGB', (1536, 52 + 8 * 238), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f'{name} | {case["prompt"]}', fill='black')
            draw.text((8, 24), 'Each pair: A then B; original frame indices. Static frame review, not realtime playback.', fill='black')
            for j, index in enumerate(indices):
                for a, label in enumerate(('A', 'B')):
                    x, y = ((j % 2) * 2 + a) * 384, 52 + (j // 2) * 238
                    draw.text((x + 5, y + 2), f'{label} frame {index} | {index/24:.3f}s', fill='black')
                    canvas.paste(frames[label][index], (x, y + 22))
            path = public / f'{name}_page{page:02d}.png'
            canvas.save(path)
            pages.append(dict(file=str(path), frames=indices))
        records.append(dict(case=name, prompt=case['prompt'], clips=clips, pages=pages))
    key_path = private / 'key.json'
    key_path.write_text(json.dumps(keys, indent=2) + '\n')
    key_path.chmod(0o600)
    result = dict(status='complete', experiment='E073', receipts=receipts, cases=records,
                  sampled_indices=FRAMES, full_video_frames=124, frames_reviewed_by_rendering=0,
                  observation_policy='Assistant must explicitly view pages; rendering is not review. No human rating.',
                  human_reviews=0, models=0, gpu=0, seconds=time.monotonic()-started)
    (public / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='complete', public_manifest=str(public/'manifest.json'),
                          cases=8, pages=sum(len(c['pages']) for c in records), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
