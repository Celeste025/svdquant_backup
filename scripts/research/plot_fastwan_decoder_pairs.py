#!/usr/bin/env python3
"""Fixed-frame E024/E025 decoder comparisons; no generation or quality filter."""
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E025/contact_sheets')
INDICES = [0, 20, 40, 60, 80]


def record(path):
    return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def frames(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    selected = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count in INDICES:
            selected.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((332, 192)))
        count += 1
    cap.release()
    assert count == 81 and len(selected) == 5
    return selected


def main():
    source_paths = [ROOT / 'results/research/E024/suite_run.json',
                    ROOT / 'results/research/E025/decode_run.json']
    sources = [json.loads(p.read_text()) for p in source_paths]
    assert all(r['status'] == 'complete' for r in sources)
    rows = [{r['trajectory_id']: r for r in source['cases']} for source in sources]
    ids = [r['trajectory_id'] for r in sources[0]['cases'] if r['replica'] == 0]
    assert len(ids) == 8 and rows[0].keys() == rows[1].keys()
    DATA.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'results/research/E025/contact_sheets.json'
    assert not report_path.exists(), 'Preserve prior visualizations'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    result = dict(sources=[record(p) for p in source_paths], indices=INDICES,
                  scope='All 8 replica0 trajectories, fixed 5 frames; not full-motion human evaluation.', sheets=[])
    for offset in (0, 4):
        canvas = Image.new('RGB', (1660, 4 * 466), 'white')
        draw = ImageDraw.Draw(canvas)
        for group, tid in enumerate(ids[offset:offset + 4]):
            y = group * 466
            draw.text((5, y + 3), tid + ' | ' + rows[0][tid]['prompt'], fill='black', font=font)
            for arm, name in enumerate(('TAEHV FP16', 'Full Wan VAE FP32')):
                row = rows[arm][tid]
                assert row['prompt'] == rows[0][tid]['prompt'] and row['seed'] == rows[0][tid]['seed']
                path = Path(row['video']['file'])
                assert record(path)['sha256'] == row['video']['sha256']
                yy = y + 30 + arm * 216
                draw.text((5, yy), name + '   frames: 0 / 20 / 40 / 60 / 80', fill='black', font=font)
                for col, frame in enumerate(frames(path)):
                    canvas.paste(frame, (332 * col, yy + 24))
        out = DATA / ('first4_r0.png' if offset == 0 else 'last4_r0.png')
        assert not out.exists()
        canvas.save(out)
        result['sheets'].append(dict(record(out), trajectories=ids[offset:offset + 4]))
    report_path.write_text(json.dumps(result, indent=2) + '\n')
    print(report_path)


if __name__ == '__main__':
    main()
