#!/usr/bin/env python3
"""Evaluate all-sparse H3 rollouts and compare to warmup arms."""
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
import sys
import time
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/wjq/workspace/svdquant-exp')
DATA=Path('/data1/models/svdquant-wjq/research/20261031/E140_draft_no_dense')
ARMS=['s70_int8_30','s70_fp4_20_int8_10']
EXPERIMENT='E140'
DENSE_STEPS=0
COMPARISON_EXPERIMENT='E139'
WORKER_NAME='run_e140_draft_no_dense.py'
COMPARISON_LABEL='Mixed: first10 steps /2 layers dense'
NEW_LABELS=['All sparse: 70% skip /30% INT8','All sparse: 70% skip /20% FP4 /10% INT8']


def main():
    deadline=time.monotonic()+1800
    for case in ['race','clap0']:
        folder=ROOT/'results/research'/EXPERIMENT/case
        for arm in ARMS:
            while not (folder/f'{arm}_video.json').is_file():
                assert time.monotonic()<deadline, 'E140 completion deadline reached'
                time.sleep(10)
        subprocess.run([sys.executable,str(ROOT/'scripts/research/evaluate_e138_videos.py'),
                        '--experiment',EXPERIMENT,'--reference-experiment','E121' if case=='race' else 'E139',
                        '--case',case,'--arms',*ARMS],check=True)
        report=json.loads((folder/'video_metrics.json').read_text())
        old=json.loads((ROOT/'results/research'/COMPARISON_EXPERIMENT/case/'video_metrics.json').read_text())
        reports={a:json.loads((folder/f'{a}_denoise.json').read_text()) for a in ARMS}
        for arm,r in reports.items():
            assert r['attention_router']['dense_first_steps']==DENSE_STEPS
            assert r['attention_router']['dense_first_layers']==0
            first_path='dense' if DENSE_STEPS else 'draft'
            assert all(row['path']==first_path for row in r['steps'][0]['attention_main_rows'])
            assert all(step['attention_main_calls']==50 for step in r['steps'])
        report['comparison_reference_summary']={'experiment':COMPARISON_EXPERIMENT,'summary':old['summary']}
        report['schedule']={'dense_first_steps':DENSE_STEPS,'dense_first_layers':0,'sparse_main_calls_per_arm':(20-DENSE_STEPS)*50}
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [ROOT/'scripts/research'/WORKER_NAME,
                                  ROOT/'scripts/research/run_e139_draftattention2.py']}
        paths=[report['sources']['bf16'],old['sources'][ARMS[1]],
               report['sources'][ARMS[0]],report['sources'][ARMS[1]]]
        labels=['BF16',COMPARISON_LABEL,*NEW_LABELS]
        target=DATA/case/f'{case}_bf16_warmup_int8_mixed.mp4'
        assert not target.exists(), target
        inputs=[av.open(path) for path in paths]
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        try:
            with av.open(str(target),'w',options={'movflags':'+faststart'}) as output:
                stream=output.add_stream('libx264',rate=24)
                stream.width,stream.height,stream.pix_fmt=2048,320,'yuv420p'
                stream.options={'crf':'18','preset':'fast'}
                count=0
                for count,frames in enumerate(zip_longest(*(c.decode(video=0) for c in inputs)),1):
                    assert count<=124 and all(f is not None for f in frames)
                    canvas=Image.new('RGB',(2048,320),'black')
                    for i,frame in enumerate(frames):
                        canvas.paste(frame.to_image().resize((512,288)),(i*512,32))
                    draw=ImageDraw.Draw(canvas)
                    for i,label in enumerate(labels):
                        draw.text((i*512+8,8),label,font=font,fill='white')
                    if count in [25,63,101]:
                        canvas.save(folder/f'comparison_frame{count:03}.jpg')
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.array(canvas),format='rgb24')):
                        output.mux(packet)
                assert count==124
                for packet in stream.encode():
                    output.mux(packet)
        finally:
            for container in inputs:
                container.close()
        report.update(comparison_video=str(target),comparison_audio='none; originals retain audio',
                      comparison_columns=labels)
        (folder/'video_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'case':case,'summary':report['summary'],'comparison_video':str(target)}),flush=True)


if __name__=='__main__':
    main()
