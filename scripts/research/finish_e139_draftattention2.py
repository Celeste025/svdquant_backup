#!/usr/bin/env python3
"""Evaluate E139 and render BF16/INT8/mixed side-by-side videos."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from itertools import zip_longest
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/wjq/workspace/svdquant-exp')
ARMS=['s70_int8_30','s70_fp4_20_int8_10']
for case in ['race','clap0']:
    if case == 'clap0':
        deadline=time.monotonic()+900
        while not (ROOT/'results/research/E139/clap0/bf16_video.json').is_file():
            assert time.monotonic()<deadline, 'new clap BF16 reference not completed'
            time.sleep(10)
    subprocess.run([sys.executable,str(ROOT/'scripts/research/evaluate_e138_videos.py'),
                    '--experiment','E139','--case',case,'--arms',*ARMS],check=True)
    folder=ROOT/'results/research/E139'/case
    report=json.loads((folder/'video_metrics.json').read_text())
    source=ROOT/'third_party/anemoi-code'
    report['source_commit']=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    report['local_compatibility_changes']=['NVCC cassert preinclude flag', 'explicit host tensor-check array type']
    files=[source/'setup.py',source/'csrc/attention/cuda/sm120/q64_attention_host.cu',
           ROOT/'scripts/research/run_e139_draftattention2.py']
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    paths=[report['sources'][name] for name in ['bf16',*ARMS]]
    dest=Path('/data1/models/svdquant-wjq/research/20261031/E139_draftattention2')/case/f'{case}_bf16_int8_mixed.mp4'
    labels=['BF16','70% skip + 30% INT8','70% skip + 20% FP4 + 10% INT8']
    assert not dest.exists(), dest
    inputs=[av.open(path) for path in paths]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    try:
        with av.open(str(dest),'w',options={'movflags':'+faststart'}) as output:
            stream=output.add_stream('libx264',rate=24)
            stream.width=1536
            stream.height=320
            stream.pix_fmt='yuv420p'
            stream.options={'crf':'18','preset':'fast'}
            count=0
            for count,frames in enumerate(zip_longest(*(c.decode(video=0) for c in inputs)),1):
                assert count<=124 and all(f is not None for f in frames)
                canvas=Image.new('RGB',(1536,320),'black')
                draw=ImageDraw.Draw(canvas)
                for i,(frame,label) in enumerate(zip(frames,labels)):
                    canvas.paste(frame.to_image().resize((512,288)),(i*512,32))
                    draw.text((i*512+8,8),label,font=font,fill='white')
                if count==63:
                    canvas.save(folder/'comparison_poster.jpg')
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.array(canvas),format='rgb24')):
                    output.mux(packet)
            assert count==124
            for packet in stream.encode():
                output.mux(packet)
    finally:
        for container in inputs:
            container.close()
    report.update(comparison_video=str(dest),comparison_audio='none; individual originals contain audio')
    (folder/'video_metrics.json').write_text(json.dumps(report,indent=2)+'\n')

