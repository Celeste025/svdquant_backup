#!/usr/bin/env python3
"""E041: unselected native-frame contact sheets for six existing E038 clips."""
import json
from pathlib import Path
import time
import cv2
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data1/models/svdquant-wjq/research/20261003/E041')
REPORT=ROOT/'results/research/E041/frame_index.json'
assert not REPORT.exists() and not OUT.exists(), 'Preserve previous rendering'
started=time.monotonic();OUT.mkdir(parents=True)
source=ROOT/'results/research/E038/evaluation_summary.json'
ref=json.loads(source.read_text());rows=[]
for replica in (0,1):
    case_id=f'vbench0161_r{replica}'
    for arm in ('bf16','global_mean','coarse16'):
        old=ref['per_case'][case_id]['arms'][arm]
        path=Path(old['video']['file']);cap=cv2.VideoCapture(str(path))
        assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS);assert abs(fps-24)<1e-5
        frames=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert bgr.shape==(576,1024,3)
            frames.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
        cap.release();assert len(frames)==124
        pages=[]
        for start in range(0,124,16):
            end=min(start+16,124)
            sheet=Image.new('RGB',(1536,952),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,4),f'{case_id} / {arm} | every original frame | 24fps | page {start//16+1}/8',fill='black')
            for idx in range(start,end):
                col=(idx-start)%4;row=(idx-start)//4
                x=col*384;y=24+row*232
                draw.text((x+4,y+2),f'frame {idx:03d} | {idx/24:.3f}s',fill='black')
                sheet.paste(frames[idx].resize((384,216),Image.Resampling.LANCZOS),(x,y+16))
            target=OUT/f'{case_id}_{arm}_page{start//16:02d}.png';sheet.save(target)
            pages.append(dict(file=str(target),first_frame=start,last_frame=end-1))
        rows.append(dict(case_id=case_id,replica=replica,arm=arm,source_video=old['video'],
            actual_frames=len(frames),fps=fps,source_width=1024,source_height=576,
            display_width=384,display_height=216,frame_stride=1,pages=pages))
        print(f'{case_id} {arm}: all 124 frames in 8 pages',flush=True)
REPORT.parent.mkdir(parents=True,exist_ok=True)
REPORT.write_text(json.dumps(dict(experiment='E041',status='complete',source_summary=str(source),
    cpu_only=True,model_calls=0,gpu_calls=0,clips=rows,seconds=time.monotonic()-started,
    limitations='Sequential native-frame visual review, not real-time video playback or blinded human assessment.'),indent=2))
print(str(REPORT),flush=True)
