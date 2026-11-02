#!/usr/bin/env python3
"""E113 matched-seed four-arm video montage and frame-paired RGB distances."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
ARMS=('bf16','sage3','native_fp4','native_mixed')
LABELS={'bf16':'BF16','sage3':'Sage3 FP4','native_fp4':'Native FP4','native_mixed':'Native mixed FP4/FP8'}
DATA=Path('/data1/models/svdquant-wjq/research/20261020/E113_h3_mixed_pv_generation')


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def frames(path):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        if (stream.width,stream.height,float(stream.average_rate))!=(1024,576,24.):
            raise RuntimeError(f'video format changed: {path}')
        yield from (frame.to_ndarray(format='rgb24') for frame in container.decode(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('race','clap0'),required=True)
    parser.add_argument('--check-only',action='store_true',
                        help='validate all inputs and decoded frame counts without writing outputs')
    args=parser.parse_args()
    case=args.case
    e101=json.loads((ROOT/'results/research/E101'/case/'video_pixel_comparison.json').read_text())
    expected_prompt,expected_seed={'race':(16,20026),'clap0':(161,38001610)}[case]
    if (e101['case'],e101['seed'],e101['frames'],e101['fps'],
        e101['video_shape'])!=(case,expected_seed,124,24,[576,1024,3]):
        raise RuntimeError('BF16 reference identity or video contract changed')
    sources={'bf16':Path(e101['inputs']['bf16'])}
    research=ROOT/'results/research/E113'/case
    reports={}
    for arm in ARMS[1:]:
        video_info=json.loads((research/f'{arm}_video.json').read_text())
        denoise_path=research/f'{arm}_denoise.json'
        denoise=json.loads(denoise_path.read_text())
        if (video_info['case'],video_info['arm'],video_info['frames'],video_info['audio_streams'])!=(case,arm,124,1):
            raise RuntimeError(f'invalid media manifest: {arm}')
        if denoise['status']!='denoise_complete' or len(denoise['steps'])!=20:
            raise RuntimeError(f'incomplete denoising: {arm}')
        if sha256(denoise_path)!=video_info['denoise_report_sha256']:
            raise RuntimeError(f'denoise report hash changed: {arm}')
        sources[arm]=Path(video_info['video'])
        if not sources[arm].is_file() or sha256(sources[arm])!=video_info['sha256']:
            raise RuntimeError(f'video missing or hash mismatch: {arm}')
        reports[arm]=denoise
    if len({(r['prompt_id'],r['seed'],json.dumps(r['settings'],sort_keys=True),
             r['initial_latent_sha256']['video'],r['initial_latent_sha256']['audio'])
            for r in reports.values()})!=1:
        raise RuntimeError('attention-arm prompt, seed, settings or initial latent differs')
    if (reports['sage3']['prompt_id'],reports['sage3']['seed'])!=(expected_prompt,expected_seed):
        raise RuntimeError('attention arms do not match the BF16 prompt/seed')
    if not sources['bf16'].is_file():
        raise FileNotFoundError(sources['bf16'])
    if args.check_only:
        frame_counts={arm:sum(1 for _ in frames(path)) for arm,path in sources.items()}
        if set(frame_counts.values())!={124}:
            raise RuntimeError(f'input frame count mismatch: {frame_counts}')
        print(json.dumps({'case':case,'prompt_id':reports['sage3']['prompt_id'],
                          'seed':reports['sage3']['seed'],'frames':frame_counts,
                          'sources':{arm:str(path) for arm,path in sources.items()}},
                         ensure_ascii=False,indent=2),flush=True)
        return
    output_dir=DATA/case
    output_dir.mkdir(parents=True,exist_ok=True)
    montage=output_dir/f'{case}_four_arm_attention_montage.mp4'
    poster=research/'four_arm_attention_poster.jpg'
    report=research/'four_arm_attention_summary.json'
    if any(path.exists() for path in (montage,poster,report)):
        raise FileExistsError('refusing to overwrite E113 summary/montage')
    font_file=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_file),18) if font_file.is_file() else ImageFont.load_default()
    w,h,band=512,288,30
    width,height=2*w,2*(h+band)
    writer=av.open(str(montage),'w')
    stream=writer.add_stream('libx264',rate=24)
    stream.width,stream.height=width,height
    stream.pix_fmt='yuv420p'
    stream.options={'crf':'20','preset':'fast'}
    pairs=(('sage3','bf16'),('native_fp4','bf16'),('native_mixed','bf16'),
           ('native_mixed','native_fp4'),('native_fp4','sage3'),('native_mixed','sage3'))
    squared={f'{a}_vs_{b}':0 for a,b in pairs}
    energy={arm:0 for arm in ARMS}
    captured=[]
    count=0
    try:
        for count,frame_set in enumerate(zip_longest(*(frames(sources[a]) for a in ARMS)),start=1):
            if count>124 or any(frame is None or frame.shape!=(576,1024,3) for frame in frame_set):
                raise RuntimeError(f'frame mismatch at {count-1}')
            images=dict(zip(ARMS,frame_set,strict=True))
            numeric={a:images[a].astype(np.int32) for a in ARMS}
            for a in ARMS:
                energy[a]+=int(np.square(numeric[a]).sum(dtype=np.int64))
            for a,b in pairs:
                squared[f'{a}_vs_{b}']+=int(np.square(numeric[a]-numeric[b]).sum(dtype=np.int64))
            canvas=Image.new('RGB',(width,height),(18,22,31))
            draw=ImageDraw.Draw(canvas)
            for index,arm in enumerate(ARMS):
                x=(index%2)*w
                y=(index//2)*(h+band)
                draw.text((x+10,y+5),LABELS[arm],font=font,fill='white')
                canvas.paste(Image.fromarray(images[arm]).resize((w,h),Image.Resampling.BILINEAR),(x,y+band))
            if count in (1,31,61,91):
                captured.append(canvas.copy())
            packet_frame=av.VideoFrame.from_image(canvas)
            for packet in stream.encode(packet_frame):
                writer.mux(packet)
        if count!=124 or len(captured)!=4:
            raise RuntimeError(f'incomplete frame sequence: {count}')
        for packet in stream.encode():
            writer.mux(packet)
    finally:
        writer.close()
    poster_canvas=Image.new('RGB',(width,4*height),'white')
    for index,image in enumerate(captured):
        poster_canvas.paste(image,(0,index*height))
    poster_canvas.save(poster,quality=90,optimize=True)
    distances={key:squared[key]/energy[key.split('_vs_')[1]] for key in squared}
    payload={'format':'e113-four-arm-attention-video-summary-v1','case':case,
             'scope':'124 aligned RGB8 frames, 1024x576, no temporal/spatial alignment',
             'interpretation':'RGB NMSE is distance to the named reference video, not a perceptual quality score',
             'sources':{a:str(sources[a].resolve()) for a in ARMS},
             'sha256':{a:sha256(sources[a]) for a in ARMS},
             'prompt_id':reports['sage3']['prompt_id'],'seed':reports['sage3']['seed'],
             'initial_latent_sha256':reports['sage3']['initial_latent_sha256'],
             'frames':count,'rgb_nmse':distances,
             'montage':str(montage.resolve()),'montage_sha256':sha256(montage),
             'poster':str(poster.resolve()),'poster_sha256':sha256(poster)}
    report.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'case':case,'rgb_nmse':distances,'montage':str(montage)},indent=2),flush=True)


if __name__=='__main__':
    main()
