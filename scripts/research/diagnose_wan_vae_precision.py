#!/usr/bin/env python3
"""Two saved BF16 DiT latents x BF16/FP32 VAE, no DiT recomputation.

CPU --check-only prepares the fixed comparison. --supervise launches a single
GPU2 worker with a 600-second outer deadline and a 60GiB allocation guard.
Original E022 sources, videos and latent artifacts are never modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data1/models/svdquant-wjq')
BASE=DATA/'models/Wan2.1-T2V-1.3B-Diffusers'
RD=ROOT/'results/research/E022'
GENERATION=RD/'video_baselines.json'
CASES=('vbench_197_r0','vbench_133_r0')
FRAMES=(0,19,38,57,76)


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def record(path):
    p=Path(path)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def tensor_sha(value,torch):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def inputs():
    generation=json.loads(GENERATION.read_text())
    require(generation['status']=='complete','Completed E022 baselines required')
    rows={key:generation['arms']['bf16']['trajectories'][key] for key in CASES}
    for key,row in rows.items():
        p=Path(row['final_latent']['path'])
        require(sha(p)==row['final_latent']['sha256'] and p.stat().st_size==row['final_latent']['bytes'],f'Saved latent changed: {key}')
    return rows


def check_only(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    rows=inputs();details={}
    for key,row in rows.items():
        x=torch.load(row['final_latent']['path'],map_location='cpu',weights_only=True)
        require(x.shape==(1,16,20,60,104) and x.dtype==torch.float64 and bool(torch.isfinite(x).all()),f'Invalid latent {key}')
        require(tensor_sha(x,torch)==row['final_latent_sha256'],f'Tensor identity changed {key}')
        details[key]=dict(prompt=row['prompt'],seed=row['seed'],input=row['final_latent'],tensor_sha256=row['final_latent_sha256'])
    compile(Path(__file__).read_text(),str(Path(__file__)),'exec')
    require(not torch.cuda.is_initialized(),'CPU prepare initialized CUDA')
    report=dict(status='complete',phase='CPU_prepare',cuda_initialized=False,source=record(Path(__file__)),
        cases=details,planned_vae_decodes=4,planned_dit_calls=0,vae_dtypes=['bfloat16','float32'],
        same_normalization_formula='saved FP64 latent -> FP32 -> VAE dtype; latent / (1 / config.latents_std in VAE dtype) + config.latents_mean in VAE dtype',
        normalization_limit='Only dtype changes. This includes normalization rounding, VAE weight rounding and arithmetic; it is not an isolated convolution-precision experiment.',
        spatial_decode=dict(latent_hw=[60,104],core=128,halo=0,actual_spatial_tiles=1),
        frames=77,fps=16,saved_frame_indices=list(FRAMES),gpu=2,wall_seconds=args.wall_seconds,
        checkpoint=str(BASE/'vae'),official_example=str(BASE/'README.md')+':175',
        purpose='Test whether the existing decode precision configuration materially changes the fixed bad/good examples. Neither input is discarded; no quantization-quality claim.')
    require(not args.check_output.exists(),f'Preserve {args.check_output}')
    save(args.check_output,report)
    print(json.dumps(dict(status='complete',output=str(args.check_output),cuda_initialized=False,planned_vae_decodes=4)),flush=True)


def supervise(args):
    supervisor=RD/'vae_precision_supervisor.json'
    log=ROOT/'results/logs/E022_vae_precision.log'
    require(not supervisor.exists() and not log.exists() and not args.output.exists() and not args.artifact_dir.exists(),'Preserve prior diagnostic artifacts')
    require(0<args.wall_seconds<=600,'Supervisor is capped at ten minutes')
    start=time.time();deadline=start+args.wall_seconds
    command=[sys.executable,str(Path(__file__).resolve()),'--deadline-unix',str(deadline),
             '--output',str(args.output),'--artifact-dir',str(args.artifact_dir)]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='2'
    log.parent.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',gpu=2,start_epoch=start,deadline_epoch=deadline,command=command,log=str(log),source=record(Path(__file__)))
    save(supervisor,state)
    with log.open('x') as stream:
        worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        state.update(status='running',pid=worker.pid);save(supervisor,state)
        timed_out=False
        try:code=worker.wait(timeout=max(0.01,deadline-time.time()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(worker.pid,signal.SIGTERM)
            try:code=worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGKILL);code=worker.wait()
    state.update(status='deadline_terminated' if timed_out else ('complete' if code==0 else 'failed'),returncode=code,seconds=time.time()-start)
    save(supervisor,state);print(json.dumps(state),flush=True)
    if code:raise SystemExit(code)


def execute(args):
    require(args.deadline_unix is not None and args.deadline_unix>time.time(),'Explicit future deadline required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root must launch this fixed diagnostic on GPU2')
    require(not args.output.exists() and not args.artifact_dir.exists(),'Preserve old outputs')
    rows=inputs()
    sys.path.insert(0,str(ROOT/'scripts'))
    import torch
    import diffusers
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from diffusers.utils import export_to_video
    from infer_rcm_wan_4step import decode_spatial_tiled
    from PIL import Image,ImageDraw,ImageFont
    import numpy as np
    require(torch.cuda.get_device_capability()==(12,0),'Expected SM120')
    require(torch.__version__=='2.11.0+cu128' and diffusers.__version__=='0.33.1','Use the E022 environment')
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();args.artifact_dir.mkdir(parents=True)
    report=dict(status='running',cases={},inputs={k:dict(final_latent=r['final_latent'],tensor_sha256=r['final_latent_sha256'],prompt=r['prompt'],seed=r['seed']) for k,r in rows.items()},
        source=record(Path(__file__)),generation=record(GENERATION),vae_config=record(BASE/'vae/config.json'),
        normalization='Exactly the historical expression in each VAE dtype: FP64 saved latent -> FP32 -> dtype; z/(1/std)+mean.',
        scope='Four VAE decodes only. Difference may include normalization, weights and VAE arithmetic precision; no DiT or quantization attribution.',
        environment=dict(python=sys.executable,torch=torch.__version__,diffusers=diffusers.__version__,gpu=torch.cuda.get_device_name(),cuda_visible_devices='2',tf32_matmul=False,tf32_cudnn=False),
        actual_vae_decode_calls=0,actual_dit_calls=0,frame_indices=list(FRAMES),limitations=['Official decoder clamps to [-1,1]; capture decoder-module output ranges before that clamp as diagnostics, not quality scores.'])
    weights=BASE/'vae/diffusion_pytorch_model.safetensors'
    report['vae_weight_metadata']=dict(path=str(weights),bytes=weights.stat().st_size,mtime_ns=weights.stat().st_mtime_ns)
    started=time.time();processor=VideoProcessor(vae_scale_factor=8)
    bf16_outputs={};bf16_normalized={};frame_images={}

    def budget():
        if time.time()>=args.deadline_unix:raise TimeoutError('VAE diagnostic deadline expired')
        if torch.cuda.max_memory_allocated()>60*1024**3:raise MemoryError('60GiB VAE allocation guard exceeded')

    def checkpoint():
        report.update(seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        save(args.output,report);budget()

    def pixel_stats(decoded):
        # decode_spatial_tiled returns CPU FP32 in [B,C,T,H,W].
        require(bool(torch.isfinite(decoded).all()),'Nonfinite decoded pixels')
        rgb=(decoded[0]+1)*0.5
        return dict(min=float(decoded.min()),max=float(decoded.max()),
            rgb_mean=rgb.mean(dim=(1,2,3)).tolist(),rgb_std=rgb.std(dim=(1,2,3),unbiased=False).tolist(),
            zero_fraction_per_channel=(rgb<=0).float().mean(dim=(1,2,3)).tolist(),
            one_fraction_per_channel=(rgb>=1).float().mean(dim=(1,2,3)).tolist(),
            per_frame_rgb_mean=rgb.mean(dim=(2,3)).t().tolist(),
            column_rgb_mean_over_time_height=rgb.mean(dim=(1,2)).tolist())

    try:
        with torch.inference_mode():
            for dtype_name,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
                budget()
                vae=AutoencoderKLWan.from_pretrained(BASE/'vae',torch_dtype=dtype,local_files_only=True).cuda().eval()
                mean=torch.tensor(vae.config.latents_mean,device='cuda',dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1)
                reciprocal_std=1.0/torch.tensor(vae.config.latents_std,device='cuda',dtype=vae.dtype).view(1,vae.config.z_dim,1,1,1)
                for key,row in rows.items():
                    budget();folder=args.artifact_dir/key/dtype_name;folder.mkdir(parents=True)
                    latent=torch.load(row['final_latent']['path'],map_location='cpu',weights_only=True)
                    require(tensor_sha(latent,torch)==row['final_latent_sha256'],'Saved latent identity changed')
                    normalized=latent.cuda().float().to(vae.dtype)/reciprocal_std+mean
                    require(bool(torch.isfinite(normalized).all()),'Nonfinite normalized input')
                    normalized_cpu=normalized.float().cpu()
                    entry=dict(normalized_input_sha256=tensor_sha(normalized,torch),normalization_dtype=str(dtype),preclamp_decoder_chunks=[])
                    report['cases'].setdefault(key,{})[dtype_name]=entry
                    if dtype_name=='bf16':bf16_normalized[key]=normalized_cpu
                    else:
                        delta=normalized_cpu-bf16_normalized.pop(key)
                        entry['normalized_input_difference_vs_bf16']=dict(mae=float(delta.abs().mean()),max_abs=float(delta.abs().max()))
                        del delta
                    def capture(module,inputs,output):
                        budget();x=output.detach()
                        finite=torch.isfinite(x)
                        entry['preclamp_decoder_chunks'].append(dict(shape=list(x.shape),dtype=str(x.dtype),finite_fraction=float(finite.float().mean()),
                            min=float(x.min()) if bool(finite.all()) else None,max=float(x.max()) if bool(finite.all()) else None,
                            fraction_below_minus_one=float((x < -1).float().mean()),fraction_above_one=float((x > 1).float().mean())))
                    hook=vae.decoder.register_forward_hook(capture)
                    vae.clear_cache();began=time.time()
                    try:decoded=decode_spatial_tiled(vae,normalized,core=128,halo=0)
                    finally:hook.remove();vae.clear_cache()
                    torch.cuda.synchronize();report['actual_vae_decode_calls']+=1
                    require(decoded.shape==(1,3,77,480,832),'Full 77-frame output required')
                    entry.update(seconds=time.time()-began,decoded_tensor_sha256=tensor_sha(decoded,torch),pixel_stats=pixel_stats(decoded))
                    if dtype_name=='bf16':bf16_outputs[key]=decoded
                    else:
                        reference=bf16_outputs.pop(key);delta=decoded-reference
                        entry['decoded_difference_vs_bf16']=dict(mae=float(delta.abs().mean()),rmse=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),per_frame_mae=delta.abs().mean(dim=(0,1,3,4)).tolist())
                        del reference,delta
                    frames=processor.postprocess_video(decoded,output_type='np')[0]
                    require(len(frames)==77,'Postprocessor frame count changed')
                    video=folder/'video.mp4';export_to_video(frames,str(video),fps=16)
                    entry['video']=record(video)|dict(frames=77,height=480,width=832,fps=16)
                    if dtype_name=='bf16':entry['video_sha_matches_original']=entry['video']['sha256']==row['video']['sha256']
                    entry['saved_frames']=[]
                    for index in FRAMES:
                        image=Image.fromarray((frames[index]*255).round().clip(0,255).astype(np.uint8))
                        path=folder/f'frame_{index:03d}.png';image.save(path)
                        entry['saved_frames'].append(record(path)|dict(frame_index=index))
                        frame_images[(key,dtype_name,index)]=image.resize((416,240),Image.Resampling.LANCZOS)
                    del latent,normalized,normalized_cpu,decoded,frames
                    gc.collect();torch.cuda.empty_cache();checkpoint()
                    print(json.dumps(dict(case=key,vae_dtype=dtype_name,seconds=entry['seconds'],rgb_mean=entry['pixel_stats']['rgb_mean'])),flush=True)
                del vae,mean,reciprocal_std;gc.collect();torch.cuda.empty_cache()
        try:
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
        except OSError:font=ImageFont.load_default(size=22)
        report['contact_sheets']=[]
        for key in CASES:
            sheet=Image.new('RGB',(2320,670),(247,248,250));draw=ImageDraw.Draw(sheet)
            draw.text((15,15),f'{key} | SAME saved BF16 DiT latent | {rows[key]["prompt"]}',font=font,fill='black')
            for col,index in enumerate(FRAMES):draw.text((200+col*422,62),f'frame {index}',font=font,fill='black')
            for row,dtype_name in enumerate(('bf16','fp32')):
                y=110+row*260;draw.text((15,y+100),f'VAE {dtype_name}',font=font,fill='black')
                for col,index in enumerate(FRAMES):sheet.paste(frame_images[(key,dtype_name,index)],(200+col*422,y))
            draw.text((15,635),'Fixed examples and frames; decode precision diagnostic, not a quantization or motion-quality conclusion.',font=font,fill='black')
            path=args.artifact_dir/(key+'_vae_precision.png');sheet.save(path);report['contact_sheets'].append(record(path))
        require(report['actual_vae_decode_calls']==4,'Expected exactly four decodes')
        report['status']='complete';checkpoint()
    except BaseException:
        report.update(status='failed_partial_preserved',error=traceback.format_exc());save(args.output,report);raise
    print(json.dumps(dict(status=report['status'],output=str(args.output),vae_decodes=4,dit_calls=0)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--supervise',action='store_true')
    parser.add_argument('--wall-seconds',type=int,default=600)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--check-output',type=Path,default=RD/'vae_precision_cpucheck.json')
    parser.add_argument('--output',type=Path,default=RD/'vae_precision_diagnostic.json')
    parser.add_argument('--artifact-dir',type=Path,default=DATA/'research/20261003/E022/vae_precision_diagnostic')
    args=parser.parse_args()
    if args.check_only:check_only(args)
    elif args.supervise:supervise(args)
    else:execute(args)


if __name__=='__main__':main()
