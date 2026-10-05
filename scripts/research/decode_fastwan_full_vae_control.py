#!/usr/bin/env python3
"""E025: full FP32 Wan VAE on all sixteen saved E024 latents; zero DiT calls.

This intervenes on decoder implementation + weights + dtype, not solely decoder
precision. It does not claim byte equivalence to FastVideo's BF16 full VAE.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq')
GENERATION = ROOT / 'results/research/E024/suite_run.json'
RD = ROOT / 'results/research/E025'
ARTIFACTS = DATA / 'research/20261003/E025/decoded'
VAE = DATA / 'models/FastWan-QAD-1.3B-621c6aeb/vae'
FRAMES = (0, 20, 40, 60, 80)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def record(path):
    p = Path(path).absolute()
    return dict(file=str(p), bytes=p.stat().st_size, sha256=sha(p))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'); tmp.replace(path)


def tensor_record(value, torch):
    x = value.detach().cpu().contiguous()
    return dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(x.isfinite().all()),
                sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def inputs(torch, budget):
    g = json.loads(GENERATION.read_text())
    assert g['status'] == 'complete' and g['phase'] == 'suite' and len(g['cases']) == 16
    assert [r['trajectory_id'] for r in g['cases']] == [r['trajectory_id'] for r in g['trajectories']]
    assets_ref = g['asset_manifest']; assert record(assets_ref['file']) == assets_ref
    assets = json.loads(Path(assets_ref['file']).read_text()); assert assets['status'] == 'complete'
    vae_weights = assets['files']['hf/vae/diffusion_pytorch_model.safetensors']
    assert vae_weights['status'] == 'verified'
    assert (VAE / 'diffusion_pytorch_model.safetensors').resolve() == Path(vae_weights['path']).resolve()
    current_weight = record(vae_weights['path'])
    assert current_weight['bytes'] == vae_weights['bytes'] and current_weight['sha256'] == vae_weights['sha256']
    config = json.loads((VAE / 'config.json').read_text()); assert config['z_dim'] == 16
    assert len(config['latents_mean']) == len(config['latents_std']) == 16 and all(v > 0 for v in config['latents_std'])
    config_ref = assets['files']['hf/vae/config.json']
    assert sha(VAE / 'config.json') == config_ref['sha256']
    rows = []
    for original in g['cases']:
        budget(); assert original['status'] == 'complete'
        ref = original['latents_artifact']; assert record(ref['file']) == ref
        value = torch.load(ref['file'], map_location='cpu', weights_only=True)['final_latents']
        tr = tensor_record(value, torch)
        assert tr == original['tensors']['final_latents']
        assert tr['shape'] == [1,16,21,60,104] and tr['dtype'] == 'torch.float32' and tr['finite']
        rows.append(dict(**{k:original[k] for k in ('trajectory_id','prompt_id','prompt','seed','replica','manifest_index')},
                         latents_artifact=ref, tensors=dict(final_latents=tr), taehv_video=original['video'], status='input_verified'))
        del value
    return g, rows, current_weight, assets_ref, config


def contact_sheet(reference_video, decoded_frames, path, identity):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    reader = imageio.get_reader(reference_video)
    try: reference = [reader.get_data(i) for i in FRAMES]
    finally: reader.close()
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except OSError: font = ImageFont.load_default()
    sheet = Image.new('RGB',(2240,625),'white'); draw=ImageDraw.Draw(sheet)
    draw.text((12,8), identity,fill='black',font=font)
    for i,n in enumerate(FRAMES):draw.text((140+i*416,45),f'frame {n}',fill='black',font=font)
    for row,images in enumerate((reference,[decoded_frames[i] for i in FRAMES])):
        draw.text((8,160+row*265),'TAEHV FP16' if row==0 else 'Wan FP32',fill='black',font=font)
        for col,array in enumerate(images):sheet.paste(Image.fromarray(array).resize((416,240)),(140+col*416,78+row*265))
    draw.text((12,590),'Same saved latent; decoder stack + dtype intervention, not a new DiT sample.',fill='black',font=font)
    sheet.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--artifact-dir',type=Path,default=ARTIFACTS);p.add_argument('--deadline-unix',type=float)
    args=p.parse_args(); output=args.output or RD/('decode_cpucheck.json' if args.check_only else 'decode_run.json')
    assert not output.exists(),output
    if args.check_only:assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    else:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and args.deadline_unix and args.deadline_unix>time.time()
    started=time.time()
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E025 external deadline exceeded')
    report=dict(experiment='E025',phase='check' if args.check_only else 'suite',status='running',cases=[],
        source=record(Path(__file__)),input_generation=record(GENERATION),sampling=dict(height=480,width=832,num_frames=81,fps=16),
        actual_dit_calls=0,actual_vae_decode_calls=0,model_loads=dict(vae=0,text_encoder=0,transformer=0),
        decoder=dict(implementation='diffusers.AutoencoderKLWan',dtype='float32',path=str(VAE),spatial_core=128,spatial_halo=0,actual_spatial_tiles=1),
        normalization='One FP32 denormalization: z / (1 / config.latents_std) + config.latents_mean, channel axis1.',
        scope='All sixteen E024 final latents, unchanged. Decoder stack + dtype control; not byte equivalence to official FastVideo BF16 VAE, not a quantization or same-latent semantic-quality proof.',
        timing_scope='Per-video decode includes preclamp finite/clip diagnostics; not a deployment speed benchmark.')
    def progress():
        report['seconds']=time.time()-started;save(output,report)
    try:
        import torch
        torch.set_num_threads(6)
        generation,rows,weight,assets,config=inputs(torch,budget)
        report.update(trajectories=generation['trajectories'],input_asset_manifest=assets,vae_weights=weight,vae_config=record(VAE/'config.json'),cases=rows,
                      environment=dict(python=sys.executable,torch=torch.__version__,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        report['decode_helper']=record(ROOT/'scripts/infer_rcm_wan_4step.py')
        if args.check_only:
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False,planned_vae_decodes=16)
            return
        assert not args.artifact_dir.exists();args.artifact_dir.mkdir(parents=True)
        sys.path.insert(0,str(ROOT/'scripts'))
        import diffusers
        import imageio.v2 as imageio
        from diffusers import AutoencoderKLWan
        from infer_rcm_wan_4step import decode_spatial_tiled
        assert torch.__version__=='2.11.0+cu128' and diffusers.__version__=='0.33.1','Use historical PTQ environment'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        prop=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction(min(1,60*2**30/prop.total_memory))
        report['environment'].update(diffusers=diffusers.__version__,device=prop.name,tf32=False)
        budget();vae=AutoencoderKLWan.from_pretrained(VAE,torch_dtype=torch.float32,local_files_only=True).cuda().eval()
        report['model_loads']['vae']=1
        mean=torch.tensor(vae.config.latents_mean,device='cuda',dtype=torch.float32).view(1,16,1,1,1)
        invstd=1.0/torch.tensor(vae.config.latents_std,device='cuda',dtype=torch.float32).view(1,16,1,1,1)
        with torch.inference_mode():
            for row in rows:
                budget();row['status']='running';progress();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                x=torch.load(row['latents_artifact']['file'],map_location='cpu',weights_only=True)['final_latents']
                assert tensor_record(x,torch)==row['tensors']['final_latents']
                z=x.cuda().float()/invstd+mean;row['denormalized_input']=tensor_record(z,torch);assert row['denormalized_input']['finite']
                chunks=[]
                def capture(module,inputs,out):
                    budget();v=out.detach();finite=bool(torch.isfinite(v).all());assert finite
                    chunks.append(dict(shape=list(v.shape),dtype=str(v.dtype),finite=finite,elements=v.numel(),
                                       below_minus_one=int((v < -1).sum()),above_one=int((v > 1).sum()),min=float(v.min()),max=float(v.max())))
                hook=vae.decoder.register_forward_hook(capture);vae.clear_cache();torch.cuda.synchronize();t0=time.perf_counter()
                try:decoded=decode_spatial_tiled(vae,z,core=128,halo=0)
                finally:hook.remove();vae.clear_cache()
                torch.cuda.synchronize();report['actual_vae_decode_calls']+=1
                row['decode_seconds_with_diagnostics']=time.perf_counter()-t0
                assert decoded.shape==(1,3,81,480,832) and bool(torch.isfinite(decoded).all())
                row['decoded_tensor']=tensor_record(decoded,torch);row['preclamp_decoder_chunks']=chunks
                n=sum(v['elements'] for v in chunks);assert n==decoded.numel()
                row['clip_rates']=dict(below_minus_one=sum(v['below_minus_one'] for v in chunks)/n,
                    above_one=sum(v['above_one'] for v in chunks)/n,
                    output_minus_one_fraction=float((decoded<=-1).float().mean()),output_plus_one_fraction=float((decoded>=1).float().mean()))
                frames=((decoded[0].permute(1,2,3,0)/2+0.5).clamp(0,1)*255).to(torch.uint8).numpy()
                video=args.artifact_dir/(row['trajectory_id']+'.mp4');imageio.mimsave(str(video),frames,fps=16,format='mp4')
                row['video']=record(video);row['frames_shape']=list(frames.shape);row['frames_sha256']=hashlib.sha256(frames.tobytes()).hexdigest()
                sheet=args.artifact_dir/(row['trajectory_id']+'_decoder_control.png')
                contact_sheet(row['taehv_video']['file'],frames,sheet,row['trajectory_id']+' | '+row['prompt'][:140]);row['contact_sheet']=record(sheet)
                row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30;row['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
                assert row['peak_allocated_gib']<=60
                row['status']='complete';progress();print(json.dumps(dict(trajectory_id=row['trajectory_id'],seconds=row['decode_seconds_with_diagnostics'],peak_gib=row['peak_allocated_gib'])),flush=True)
                del x,z,decoded,frames;gc.collect()
        assert report['actual_vae_decode_calls']==16 and report['actual_dit_calls']==0
        report.update(status='complete',cuda_initialized=torch.cuda.is_initialized());budget()
    except BaseException:
        report.update(status='failed_partial_preserved',error=traceback.format_exc());raise
    finally:
        progress();print(json.dumps(dict(status=report['status'],output=str(output))),flush=True)


if __name__=='__main__':main()
