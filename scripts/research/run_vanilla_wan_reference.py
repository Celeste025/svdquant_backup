#!/usr/bin/env python3
"""E043: direct original WanPipeline teacher; two fixed seeds per prompt worker."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E043_vanilla_wan_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E043_vanilla_wan_reference_plan.md'
RECEIPT = ROOT/'results/research/E024/assets_manifest_attempt3.json'
TRANSFORMER_RECEIPT = ROOT/'results/research/E043/base_transformer_provenance.json'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def record(path):
    p = Path(path).resolve()
    return dict(file=str(p), bytes=p.stat().st_size, sha256=sha(p))


def tensor_record(x):
    x = x.detach().cpu().contiguous()
    return dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(x.isfinite().all()),
                sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def save_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.replace(path)


def config_dict(config):
    return config.to_dict() if hasattr(config, 'to_dict') else dict(config)


def budget(args):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix, 'Worker deadline reached')


def prerequisites(args, report):
    m = json.loads(args.manifest.read_text())
    cases = [c for c in m['cases'] if c['prompt_id'] == args.prompt_id]
    require(len(cases) == 2 and len({c['prompt'] for c in cases}) == 1, 'Expected one fixed prompt/two seeds')
    require(len({c['seed'] for c in cases}) == 2, 'Seeds must differ')
    for c in cases:
        require(hashlib.sha256(c['prompt'].encode()).hexdigest() == c['prompt_sha256'], 'Prompt hash differs')
    require(hashlib.sha256(m['negative_prompt'].encode()).hexdigest() == m['negative_prompt_sha256'], 'Negative hash differs')
    base = Path(m['model_dir'])
    configs = {str(p.relative_to(base)): json.loads(p.read_text()) for p in
               [base/'model_index.json', base/'transformer/config.json', base/'vae/config.json',
                base/'text_encoder/config.json', base/'scheduler/scheduler_config.json']}
    require(configs['model_index.json']['_class_name'] == 'WanPipeline', 'Not a WanPipeline checkpoint')
    require(configs['transformer/config.json']['num_layers'] == 30, 'Expected original 1.3B geometry')
    assets = []
    for p in sorted(base.rglob('*')):
        if p.is_file() and (p.suffix in ('.safetensors', '.json', '.model')):
            st = p.stat()
            assets.append(dict(file=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    from diffusers import WanTransformer3DModel
    from transformers import UMT5EncoderModel, AutoConfig
    text_config = AutoConfig.from_pretrained(str(base/'text_encoder'), local_files_only=True)
    require(config_dict(text_config)['model_type'] == 'umt5', 'Text config serialization differs')
    vae_init_source = inspect.getsource(AutoencoderKLWan.__init__)
    require('self.use_tiling = False' in vae_init_source and 'self.use_slicing = False' in vae_init_source,
            'VAE default tiling/slicing attributes differ')
    source_paths = [Path(__file__), args.manifest, PLAN, Path(inspect.getsourcefile(WanPipeline)),
                    Path(inspect.getsourcefile(WanTransformer3DModel)), Path(inspect.getsourcefile(AutoencoderKLWan)),
                    Path(inspect.getsourcefile(UniPCMultistepScheduler)), Path(inspect.getsourcefile(UMT5EncoderModel))]
    import diffusers
    report.update(manifest=record(args.manifest), sources=[record(p) for p in source_paths],
                  inherited_te_vae_asset_receipt=record(RECEIPT), base_transformer_receipt=record(TRANSFORMER_RECEIPT),
                  assets_stat=assets, disk_configs=configs,
                  versions=dict(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__,
                                imageio=imageio.__version__ if hasattr(imageio, '__version__') else 'v2 API',
                                ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(), diffusers_path=diffusers.__file__),
                  settings=m['settings'], prompt_id=args.prompt_id, selected_cases=cases)
    schedule = UniPCMultistepScheduler.from_config(configs['scheduler/scheduler_config.json'], flow_shift=8.)
    schedule.set_timesteps(m['settings']['num_inference_steps'], device='cpu')
    report['cpu_schedule'] = dict(config=dict(schedule.config), timesteps=schedule.timesteps.tolist(),
                                  sigmas=schedule.sigmas.tolist())
    require(len(schedule.timesteps) == 50 and schedule.config.flow_shift == 8., 'Scheduler contract differs')
    require(all(hasattr(WanPipeline, k) for k in ('encode_prompt', 'prepare_latents', '__call__')), 'Pipeline API missing')
    return m, cases


def write_media(frames, case_dir, fps):
    require(frames.shape == (81, 480, 832, 3) and np.isfinite(frames).all(), 'Decoded video shape/finite mismatch')
    pixels = np.rint(np.clip(frames, 0, 1)*255).astype(np.uint8)
    video = case_dir/'video.mp4'
    with imageio.get_writer(str(video), format='FFMPEG', mode='I', fps=fps, codec='libx264',
                            pixelformat='yuv420p', output_params=['-crf', '18', '-preset', 'medium']) as writer:
        for frame in pixels:
            writer.append_data(frame)
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    metadata = next(reader)
    count = 0
    for frame in reader:
        require(len(frame) == 480*832*3, 'Encoded frame byte count differs')
        count += 1
    media = dict(frames=count, width=metadata['size'][0], height=metadata['size'][1],
                 fps=float(metadata['fps']), audio_streams=0)
    require(media == dict(frames=81, width=832, height=480, fps=16., audio_streams=0), 'Encoded media contract differs')
    ids = list(range(0, 81, 10))
    sheet = Image.new('RGB', (3*416, 3*(240+24)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, index in enumerate(ids):
        left, top = (i % 3)*416, (i//3)*264
        sheet.paste(Image.fromarray(pixels[index]).resize((416, 240)), (left, top+24))
        draw.text((left+5, top+5), f'frame {index} / {index/fps:.3f} s', fill='black')
    image = case_dir/'contact_sheet.png'
    sheet.save(image)
    return dict(video=record(video), media=media, media_decoder='imageio_ffmpeg bundled FFmpeg',
                audio_stream_basis='Video-only imageio writer; no audio input supplied',
                contact_sheet=record(image), contact_frames=ids,
                decoded_shape=list(frames.shape), decoded_finite=True,
                normalized_pixel_min=float(frames.min()), normalized_pixel_max=float(frames.max()))


@torch.inference_mode()
def run(args, m, cases, report):
    budget(args)
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    stage = Path(m['data_dir'])/f'prompt_{args.prompt_id}'
    stage.mkdir(parents=True, exist_ok=False)
    counts = report['actual_counts']
    vae = AutoencoderKLWan.from_pretrained(m['model_dir'], subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(m['model_dir'], vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=m['settings']['flow_shift'])
    pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
    require(pipe.transformer.dtype == torch.bfloat16 and pipe.text_encoder.dtype == torch.bfloat16 and
            pipe.vae.dtype == torch.float32, 'Model dtype contract differs')
    require(not pipe.vae.use_tiling and not pipe.vae.use_slicing, 'Unexpected VAE tiling/slicing')
    report['actual_configs'] = {k: config_dict(getattr(pipe, k).config) for k in ('transformer','vae','text_encoder','scheduler')}
    report['pipeline_config'] = dict(pipe.config)
    report['resident_allocated_bytes'] = torch.cuda.memory_allocated()
    original_dit, original_te = pipe.transformer.forward, pipe.text_encoder.forward
    original_step, original_decode = pipe.scheduler.step, pipe.vae.decode

    def dit(*a, **kw):
        budget(args); counts['dit'] += 1
        require(kw['hidden_states'].dtype == torch.bfloat16, 'DiT input must be BF16')
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return original_dit(*a, **kw)

    def te(*a, **kw):
        budget(args); counts['text_encoder'] += 1
        return original_te(*a, **kw)

    def step(*a, **kw):
        budget(args); counts['scheduler'] += 1
        return original_step(*a, **kw)

    def decode(*a, **kw):
        budget(args); counts['public_vae_decode'] += 1
        require(a[0].dtype == torch.float32, 'VAE input must be FP32')
        output = original_decode(*a, **kw)
        decoded = output[0]
        report['cases'][-1]['raw_decoder'] = dict(shape=list(decoded.shape), dtype=str(decoded.dtype),
            finite=bool(decoded.isfinite().all()), min=float(decoded.min()), max=float(decoded.max()))
        require(report['cases'][-1]['raw_decoder']['finite'], 'Nonfinite raw VAE output')
        return output

    pipe.transformer.forward, pipe.text_encoder.forward = dit, te
    pipe.scheduler.step, pipe.vae.decode = step, decode
    started = time.monotonic()
    positive, negative = pipe.encode_prompt(prompt=cases[0]['prompt'], negative_prompt=m['negative_prompt'],
        do_classifier_free_guidance=True, device=torch.device('cuda'), max_sequence_length=512)
    torch.cuda.synchronize()
    require(bool(positive.isfinite().all() and negative.isfinite().all()), 'Nonfinite embeddings')
    embedding_path = stage/'embeddings.pt'
    torch.save(dict(prompt_embeds=positive.cpu(), negative_prompt_embeds=negative.cpu()), embedding_path)
    report['embeddings'] = dict(artifact=record(embedding_path), prompt_embeds=tensor_record(positive),
        negative_prompt_embeds=tensor_record(negative), seconds=time.monotonic()-started,
        actual_text_encoder_calls=counts['text_encoder'])
    report['cases'] = []
    for case in cases:
        budget(args)
        case_dir = Path(m['data_dir'])/case['case_id']
        case_dir.mkdir(parents=True, exist_ok=False)
        row = dict(**case, status='running', variant='vanilla_wan_bf16', steps=[])
        report['cases'].append(row)
        initial = torch.randn((1,16,21,60,104), generator=torch.Generator(device='cpu').manual_seed(case['seed']), dtype=torch.float32)
        initial_path = case_dir/'initial_noise.pt'
        torch.save(initial, initial_path)
        row['initial_noise'] = dict(artifact=record(initial_path), tensor=tensor_record(initial))
        row['embeddings'] = report['embeddings']['artifact']
        before = counts.copy()
        last = {}

        def callback(pipeline, index, t, values):
            budget(args)
            x = values['latents']
            require(x.dtype == torch.float32 and bool(x.isfinite().all()), 'Nonfinite/non-FP32 sampler state')
            row['steps'].append(dict(index=index, timestep=float(t), sigma=float(pipeline.scheduler.sigmas[index]),
                next_sigma=float(pipeline.scheduler.sigmas[index+1]), dtype=str(x.dtype), finite=True,
                rms=float(x.square().mean().sqrt()), min=float(x.min()), max=float(x.max())))
            if index == m['settings']['num_inference_steps']-1:
                last['latents'] = x.detach().cpu().clone()
            if (index+1) % 10 == 0:
                print(f"{case['case_id']} {index+1}/50 scheduler steps", flush=True)
            return values

        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        frames = pipe(prompt_embeds=positive, negative_prompt_embeds=negative, latents=initial,
            height=480, width=832, num_frames=81, num_inference_steps=50, guidance_scale=6.,
            output_type='np', callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'],
            max_sequence_length=512).frames[0]
        torch.cuda.synchronize()
        row.update(seconds_pipeline_including_diagnostics=time.monotonic()-began,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   actual_counts={k: counts[k]-before[k] for k in counts},
                   schedule=dict(timesteps=pipe.scheduler.timesteps.cpu().tolist(), sigmas=pipe.scheduler.sigmas.cpu().tolist()))
        require(row['actual_counts'] == dict(dit=100, scheduler=50, public_vae_decode=1, text_encoder=0), 'Per-video call count differs')
        require(len(row['steps']) == 50, 'Incomplete callback chain')
        final_path = case_dir/'final_latents.pt'
        torch.save(last['latents'], final_path)
        row['final_latents'] = dict(artifact=record(final_path), tensor=tensor_record(last['latents']))
        budget(args); row.update(write_media(frames, case_dir, m['settings']['fps']))
        row['status'] = 'complete'
        save_json(report, args.output)
        del frames, last, initial
    require(counts['dit'] == 200 and counts['scheduler'] == 100 and counts['public_vae_decode'] == 2, 'Worker count differs')
    budget(args)
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--prompt-id', type=int, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.output = args.output or ROOT/f'results/research/E043/{"check" if args.check else "worker"}_{args.prompt_id}.json'
    require(not args.output.exists(), 'Refusing to overwrite previous result')
    report = dict(experiment='E043', status='running', phase='check' if args.check else 'generate',
                  actual_counts=dict(dit=0, scheduler=0, public_vae_decode=0, text_encoder=0), cases=[])
    started = time.monotonic()
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        manifest, cases = prerequisites(args, report)
        if args.check:
            require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CPU check requires CUDA hidden')
            report.update(status='complete', cuda_available=False, model_weights_loaded=False)
        else:
            run(args, manifest, cases, report)
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        save_json(report, args.output)
        print(json.dumps(dict(status=report['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
