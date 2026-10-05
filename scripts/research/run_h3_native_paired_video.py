#!/usr/bin/env python3
"""E010: frozen two-prompt H3 BF16/native free-rollout closure, phased processes.

Uses the original MiniMaxH3Pipeline.__call__, original schedules and original
step function. Only model placement and observation are intercepted; decoding
is deferred until a separate process after both denoisers have exited.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
DS = Path('/home/wjq/workspace/DiffSynth-Studio')
MODEL_ROOT = DS/'models/MiniMax/MiniMax-H3'
DIT_PATH = DS/'models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors'
for key, value in {'DIFFSYNTH_SKIP_DOWNLOAD': 'True', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION': 'torch',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
    os.environ[key] = value
os.environ.setdefault('DIFFSYNTH_ROOT', str(DS))
os.environ.setdefault('MINIMAX_H3_MODEL_ROOT', str(MODEL_ROOT))
os.environ.setdefault('MINIMAX_H3_DIT_PATH', str(DIT_PATH))
os.environ.setdefault('SVDQUANT_DATA_ROOT', '/data1/models/svdquant-wjq')
if os.environ.get('DIFFSYNTH_FLASH_ATTN_KERNEL_REPO_ID'):
    raise RuntimeError('Remove custom downloadable attention kernel override for this offline protocol')
sys.path[:0] = [str(ROOT/'scripts'), str(DS)]

import torch
from minimax_h3_svdquant_common import load_h3_pipeline, disk_config, tree_cpu
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_NoiseInitializer, ModelConfig)
from probe_h3_modality_pulse import sha256, tensor_sha, save
from bench_h3_native_nvfp4 import make_h3_resident, RuntimeAudit, memory_guard as legacy_memory_guard
from h3_native_nvfp4 import install_native_h3
from h3_nvfp4_zero_sf_compat import pack_activation_fast, collect_fastpack_checks
from profile_h3_native_nvfp4 import correctness_binding

MANIFEST = ROOT/'research_state/06_experiments/E010_h3_heldout_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E010_h3_heldout_paired_plan.md'
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E010')
EXCLUDE = {1, 11, 20, 25, 46, 48, 105, 116}


def memory_guard():
    """Check current and historical peak allocations at stage checkpoints."""
    legacy_memory_guard()
    peak_gib = torch.cuda.max_memory_allocated()/1024**3
    if peak_gib > 60.:
        raise RuntimeError(f'Peak CUDA allocation exceeded 60 GiB before checkpoint: {peak_gib:.3f} GiB')


def source_paths():
    files = [Path(__file__), MANIFEST, PLAN, ROOT/'scripts/minimax_h3_svdquant_common.py',
             ROOT/'scripts/infer_minimax_h3_svdquant_standard.py']
    for name in ['bench_h3_native_nvfp4.py', 'profile_h3_native_nvfp4.py', 'h3_native_nvfp4.py',
                 'wan_native_nvfp4.py', 'h3_nvfp4_fastpack.py', 'h3_nvfp4_zero_sf_compat.py',
                 'probe_h3_native_contract.py', 'probe_h3_modality_pulse.py']:
        files.append(Path(__file__).with_name(name))
    for name in ['diffsynth.pipelines.minimax_h3_audio_video', 'diffsynth.diffusion.base_pipeline',
                 'diffsynth.diffusion.flow_match', 'diffsynth.models.minimax_h3_dit',
                 'diffsynth.models.minimax_h3_dit_comfy', 'diffsynth.models.minimax_h3_text_encoder',
                 'diffsynth.models.minimax_h3_video_vae', 'diffsynth.models.minimax_h3_audio_vae',
                 'diffsynth.core.attention.attention', 'diffsynth.core.vram.layers',
                 'diffsynth.utils.data.audio_video']:
        files.append(Path(importlib.import_module(name).__file__))
    return files


def header_sha(path):
    import hashlib
    with path.open('rb') as stream:
        n = struct.unpack('<Q', stream.read(8))[0]
        raw = stream.read(n)
    header = json.loads(raw)
    expected = 8+n+max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
    if expected != path.stat().st_size:
        raise RuntimeError(f'Truncated safetensors file: {path}')
    return hashlib.sha256(raw).hexdigest()


def phase_report(args, phase, arm=None):
    tag = phase + ('_'+arm if arm else '')
    return args.report_dir/f'E010_h3_{tag}.json'


def load_complete(path):
    value = json.loads(path.read_text())
    if value['status'] != 'complete':
        raise RuntimeError(f'Prerequisite incomplete: {path}')
    return value


def prerequisites(args, report):
    import hashlib
    manifest = json.loads(MANIFEST.read_text())
    if [r['prompt_id'] for r in manifest['cases']] != [30, 36]:
        raise RuntimeError('Fixed preregistered cases changed')
    original = Path(manifest['source_prompt_file'])
    if sha256(original) != manifest['source_prompt_file_sha256']:
        raise RuntimeError('Original 124-prompt source changed')
    original_rows = {int(r['prompt_id']): r for r in map(json.loads, original.open())}
    for row in manifest['cases']:
        source = original_rows[row['prompt_id']]
        if row['prompt_id'] in EXCLUDE or row['prompt'] != source['prompt'] or row['seed'] != int(source['seed']):
            raise RuntimeError('Heldout prompt/seed provenance changed')
        if hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']:
            raise RuntimeError('Prompt hash mismatch')
    for info in manifest['assets_cpu_header_inventory']:
        path = Path(info['path'])
        if path.stat().st_size != info['bytes']:
            raise RuntimeError(f'Asset size changed: {path}')
        if 'header_sha256' in info and header_sha(path) != info['header_sha256']:
            raise RuntimeError(f'Asset header changed: {path}')
        if 'sha256' in info and sha256(path) != info['sha256']:
            raise RuntimeError(f'Processor asset changed: {path}')
    binding_args = types.SimpleNamespace(arm=args.arm or 'native',
        correctness=ROOT/'results/research/E009_h3_full.json',
        fast_proof=ROOT/'results/research/E009_h3_native_resume.json')
    reference, _resume, _expected = correctness_binding(binding_args, report)
    export_manifest = args.export_dir/'manifest.json'
    export_sha = sha256(export_manifest)
    if export_sha != reference['export_manifest_sha256']:
        raise RuntimeError('Native export manifest differs from the E009 correctness reference')
    report['export_manifest'] = {'file': str(export_manifest), 'sha256': export_sha}
    sdpa_enabled = {'flash': torch.backends.cuda.flash_sdp_enabled(),
                    'math': torch.backends.cuda.math_sdp_enabled(),
                    'mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
                    'cudnn': torch.backends.cuda.cudnn_sdp_enabled()}
    if sdpa_enabled != reference['sdpa_enabled']:
        raise RuntimeError(f'SDPA backend flags differ from E009: {sdpa_enabled}')
    if os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'] != 'torch':
        raise RuntimeError('E010 requires explicit torch attention implementation')
    report['sdpa_enabled'] = sdpa_enabled
    native_profile = load_complete(ROOT/'results/research/E009_profile_native.json')
    load_complete(ROOT/'results/research/E009_profile_bf16.json')
    report['profile_precondition'] = {'native': str(ROOT/'results/research/E009_profile_native.json'),
                                     'sha256': sha256(ROOT/'results/research/E009_profile_native.json')}
    for name in ['h3_native_nvfp4.py', 'wan_native_nvfp4.py', 'h3_nvfp4_fastpack.py', 'h3_nvfp4_zero_sf_compat.py']:
        path = Path(__file__).with_name(name)
        if native_profile['files'][str(path)]['sha256'] != sha256(path):
            raise RuntimeError('Runtime differs from completed native profile')
    if os.environ['MINIMAX_H3_DIT_PATH'] != str(DIT_PATH):
        raise RuntimeError('Fixed local pruned DiT path required')
    report['sources'] = {str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in source_paths()}
    report['settings'] = manifest['settings']
    report['cases'] = []
    report['torch'] = torch.__version__
    if torch.__version__ != reference['torch']:
        raise RuntimeError('Torch differs from E009')
    report['environment'] = {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'DIFFSYNTH_ROOT',
        'MINIMAX_H3_MODEL_ROOT', 'MINIMAX_H3_DIT_PATH', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION',
        'DIFFSYNTH_SKIP_DOWNLOAD', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']}
    return manifest, reference


def snapshot_sources(report, directory):
    directory.mkdir(parents=True, exist_ok=False)
    for path, info in report['sources'].items():
        target = directory/(info['sha256'][:12]+'_'+Path(path).name)
        shutil.copyfile(path, target)
        info['snapshot'] = str(target)


def initial_noise(settings, seed):
    # Use the existing initializer verbatim, on a CPU-only empty pipeline.
    # It creates separate generators with the same seed for video and audio.
    cpu = MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    return MiniMaxH3Unit_NoiseInitializer().process(cpu, seed, settings['num_frames'],
                                                  settings['height'], settings['width'], 'cpu')


def cpu_contract(manifest, report):
    from transformers import AutoProcessor
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va
    processor = AutoProcessor.from_pretrained(str(MODEL_ROOT/'FL2VA/processor'), local_files_only=True)
    pipe = MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    settings = manifest['settings']
    pipe.scheduler.set_timesteps(20, shift=12.)
    pipe.scheduler_audio.set_timesteps(20, shift=3.)
    report['schedule_cpu'] = {'video': pipe.scheduler.timesteps.tolist(),
                              'audio': pipe.scheduler_audio.timesteps.tolist()}
    for row in manifest['cases']:
        ids, tags = presentation_t2va(processor.tokenizer, row['prompt'])
        noise = initial_noise(settings, row['seed'])
        repeated = initial_noise(settings, row['seed'])
        assert all(torch.equal(noise[k], repeated[k]) for k in noise)
        assert tags.unique().tolist() == [1]
        report['cases'].append({'prompt_id': row['prompt_id'], 'seed': row['seed'],
            'text_tokens': ids.numel(), 'text_tag_values': tags.unique().tolist(),
            'initial_noise': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': tensor_sha(v)} for k, v in noise.items()}})
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    report['cuda_initialized'] = False
    report['status'] = 'complete'


@torch.inference_mode()
def prepare(args, manifest, reference, report):
    stage = args.data_dir/'prepare'
    stage.mkdir(parents=True, exist_ok=False)
    snapshot_sources(report, stage/'sources')
    # Hash all local assets before opening a CUDA model; no downloading allowed.
    assets = [Path(v['path']) for v in manifest['assets_cpu_header_inventory']]+[DIT_PATH]
    report['assets'] = {str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size,
                               'mtime_ns': p.stat().st_mtime_ns} for p in assets}
    if report['assets'][str(DIT_PATH)]['sha256'] != reference['files'][str(DIT_PATH)]['sha256']:
        raise RuntimeError('DiT file differs from E009')
    save(report, args.output)
    shards = sorted((MODEL_ROOT/'FL2VA/text_encoder').glob('model*.safetensors'))
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=[str(p) for p in shards], **disk_config())],
        processor_config=ModelConfig(path=str(MODEL_ROOT/'FL2VA/processor')), vram_limit=30.)
    if pipe.dit is not None or pipe.video_vae is not None or pipe.audio_vae is not None:
        raise RuntimeError('Prepare must load only the text encoder')
    pipe.text_encoder.eval()
    unit = MiniMaxH3Unit_PromptEmbedder()
    for case in manifest['cases']:
        print(f'E010 prepare p{case["prompt_id"]}', flush=True)
        started = time.monotonic()
        embedded = unit.process(pipe, prompt=case['prompt'], height=576, width=1024)
        embedding = embedded['prompt_embeds'].detach().cpu()
        tags = embedded['text_token_tags'].detach().cpu()
        if not bool(torch.isfinite(embedding).all()) or tags.unique().tolist() != [1]:
            raise RuntimeError('Text embedding invalid or cached bypass would change tags')
        payload = {**case, 'embedding': embedding, 'text_token_tags': tags,
                   **initial_noise(manifest['settings'], case['seed'])}
        path = stage/f'p{case["prompt_id"]:03d}.pt'
        torch.save(payload, path)
        info = {'prompt_id': case['prompt_id'], 'seed': case['seed'], 'file': str(path),
                'sha256': sha256(path), 'embedding_shape': list(embedding.shape),
                'embedding_sha256': tensor_sha(embedding),
                'initial_noise_sha256': {k: tensor_sha(payload[k]) for k in ('video_latents', 'audio_latents')},
                'seconds': time.monotonic()-started}
        report['cases'].append(info)
        save(report, args.output)
        memory_guard()
    report['status'] = 'complete'


def prepared_inputs(args, report):
    path = phase_report(args, 'prepare')
    prep = load_complete(path)
    for source, info in prep['sources'].items():
        if sha256(source) != info['sha256']:
            raise RuntimeError(f'Source changed after prepare: {source}')
    for asset, info in prep['assets'].items():
        st = Path(asset).stat()
        if st.st_size != info['bytes'] or st.st_mtime_ns != info['mtime_ns']:
            raise RuntimeError(f'Asset changed after full prepare SHA: {asset}')
    report['prepared_reference'] = {'file': str(path), 'sha256': sha256(path),
                                  'assets': prep['assets'], 'source_hashes': prep['sources']}
    return prep


class DenoiseComplete(RuntimeError):
    pass


@torch.inference_mode()
def denoise(args, manifest, report):
    prep = prepared_inputs(args, report)
    stage = args.data_dir/('denoise_'+args.arm)
    stage.mkdir(parents=True, exist_ok=False)
    snapshot_sources(report, stage/'sources')
    pipe = load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    report['resident_conversion'] = make_h3_resident(pipe.dit)
    if args.arm == 'native':
        report['native_installation'] = install_native_h3(pipe.dit, args.export_dir,
                                                         activation_packer=pack_activation_fast, chunk_rows=1024)
    gc.collect()
    torch.cuda.empty_cache()
    original_step, original_forward = pipe.step, pipe.dit.forward
    settings = manifest['settings']
    with RuntimeAudit().installed() as audit:
        for case, cached_info in zip(manifest['cases'], prep['cases'], strict=True):
            if case['prompt_id'] != cached_info['prompt_id'] or sha256(cached_info['file']) != cached_info['sha256']:
                raise RuntimeError('Prepared case order/content changed')
            cached = torch.load(cached_info['file'], map_location='cpu', weights_only=True, mmap=True)
            directory = stage/f'p{case["prompt_id"]:03d}'
            directory.mkdir()
            info = {'prompt_id': case['prompt_id'], 'seed': case['seed'], 'variant': args.arm,
                    'prompt': case['prompt'], 'initial_noise_sha256': cached_info['initial_noise_sha256'],
                    'embedding_sha256': cached_info['embedding_sha256'], 'steps': [], 'dit_calls': [], 'status': 'running'}
            report['cases'].append(info)
            noise_calls, final = [], {}
            phase_key = f'{args.arm}_p{case["prompt_id"]}'
            audit.phase = phase_key

            def cached_noise(_self, shape, seed=None, rand_device='cpu', rand_torch_dtype=torch.float32,
                             device=None, torch_dtype=None):
                index = len(noise_calls)
                key = ('video_latents', 'audio_latents')[index] if index < 2 else None
                if (key is None or tuple(shape) != tuple(cached[key].shape) or seed != case['seed']
                        or rand_device != 'cpu' or rand_torch_dtype != torch.bfloat16):
                    raise RuntimeError('Original noise initializer contract changed')
                noise_calls.append(key)
                return cached[key].clone().to(device=device or pipe.device, dtype=torch_dtype or pipe.torch_dtype)

            def resident_placement(_self, names):
                if tuple(names) == tuple(pipe.in_iteration_models):
                    return
                if list(names) == ['video_vae']:
                    if len(info['steps']) != 40 or len(info['dit_calls']) != 20:
                        raise RuntimeError('Unexpected sampling completion counts')
                    raise DenoiseComplete('Original sampler finished; decoding deferred to separate process')
                raise RuntimeError(f'Unexpected component load during resident sampling: {names}')

            def observed_forward(_self, *pos, **kwargs):
                before = dict(audit.row())
                cu = kwargs['packed_seq_params']['cu_seqlens_q']
                refcu = kwargs['refiner_packed_seq_params']['cu_seqlens_q']
                expected_sdpa = 50*int((cu[1:] > cu[:-1]).sum())+2*int((refcu[1:] > refcu[:-1]).sum())
                context = collect_fastpack_checks() if args.arm == 'native' else nullcontext(None)
                with context as checks:
                    output = original_forward(*pos, **kwargs)
                now = audit.row()
                counts = {k: now[k]-before[k] for k in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')}
                if counts != {'sdpa_calls': expected_sdpa, 'scaled_mm_calls': 200 if args.arm == 'native' else 0, 'disk_loads': 0}:
                    raise RuntimeError(f'Unexpected actual resident denoiser path: {counts}')
                if args.arm == 'native' and checks.summary['checked_calls'] != 200:
                    raise RuntimeError('Incomplete per-forward fast checks')
                if any(not bool(torch.isfinite(x).all()) for x in output):
                    raise RuntimeError('Nonfinite denoiser output')
                info['dit_calls'].append({'step': len(info['dit_calls']), 'counts': counts,
                    'zero_sf_checks': checks.summary if checks is not None else None,
                    'video_sha256': tensor_sha(output[0]), 'audio_sha256': tensor_sha(output[1])})
                return output

            def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
                name = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
                if name is None or not 0 <= progress_id < 20:
                    raise RuntimeError('Unexpected scheduler call')
                if progress_id == 0 and tensor_sha(latents) != cached_info['initial_noise_sha256'][name+'_latents']:
                    raise RuntimeError('Initial noise differs between arms')
                updated = original_step(scheduler, latents, progress_id, noise_pred, **kwargs)
                if not bool(torch.isfinite(updated).all()):
                    raise RuntimeError('Nonfinite recurrent latent')
                path = directory/f's{progress_id:02d}_{name}.pt'
                torch.save({'latents_before': latents.detach().cpu(), 'noise_pred': noise_pred.detach().cpu(),
                            'latents_after': updated.detach().cpu(), 'timestep': scheduler.timesteps[progress_id].cpu(),
                            'sigma': scheduler.sigmas[progress_id].cpu()}, path)
                info['steps'].append({'step': progress_id, 'modality': name, 'file': str(path),
                                      'sha256': sha256(path), 'latent_sha256': tensor_sha(updated)})
                if progress_id == 19:
                    final[name+'_latents'] = updated.detach().cpu()
                save(report, args.output)
                memory_guard()
                return updated

            pipe.generate_noise = types.MethodType(cached_noise, pipe)
            pipe.load_models_to_device = types.MethodType(resident_placement, pipe)
            pipe.step = types.MethodType(observed_step, pipe)
            pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
            started = time.monotonic()
            print(f'E010 {args.arm} p{case["prompt_id"]}: original 20-step sampler', flush=True)
            try:
                pipe(prompt=None, text_embedding=cached['embedding'], seed=case['seed'],
                     height=settings['height'], width=settings['width'], num_frames=settings['num_frames'],
                     num_inference_steps=20, cfg_scale=1., flow_shift=12., audio_flow_shift=3.,
                     rand_device='cpu', tiled=True, tile_size=256, tile_overlap=64)
            except DenoiseComplete:
                pass
            else:
                raise RuntimeError('Expected stop before VAE load')
            if noise_calls != ['video_latents', 'audio_latents'] or set(final) != {'video_latents', 'audio_latents'}:
                raise RuntimeError('Incomplete paired-noise/final-latent contract')
            path = directory/'final_latents.pt'
            torch.save(final, path)
            info.update(status='complete', final_latents=str(path), final_latents_sha256=sha256(path),
                        seconds_including_diagnostics=time.monotonic()-started,
                        video_timesteps=pipe.scheduler.timesteps.tolist(), audio_timesteps=pipe.scheduler_audio.timesteps.tolist(),
                        runtime_audit=dict(audit.row()))
            save(report, args.output)
            gc.collect()
            torch.cuda.empty_cache()
    report['status'] = 'complete'


@torch.inference_mode()
def decode_videos(args, manifest, report):
    from diffsynth.utils.data.audio_video import write_video_audio
    import av
    prep = prepared_inputs(args, report)
    denoisers = {arm: load_complete(phase_report(args, 'denoise', arm)) for arm in ('bf16', 'native')}
    report['denoiser_reports'] = {arm: {'file': str(phase_report(args, 'denoise', arm)),
                                      'sha256': sha256(phase_report(args, 'denoise', arm))} for arm in denoisers}
    stage = args.data_dir/'decode'
    stage.mkdir(parents=True, exist_ok=False)
    snapshot_sources(report, stage/'sources')
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=str(MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'), **disk_config()),
                       ModelConfig(path=str(MODEL_ROOT/'FL2VA/audio_vae/model.safetensors'), **disk_config())],
        processor_config=None, vram_limit=30.)
    if pipe.dit is not None or pipe.text_encoder is not None:
        raise RuntimeError('Decode process must contain VAEs only')
    if type(pipe.video_vae).__name__ != 'MiniMaxH3VideoVAE' or pipe.audio_vae.sample_rate != 32000:
        raise RuntimeError('Original VAE class/audio rate changed')
    pipe.video_vae.eval()
    pipe.audio_vae.eval()
    for index, case in enumerate(manifest['cases']):
        a, b = [denoisers[arm]['cases'][index] for arm in ('bf16', 'native')]
        if a['initial_noise_sha256'] != b['initial_noise_sha256'] or a['embedding_sha256'] != b['embedding_sha256']:
            raise RuntimeError('Arms did not share the frozen initial condition')
        for arm in ('bf16', 'native'):
            record = denoisers[arm]['cases'][index]
            if record['prompt_id'] != case['prompt_id'] or sha256(record['final_latents']) != record['final_latents_sha256']:
                raise RuntimeError('Final latent identity changed')
            latents = torch.load(record['final_latents'], map_location='cpu', weights_only=True, mmap=True)
            pipe.load_models_to_device(['video_vae'])
            recon = pipe.video_vae.decode_video(latents['video_latents'].cuda(), dtype=torch.bfloat16,
                                                tiled=True, tile_size=256, tile_overlap=64)
            if not bool(torch.isfinite(recon).all()):
                raise RuntimeError('Nonfinite decoded video')
            video = pipe.vae_output_to_video(recon, min_value=0, max_value=1)
            del recon
            if len(video) != 124 or any(frame.size != (1024, 576) for frame in video):
                raise RuntimeError('Video frames/shape changed')
            pipe.load_models_to_device(['audio_vae'])
            waveform = pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(), dtype=torch.bfloat16)
            audio = pipe.output_audio_format_check(waveform)
            if not bool(torch.isfinite(audio).all()) or audio.shape[0] != 2:
                raise RuntimeError('Invalid decoded stereo audio')
            path = stage/f'p{case["prompt_id"]:03d}_{arm}.mp4'
            write_video_audio(video, audio, str(path), fps=24, audio_sample_rate=32000)
            with av.open(str(path)) as container:
                frames = sum(1 for _ in container.decode(video=0))
                vs = container.streams.video[0]
                actual = {'frames': frames, 'width': vs.width, 'height': vs.height,
                          'fps': float(vs.average_rate), 'audio_streams': len(container.streams.audio),
                          'audio_sample_rate': container.streams.audio[0].codec_context.sample_rate}
            if actual != {'frames': 124, 'width': 1024, 'height': 576, 'fps': 24., 'audio_streams': 1, 'audio_sample_rate': 32000}:
                raise RuntimeError(f'Encoded media metadata mismatch: {actual}')
            audio_path = path.with_suffix('.audio.pt')
            torch.save({'waveform': audio, 'sample_rate': 32000}, audio_path)
            case_json = {**case, 'video': str(path), 'variant': arm, 'settings': manifest['settings'],
                         'video_sha256': sha256(path), 'status': 'complete', 'media': actual,
                         'audio_samples_before_aac': audio.shape[-1], 'audio_pcm_sha256': tensor_sha(audio),
                         'audio_pcm_file': str(audio_path)}
            save(case_json, path.with_suffix('.json'))
            report['cases'].append(case_json)
            save(report, args.output)
            memory_guard()
            del video, audio, waveform, latents
            gc.collect()
            torch.cuda.empty_cache()
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['check', 'prepare', 'denoise', 'decode'], required=True)
    parser.add_argument('--arm', choices=['bf16', 'native'])
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/research')
    parser.add_argument('--export-dir', type=Path, default=Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export'))
    args = parser.parse_args()
    if (args.phase == 'denoise') != (args.arm is not None):
        parser.error('--arm is required only for denoise')
    args.output = phase_report(args, args.phase, args.arm)
    if args.output.exists():
        raise FileExistsError(f'Preserve earlier report; choose a new --report-dir: {args.output}')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'experiment': 'E010', 'phase': args.phase, 'arm': args.arm, 'status': 'running',
              'scope': 'two original-PTQ-heldout prompts; paired free generation diagnostic, no benchmark claim'}
    started = time.monotonic()
    try:
        manifest, reference = prerequisites(args, report)
        if args.phase == 'check':
            cpu_contract(manifest, report)
        elif args.phase == 'prepare':
            prepare(args, manifest, reference, report)
        elif args.phase == 'denoise':
            denoise(args, manifest, report)
        else:
            decode_videos(args, manifest, report)
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        print(json.dumps({'status': report['status'], 'report': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
