#!/usr/bin/env python3
"""E017: frozen E010 sampler with E016 native FP4 attention, VAE in a separate process.

Only experiment observation/orchestration is new. All recurrent tensors are
saved on CPU. No quality metrics, extra teacher forward, or profiler runs here.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_h3_native_paired_video as old
import probe_h3_plain_baseline as base
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
from h3_native_fp4_attention import install_h3_fp4_attention

MANIFEST = ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json'
PLAN = MANIFEST.with_name('E017_h3_fp4_video_plan.md')
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E017')
REPORTS = ROOT/'results/research/E017'
ARMS = ('block_mean', 'global_mean')
MODALITIES = ('video', 'audio')
file_record, tensor_record, tree_signature = base.file_record, base.tensor_record, base.tree_signature
save, sha256 = old.save, old.sha256


def require(value, message):
    if not value:
        raise RuntimeError(message)


def checked_json(reference):
    path = base.verify_file(reference)
    value = json.loads(path.read_text())
    require(value.get('status') == 'complete', f'Incomplete prerequisite: {path}')
    return value


def finite(value, message):
    require(bool(torch.isfinite(value).all()), message)


def budget(args, report, *, before_forward=False):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix,
            'GPU phase requires an unexpired original shared deadline')
    if before_forward:
        require(report['attempted_dit_calls'] < 40, 'Per-arm 40-DiT allocation exhausted')
    old.memory_guard()


def schedule(pipe, settings):
    pipe.scheduler.set_timesteps(settings['num_inference_steps'], shift=settings['flow_shift'])
    pipe.scheduler_audio.set_timesteps(settings['num_inference_steps'], shift=settings['audio_flow_shift'])
    return {m: {'timesteps': s.timesteps.tolist(), 'sigmas': s.sigmas.tolist()}
            for m, s in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio))}


def load_cached(row):
    base.verify_file(row)
    return torch.load(row['file'], map_location='cpu', weights_only=True, mmap=True)


def prerequisites(args, report):
    manifest = json.loads(args.manifest.read_text())
    require(manifest['experiment'] == 'E017' and manifest['arms'] == list(ARMS), 'Wrong protocol')
    require(file_record(args.plan) == manifest['plan'], 'Plan binding changed')
    require(args.data_dir == Path(manifest['data_dir']) and args.report_dir == Path(manifest['report_dir']),
            'Data/report directories differ from protocol')
    require(manifest['budget'] == dict(wall_seconds=1800, gpu=5, max_dit_calls=80,
        max_peak_allocated_gib=60, decode_videos=4, evaluation_questions=232), 'Budget changed')
    e010 = json.loads(base.verify_file(manifest['e010_manifest']).read_text())
    prep = checked_json(manifest['e010_prepare'])
    e016 = json.loads(base.verify_file(manifest['e016_manifest']).read_text())
    prior_check = checked_json(manifest['e016_check'])
    checked_json(manifest['e010_summary'])
    checked_json(manifest['e016_summary'])
    freeze = checked_json(manifest['e016_freeze'])
    for path, digest in freeze['files'].items():
        require(sha256(path) == digest, f'E016 frozen source drift: {path}')
    require(manifest['settings'] == e010['settings'] and manifest['cases'] == e010['cases']
            and manifest['prepared_cases'] == prep['cases'], 'E010 conditions/settings changed')
    require([(r['prompt_id'], r['seed']) for r in manifest['cases']] == [(30, 49771), (36, 59526)],
            'Fixed case table changed')
    require(manifest['export_dir'] == e016['svd_export_dir'], 'Native export directory changed')
    require(str(Path(manifest['export_dir'])/'manifest.json') == prep['export_manifest']['file'] and
            sha256(prep['export_manifest']['file']) == prep['export_manifest']['sha256'], 'Native export manifest changed')
    for arm in ('bf16', 'native'):
        checked_json(manifest['e010_denoisers'][arm])
    checked_json(manifest['e010_decode'])
    # Hash known source paths statically: native environment intentionally has
    # no av, so never call the old source_paths() importer for media modules.
    sources = {**prep['sources'], **prior_check['sources']}
    base.check_sources(prep['sources'])
    base.check_sources(prior_check['sources'])
    source_paths = {Path(p) for p in sources} | {Path(__file__), args.manifest, args.plan}
    sources = {str(p.resolve()): file_record(p) for p in sorted(source_paths)}
    for path, rec in prep['assets'].items():
        st = Path(path).stat()
        require(st.st_size == rec['bytes'] and st.st_mtime_ns == rec['mtime_ns'],
                f'Asset differs from inherited full-SHA inventory: {path}')
    for case, cached in zip(manifest['cases'], prep['cases'], strict=True):
        require(case['prompt_id'] == cached['prompt_id'] and case['seed'] == cached['seed'], 'Prepared case order changed')
        require(sha256(cached['file']) == cached['sha256'], 'Prepared file content changed')
    sdp = {name: getattr(torch.backends.cuda, name+'_sdp_enabled')() for name in ('flash', 'math', 'mem_efficient', 'cudnn')}
    require(sdp == prior_check['e014_inventory_recheck']['actual_sdpa_enabled'], 'Torch SDPA flags changed')
    require(torch.__version__ == '2.11.0+cu128' and torch.version.cuda == '12.8'
            and os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'] == 'torch', 'Torch/attention runtime changed')
    env = dict(python=sys.executable, torch=torch.__version__, torch_cuda=torch.version.cuda,
               attention_implementation=os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'], sdpa_enabled=sdp)
    require(sys.executable == manifest['python_decode' if args.phase == 'decode' else 'python'], 'Unexpected Python environment')
    if args.phase != 'decode':
        packages = {name: importlib.metadata.version(name) for name in ('torch', 'triton', 'flashinfer-python')}
        require(packages == prior_check['environment']['packages'] and
                file_record(torch.__file__) == prior_check['environment']['torch_file'], 'Native packages changed from E016')
        env['packages'] = packages
        # CUDA/JIT variables are explicit even during CPU preflight; no JIT is
        # invoked here, and the outer launcher owns process scheduling.
        for key, value in manifest['environment'].items():
            require(os.environ.get(key) == value, f'Environment variable changed: {key}')
    if args.phase != 'check':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(manifest['budget']['gpu']), 'Only GPU5 is authorized')
    references = {key: manifest[key] for key in ('e010_manifest', 'e010_prepare', 'e010_summary',
        'e010_denoisers', 'e010_decode', 'e016_manifest', 'e016_check', 'e016_summary', 'e016_freeze')}
    report.update(manifest=file_record(args.manifest), plan=file_record(args.plan), sources=sources,
        inherited_references=references, prepared_reference=manifest['e010_prepare'],
        inherited_assets=prep['assets'], asset_policy='Inherited E010 full SHA; current size/mtime checked, no repeated model hash',
        export_manifest=file_record(prep['export_manifest']['file']), settings=manifest['settings'], environment=env)
    checked = None
    if args.phase != 'check':
        checked = old.load_complete(args.check_report)
        require(checked['cuda_initialized'] is False and checked['sources'] == sources
                and checked['manifest'] == report['manifest'] and checked['plan'] == report['plan'], 'CPU source/protocol gate changed')
        if args.phase == 'denoise':
            require(checked['environment'] == env, 'Denoise environment differs from CPU preflight')
        report['cpu_check_reference'] = file_record(args.check_report)
    return manifest, prep, checked


def cpu_check(manifest, prep, report):
    pipe = old.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['schedule_cpu'] = schedule(pipe, manifest['settings'])
    report['cases'] = []
    for case, cached_info in zip(manifest['cases'], prep['cases'], strict=True):
        cached = load_cached({**file_record(cached_info['file']), 'sha256': cached_info['sha256']})
        embedding, tags = cached['embedding'], cached['text_token_tags']
        require(embedding.ndim == 2 and embedding.shape[1] == 5120 and embedding.dtype == torch.bfloat16,
                'Expected saved BF16 embedding [tokens,5120]')
        require(tags.ndim == 1 and tags.numel() == embedding.shape[0] and tags.unique().tolist() == [1],
                'Embedding bypass requires the original all-positive text tags')
        require(tensor_record(embedding)['sha256'] == cached_info['embedding_sha256'], 'Embedding SHA differs')
        finite(embedding, 'Nonfinite prepared embedding')
        generated = old.initial_noise(manifest['settings'], case['seed'])
        for key in ('video_latents', 'audio_latents'):
            require(tensor_record(cached[key]) == tensor_record(generated[key]) and
                    tensor_record(cached[key])['sha256'] == cached_info['initial_noise_sha256'][key],
                    'Original paired CPU noise does not replay exact')
        packed = MiniMaxH3Unit_PackedSequenceBuilder().process(pipe, prompt_embeds=embedding,
            text_token_tags=tags, video_latents=cached['video_latents'], audio_latents=cached['audio_latents'])['packed']
        contract = dict(expected_cu=packed['cu_seqlens'].tolist(),
                        expected_refiner_cu=[0, embedding.shape[0], embedding.shape[0]])
        require(contract == manifest['attention_contracts'][str(case['prompt_id'])], 'CPU attention layout differs')
        require(len(report['schedule_cpu']['video']['timesteps']) == 20 and
                len(report['schedule_cpu']['audio']['timesteps']) == 20, 'Wrong scheduler length')
        report['cases'].append(dict(prompt_id=case['prompt_id'], seed=case['seed'],
            prepared=file_record(cached_info['file']), embedding=tensor_record(embedding),
            text_token_tags=tensor_record(tags), initial_noise={k: tensor_record(v) for k, v in generated.items()},
            attention_contract=contract, packed_signature=tree_signature(packed)))
    # Actual saved scalar fixtures exercise the historical 0-D hash failure.
    report['scalar_fixture'] = {m: tensor_record(s.timesteps[0])
                               for m, s in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio))}
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False)


@torch.inference_mode()
def denoise(args, manifest, prep, checked, report):
    budget(args, report)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    require(torch.cuda.get_device_capability() == (12, 0), 'Native path requires SM120')
    report['device'] = dict(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    stage = args.data_dir/f'denoise_{args.arm}'
    stage.mkdir(parents=True, exist_ok=False)
    pipe = old.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    report['resident_conversion'] = old.make_h3_resident(pipe.dit)
    identities = base.non_target_identity(pipe.dit)
    report['native_installation'] = old.install_native_h3(pipe.dit, Path(manifest['export_dir']),
        activation_packer=old.pack_activation_fast, chunk_rows=1024)
    require(report['native_installation']['target_count'] == 200 and
            report['native_installation']['exact_roundtrip_count'] == 200, 'Incomplete native linear installation')
    require(base.non_target_identity(pipe.dit) == identities, 'Native installation changed non-target tensors')
    from diffsynth.core.vram.layers import AutoTorchModule
    require(not any(isinstance(m, AutoTorchModule) for m in pipe.dit.modules()), 'Residual offload module')
    report['non_target_identity_preserved'] = True
    router = install_h3_fp4_attention(pipe.dit, mode=args.arm)
    report['attention_installation'] = router.manifest
    gc.collect()
    torch.cuda.empty_cache()
    original_step, original_forward = pipe.step, pipe.dit.forward
    cfg = manifest['settings']
    report['cases'] = []
    audit = old.RuntimeAudit()
    audit.phase = 'native'
    try:
        with audit.installed():
            for case, cached_info, cpu in zip(manifest['cases'], prep['cases'], checked['cases'], strict=True):
                budget(args, report)
                cached = load_cached(cpu['prepared'])
                directory = stage/f'p{case["prompt_id"]:03d}'
                directory.mkdir()
                info = dict(prompt_id=case['prompt_id'], seed=case['seed'], prompt=case['prompt'],
                    variant=args.arm, status='running', prepared=cpu['prepared'],
                    embedding_sha256=cached_info['embedding_sha256'],
                    initial_noise_sha256=cached_info['initial_noise_sha256'], steps=[], dit_calls=[])
                report['cases'].append(info)
                before_case = dict(audit.row())
                noise_calls, final, previous = [], {}, {}

                def cached_noise(_self, shape, seed=None, rand_device='cpu', rand_torch_dtype=torch.float32,
                                 device=None, torch_dtype=None):
                    index = len(noise_calls)
                    key = ('video_latents', 'audio_latents')[index] if index < 2 else None
                    require(key is not None and tuple(shape) == tuple(cached[key].shape) and seed == case['seed']
                            and rand_device == 'cpu' and rand_torch_dtype == torch.bfloat16, 'Noise initializer contract changed')
                    noise_calls.append(key)
                    return cached[key].clone().to(device=device or pipe.device, dtype=torch_dtype or pipe.torch_dtype)

                def resident_placement(_self, names):
                    if tuple(names) == tuple(pipe.in_iteration_models):
                        return
                    if list(names) == ['video_vae']:
                        require(len(info['steps']) == 40 and len(info['dit_calls']) == 20, 'Unexpected sampling counts')
                        raise old.DenoiseComplete('Sampler finished; VAE runs in separate process')
                    raise RuntimeError(f'Unexpected resident component load: {names}')

                def observed_forward(_self, *pos, **kwargs):
                    budget(args, report, before_forward=True)
                    require(len(info['dit_calls']) < 20, 'Unexpected extra DiT call in case')
                    actual = tree_signature({'args': pos, 'kwargs': kwargs})
                    before = dict(audit.row())
                    report['attempted_dit_calls'] += 1
                    with old.collect_fastpack_checks() as checks, router.forward_context(
                            **cpu['attention_contract'], diagnostics=False) as attention:
                        output = original_forward(*pos, **kwargs)
                        torch.cuda.synchronize()
                    now = audit.row()
                    counts = {k: now[k]-before[k] for k in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')}
                    require(counts == dict(sdpa_calls=52, scaled_mm_calls=200, disk_loads=0), 'Unexpected native execution path')
                    require(checks.summary['checked_calls'] == 200 and checks.summary['invalid_calls'] == 0,
                            'Incomplete/invalid activation packing checks')
                    require(attention.summary['fp4_calls'] == 50 and attention.summary['original_bf16_segments'] == 52
                            and attention.summary['finite_flag_count'] == 0, 'Unexpected attention routes/diagnostic cost')
                    for value in output:
                        finite(value, 'Nonfinite DiT output')
                    info['dit_calls'].append(dict(step=len(info['dit_calls']), counts=counts,
                        zero_sf_checks=checks.summary, attention=attention.summary, actual_dit_inputs=actual,
                        raw_outputs={m: tensor_record(v) for m, v in zip(MODALITIES, output, strict=True)}, finite=True))
                    report['complete_dit_calls'] += 1
                    return output

                def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
                    budget(args, report)
                    modality = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
                    require(modality is not None and 0 <= progress_id < 20, 'Unexpected scheduler call')
                    require(len(info['steps']) == 2*progress_id+MODALITIES.index(modality), 'Scheduler modality/order changed')
                    before_sha = tensor_record(latents)['sha256']
                    expected = previous.get(modality, cached_info['initial_noise_sha256'][modality+'_latents'])
                    require(before_sha == expected, 'Recurrent latent chain or initial noise changed')
                    updated = original_step(scheduler, latents, progress_id, noise_pred, **kwargs)
                    finite(noise_pred, 'Nonfinite denoiser velocity')
                    finite(updated, 'Nonfinite recurrent latent')
                    payload = {'latents_before': latents.detach().cpu(), 'noise_pred': noise_pred.detach().cpu(),
                        'latents_after': updated.detach().cpu(), 'timestep': scheduler.timesteps[progress_id].cpu(),
                        'sigma': scheduler.sigmas[progress_id].cpu()}
                    records = tree_signature(payload)
                    path = directory/f's{progress_id:02d}_{modality}.pt'
                    torch.save(payload, path)
                    previous[modality] = records['latents_after']['sha256']
                    info['steps'].append(dict(step=progress_id, modality=modality, **file_record(path),
                        tensors=records, latent_sha256=previous[modality], finite=True,
                        scheduler_extra_signature=tree_signature(kwargs)))
                    if progress_id == 19:
                        final[modality+'_latents'] = payload['latents_after']
                    save(report, args.output)
                    return updated

                pipe.generate_noise = types.MethodType(cached_noise, pipe)
                pipe.load_models_to_device = types.MethodType(resident_placement, pipe)
                pipe.step = types.MethodType(observed_step, pipe)
                pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
                started = time.monotonic()
                print(f'E017 {args.arm} p{case["prompt_id"]}: original 20-step sampler', flush=True)
                try:
                    pipe(prompt=None, text_embedding=cached['embedding'], seed=case['seed'],
                        height=cfg['height'], width=cfg['width'], num_frames=cfg['num_frames'],
                        num_inference_steps=cfg['num_inference_steps'], cfg_scale=cfg['cfg_scale'],
                        flow_shift=cfg['flow_shift'], audio_flow_shift=cfg['audio_flow_shift'],
                        rand_device=cfg['rand_device'], tiled=cfg['tiled'],
                        tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'])
                except old.DenoiseComplete:
                    pass
                else:
                    raise RuntimeError('Sampler did not stop at the original VAE boundary')
                require(noise_calls == ['video_latents', 'audio_latents'] and set(final) == {'video_latents', 'audio_latents'},
                        'Missing shared noise/final modalities')
                actual_schedule = {m: {'timesteps': s.timesteps.tolist(), 'sigmas': s.sigmas.tolist()}
                    for m, s in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio))}
                require(actual_schedule == checked['schedule_cpu'], 'Runtime schedule differs from original CPU schedule')
                path = directory/'final_latents.pt'
                torch.save(final, path)
                info.update(status='complete', final_latents=file_record(path), final_tensors=tree_signature(final),
                    schedule=actual_schedule, seconds_including_diagnostics=time.monotonic()-started,
                    runtime_audit={k: audit.row()[k]-before_case[k] for k in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')})
                save(report, args.output)
                gc.collect()
                torch.cuda.empty_cache()
        require(report['complete_dit_calls'] == report['attempted_dit_calls'] == 40, 'Arm allocation incomplete')
        report['runtime_audit'] = dict(audit.row())
        report['status'] = 'complete'
    finally:
        router.close()


@torch.inference_mode()
def decode(args, manifest, report):
    import av
    from diffsynth.utils.data.audio_video import write_video_audio
    budget(args, report)
    source = args.report_dir/f'E017_denoise_{args.arm}.json'
    denoiser = old.load_complete(source)
    require(denoiser['manifest'] == report['manifest'] and denoiser['sources'] == report['sources']
            and denoiser['arm'] == args.arm and denoiser['complete_dit_calls'] == 40, 'Denoiser binding differs')
    report['denoiser_reference'] = file_record(source)
    report['av_version'] = av.__version__
    stage = args.data_dir/f'decode_{args.arm}'
    stage.mkdir(parents=True, exist_ok=False)
    pipe = old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'), **old.disk_config()),
                       old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/audio_vae/model.safetensors'), **old.disk_config())],
        processor_config=None, vram_limit=30.)
    require(pipe.dit is None and pipe.text_encoder is None and type(pipe.video_vae).__name__ == 'MiniMaxH3VideoVAE'
            and pipe.audio_vae.sample_rate == 32000, 'Decode must contain original VAEs only')
    pipe.video_vae.eval()
    pipe.audio_vae.eval()
    cfg = manifest['settings']
    report['cases'] = []
    for case, row in zip(manifest['cases'], denoiser['cases'], strict=True):
        budget(args, report)
        require(row['status'] == 'complete' and row['prompt_id'] == case['prompt_id'], 'Denoiser case order differs')
        latents = load_cached(row['final_latents'])
        require(tree_signature(latents) == row['final_tensors'], 'Final latent tensor SHA changed')
        pipe.load_models_to_device(['video_vae'])
        recon = pipe.video_vae.decode_video(latents['video_latents'].cuda(), dtype=torch.bfloat16,
            tiled=cfg['tiled'], tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'])
        finite(recon, 'Nonfinite decoded video')
        video = pipe.vae_output_to_video(recon, min_value=0, max_value=1)
        del recon
        require(len(video) == 124 and all(frame.size == (1024, 576) for frame in video), 'Decoded video layout differs')
        budget(args, report)
        pipe.load_models_to_device(['audio_vae'])
        waveform = pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(), dtype=torch.bfloat16)
        audio = pipe.output_audio_format_check(waveform)
        finite(audio, 'Nonfinite decoded audio')
        require(audio.ndim == 2 and audio.shape[0] == 2, 'Expected stereo PCM')
        path = stage/f'p{case["prompt_id"]:03d}_{args.arm}.mp4'
        write_video_audio(video, audio, str(path), fps=cfg['fps'], audio_sample_rate=cfg['audio_sample_rate'])
        with av.open(str(path)) as container:
            frames = sum(1 for _ in container.decode(video=0))
            vs, aus = container.streams.video[0], container.streams.audio[0]
            media = dict(frames=frames, width=vs.width, height=vs.height, fps=float(vs.average_rate),
                audio_streams=len(container.streams.audio), audio_sample_rate=aus.codec_context.sample_rate,
                audio_channels=aus.codec_context.channels)
        require(media == dict(frames=124, width=1024, height=576, fps=24., audio_streams=1,
                             audio_sample_rate=32000, audio_channels=2), 'Encoded media contract differs')
        pcm_path = path.with_suffix('.audio.pt')
        torch.save({'waveform': audio.detach().cpu(), 'sample_rate': 32000}, pcm_path)
        result = dict(**case, status='complete', variant=args.arm, settings=cfg, video=str(path),
            video_sha256=sha256(path), video_file=file_record(path), media=media, final_latents=row['final_latents'],
            audio_pcm_file=file_record(pcm_path), audio_pcm=tensor_record(audio), audio_samples_before_aac=audio.shape[-1])
        save(result, path.with_suffix('.json'))
        report['cases'].append(result)
        save(report, args.output)
        del video, audio, waveform, latents
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'denoise', 'decode'), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=REPORTS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.output = args.output or args.report_dir/f'E017_{args.phase}_{args.arm}.json'
    args.check_report = args.check_report or args.report_dir/'E017_check_block_mean.json'
    require(not args.output.exists(), f'Refusing to overwrite prior report: {args.output}')
    report = dict(experiment='E017', phase=args.phase, arm=args.arm, status='running',
        attempted_dit_calls=0, complete_dit_calls=0, deadline_unix=args.deadline_unix,
        scope='Matched-initial-condition free rollout of two existing official attention recipes; no new method')
    started = time.monotonic()
    try:
        manifest, prep, checked = prerequisites(args, report)
        if args.phase == 'check':
            cpu_check(manifest, prep, report)
        elif args.phase == 'denoise':
            denoise(args, manifest, prep, checked, report)
        else:
            decode(args, manifest, report)
        if args.phase != 'check':
            budget(args, report)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save(report, args.output)
        print(json.dumps(dict(status=report['status'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
