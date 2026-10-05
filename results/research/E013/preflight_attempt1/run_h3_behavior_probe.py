#!/usr/bin/env python3
"""E013: case-isolated H3 behavior probe using the frozen E010 model pipeline.

No sampler/attention/quantization/VAE implementation changes. Intermediate
latents are hashed, not saved; only initial and final tensors are persisted.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
FROZEN = Path(__file__).with_name('run_h3_native_paired_video.py')
if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != '41e6e4eb1d9df554e5ea109b9bad3d1df83e9b64788137511262a82a0f34431f':
    raise RuntimeError('Frozen E010 pipeline wrapper changed')
import run_h3_native_paired_video as old
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder

MANIFEST = ROOT/'research_state/06_experiments/E013_h3_behavior_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E013_h3_behavior_plan.md'
E010_PREPARE = ROOT/'results/research/E010_h3_prepare.json'
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E013')
sha256, save = old.sha256, old.save


def require(value, message):
    if not value:
        raise RuntimeError(message)


def file_record(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def tensor_record(value):
    require(bool(torch.isfinite(value).all()), 'Nonfinite tensor')
    return {'sha256': old.tensor_sha(value.reshape(-1)), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def check_sources(sources):
    for path, row in sources.items():
        require(sha256(path) == row['sha256'], f'Source drift: {path}')


def report_path(args, phase, case_id=None):
    if phase in ('check', 'prepare'):
        return args.report_dir/f'E013_h3_{phase}.json'
    return args.report_dir/f'p{case_id:03d}'/f'E013_h3_{phase}_{args.arm}.json'


def checked_json(reference):
    require(sha256(reference['file']) == reference['sha256'], 'Report reference changed')
    return old.load_complete(Path(reference['file']))


def prerequisites(args):
    # Reuse the established code/export/attention contracts. This reads model
    # headers and small source files, never rehashes 117 GB of model weights.
    report = {'experiment': 'E013', 'phase': args.phase, 'arm': args.arm,
              'status': 'running', 'scope': 'Behavior feasibility; no paper or quality-equivalence claim'}
    inherited_args = types.SimpleNamespace(arm=args.arm or 'native', export_dir=args.export_dir)
    e010_manifest, _reference = old.prerequisites(inherited_args, report)
    manifest = json.loads(MANIFEST.read_text())
    require(manifest['experiment'] == 'E013', 'Wrong manifest')
    require([c['prompt_id'] for c in manifest['cases']] == [1, 2, 3, 4], 'E013 fixed case order changed')
    require([c['seed'] for c in manifest['cases']] == [49771, 49771, 59526, 59526], 'E013 fixed paired seeds changed')
    require(manifest['cases'][0]['prompt'] == manifest['cases'][2]['prompt'] and
            manifest['cases'][1]['prompt'] == manifest['cases'][3]['prompt'], 'Direction prompt differs across seeds')
    expected_settings = {**e010_manifest['settings'], 'num_inference_steps': 50}
    require(manifest['settings'] == expected_settings, 'Only the preregistered step-count change is allowed')
    for case in manifest['cases']:
        require(hashlib.sha256(case['prompt'].encode()).hexdigest() == case['prompt_sha256'], 'Prompt SHA mismatch')
    require(Path(manifest['asset_reference_report']).resolve() == E010_PREPARE.resolve()
            and sha256(E010_PREPARE) == manifest['asset_reference_sha256'], 'Existing full-SHA asset inventory changed')
    asset_reference = old.load_complete(E010_PREPARE)
    check_sources(asset_reference['sources'])
    for path, row in asset_reference['assets'].items():
        st = Path(path).stat()
        require(st.st_size == row['bytes'] and st.st_mtime_ns == row['mtime_ns'], f'Asset differs from existing full-SHA audit: {path}')
    report.update(manifest=manifest, manifest_file=file_record(MANIFEST), plan=file_record(PLAN),
                  settings=manifest['settings'], assets=asset_reference['assets'],
                  inherited_asset_sha_reference=file_record(E010_PREPARE),
                  asset_validation='E010 full SHA reused; every size/mtime checked; no repeated full-weight hash')
    report['sources'].update({str(p.resolve()): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                              for p in (Path(__file__), MANIFEST, PLAN)})
    if args.phase != 'check':
        path = report_path(args, 'check')
        checked = old.load_complete(path)
        check_sources(checked['sources'])
        require(checked['manifest_file'] == report['manifest_file'] and checked['plan'] == report['plan'], 'Preflight differs from manifest/plan')
        require(checked['cuda_initialized'] is False, 'CPU-only preflight missing')
        report['cpu_check_reference'] = file_record(path)
        report['cpu_cases'] = checked['cases']
        report['schedule_cpu'] = checked['schedule_cpu']
    return manifest, report


def cpu_check(manifest, report):
    from transformers import AutoProcessor
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va
    processor = AutoProcessor.from_pretrained(str(old.MODEL_ROOT/'FL2VA/processor'), local_files_only=True)
    pipe = old.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    cfg = manifest['settings']
    pipe.scheduler.set_timesteps(cfg['num_inference_steps'], shift=cfg['flow_shift'])
    pipe.scheduler_audio.set_timesteps(cfg['num_inference_steps'], shift=cfg['audio_flow_shift'])
    report['schedule_cpu'] = {'video': pipe.scheduler.timesteps.tolist(), 'audio': pipe.scheduler_audio.timesteps.tolist()}
    require(all(len(s) == 50 for s in report['schedule_cpu'].values()), 'Wrong schedule length')
    for case in manifest['cases']:
        ids, tags = presentation_t2va(processor.tokenizer, case['prompt'])
        require(tags.unique().tolist() == [1], 'Cached embedding bypass requires all-positive text tags')
        noise = old.initial_noise(cfg, case['seed'])
        repeated = old.initial_noise(cfg, case['seed'])
        require(all(torch.equal(noise[k], repeated[k]) for k in noise), 'Initial-noise replay mismatch')
        dummy = torch.empty(1, ids.numel(), 3584, dtype=torch.bfloat16)
        packed = MiniMaxH3Unit_PackedSequenceBuilder().process(pipe, prompt_embeds=dummy,
            text_token_tags=tags, video_latents=noise['video_latents'], audio_latents=noise['audio_latents'])['packed']
        cu = packed['cu_seqlens']
        sdpa = 50*int((cu[1:] > cu[:-1]).sum()) + 2
        require(sdpa == 102, f'Expected 102 SDPA calls, got {sdpa}')
        report['cases'].append({'prompt_id': case['prompt_id'], 'seed': case['seed'],
            'text_tokens': ids.numel(), 'text_tag_values': tags.unique().tolist(), 'tags': tensor_record(tags),
            'token_ids': ids.tolist(), 'cu_seqlens': cu.tolist(), 'seq_len': packed['seq_len'],
            'expected_sdpa_calls_per_dit': sdpa,
            'initial_noise': {k: tensor_record(v) for k, v in noise.items()}})
    for a, b in ((0, 1), (2, 3)):
        require(report['cases'][a]['initial_noise'] == report['cases'][b]['initial_noise'], 'Direction pair noise differs')
    require(not torch.cuda.is_initialized(), 'CPU preflight initialized CUDA')
    report.update(status='complete', cuda_initialized=False)


@torch.inference_mode()
def prepare(args, manifest, report):
    stage = args.data_dir/'prepare'
    stage.mkdir(parents=True, exist_ok=False)
    old.snapshot_sources(report, stage/'sources')
    shards = sorted((old.MODEL_ROOT/'FL2VA/text_encoder').glob('model*.safetensors'))
    pipe = old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[old.ModelConfig(path=[str(p) for p in shards], **old.disk_config())],
        processor_config=old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/processor')), vram_limit=30.)
    require(pipe.dit is None and pipe.video_vae is None and pipe.audio_vae is None, 'Prepare must contain TE only')
    pipe.text_encoder.eval()
    unit, reused = old.MiniMaxH3Unit_PromptEmbedder(), {}
    for case, cpu in zip(manifest['cases'], report['cpu_cases'], strict=True):
        if case['prompt_sha256'] not in reused:
            embedded = unit.process(pipe, prompt=case['prompt'], height=manifest['settings']['height'], width=manifest['settings']['width'])
            reused[case['prompt_sha256']] = (embedded['prompt_embeds'].detach().cpu(), embedded['text_token_tags'].detach().cpu())
        embedding, tags = reused[case['prompt_sha256']]
        require(bool(torch.isfinite(embedding).all()) and tensor_record(tags) == cpu['tags'], 'Prepared embedding/tags invalid')
        require(embedding.shape[1] == cpu['text_tokens'], 'Embedding token length differs')
        payload = {**case, 'embedding': embedding, 'text_token_tags': tags,
                   **old.initial_noise(manifest['settings'], case['seed'])}
        noise = {k: tensor_record(payload[k]) for k in ('video_latents', 'audio_latents')}
        require(noise == cpu['initial_noise'], 'Prepared noise differs from CPU preflight')
        path = stage/f'p{case["prompt_id"]:03d}.pt'
        torch.save(payload, path)
        report['cases'].append({'prompt_id': case['prompt_id'], 'seed': case['seed'], **file_record(path),
            'embedding': tensor_record(embedding), 'tags': tensor_record(tags), 'initial_noise': noise})
        save(report, report_path(args, 'prepare'))
        old.memory_guard()
    report.update(status='complete', unique_text_encoder_calls=len(reused))


def prepared(args, report):
    path = report_path(args, 'prepare')
    prep = old.load_complete(path)
    check_sources(prep['sources'])
    require(prep['manifest_file'] == report['manifest_file'] and prep['assets'] == report['assets'], 'Prepared provenance drift')
    require([c['prompt_id'] for c in prep['cases']] == [1, 2, 3, 4], 'Prepared case table incomplete')
    report['prepared_reference'] = file_record(path)
    return {row['prompt_id']: row for row in prep['cases']}


@torch.inference_mode()
def denoise_case(args, case, cached_info, pipe, audit, original_step, original_forward, report):
    cid, cfg = case['prompt_id'], report['settings']
    path = report_path(args, 'denoise', cid)
    stage = args.data_dir/f'p{cid:03d}'/f'denoise_{args.arm}'
    stage.mkdir(parents=True, exist_ok=False)
    old.snapshot_sources(report, stage/'sources')
    require(sha256(cached_info['file']) == cached_info['sha256'], 'Prepared tensor file changed')
    cached = torch.load(cached_info['file'], map_location='cpu', weights_only=True, mmap=True)
    for key in ('prompt_id', 'prompt', 'seed', 'prompt_sha256'):
        require(cached[key] == case[key], f'Prepared case differs: {key}')
    require(tensor_record(cached['embedding']) == cached_info['embedding'], 'Embedding SHA mismatch')
    report.update(case=case, prepared_case=cached_info, steps=[], dit_calls=[])
    final, noise_calls = {}, []
    audit.phase = f'{args.arm}_p{cid:03d}'

    def cached_noise(_self, shape, seed=None, rand_device='cpu', rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        index = len(noise_calls)
        key = ('video_latents', 'audio_latents')[index] if index < 2 else None
        require(key is not None and tuple(shape) == tuple(cached[key].shape) and seed == case['seed']
                and rand_device == 'cpu' and rand_torch_dtype == torch.bfloat16, 'Original noise initializer contract changed')
        noise_calls.append(key)
        return cached[key].clone().to(device=device or pipe.device, dtype=torch_dtype or pipe.torch_dtype)

    def placement(_self, names):
        if tuple(names) == tuple(pipe.in_iteration_models):
            return
        if list(names) == ['video_vae']:
            require(len(report['steps']) == 100 and len(report['dit_calls']) == 50, 'Incomplete original sampler')
            raise old.DenoiseComplete('Original sampler complete; decode in separate process')
        raise RuntimeError(f'Unexpected resident component load: {names}')

    def forward(_self, *pos, **kwargs):
        before = dict(audit.row())
        cu, rcu = kwargs['packed_seq_params']['cu_seqlens_q'], kwargs['refiner_packed_seq_params']['cu_seqlens_q']
        expected = 50*int((cu[1:] > cu[:-1]).sum()) + 2*int((rcu[1:] > rcu[:-1]).sum())
        require(expected == 102 and len(report['dit_calls']) < 50, 'Unexpected packed layout/extra DiT call')
        with old.collect_fastpack_checks() if args.arm == 'native' else nullcontext(None) as checks:
            result = original_forward(*pos, **kwargs)
        now = audit.row()
        counts = {k: now[k]-before[k] for k in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')}
        require(counts == {'sdpa_calls': 102, 'scaled_mm_calls': 200 if args.arm == 'native' else 0, 'disk_loads': 0}, 'Actual execution path changed')
        require(args.arm != 'native' or checks.summary['checked_calls'] == 200, 'Native checks incomplete')
        report['dit_calls'].append({'step': len(report['dit_calls']), 'counts': counts,
            'zero_sf_checks': checks.summary if checks is not None else None,
            'video': tensor_record(result[0]), 'audio': tensor_record(result[1])})
        return result

    def step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
        name = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
        require(name is not None and 0 <= progress_id < 50, 'Unexpected scheduler call')
        same_modality = [r for r in report['steps'] if r['modality'] == name]
        require(len(same_modality) == progress_id, 'Scheduler step missing/duplicated/out of order')
        before = tensor_record(latents)
        if progress_id == 0:
            require(before == cached_info['initial_noise'][name+'_latents'], 'Initial noise differs from frozen input')
        else:
            require(before == same_modality[-1]['latents_after'], 'Latent chain discontinuity')
        updated = original_step(scheduler, latents, progress_id, noise_pred, **kwargs)
        row = {'step': progress_id, 'modality': name, 'latents_before': before,
            'noise_pred': tensor_record(noise_pred), 'latents_after': tensor_record(updated),
            'timestep': float(scheduler.timesteps[progress_id]), 'sigma': float(scheduler.sigmas[progress_id]),
            'next_sigma': float(scheduler.sigmas[progress_id+1]) if progress_id+1 < len(scheduler.sigmas) else None}
        report['steps'].append(row)
        if progress_id == 49:
            final[name+'_latents'] = updated.detach().cpu()
        save(report, path)
        old.memory_guard()
        return updated

    pipe.generate_noise = types.MethodType(cached_noise, pipe)
    pipe.load_models_to_device = types.MethodType(placement, pipe)
    pipe.step = types.MethodType(step, pipe)
    pipe.dit.forward = types.MethodType(forward, pipe.dit)
    print(f'E013 {args.arm} case={cid}: unchanged original pipeline, 50 steps', flush=True)
    try:
        pipe(prompt=None, text_embedding=cached['embedding'], seed=case['seed'], height=cfg['height'], width=cfg['width'],
             num_frames=cfg['num_frames'], num_inference_steps=cfg['num_inference_steps'], cfg_scale=cfg['cfg_scale'],
             flow_shift=cfg['flow_shift'], audio_flow_shift=cfg['audio_flow_shift'], rand_device=cfg['rand_device'],
             tiled=cfg['tiled'], tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'])
    except old.DenoiseComplete:
        pass
    else:
        raise RuntimeError('Did not stop before VAE load')
    require(noise_calls == ['video_latents', 'audio_latents'] and set(final) == {'video_latents', 'audio_latents'}, 'Sampling incomplete')
    require(pipe.scheduler.timesteps.tolist() == report['schedule_cpu']['video'] and
            pipe.scheduler_audio.timesteps.tolist() == report['schedule_cpu']['audio'], 'Runtime schedule differs from CPU preflight')
    final_path = stage/'final_latents.pt'
    torch.save(final, final_path)
    report.update(status='complete', final_latents=file_record(final_path),
                  final_latent_tensors={k: tensor_record(v) for k, v in final.items()}, runtime_audit=dict(audit.row()))


def finish(report, path, started):
    report['seconds_including_diagnostics'] = time.monotonic()-started
    if torch.cuda.is_initialized():
        report['peak_allocated_gib_process'] = torch.cuda.max_memory_allocated()/1024**3
    save(report, path)
    print(json.dumps({'status': report['status'], 'report': str(path)}), flush=True)


@torch.inference_mode()
def denoise(args, cases, template):
    cache = prepared(args, template)
    pipe = old.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    template['resident_conversion'] = old.make_h3_resident(pipe.dit)
    if args.arm == 'native':
        template['native_installation'] = old.install_native_h3(pipe.dit, args.export_dir,
            activation_packer=old.pack_activation_fast, chunk_rows=1024)
    original_step, original_forward = pipe.step, pipe.dit.forward
    gc.collect()
    torch.cuda.empty_cache()
    with old.RuntimeAudit().installed() as audit:
        for case in cases:
            report, started = copy.deepcopy(template), time.monotonic()
            path = report_path(args, 'denoise', case['prompt_id'])
            try:
                denoise_case(args, case, cache[case['prompt_id']], pipe, audit, original_step, original_forward, report)
            except BaseException as exc:
                report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
                raise
            finally:
                finish(report, path, started)
            gc.collect()
            torch.cuda.empty_cache()


@torch.inference_mode()
def decode(args, cases, template):
    from diffsynth.utils.data.audio_video import write_video_audio
    import av
    cache = prepared(args, template)
    denoisers = {}
    for case in cases:
        reference = file_record(report_path(args, 'denoise', case['prompt_id']))
        report = checked_json(reference)
        check_sources(report['sources'])
        require(report['case'] == case and report['prepared_case'] == cache[case['prompt_id']], 'Decode case/input provenance differs')
        denoisers[case['prompt_id']] = (reference, report)
    pipe = old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'), **old.disk_config()),
                       old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/audio_vae/model.safetensors'), **old.disk_config())],
        processor_config=None, vram_limit=30.)
    require(pipe.dit is None and pipe.text_encoder is None and type(pipe.video_vae).__name__ == 'MiniMaxH3VideoVAE'
            and pipe.audio_vae.sample_rate == 32000, 'Original VAE-only contract changed')
    pipe.video_vae.eval()
    pipe.audio_vae.eval()
    for case in cases:
        report, started = copy.deepcopy(template), time.monotonic()
        cid, cfg = case['prompt_id'], template['settings']
        output = report_path(args, 'decode', cid)
        stage = args.data_dir/f'p{cid:03d}'/f'decode_{args.arm}'
        try:
            stage.mkdir(parents=True, exist_ok=False)
            old.snapshot_sources(report, stage/'sources')
            reference, denoised = denoisers[cid]
            record = denoised['final_latents']
            require(sha256(record['file']) == record['sha256'], 'Final latent file changed')
            latents = torch.load(record['file'], map_location='cpu', weights_only=True, mmap=True)
            require({k: tensor_record(v) for k, v in latents.items()} == denoised['final_latent_tensors'], 'Final tensor hashes differ')
            pipe.load_models_to_device(['video_vae'])
            recon = pipe.video_vae.decode_video(latents['video_latents'].cuda(), dtype=torch.bfloat16,
                tiled=cfg['tiled'], tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'])
            require(bool(torch.isfinite(recon).all()), 'Nonfinite decoded video')
            video = pipe.vae_output_to_video(recon, min_value=0, max_value=1)
            del recon
            require(len(video) == cfg['num_frames'] and all(frame.size == (cfg['width'], cfg['height']) for frame in video), 'Decoded video layout changed')
            pipe.load_models_to_device(['audio_vae'])
            waveform = pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(), dtype=torch.bfloat16)
            audio = pipe.output_audio_format_check(waveform)
            require(bool(torch.isfinite(audio).all()) and audio.shape[0] == 2, 'Invalid decoded audio')
            path = stage/f'p{cid:03d}_{args.arm}.mp4'
            write_video_audio(video, audio, str(path), fps=cfg['fps'], audio_sample_rate=cfg['audio_sample_rate'])
            with av.open(str(path)) as container:
                frames = sum(1 for _ in container.decode(video=0))
                vs = container.streams.video[0]
                media = {'frames': frames, 'width': vs.width, 'height': vs.height, 'fps': float(vs.average_rate),
                         'audio_streams': len(container.streams.audio), 'audio_sample_rate': container.streams.audio[0].codec_context.sample_rate}
            require(media == {'frames': 124, 'width': 1024, 'height': 576, 'fps': 24., 'audio_streams': 1, 'audio_sample_rate': 32000}, 'Encoded media differs')
            audio_path = stage/'audio.pt'
            torch.save({'waveform': audio, 'sample_rate': cfg['audio_sample_rate']}, audio_path)
            report.update(status='complete', case=case, denoiser_reference=reference, video=str(path),
                video_file=file_record(path), media=media, audio_pcm_file=file_record(audio_path),
                audio_pcm=tensor_record(audio), audio_samples_before_aac=audio.shape[-1])
            save({**case, 'variant': args.arm, 'video': str(path), 'video_sha256': sha256(path), 'settings': cfg}, path.with_suffix('.json'))
            old.memory_guard()
            del video, audio, waveform, latents
        except BaseException as exc:
            report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            finish(report, output, started)
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'prepare', 'denoise', 'decode'))
    parser.add_argument('--case-ids', nargs='+', type=int)
    parser.add_argument('--arm', choices=('bf16', 'native'))
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/research/E013')
    parser.add_argument('--export-dir', type=Path, default=Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export'))
    args = parser.parse_args()
    if args.phase in ('denoise', 'decode'):
        if args.arm is None or not args.case_ids:
            parser.error('denoise/decode require --arm and explicit --case-ids')
    elif args.arm is not None or args.case_ids not in (None, [1, 2, 3, 4]):
        parser.error('check/prepare cover all four cases and have no arm')
    if args.case_ids and (len(args.case_ids) != len(set(args.case_ids)) or not set(args.case_ids) <= {1, 2, 3, 4}):
        parser.error('Case ids must be unique members of 1,2,3,4')
    paths = [report_path(args, args.phase)] if args.phase in ('check', 'prepare') else [report_path(args, args.phase, cid) for cid in args.case_ids]
    for path in paths:
        require(not path.exists(), f'Preserve existing report: {path}')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest, report = prerequisites(args)
    if args.phase in ('check', 'prepare'):
        started = time.monotonic()
        try:
            cpu_check(manifest, report) if args.phase == 'check' else prepare(args, manifest, report)
        except BaseException as exc:
            report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            finish(report, paths[0], started)
    else:
        cases = [next(c for c in manifest['cases'] if c['prompt_id'] == cid) for cid in args.case_ids]
        started = time.monotonic()
        try:
            denoise(args, cases, report) if args.phase == 'denoise' else decode(args, cases, report)
        except BaseException as exc:
            # A load/setup failure can occur before the per-case handler starts.
            # Preserve it in the first unstarted requested case without touching
            # an already completed or failed report.
            missing = next((p for p in paths if not p.exists()), None)
            if missing is not None:
                report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc(),
                              requested_case_ids=args.case_ids, failure_scope='setup_or_unstarted_case')
                finish(report, missing, started)
            raise


if __name__ == '__main__':
    main()
