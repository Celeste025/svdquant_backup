#!/usr/bin/env python3
"""E073: original BF16 references for the frozen E038 eight action cases.

Original H3 sampler/step semantics; actual prepared noise is injected, not
regenerated from the seed. Only scalar step receipts and final latents are saved.
"""
from __future__ import annotations
import argparse
import gc
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
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
from probe_h3_plain_baseline import file_record, tensor_record, tree_signature
from h3_coarse16_attention import install_h3_center_attention
ARMS = ('full_bf16',)

MANIFEST = ROOT/'research_state/06_experiments/E073_behavior_manifest.json'
MODALITIES = ('video', 'audio')
save = old.save


def require(value, message):
    if not value:
        raise RuntimeError(message)


def schedules(pipe, cfg):
    pipe.scheduler.set_timesteps(cfg['num_inference_steps'], shift=cfg['flow_shift'])
    pipe.scheduler_audio.set_timesteps(cfg['num_inference_steps'], shift=cfg['audio_flow_shift'])
    return actual_schedules(pipe)


def actual_schedules(pipe):
    return {m: dict(timesteps=s.timesteps.tolist(), sigmas=s.sigmas.tolist())
        for m, s in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio))}


def budget(args, report, before_forward=False):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix,
            'Original worker deadline expired or absent')
    if before_forward:
        require(report['attempted_dit_calls'] < report['allocated_dit_calls'], 'DiT allocation exhausted')
    old.memory_guard()


def tensor_scalars(value):
    x = value.detach().float()
    finite = bool(torch.isfinite(x).all())
    require(finite, 'Nonfinite recurrent tensor')
    return dict(finite=True, rms=float(x.square().mean().sqrt()), max_abs=float(x.abs().max()))


def read_manifest(args, report):
    manifest = json.loads(args.manifest.read_text())
    require(manifest['experiment'] == 'E073' and manifest['arms'] == list(ARMS), 'Wrong E073 protocol')
    require(len(manifest['cases']) == 8 and len({r['case_id'] for r in manifest['cases']}) == 8,
            'Expected eight unique fixed trajectories')
    cfg = manifest['settings']
    require((cfg['height'], cfg['width'], cfg['num_frames'], cfg['num_inference_steps'],
             cfg['cfg_scale'], cfg['flow_shift'], cfg['audio_flow_shift']) == (576,1024,124,20,1.,12.,3.),
            'Original E017 sampling contract changed')
    require(cfg['rand_device'] == 'cpu' and cfg['noise_dtype'] == 'bfloat16', 'Noise contract changed')
    args.data_dir = args.data_dir or Path(manifest['data_dir'])
    args.report_dir = args.report_dir or Path(manifest['report_dir'])
    args.prepare_report = args.prepare_report or ROOT/'results/research/E038/prepare.json'
    args.output = args.output or args.report_dir/f'{args.phase}_{args.arm}_r{args.replica}.json'
    require(not args.output.exists(), f'Preserve prior report: {args.output}')
    args.report_writable = True
    require(args.data_dir == Path(manifest['data_dir']) and args.report_dir == Path(manifest['report_dir']),
            'Data/report paths differ from manifest')
    report.update(manifest=file_record(args.manifest), settings=cfg,
        sources={p.name:file_record(p) for p in (Path(__file__), HERE/'h3_coarse16_attention.py',
            HERE/'h3_native_fp4_attention.py', HERE/'codebook_attention_sm120.py')},
        allocated_dit_calls=4*cfg['num_inference_steps'],
        environment=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            attention_implementation=os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'],
            sdpa_enabled={name:getattr(torch.backends.cuda,name+'_sdp_enabled')()
                for name in ('flash','math','mem_efficient','cudnn')}))
    return manifest


def prepare_bindings(args, manifest, report, required):
    """Only actual input files/tensors and CPU packing; no model/source-tree audit."""
    pipe = old.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['schedule_cpu'] = schedules(pipe, manifest['settings'])
    if not args.prepare_report.exists():
        require(not required, 'Preparation report not available')
        report['prepare_available'] = False
        precheck = args.report_dir/'prepare_check.json'
        lengths = []
        if precheck.exists():
            check = json.loads(precheck.read_text())
            for row in check.get('cases', []):
                contract = row.get('attention_contract')
                if contract:
                    lengths.append(contract['expected_cu'][1])
        return [], lengths
    prep = old.load_complete(args.prepare_report)
    if 'manifest' in prep:
        require(prep['manifest']['sha256'] == manifest['source_manifest']['sha256'], 'E038 prepare manifest changed')
    rows = {r['case_id']:r for r in prep['cases']}
    require(set(rows) == {r['case_id'] for r in manifest['cases']}, 'Prepared case set differs')
    bound, lengths = [], []
    for case in manifest['cases']:
        if case['replica'] != args.replica:
            continue
        row = rows[case['case_id']]
        require((row['prompt_id'], row['seed']) == (case['prompt_id'], case['seed']), 'Prepared identity changed')
        record = file_record(row['file'])
        require(record['sha256'] == row['sha256'], 'Prepared input file SHA differs')
        cached = torch.load(row['file'], map_location='cpu', weights_only=True, mmap=True)
        require(cached['case_id'] == case['case_id'] and cached['seed'] == case['seed']
                and cached['prompt'] == case['prompt'], 'Prepared payload identity changed')
        embedding, tags = cached['embedding'], cached['text_token_tags']
        require(embedding.ndim == 2 and embedding.shape[1] == 5120 and embedding.dtype == torch.bfloat16,
                'Expected real BF16 text embedding [tokens,5120]')
        require(tags.shape == (embedding.shape[0],) and tags.unique().tolist() == [1],
                'Embedding bypass would change actual text tags')
        require(tensor_record(embedding)['sha256'] == row['embedding_sha256'], 'Embedding SHA differs')
        for key in ('video_latents','audio_latents'):
            require(cached[key].dtype == torch.bfloat16 and bool(torch.isfinite(cached[key]).all()), 'Invalid actual noise')
            require(tensor_record(cached[key])['sha256'] == row['initial_noise_sha256'][key], 'Actual shared noise SHA differs')
        packed = MiniMaxH3Unit_PackedSequenceBuilder().process(pipe, prompt_embeds=embedding,
            text_token_tags=tags, video_latents=cached['video_latents'], audio_latents=cached['audio_latents'])['packed']
        contract = dict(expected_cu=packed['cu_seqlens'].tolist(),
            expected_refiner_cu=[0,embedding.shape[0],embedding.shape[0]])
        if 'attention_contract' in row:
            require(contract == row['attention_contract'], 'Actual prepared layout differs')
        lengths.append(contract['expected_cu'][1])
        binding_devices = {key: str(cached[key].device) for key in
            ('embedding', 'text_token_tags', 'video_latents', 'audio_latents')}
        binding_devices['packed_cu_seqlens'] = str(packed['cu_seqlens'].device)
        bound.append(dict(case_id=case['case_id'], prepared=record, attention_contract=contract,
            initial_noise_sha256=row['initial_noise_sha256'], embedding_sha256=row['embedding_sha256'],
            binding_tensor_devices=binding_devices))
        del cached, packed, embedding, tags
    if args.phase == 'check':
        require(not torch.cuda.is_initialized(), 'CPU input binding initialized CUDA')
    report.update(prepare_available=True, prepared_reference=file_record(args.prepare_report), prepared_cases=bound)
    return bound, lengths


@torch.inference_mode()
def denoise(args, manifest, bound, report):
    budget(args, report)
    require(torch.cuda.get_device_capability() == (12,0), 'Native attention requires SM120')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(manifest['budget']['generation_gpus'][str(args.replica)]),
            'Visible GPU differs from fixed arm allocation')
    torch.cuda.set_per_process_memory_fraction(min(1.,60*2**30/torch.cuda.get_device_properties(0).total_memory))
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    stage = args.data_dir/f'denoise_{args.arm}_r{args.replica}'
    stage.mkdir(parents=True, exist_ok=False)
    pipe = old.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    report['resident_conversion'] = old.make_h3_resident(pipe.dit)
    report['precision'] = 'Original BF16 main weights; no native linear installation'
    router = install_h3_center_attention(pipe.dit, 'bf16')
    report['attention_installation'] = router.manifest
    gc.collect()
    torch.cuda.empty_cache()
    original_step, original_forward = pipe.step, pipe.dit.forward
    cfg = manifest['settings']
    audit = old.RuntimeAudit()
    audit.phase = 'full_bf16'
    try:
        with audit.installed():
            for case, cpu in zip(manifest['cases'], bound, strict=True):
                budget(args, report)
                cached = torch.load(cpu['prepared']['file'], map_location='cpu', weights_only=True, mmap=True)
                directory = stage/case['case_id']
                directory.mkdir()
                info = dict(**case, variant=args.arm, status='running', prepared=cpu['prepared'],
                    initial_noise_sha256=cpu['initial_noise_sha256'], embedding_sha256=cpu['embedding_sha256'],
                    attention_contract=cpu['attention_contract'], steps=[], dit_calls=[])
                report['cases'].append(info)
                before_case = dict(audit.row())
                noise_calls, final = [], {}

                def cached_noise(_self, shape, seed=None, rand_device='cpu', rand_torch_dtype=torch.float32,
                                 device=None, torch_dtype=None):
                    index = len(noise_calls)
                    key = ('video_latents','audio_latents')[index] if index < 2 else None
                    require(key is not None and tuple(shape) == tuple(cached[key].shape) and seed == case['seed']
                        and rand_device == 'cpu' and rand_torch_dtype == torch.bfloat16, 'Noise initializer contract changed')
                    noise_calls.append(key)
                    return cached[key].clone().to(device=device or pipe.device, dtype=torch_dtype or pipe.torch_dtype)

                def resident_placement(_self, names):
                    if tuple(names) == tuple(pipe.in_iteration_models):
                        return
                    if list(names) == ['video_vae']:
                        require(len(info['steps']) == 40 and len(info['dit_calls']) == 20, 'Sampling completion count changed')
                        raise old.DenoiseComplete('Original sampler complete; decode in separate process')
                    raise RuntimeError(f'Unexpected resident component load: {names}')

                def observed_forward(_self, *pos, **kwargs):
                    budget(args, report, before_forward=True)
                    require(len(info['dit_calls']) < 20, 'Extra per-case DiT call')
                    before = dict(audit.row())
                    report['attempted_dit_calls'] += 1
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    with router.forward_context(
                            **cpu['attention_contract'], diagnostics=False) as attention:
                        output = original_forward(*pos, **kwargs)
                        torch.cuda.synchronize()
                    elapsed = time.monotonic()-started
                    counts = {k:audit.row()[k]-before[k] for k in ('sdpa_calls','scaled_mm_calls','disk_loads')}
                    low = False
                    require(counts == dict(sdpa_calls=52 if low else 102, scaled_mm_calls=0, disk_loads=0),
                            f'Unexpected native path: {counts}')
                    summary = attention.summary
                    require(summary['fp4_calls'] == (50 if low else 0) and summary['finite_flag_count'] == 0,
                            'Unexpected attention route')
                    for x in output:
                        require(bool(torch.isfinite(x).all()), 'Nonfinite raw DiT output')
                    # Every layer is counted by the actual router; retain full layout once per case.
                    if not info['dit_calls']:
                        info['first_dit_attention_routes'] = summary['main_rows']
                    info['dit_calls'].append(dict(step=len(info['dit_calls']), counts=counts,
                        zero_sf_checks=None, attention={k:v for k,v in summary.items() if k!='main_rows'},
                        synchronized_dit_seconds=elapsed, finite=True))
                    report['complete_dit_calls'] += 1
                    return output

                def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
                    budget(args, report)
                    modality = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
                    require(modality is not None and 0 <= progress_id < 20, 'Unexpected scheduler call')
                    require(len(info['steps']) == 2*progress_id+MODALITIES.index(modality), 'Scheduler order changed')
                    if progress_id == 0:
                        require(tensor_record(latents)['sha256'] == cpu['initial_noise_sha256'][modality+'_latents'],
                                'Actual first sampler state differs from shared prepared noise')
                    updated = original_step(scheduler, latents, progress_id, noise_pred, **kwargs)
                    info['steps'].append(dict(step=progress_id, modality=modality,
                        timestep=float(scheduler.timesteps[progress_id]), sigma=float(scheduler.sigmas[progress_id]),
                        latent_before=tensor_scalars(latents), velocity=tensor_scalars(noise_pred),
                        latent_after=tensor_scalars(updated)))
                    if progress_id == 19:
                        final[modality+'_latents'] = updated.detach().cpu()
                    if modality == 'audio':
                        save(report, args.output)
                        if progress_id % 5 == 4:
                            print(f'E073 {args.arm} {case["case_id"]}: {progress_id+1}/20 steps', flush=True)
                    return updated

                pipe.generate_noise = types.MethodType(cached_noise, pipe)
                pipe.load_models_to_device = types.MethodType(resident_placement, pipe)
                pipe.step = types.MethodType(observed_step, pipe)
                pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
                started = time.monotonic()
                print(f'E073 {args.arm} {case["case_id"]}: original free rollout', flush=True)
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
                    raise RuntimeError('Expected stop at original VAE boundary')
                require(noise_calls == ['video_latents','audio_latents']
                    and set(final) == {'video_latents','audio_latents'}, 'Incomplete shared-input/final output')
                require(actual_schedules(pipe) == report['schedule_cpu'], 'Runtime schedules changed')
                path = directory/'final_latents.pt'
                torch.save(final, path)
                info.update(status='complete', final_latents=file_record(path), final_tensors=tree_signature(final),
                    schedule=actual_schedules(pipe), seconds_including_diagnostics=time.monotonic()-started,
                    runtime_audit={k:audit.row()[k]-before_case[k] for k in ('sdpa_calls','scaled_mm_calls','disk_loads')})
                save(report, args.output)
                del cached, final
                gc.collect()
                torch.cuda.empty_cache()
        require(report['complete_dit_calls'] == report['attempted_dit_calls'] == report['allocated_dit_calls'],
                'Incomplete allocated rollout')
        report['runtime_audit'] = dict(audit.row())
        report['actual_fp4_attention_calls'] = sum(d['attention']['fp4_calls'] for c in report['cases'] for d in c['dit_calls'])
        report['status'] = 'complete'
    finally:
        router.close()


def decode(args, manifest, report):
    import av
    from diffsynth.utils.data.audio_video import write_video_audio
    budget(args, report)
    source = args.report_dir/f'denoise_{args.arm}_r{args.replica}.json'
    denoiser = old.load_complete(source)
    require(denoiser['manifest'] == report['manifest'] and denoiser['arm'] == args.arm, 'Denoiser binding differs')
    require(denoiser['complete_dit_calls'] == 80 and len(denoiser['cases']) == 4, 'Incomplete denoising allocation')
    prep = old.load_complete(args.prepare_report)
    require(prep['manifest']['sha256'] == manifest['source_manifest']['sha256'], 'Prepared manifest differs')
    report.update(denoiser_reference=file_record(source), prepared_reference=file_record(args.prepare_report), av_version=av.__version__)
    stage = args.data_dir/f'decode_{args.arm}_r{args.replica}'
    stage.mkdir(parents=True, exist_ok=False)
    pipe = old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'), **old.disk_config()),
                       old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/audio_vae/model.safetensors'), **old.disk_config())],
        processor_config=None, vram_limit=30.)
    require(pipe.dit is None and pipe.text_encoder is None and type(pipe.video_vae).__name__ == 'MiniMaxH3VideoVAE'
            and pipe.audio_vae.sample_rate == 32000, 'Decode must contain original VAEs only')
    pipe.video_vae.eval(); pipe.audio_vae.eval()
    cfg = manifest['settings']
    report['cases'] = []
    for case, row in zip(manifest['cases'], denoiser['cases'], strict=True):
        budget(args, report); started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        require(row['status'] == 'complete' and all(row[k] == case[k] for k in ('case_id','prompt_id','seed','prompt')),
                'Denoiser case identity changed')
        require(file_record(row['final_latents']['file']) == row['final_latents'], 'Final latent file changed')
        latents = torch.load(row['final_latents']['file'], map_location='cpu', weights_only=True, mmap=True)
        require(tree_signature(latents) == row['final_tensors'], 'Final latent tensor signatures differ')
        pipe.load_models_to_device(['video_vae'])
        recon = pipe.video_vae.decode_video(latents['video_latents'].cuda(), dtype=torch.bfloat16,
            tiled=cfg['tiled'], tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'])
        require(bool(recon.isfinite().all()), 'Nonfinite decoded video')
        video = pipe.vae_output_to_video(recon, min_value=0, max_value=1)
        del recon
        require(len(video) == cfg['num_frames'] and all(im.size == (cfg['width'], cfg['height']) for im in video), 'Video shape differs')
        budget(args, report); pipe.load_models_to_device(['audio_vae'])
        waveform = pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(), dtype=torch.bfloat16)
        audio = pipe.output_audio_format_check(waveform)
        require(bool(audio.isfinite().all()) and audio.ndim == 2 and audio.shape[0] == 2, 'Invalid stereo PCM')
        path = stage/f'{case["case_id"]}.mp4'
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
        result = dict(**case, trajectory_id=case['case_id']+'_'+args.arm, status='complete', variant=args.arm,
            settings=cfg, video=file_record(path), video_path=str(path), video_sha256=old.sha256(path), media=media,
            final_latents=row['final_latents'], final_tensors=row['final_tensors'], audio_pcm_file=file_record(pcm_path),
            audio_pcm=tensor_record(audio), audio_samples_before_aac=audio.shape[-1],
            seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        old.save(result, path.with_suffix('.json')); report['cases'].append(result)
        report['actual_video_vae_calls'] += 1; report['actual_audio_vae_calls'] += 1
        old.save(report, args.output)
        del video, audio, waveform, latents
        gc.collect(); torch.cuda.empty_cache()
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check','denoise','decode'), required=True)
    parser.add_argument('--arm', choices=ARMS, default='full_bf16')
    parser.add_argument('--replica', type=int, choices=(0,1), required=True)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--prepare-report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.report_writable = False
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(experiment='E073', phase=args.phase, arm=args.arm, replica=args.replica, status='running', cases=[],
        actual_video_vae_calls=0, actual_audio_vae_calls=0,
        attempted_dit_calls=0, complete_dit_calls=0, deadline_unix=args.deadline_unix,
        scope='Original BF16 reference; paired native linears/BF16 attention videos reused from E038.')
    started = time.monotonic()
    try:
        manifest = read_manifest(args, report)
        bound, lengths = prepare_bindings(args, manifest, report, required=True)
        require(report['schedule_cpu'] == manifest['expected_schedule'], 'E038 schedules changed')
        for rec in manifest['frozen_runtime_sources']:
            require(file_record(rec['file']) == rec, 'Frozen runtime source changed: '+rec['file'])
        manifest = dict(manifest, cases=[c for c in manifest['cases'] if c['replica'] == args.replica])
        if args.phase == 'check':
            report.update(status='complete', cuda_initialized=torch.cuda.is_initialized())
            require(not report['cuda_initialized'], 'CPU check initialized CUDA')
        elif args.phase == 'decode':
            decode(args, manifest, report)
        else:
            denoise(args, manifest, bound, report)
            budget(args, report)
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        if args.report_writable:
            save(report, args.output)
        print(json.dumps(dict(status=report['status'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
