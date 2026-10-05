#!/usr/bin/env python3
"""E068: four frozen carry/restart videos using the unchanged E010 pipeline.

Preparation is read-only reuse. Each arm runs one byte-exact E065b teacher
replay and two original 20-step trajectories, then VAEs in a fresh process.
No fitting, new scheduler, quality metric, or BF16/TE forward is introduced.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_h3_native_paired_video as e010
import probe_h3_plain_baseline as base
import probe_h3_activation_residual_oracle as inputs
from h3_native_nvfp4 import FORMAT, RECIPE, TARGET_NAMES, install_native_h3
import torch

DATA = Path('/data1/models/svdquant-wjq/research/20261004/E068')
REPORTS = ROOT/'results/research/E068'
PLAN = ROOT/'research_state/06_experiments/E068_h3_carry_pair_video_plan.md'
PARENT = ROOT/'results/research/E065b'
ARMS = ('restart', 'carry')
REPLAY_KEY = 'e010_p030_s05/source_teacher'
MAX_DATA_BYTES = 10*1024**3
WALL_SECONDS = 900
MAX_CALLS = 41
save, sha256, tensor_sha = e010.save, e010.sha256, e010.tensor_sha
file_record, signature = base.file_record, base.tree_signature


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def phase_report(phase, arm=None):
    return REPORTS/(phase+('_'+arm if arm else '')+'.json')


def data_bytes():
    return sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()) if DATA.exists() else 0


def guard(args, report, *, before_forward=False):
    require(time.time() < args.deadline_unix, 'E068 shared absolute deadline reached')
    require(report['attempted_dit_calls'] <= MAX_CALLS and
            (not before_forward or report['attempted_dit_calls'] < MAX_CALLS), 'E068 41-call arm budget exhausted')
    report['data_bytes'] = data_bytes()
    require(report['data_bytes'] <= MAX_DATA_BYTES, 'E068 combined new-data budget exceeded 10 GiB')
    if torch.cuda.is_initialized():
        e010.memory_guard()
        require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E068 peak allocation reached 60 GiB')


def write_tensor(payload, path):
    require(not path.exists(), f'Refusing artifact overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return file_record(path)


def verify_sha(path, expected):
    record = file_record(path)
    require(record['sha256'] == expected, f'Historical file changed: {path}')
    return record


def prerequisites(report):
    require(PLAN.is_file(), 'E068 plan must be frozen before check/run')
    # The original provenance helper checks prompts, assets, runtime sources,
    # Torch and SDPA flags. Its legacy-export record is kept under a separate
    # namespace; it never describes the selected E068 arm.
    binding = {}
    manifest, _ = e010.prerequisites(types.SimpleNamespace(arm='native',
        export_dir=Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export')), binding)
    prep = e010.prepared_inputs(types.SimpleNamespace(report_dir=ROOT/'results/research'), binding)
    require([(c['prompt_id'], c['seed']) for c in manifest['cases']] == [(30, 49771), (36, 59526)],
            'E068 fixed prompt/seed pairs changed')
    expected_settings = dict(height=576, width=1024, num_frames=124, num_inference_steps=20,
        cfg_scale=1., flow_shift=12., audio_flow_shift=3., rand_device='cpu', noise_dtype='bfloat16',
        tiled=True, tile_size=256, tile_overlap=64, fps=24, audio_sample_rate=32000)
    require(manifest['settings'] == expected_settings, 'Original generation settings changed')
    parent_record = file_record(PARENT/'evaluate.json')
    parent = e010.load_complete(base.verify_file(parent_record))
    require(parent['experiment'] == 'E065b' and parent['complete_dit_calls'] == 9 and
            parent['completed_candidate_native_calls'] == 25600 and len(parent['search_layers']) == 200,
            'Complete E065b selected exports required')
    base.check_sources(parent['sources'])
    checks = {}
    for name in ('outputs', 'selection'):
        checks[name] = file_record(PARENT/f'independent_{name}.json')
        checked = e010.load_complete(base.verify_file(checks[name]))
        require(checked['evaluation_sha256'] == parent_record['sha256'] and
                checked['cuda_initialized'] is False and checked['new_model_forwards'] == 0,
                'E065b independent verification binding changed')
        require(sha256(HERE/f'check_h3_carry_q_{name}.py') == checked['script_sha256'],
                'Executed E065b checker source changed')
        if name == 'selection':
            require(checked['final_manifest_selection_bindings'] == 400 and
                    checked['candidate_zero_exact_layers'] == 200, 'Incomplete E065b selection check')
        else:
            require(checked['legacy_replay_byte_exact'], 'E065b legacy replay not exact')
    exit_record = file_record(PARENT/'process_exit.json')
    exited = json.loads(base.verify_file(exit_record).read_text())
    require(exited['exit_code'] == 0 and exited['timeout_exit'] is False, 'E065b did not exit normally')
    selected = {}
    for arm in ARMS:
        path = base.verify_file(parent['exports'][arm])
        exported = e010.load_complete(path)
        require(exported['format'] == FORMAT and exported['recipe'] == RECIPE and
                exported['target_count'] == exported['exact_roundtrip_count'] == 200 and
                [r['name'] for r in exported['layers']] == list(TARGET_NAMES), 'Selected export ABI changed')
        selected[arm] = {}
        for row in exported['layers']:
            require(row['roundtrip']['exact'] and row['roundtrip']['changed_elements'] == 0,
                    'Selected weight roundtrip is not exact')
            selected[arm][row['name']] = dict(file=str((path.parent/row['file']).resolve()),
                sha256=row['file_sha256'], bytes=row['file_bytes'])
    bf16_path = ROOT/'results/research/E010_h3_denoise_bf16.json'
    decode_path = ROOT/'results/research/E010_h3_decode.json'
    bf16, decoded = e010.load_complete(bf16_path), e010.load_complete(decode_path)
    base.check_sources(bf16['sources'])
    base.check_sources(decoded['sources'])
    refs, prepared = [], []
    for case, cached in zip(manifest['cases'], prep['cases'], strict=True):
        require(case['prompt_id'] == cached['prompt_id'] and case['seed'] == cached['seed'], 'Prepared pair identity changed')
        prepared.append(verify_sha(cached['file'], cached['sha256']))
        den = next(r for r in bf16['cases'] if r['prompt_id'] == case['prompt_id'])
        media = next(r for r in decoded['cases'] if r['prompt_id'] == case['prompt_id'] and r['variant'] == 'bf16')
        require(den['status'] == media['status'] == 'complete' and den['variant'] == 'bf16' and
                den['initial_noise_sha256'] == cached['initial_noise_sha256'] and
                den['embedding_sha256'] == cached['embedding_sha256'] and media['settings'] == manifest['settings'],
                'BF16 media does not share the original input/recipe')
        refs.append(dict(prompt_id=case['prompt_id'], seed=case['seed'],
            final_latents=verify_sha(den['final_latents'], den['final_latents_sha256']),
            video=verify_sha(media['video'], media['video_sha256']),
            audio_pcm_file=file_record(media['audio_pcm_file']), audio_pcm_sha256=media['audio_pcm_sha256'],
            media=media['media']))
    e014_path = ROOT/'research_state/06_experiments/E014_h3_plain_baseline_manifest.json'
    replay_case = next(r for r in json.loads(e014_path.read_text())['cases'] if r['id'] == 'e010_p030_s05')
    base.verify_file(replay_case['prepared'])
    for record in replay_case['state'].values():
        base.verify_file(record)
    replays = {}
    for arm in ARMS:
        matches = [r for r in parent['cases'] if (r['case_id'], r['position'], r['arm']) ==
                   ('e010_p030_s05', 'source_teacher', arm)]
        require(len(matches) == 1 and matches[0]['status'] == 'complete', 'Missing E065b selected-arm replay reference')
        replays[arm] = matches[0]
        base.verify_file(replays[arm]['artifact'])
    sources = dict(parent['sources'])
    sources.update({p: file_record(p) for p in binding['sources']})
    for p in (Path(__file__), PLAN, Path(base.__file__), Path(inputs.__file__), e014_path,
              PARENT/'evaluate.json', PARENT/'process_exit.json', bf16_path, decode_path,
              ROOT/'results/research/E010_h3_prepare.json', ROOT/'results/research/E010_h3_check.json',
              HERE/'check_h3_carry_q_outputs.py', HERE/'check_h3_carry_q_selection.py',
              *[Path(r['file']) for r in checks.values()]):
        sources[str(p.resolve())] = file_record(p)
    report.update(plan=file_record(PLAN), sources=sources, settings=manifest['settings'],
        e010_binding=binding, e010_prepare=file_record(ROOT/'results/research/E010_h3_prepare.json'),
        e010_cpu_check=file_record(ROOT/'results/research/E010_h3_check.json'),
        prepared_inputs=prepared, bf16_reference_reports=dict(denoise=file_record(bf16_path), decode=file_record(decode_path)),
        bf16_references=refs, source_evaluation=parent_record, source_checks=checks, source_process_exit=exit_record,
        export_manifests=parent['exports'], selected_exports=selected, state=parent['state'],
        replay_case=replay_case, replay_references=replays, replay_input=parent['inputs'][REPLAY_KEY],
        sdpa_enabled=binding['sdpa_enabled'], torch=torch.__version__,
        environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'DIFFSYNTH_ROOT',
            'DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'MINIMAX_H3_DIT_PATH', 'HF_HUB_OFFLINE')})
    return manifest, prep


@torch.inference_mode()
def cpu_check(report, manifest, prep):
    require(not torch.cuda.is_initialized(), 'E068 check must start without CUDA')
    contract = e010.load_complete(base.verify_file(report['e010_cpu_check']))
    require(contract['cuda_initialized'] is False, 'Original E010 check was not CPU-only')
    base.check_sources(contract['sources'])
    cpu_schedule = e010.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    cpu_schedule.scheduler.set_timesteps(20, shift=12.)
    cpu_schedule.scheduler_audio.set_timesteps(20, shift=3.)
    require(dict(video=cpu_schedule.scheduler.timesteps.tolist(), audio=cpu_schedule.scheduler_audio.timesteps.tolist()) ==
            contract['schedule_cpu'], 'Original CPU scheduler timesteps changed')
    report['schedule_cpu'] = contract['schedule_cpu']
    report['inputs'] = {}
    for case, info, noise_row in zip(manifest['cases'], prep['cases'], contract['cases'], strict=True):
        cached = torch.load(info['file'], map_location='cpu', weights_only=True, mmap=True)
        require(cached['prompt_id'] == case['prompt_id'] and cached['seed'] == case['seed'] and
                cached['prompt'] == case['prompt'] and cached['text_token_tags'].unique().tolist() == [1],
                'Original embedding/noise payload identity changed')
        require(cached['embedding'].dtype == torch.bfloat16 and bool(torch.isfinite(cached['embedding']).all()) and
                tensor_sha(cached['embedding']) == info['embedding_sha256'], 'Embedding changed')
        for name in ('video_latents', 'audio_latents'):
            require(cached[name].device.type == 'cpu' and cached[name].dtype == torch.bfloat16 and
                    bool(torch.isfinite(cached[name]).all()) and
                    tensor_sha(cached[name]) == info['initial_noise_sha256'][name] == noise_row['initial_noise'][name]['sha256'],
                    'Frozen noise differs from original CPU-check signature')
        report['inputs'][str(case['prompt_id'])] = dict(artifact=file_record(info['file']),
            embedding=base.tensor_record(cached['embedding']), text_token_tags=base.tensor_record(cached['text_token_tags']),
            initial_noise={k: base.tensor_record(cached[k]) for k in ('video_latents', 'audio_latents')})
    for rows in report['selected_exports'].values():
        for record in rows.values():
            base.verify_file(record)
    base.verify_file(report['state'])
    value = base.load_case(report['replay_case'])
    cpu_pipe = e010.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    actual = inputs.capture_cpu_inputs(cpu_pipe, value)
    validated = base.validate_case(report['replay_case'], value, cpu_pipe)
    require(actual == report['replay_input']['actual_dit_input_signature'] and
            validated == report['replay_input']['input_signature'], 'CPU replay input differs from E065b')
    for arm, record in report['replay_references'].items():
        payload = torch.load(base.verify_file(record['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(signature(payload['actual_dit_inputs']) == actual, 'Saved E065b actual input changed')
        for field in ('raw_outputs', 'velocities'):
            require(signature(payload[field]) == record[field], 'Saved selected-arm output changed')
    for ref in report['bf16_references']:
        pcm = torch.load(base.verify_file(ref['audio_pcm_file']), map_location='cpu', weights_only=True, mmap=True)
        require(pcm['sample_rate'] == 32000 and tensor_sha(pcm['waveform']) == ref['audio_pcm_sha256'],
                'Historical BF16 PCM changed')
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, checked_selected_exports=400,
        new_model_forwards=0, replay_actual_input_signature=actual)


def bind_check(report):
    path = phase_report('check')
    checked = e010.load_complete(path)
    require(checked['cuda_initialized'] is False and checked['new_model_forwards'] == 0 and
            checked['checked_selected_exports'] == 400, 'Complete CPU-only E068 check required')
    for field in ('sources', 'plan', 'settings', 'e010_cpu_check', 'prepared_inputs', 'bf16_references', 'source_evaluation',
                  'source_checks', 'source_process_exit', 'export_manifests', 'selected_exports', 'state',
                  'replay_case', 'replay_references', 'replay_input', 'sdpa_enabled', 'torch'):
        require(checked[field] == report[field], f'E068 CPU binding changed: {field}')
    report['cpu_check_reference'] = file_record(path)
    report['checked_inputs'] = checked['inputs']
    report['schedule_cpu'] = checked['schedule_cpu']
    return checked


def expected_sdpa(kwargs):
    def positive(field):
        cu = kwargs[field]['cu_seqlens_q']
        require(cu.ndim == 1 and int(cu[0]) == 0 and bool((cu[1:] >= cu[:-1]).all()), 'Invalid packed segments')
        return int((cu[1:] > cu[:-1]).sum())
    return 50*positive('packed_seq_params')+2*positive('refiner_packed_seq_params')


def record_call(report, counts, fastpack):
    require(counts['scaled_mm_calls'] == 200 and counts['disk_loads'] == 0 and
            fastpack['checked_calls'] == 200, 'Native full-forward audit failed')
    report['complete_dit_calls'] += 1
    report['activation_packs'] += fastpack['checked_calls']
    for name in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads'):
        report['runtime_totals'][name] += counts[name]


@torch.inference_mode()
def replay_teacher(pipe, args, report, stage):
    guard(args, report, before_forward=True)
    value = base.load_case(report['replay_case'])
    cpu_pipe = e010.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    require(base.validate_case(report['replay_case'], value, cpu_pipe) == report['replay_input']['input_signature'],
            'Teacher input changed after CPU check')
    forward = base.gpu_call(pipe, {'kind': 'model_fn'}, value)
    actual, raw_capture, sdpa_expected = [], [], []
    def pre(_module, pos, kw):
        captured = e010.tree_cpu(dict(args=pos, kwargs=kw))
        require(signature(captured) == report['replay_input']['actual_dit_input_signature'], 'Replay actual input mismatch')
        actual.append(captured)
        sdpa_expected.append(expected_sdpa(kw))
    def post(_module, _pos, output):
        raw_capture.append(output)
    prehook = pipe.dit.register_forward_pre_hook(pre, with_kwargs=True)
    posthook = pipe.dit.register_forward_hook(post)
    audit = e010.RuntimeAudit()
    audit.phase = 'teacher_replay_'+args.arm
    report['attempted_dit_calls'] += 1
    save(report, args.output)
    try:
        with audit.installed(), e010.collect_fastpack_checks() as checks:
            output = forward()
            torch.cuda.synchronize()
    finally:
        prehook.remove()
        posthook.remove()
    require(len(actual) == len(raw_capture) == len(sdpa_expected) == 1, 'Replay must make exactly one DiT call')
    counts = dict(audit.row())
    require(counts['sdpa_calls'] == sdpa_expected[0], 'Replay SDPA count mismatch')
    record_call(report, counts, checks.summary)
    raw, raw_sig = base.cpu_outputs(raw_capture[0])
    velocity, velocity_sig = base.cpu_outputs(output)
    reference = report['replay_references'][args.arm]
    require(raw_sig == reference['raw_outputs'] and velocity_sig == reference['velocities'],
            'Selected-arm replay is not byte-exact in raw AND velocity')
    artifact = write_tensor(dict(arm=args.arm, actual_dit_inputs=actual[0], raw_outputs=raw,
        velocities=velocity, reference=reference['artifact']), stage/'teacher_replay.pt')
    report['teacher_replay'] = dict(arm=args.arm, case_id='e010_p030_s05', position='source_teacher',
        artifact=artifact, actual_input_signature=signature(actual[0]), raw_outputs=raw_sig,
        velocities=velocity_sig, byte_exact=True, counts=counts, fastpack_checks=checks.summary)
    report['teacher_replay_calls'] = 1
    save(report, args.output)
    del forward, value, output, actual, raw_capture, raw, velocity
    gc.collect()
    torch.cuda.empty_cache()
    guard(args, report)


@torch.inference_mode()
def denoise(args, manifest, prep, report):
    guard(args, report)
    require(torch.cuda.get_device_capability() == (12, 0), 'E068 requires SM120 native NVFP4')
    stage = DATA/('denoise_'+args.arm)
    stage.mkdir(parents=True, exist_ok=False)
    e010.snapshot_sources(report, stage/'sources')
    pipe = e010.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    require(pipe.text_encoder is None and pipe.video_vae is None and pipe.audio_vae is None,
            'Denoise process must load DiT only')
    report['resident_conversion'] = e010.make_h3_resident(pipe.dit)
    before = base.non_target_identity(pipe.dit)
    export_dir = Path(report['export_manifests'][args.arm]['file']).parent
    report['native_installation'] = install_native_h3(pipe.dit, export_dir,
        activation_packer=e010.pack_activation_fast, chunk_rows=1024)
    require(report['native_installation']['target_count'] == 200 and
            base.non_target_identity(pipe.dit) == before, 'Selected installation altered non-target tensors')
    report['installed_arm'] = args.arm
    gc.collect()
    torch.cuda.empty_cache()
    replay_teacher(pipe, args, report, stage)
    original_step, original_forward = pipe.step, pipe.dit.forward
    settings = manifest['settings']
    with e010.RuntimeAudit().installed() as audit:
        for case, cached_info in zip(manifest['cases'], prep['cases'], strict=True):
            guard(args, report)
            verify_sha(cached_info['file'], cached_info['sha256'])
            cached = torch.load(cached_info['file'], map_location='cpu', weights_only=True, mmap=True)
            directory = stage/f'p{case["prompt_id"]:03d}'
            directory.mkdir()
            info = dict(prompt_id=case['prompt_id'], seed=case['seed'], variant=args.arm, prompt=case['prompt'],
                initial_noise_sha256=cached_info['initial_noise_sha256'], embedding_sha256=cached_info['embedding_sha256'],
                steps=[], dit_calls=[], status='running')
            report['cases'].append(info)
            noise_calls, final = [], {}
            audit.phase = f'{args.arm}_p{case["prompt_id"]}'

            def cached_noise(_self, shape, seed=None, rand_device='cpu', rand_torch_dtype=torch.float32,
                             device=None, torch_dtype=None):
                index = len(noise_calls)
                name = ('video_latents', 'audio_latents')[index] if index < 2 else None
                require(name is not None and tuple(shape) == tuple(cached[name].shape) and seed == case['seed'] and
                        rand_device == 'cpu' and rand_torch_dtype == torch.bfloat16, 'Original noise initializer changed')
                noise_calls.append(name)
                return cached[name].clone().to(device=device or pipe.device, dtype=torch_dtype or pipe.torch_dtype)

            def resident_placement(_self, names):
                if tuple(names) == tuple(pipe.in_iteration_models):
                    return
                if list(names) == ['video_vae']:
                    require(len(info['steps']) == 40 and len(info['dit_calls']) == 20, 'Incomplete original sampler')
                    raise e010.DenoiseComplete('Original sampler finished; defer original VAE to fresh process')
                raise RuntimeError(f'Unexpected component load during resident sampling: {names}')

            def observed_forward(_self, *pos, **kwargs):
                guard(args, report, before_forward=True)
                count_expected = expected_sdpa(kwargs)
                before_call = dict(audit.row())
                report['attempted_dit_calls'] += 1
                with e010.collect_fastpack_checks() as checks:
                    output = original_forward(*pos, **kwargs)
                now = audit.row()
                counts = {k: now[k]-before_call[k] for k in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')}
                require(counts['sdpa_calls'] == count_expected, 'Resident SDPA count changed')
                require(all(bool(torch.isfinite(x).all()) for x in output), 'Nonfinite raw denoiser output')
                record_call(report, counts, checks.summary)
                report['rollout_dit_calls'] += 1
                info['dit_calls'].append(dict(step=len(info['dit_calls']), counts=counts,
                    zero_sf_checks=checks.summary, video_sha256=tensor_sha(output[0]), audio_sha256=tensor_sha(output[1])))
                return output

            def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
                guard(args, report)
                name = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
                require(name is not None and 0 <= progress_id < 20, 'Unexpected original scheduler call')
                expected_order = [(step, modality) for step in range(20) for modality in ('video', 'audio')]
                require(len(info['steps']) < 40 and (progress_id, name) == expected_order[len(info['steps'])],
                        'Scheduler update order changed')
                if progress_id == 0:
                    require(tensor_sha(latents) == cached_info['initial_noise_sha256'][name+'_latents'],
                            'Initial CPU noise differs between arms')
                require(bool(torch.isfinite(noise_pred).all()), 'Nonfinite scheduler velocity')
                updated = original_step(scheduler, latents, progress_id, noise_pred, **kwargs)
                require(bool(torch.isfinite(updated).all()), 'Nonfinite recurrent latent')
                path = directory/f's{progress_id:02d}_{name}.pt'
                artifact = write_tensor(dict(latents_before=latents.detach().cpu(), noise_pred=noise_pred.detach().cpu(),
                    latents_after=updated.detach().cpu(), timestep=scheduler.timesteps[progress_id].cpu(),
                    sigma=scheduler.sigmas[progress_id].cpu()), path)
                info['steps'].append(dict(step=progress_id, modality=name, file=str(path), sha256=artifact['sha256'],
                    bytes=artifact['bytes'], latent_sha256=tensor_sha(updated), velocity_sha256=tensor_sha(noise_pred)))
                report['scheduler_updates'] += 1
                if progress_id == 19:
                    final[name+'_latents'] = updated.detach().cpu()
                guard(args, report)
                save(report, args.output)
                return updated

            pipe.generate_noise = types.MethodType(cached_noise, pipe)
            pipe.load_models_to_device = types.MethodType(resident_placement, pipe)
            pipe.step = types.MethodType(observed_step, pipe)
            pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
            started = time.monotonic()
            print(f'E068 {args.arm} p{case["prompt_id"]}: original 20-step sampler', flush=True)
            try:
                pipe(prompt=None, text_embedding=cached['embedding'], seed=case['seed'],
                     height=settings['height'], width=settings['width'], num_frames=settings['num_frames'],
                     num_inference_steps=20, cfg_scale=1., flow_shift=12., audio_flow_shift=3.,
                     rand_device='cpu', tiled=True, tile_size=256, tile_overlap=64)
            except e010.DenoiseComplete:
                pass
            else:
                raise RuntimeError('Expected original pipeline stop before VAE load')
            require(noise_calls == ['video_latents', 'audio_latents'] and set(final) == {'video_latents', 'audio_latents'},
                    'Incomplete paired-noise/final-latent contract')
            require(pipe.scheduler.timesteps.tolist() == report['schedule_cpu']['video'] and
                    pipe.scheduler_audio.timesteps.tolist() == report['schedule_cpu']['audio'], 'Original schedules changed')
            path = directory/'final_latents.pt'
            artifact = write_tensor(final, path)
            info.update(status='complete', final_latents=str(path), final_latents_sha256=artifact['sha256'],
                seconds_including_diagnostics=time.monotonic()-started,
                video_timesteps=pipe.scheduler.timesteps.tolist(), audio_timesteps=pipe.scheduler_audio.timesteps.tolist(),
                runtime_audit=dict(audit.row()))
            guard(args, report)
            save(report, args.output)
            del final, cached
            gc.collect()
            torch.cuda.empty_cache()
    require(report['attempted_dit_calls'] == report['complete_dit_calls'] == 41 and
            report['rollout_dit_calls'] == 40 and report['teacher_replay_calls'] == 1 and
            report['activation_packs'] == report['runtime_totals']['scaled_mm_calls'] == 8200 and
            report['runtime_totals']['sdpa_calls'] == 4182 and report['scheduler_updates'] == 80,
            'Completed arm count contract failed')
    report['status'] = 'complete'


@torch.inference_mode()
def decode_videos(args, manifest, report):
    from diffsynth.utils.data.audio_video import write_video_audio
    import av
    den_path = phase_report('denoise', args.arm)
    den = e010.load_complete(den_path)
    require(den['arm'] == args.arm and den['complete_dit_calls'] == 41 and den['teacher_replay']['byte_exact'],
            'Selected denoiser prerequisite incomplete')
    for field in ('plan', 'prepared_inputs', 'export_manifests', 'source_evaluation', 'cpu_check_reference',
                  'budget_start_unix', 'deadline_unix'):
        require(den[field] == report[field], f'Decode binding differs from denoise: {field}')
    # Source snapshots are observational fields added only after CPU binding.
    require({p: r['sha256'] for p, r in den['sources'].items()} ==
            {p: r['sha256'] for p, r in report['sources'].items()}, 'Decode runtime source set changed')
    report['denoiser_reports'] = {args.arm: file_record(den_path)}
    report['encoding_environment'] = dict(pyav=av.__version__, library_versions=av.library_versions,
        h264_encoder=av.codec.Codec('libx264', 'w').name, aac_encoder=av.codec.Codec('aac', 'w').name,
        historical_encoding_versions_recorded=False, historical_media_byte_replay_claim=False)
    stage = DATA/('decode_'+args.arm)
    stage.mkdir(parents=True, exist_ok=False)
    e010.snapshot_sources(report, stage/'sources')
    guard(args, report)
    pipe = e010.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[e010.ModelConfig(path=str(e010.MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'), **e010.disk_config()),
                       e010.ModelConfig(path=str(e010.MODEL_ROOT/'FL2VA/audio_vae/model.safetensors'), **e010.disk_config())],
        processor_config=None, vram_limit=30.)
    require(pipe.dit is None and pipe.text_encoder is None, 'Decode process must contain VAEs only')
    require(type(pipe.video_vae).__name__ == 'MiniMaxH3VideoVAE' and pipe.audio_vae.sample_rate == 32000,
            'Original VAE class/audio sample rate changed')
    pipe.video_vae.eval()
    pipe.audio_vae.eval()
    for case, record in zip(manifest['cases'], den['cases'], strict=True):
        guard(args, report)
        require(record['prompt_id'] == case['prompt_id'] and record['seed'] == case['seed'] and
                record['variant'] == args.arm and record['status'] == 'complete', 'Denoised case identity changed')
        verify_sha(record['final_latents'], record['final_latents_sha256'])
        latents = torch.load(record['final_latents'], map_location='cpu', weights_only=True, mmap=True)
        pipe.load_models_to_device(['video_vae'])
        recon = pipe.video_vae.decode_video(latents['video_latents'].cuda(), dtype=torch.bfloat16,
            tiled=True, tile_size=256, tile_overlap=64)
        require(bool(torch.isfinite(recon).all()), 'Nonfinite decoded video')
        video = pipe.vae_output_to_video(recon, min_value=0, max_value=1)
        report['video_vae_calls'] += 1
        del recon
        require(len(video) == 124 and all(frame.size == (1024, 576) for frame in video), 'Video frame/shape changed')
        guard(args, report)
        pipe.load_models_to_device(['audio_vae'])
        waveform = pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(), dtype=torch.bfloat16)
        audio = pipe.output_audio_format_check(waveform)
        report['audio_vae_calls'] += 1
        require(bool(torch.isfinite(audio).all()) and audio.shape[0] == 2, 'Invalid decoded stereo audio')
        path = stage/f'p{case["prompt_id"]:03d}_{args.arm}.mp4'
        require(not path.exists(), 'Refusing media overwrite')
        write_video_audio(video, audio, str(path), fps=24, audio_sample_rate=32000)
        with av.open(str(path)) as container:
            frames = sum(1 for _ in container.decode(video=0))
            vs = container.streams.video[0]
            actual = dict(frames=frames, width=vs.width, height=vs.height, fps=float(vs.average_rate),
                audio_streams=len(container.streams.audio),
                audio_sample_rate=container.streams.audio[0].codec_context.sample_rate)
            codecs = dict(video=vs.codec_context.name, audio=container.streams.audio[0].codec_context.name,
                          audio_channels=container.streams.audio[0].codec_context.channels)
        require(actual == dict(frames=124, width=1024, height=576, fps=24., audio_streams=1, audio_sample_rate=32000),
                f'Encoded media metadata mismatch: {actual}')
        require(codecs == dict(video='h264', audio='aac', audio_channels=2), f'Encoded codecs changed: {codecs}')
        audio_path = path.with_suffix('.audio.pt')
        audio_record = write_tensor(dict(waveform=audio, sample_rate=32000), audio_path)
        row = dict(**case, video=str(path), variant=args.arm, settings=manifest['settings'],
            video_sha256=sha256(path), status='complete', media=actual, codecs=codecs,
            audio_samples_before_aac=audio.shape[-1], audio_pcm_sha256=tensor_sha(audio),
            audio_pcm_file=str(audio_path), audio_pcm_artifact=audio_record,
            initial_noise_sha256=record['initial_noise_sha256'], embedding_sha256=record['embedding_sha256'],
            final_latents=dict(file=record['final_latents'], sha256=record['final_latents_sha256']),
            encoding_environment=report['encoding_environment'])
        save(row, path.with_suffix('.json'))
        report['cases'].append(row)
        guard(args, report)
        save(report, args.output)
        del video, audio, waveform, latents
        gc.collect()
        torch.cuda.empty_cache()
    require(report['video_vae_calls'] == report['audio_vae_calls'] == 2 and report['complete_dit_calls'] == 0,
            'Decode-only call budget failed')
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'denoise', 'decode'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--budget-start-unix', type=float)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if (args.phase != 'check') != (args.arm is not None):
        parser.error('--arm is required exactly for denoise/decode')
    if args.phase != 'check' and (args.budget_start_unix is None or args.deadline_unix is None or
            not 0 < args.deadline_unix-args.budget_start_unix <= WALL_SECONDS or
            not args.budget_start_unix <= time.time() < args.deadline_unix):
        parser.error('GPU phases require the same original start and deadline, at most 900 seconds apart')
    args.output = phase_report(args.phase, args.arm)
    require(not args.output.exists(), f'Refusing report overwrite: {args.output}')
    if args.phase != 'check':
        require(not (DATA/(args.phase+'_'+args.arm)).exists(), 'Refusing existing phase data directory')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    report = dict(experiment='E068', phase=args.phase, arm=args.arm, status='running',
        budget_start_unix=args.budget_start_unix, deadline_unix=args.deadline_unix,
        wall_budget_seconds=WALL_SECONDS, data_byte_limit=MAX_DATA_BYTES, data_bytes=data_bytes(),
        attempted_dit_calls=0, complete_dit_calls=0, teacher_replay_calls=0, rollout_dit_calls=0,
        bf16_dit_calls=0, text_encoder_calls=0,
        scheduler_updates=0, activation_packs=0, video_vae_calls=0, audio_vae_calls=0,
        runtime_totals=dict(sdpa_calls=0, scaled_mm_calls=0, disk_loads=0), cases=[],
        scope='Two prior diagnostic prompts, four free videos; task-relevance check, not innovation or an independent quality benchmark')
    started = time.monotonic()
    try:
        if args.phase != 'check':
            def expired(_signum, _frame):
                raise TimeoutError('E068 original shared 900-second deadline reached')
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
            for arm in ARMS:
                for phase in ('denoise', 'decode'):
                    prior_path = phase_report(phase, arm)
                    if prior_path.is_file():
                        prior = json.loads(prior_path.read_text())
                        require(prior['budget_start_unix'] == args.budget_start_unix and
                                prior['deadline_unix'] == args.deadline_unix,
                                'All E068 GPU phases must share the original root-specified budget')
        manifest, prep = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, manifest, prep)
        else:
            bind_check(report)
            guard(args, report)
            if args.phase == 'denoise':
                denoise(args, manifest, prep, report)
            else:
                decode_videos(args, manifest, report)
            guard(args, report)
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if args.phase != 'check':
            signal.setitimer(signal.ITIMER_REAL, 0)
        report['seconds_total'] = time.monotonic()-started
        report['data_bytes'] = data_bytes()
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if torch.cuda.is_initialized():
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        print(json.dumps(dict(status=report['status'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
