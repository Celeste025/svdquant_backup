#!/usr/bin/env python3
"""Independent CPU artifact checks for E068; never imports its runner.

Run only after both supervisors and all four GPU phases complete successfully.
The private A/B commitment is hashed as bytes, never parsed or printed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import torch
import check_h3_swiglu_interaction as io

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E068'
DEST = REPORTS/'independent.json'
ARMS = ('restart', 'carry')
PAIRS = ((30, 49771), (36, 59526))
MODALITIES = ('video', 'audio')
SHAPES = {'video': (1, 24, 37, 36, 64), 'audio': (2, 32, 207)}
NAMES = [f'blocks.{i}.{kind}' for i in range(50) for kind in
         ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')]
ORDER = [(step, modality) for step in range(20) for modality in MODALITIES]


def require(value, message):
    if not value:
        raise AssertionError(message)


def finite_tensor(value, shape, dtype):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            tuple(value.shape) == tuple(shape) and value.dtype == dtype and
            bool(torch.isfinite(value).all()), 'Tensor shape/dtype/device/finite mismatch')


def fastpack(row):
    require(row['checked_calls'] == 200 and row['invalid_calls'] == 0, 'Fastpack count or validity mismatch')


def native_counts(row, full_calls):
    expected = dict(sdpa_calls=102*full_calls, scaled_mm_calls=200*full_calls, disk_loads=0)
    require({k: row[k] for k in expected} == expected, 'Native runtime counts mismatch')


def source_identity(sources):
    return {str(Path(p).resolve()): (r['sha256'], r.get('bytes')) for p, r in sources.items()}


def main():
    require(not DEST.exists(), 'Preserve prior independent result')
    started = time.monotonic()
    torch.set_num_threads(2)
    result = dict(experiment='E068', status='running', reports={}, files={}, arms={},
                  new_model_forwards=0, cuda_initialized=False,
                  script_sha256=io.sha(Path(__file__)), io_helper_sha256=io.sha(Path(io.__file__)))

    def verify(record):
        path = Path(record['file']).resolve()
        key = str(path)
        if key not in result['files']:
            result['files'][key] = dict(file=key, sha256=io.sha(path), bytes=path.stat().st_size)
        actual = result['files'][key]
        require(actual['sha256'] == record['sha256'] and
                ('bytes' not in record or actual['bytes'] == record['bytes']), f'File identity mismatch: {path}')
        return path

    def rec(path):
        path = Path(path).resolve()
        actual = dict(file=str(path), sha256=io.sha(path), bytes=path.stat().st_size)
        return actual

    def read_report(path, expected=None, complete=True):
        record = rec(path) if expected is None else expected
        path = verify(record)
        result['reports'][path.name] = result['files'][str(path)]
        value = json.loads(path.read_text())
        if complete:
            require(value['status'] == 'complete', f'Incomplete report: {path}')
        return value

    def load(record):
        return torch.load(verify(record), map_location='cpu', weights_only=True, mmap=True)

    try:
        require(not torch.cuda.is_initialized(), 'Checker started with CUDA initialized')
        launch = read_report(REPORTS/'launch.json', complete=False)
        check = read_report(REPORTS/'check.json', dict(file=str(REPORTS/'check.json'),
                                                     sha256=launch['cpu_check_sha256']))
        require(launch['experiment'] == check['experiment'] == 'E068', 'Wrong experiment')
        require(check['phase'] == 'check' and check['cuda_initialized'] is False and
                check['new_model_forwards'] == check['complete_dit_calls'] == 0 and
                check['checked_selected_exports'] == 400, 'CPU preflight contract mismatch')
        start, deadline = launch['budget_start_unix'], launch['deadline_unix']
        require(0 < deadline-start <= 900 and set(launch['arms']) == set(ARMS), 'Launch budget mismatch')
        require(len({launch['arms'][arm]['gpu'] for arm in ARMS}) == 2, 'Arms require distinct GPUs')
        verify(dict(file=launch['runner'], sha256=launch['source_sha256']))
        verify(dict(file=launch['plan'], sha256=launch['plan_sha256']))
        verify(dict(file=str(ROOT/'scripts/research/supervise_h3_carry_pair_video.py'),
                    sha256=launch['supervisor_sha256']))
        verify(dict(file=str(REPORTS/'private/blind_key.json'), sha256=launch['blind_commitment_sha256']))
        result['blind_commitment'] = dict(sha256=launch['blind_commitment_sha256'], verified=True, mapping_parsed=False)
        require(check['plan']['sha256'] == launch['plan_sha256'] and
                check['sources'][launch['runner']]['sha256'] == launch['source_sha256'], 'Launch/check source mismatch')
        for path, record in check['sources'].items():
            verify(dict(record, file=path))
        parent = read_report(check['source_evaluation']['file'], check['source_evaluation'])
        require(parent['experiment'] == 'E065b' and parent['exports'] == check['export_manifests'], 'Wrong frozen exports')
        parent_exit = read_report(check['source_process_exit']['file'], check['source_process_exit'], complete=False)
        require(parent_exit['exit_code'] == 0 and parent_exit['timeout_exit'] is False, 'Parent did not exit normally')
        for name, record in check['source_checks'].items():
            verified = read_report(record['file'], record)
            require(verified['evaluation_sha256'] == check['source_evaluation']['sha256'] and
                    verified['cuda_initialized'] is False and verified['new_model_forwards'] == 0,
                    'Parent independent check binding mismatch')
            require(io.sha(ROOT/f'scripts/research/check_h3_carry_q_{name}.py') == verified['script_sha256'],
                    'Parent independent checker source changed')
        prep = read_report(check['e010_prepare']['file'], check['e010_prepare'])
        bf = read_report(check['bf16_reference_reports']['denoise']['file'], check['bf16_reference_reports']['denoise'])
        bf_decode = read_report(check['bf16_reference_reports']['decode']['file'], check['bf16_reference_reports']['decode'])
        settings = dict(height=576, width=1024, num_frames=124, num_inference_steps=20,
                        cfg_scale=1., flow_shift=12., audio_flow_shift=3., rand_device='cpu', noise_dtype='bfloat16',
                        tiled=True, tile_size=256, tile_overlap=64, fps=24, audio_sample_rate=32000)
        require(check['settings'] == prep['settings'] == bf['settings'] == bf_decode['settings'] == settings,
                'Frozen generation or decoding settings differ')
        require([(r['prompt_id'], r['seed']) for r in prep['cases']] == list(PAIRS), 'Prepared case order differs')
        prepared, history, bf_final, bf_pcm = {}, {}, {}, {}
        for prep_row, (pid, seed), ref in zip(prep['cases'], PAIRS, check['bf16_references'], strict=True):
            input_row = check['inputs'][str(pid)]
            require(input_row['artifact']['file'] == prep_row['file'] and
                    input_row['artifact']['sha256'] == prep_row['sha256'], 'Prepared artifact binding mismatch')
            value = load(input_row['artifact'])
            require((value['prompt_id'], value['seed']) == (pid, seed), 'Prepared payload identity mismatch')
            finite_tensor(value['embedding'], (501 if pid == 30 else 813, 5120), torch.bfloat16)
            require(io.signature(value['embedding']) == input_row['embedding'] and
                    io.signature(value['text_token_tags']) == input_row['text_token_tags'] and
                    value['text_token_tags'].unique().tolist() == [1], 'Embedding/tags mismatch')
            for modality in MODALITIES:
                name = modality+'_latents'
                finite_tensor(value[name], SHAPES[modality], torch.bfloat16)
                require(io.signature(value[name]) == input_row['initial_noise'][name] and
                        io.signature(value[name])['sha256'] == prep_row['initial_noise_sha256'][name], 'Noise identity mismatch')
            prepared[pid] = value
            old = next(r for r in bf['cases'] if r['prompt_id'] == pid)
            old_media = next(r for r in bf_decode['cases'] if r['prompt_id'] == pid and r['variant'] == 'bf16')
            require((old['seed'], old_media['seed'], ref['seed']) == (seed, seed, seed) and ref['prompt_id'] == pid,
                    'Historical BF16 case mismatch')
            require(old['prompt'] == old_media['prompt'] == value['prompt'] and
                    old['initial_noise_sha256'] == prep_row['initial_noise_sha256'] and
                    old['embedding_sha256'] == prep_row['embedding_sha256'], 'Historical BF16 input mismatch')
            require([(r['step'], r['modality']) for r in old['steps']] == ORDER, 'Historical scheduler order mismatch')
            history[pid] = {}
            for row in old['steps']:
                payload = load(row)
                for field in ('timestep', 'sigma'):
                    finite_tensor(payload[field], (), torch.float32)
                history[pid][(row['step'], row['modality'])] = {k: payload[k].clone() for k in ('timestep', 'sigma')}
            require(ref['final_latents']['file'] == old['final_latents'] and
                    ref['final_latents']['sha256'] == old['final_latents_sha256'], 'BF16 terminal reference mismatch')
            bf_final[pid] = load(ref['final_latents'])
            for modality in MODALITIES:
                finite_tensor(bf_final[pid][modality+'_latents'], SHAPES[modality], torch.bfloat16)
            require(ref['video']['file'] == old_media['video'] and
                    ref['video']['sha256'] == old_media['video_sha256'], 'BF16 video reference mismatch')
            verify(ref['video'])
            pcm = load(ref['audio_pcm_file'])
            finite_tensor(pcm['waveform'], (2, 165600), torch.float32)
            require(pcm['sample_rate'] == 32000 and io.signature(pcm['waveform'])['sha256'] ==
                    ref['audio_pcm_sha256'] == old_media['audio_pcm_sha256'], 'Historical PCM mismatch')
            bf_pcm[pid] = io.signature(pcm['waveform'])
        # Saved selection manifests + installation receipts are bound here.
        # Packed weights were hashed by the frozen preflight; no second full weight read.
        exports = {}
        for arm in ARMS:
            reference = check['export_manifests'][arm]
            manifest = read_report(reference['file'], reference)
            require(manifest['arm'] == arm and manifest['target_count'] == manifest['exact_roundtrip_count'] == 200 and
                    [r['name'] for r in manifest['layers']] == NAMES, 'Selected export identity/count mismatch')
            require(set(check['selected_exports'][arm]) == set(NAMES), 'Selected source layer set mismatch')
            for row in manifest['layers']:
                path = (Path(reference['file']).parent/row['file']).resolve()
                binding = check['selected_exports'][arm][row['name']]
                require(binding == dict(file=str(path), sha256=row['file_sha256'], bytes=row['file_bytes']),
                        'Selected export path/hash/size binding mismatch')
            exports[arm] = manifest
        shared_fields = ('plan', 'settings', 'prepared_inputs', 'bf16_references', 'bf16_reference_reports',
                         'source_evaluation', 'source_checks', 'source_process_exit', 'export_manifests',
                         'selected_exports', 'state', 'replay_case', 'replay_references', 'replay_input',
                         'sdpa_enabled', 'torch', 'e010_prepare')
        encoding = None
        import av
        result['checker_media_environment'] = dict(pyav=av.__version__, library_versions=av.library_versions)
        for arm in ARMS:
            exit_row = read_report(REPORTS/f'process_exit_{arm}.json', complete=False)
            require(exit_row['arm'] == arm and exit_row['exit_code'] == 0 and 'supervisor_error' not in exit_row,
                    'Supervisor did not complete normally')
            require(exit_row['budget_start_unix'] == start and exit_row['deadline_unix'] == deadline and
                    start <= exit_row['start_epoch'] <= exit_row['end_epoch'] <= deadline, 'Supervisor budget mismatch')
            require([r['phase'] for r in exit_row['phases']] == ['denoise', 'decode'], 'Supervisor phase sequence differs')
            for phase in exit_row['phases']:
                require(phase['exit_code'] == 0 and start <= phase['start_epoch'] <= phase['end_epoch'] <= deadline,
                        'Phase failed or exceeded original deadline')
                cmd = launch['arms'][arm]['commands'][phase['phase']]
                require(phase['command'][:2] == ['timeout', '--kill-after=15s'] and phase['command'][3:] == cmd,
                        'Executed command differs from launch')
                require(cmd[cmd.index('--phase')+1] == phase['phase'] and cmd[cmd.index('--arm')+1] == arm and
                        float(cmd[cmd.index('--budget-start-unix')+1]) == start and
                        float(cmd[cmd.index('--deadline-unix')+1]) == deadline and launch['runner'] in cmd,
                        'Launched phase/arm/deadline differs')
            require(exit_row['phases'][0]['end_epoch'] <= exit_row['phases'][1]['start_epoch'], 'GPU phases overlap')
            den = read_report(REPORTS/f'denoise_{arm}.json')
            dec = read_report(REPORTS/f'decode_{arm}.json')
            for phase, report in (('denoise', den), ('decode', dec)):
                require(report['experiment'] == 'E068' and report['phase'] == phase and report['arm'] == arm,
                        'Phase identity mismatch')
                require(report['budget_start_unix'] == start and report['deadline_unix'] == deadline and
                        report['wall_budget_seconds'] == 900 and report['peak_allocated_gib'] < 60 and
                        report['data_byte_limit'] == 10*1024**3 and report['data_bytes'] <= 10*1024**3,
                        'Phase resource contract mismatch')
                require(report['cpu_check_reference']['sha256'] == launch['cpu_check_sha256'] and
                        report['environment']['CUDA_VISIBLE_DEVICES'] == str(launch['arms'][arm]['gpu']) and
                        report['bf16_dit_calls'] == report['text_encoder_calls'] == 0, 'Phase check/device/forward binding mismatch')
                for field in shared_fields:
                    require(report[field] == check[field], f'CPU binding changed: {field}')
                require(source_identity(report['sources']) == source_identity(check['sources']), 'Phase source binding differs')
                require(report['checked_inputs'] == check['inputs'] and report['schedule_cpu'] == check['schedule_cpu'],
                        'Checked inputs/schedule differ')
                for path, source in report['sources'].items():
                    verify(dict(source, file=path))
                    if 'snapshot' in source:
                        verify(dict(file=source['snapshot'], sha256=source['sha256'], bytes=source['bytes']))
            require(den['attempted_dit_calls'] == den['complete_dit_calls'] == 41 and
                    den['rollout_dit_calls'] == 40 and den['teacher_replay_calls'] == 1 and
                    den['activation_packs'] == 8200 and den['scheduler_updates'] == 80 and
                    den['video_vae_calls'] == den['audio_vae_calls'] == 0 and den['installed_arm'] == arm,
                    'Denoiser call contract mismatch')
            native_counts(den['runtime_totals'], 41)
            installed = den['native_installation']
            require(installed['manifest_sha256'] == check['export_manifests'][arm]['sha256'] and
                    installed['target_count'] == installed['exact_roundtrip_count'] == 200 and
                    [r['name'] for r in installed['layers']] == NAMES, 'Installed export mismatch')
            for got, expected in zip(installed['layers'], exports[arm]['layers'], strict=True):
                require(got['file_sha256'] == expected['file_sha256'] and got['roundtrip'] == expected['roundtrip'] and
                        got['old_weight_parameter_cleared'], 'Installed layer receipt mismatch')
            replay = den['teacher_replay']
            reference = next(r for r in parent['cases'] if
                             (r['case_id'], r['position'], r['arm']) == ('e010_p030_s05', 'source_teacher', arm))
            require(replay['arm'] == arm and replay['case_id'] == 'e010_p030_s05' and
                    replay['position'] == 'source_teacher' and replay['byte_exact'], 'Replay identity differs')
            a, b = load(replay['artifact']), load(reference['artifact'])
            require(a['arm'] == arm and a['reference'] == reference['artifact'], 'Replay artifact reference differs')
            for field in ('actual_dit_inputs', 'raw_outputs', 'velocities'):
                require(io.byte_equal(a[field], b[field]), f'Replay is not byte exact: {field}')
                key = 'actual_input_signature' if field == 'actual_dit_inputs' else field
                require(io.signature(a[field]) == replay[key], 'Replay tensor signature differs')
            require(replay['actual_input_signature'] == check['replay_actual_input_signature'], 'Replay CPU actual input differs')
            native_counts(replay['counts'], 1)
            fastpack(replay['fastpack_checks'])
            del a, b
            require([(c['prompt_id'], c['seed']) for c in den['cases']] == list(PAIRS) and
                    [(c['prompt_id'], c['seed']) for c in dec['cases']] == list(PAIRS), 'Rollout case set/order differs')
            require(dec['video_vae_calls'] == dec['audio_vae_calls'] == 2 and
                    all(dec[k] == 0 for k in ('attempted_dit_calls', 'complete_dit_calls', 'teacher_replay_calls',
                                             'rollout_dit_calls', 'scheduler_updates', 'activation_packs')),
                    'Decode-only call contract mismatch')
            native_counts(dec['runtime_totals'], 0)
            require(dec['denoiser_reports'][arm] == result['reports'][f'denoise_{arm}.json'], 'Decode references wrong denoiser')
            if encoding is None:
                encoding = dec['encoding_environment']
            require(dec['encoding_environment'] == encoding and encoding['historical_encoding_versions_recorded'] is False and
                    encoding['historical_media_byte_replay_claim'] is False, 'Encoding environments differ or scope overstated')
            arm_rows = []
            for case, media, (pid, seed) in zip(den['cases'], dec['cases'], PAIRS, strict=True):
                frozen = prepared[pid]
                for row in (case, media):
                    require(row['status'] == 'complete' and row['variant'] == arm and row['prompt'] == frozen['prompt'] and
                            row['embedding_sha256'] == check['inputs'][str(pid)]['embedding']['sha256'] and
                            row['initial_noise_sha256'] == {m+'_latents': check['inputs'][str(pid)]['initial_noise'][m+'_latents']['sha256']
                                                           for m in MODALITIES}, 'Rollout/media input identity mismatch')
                require([(r['step'], r['modality']) for r in case['steps']] == ORDER and
                        [r['step'] for r in case['dit_calls']] == list(range(20)), 'Rollout step/DiT order differs')
                native_counts(case['runtime_audit'], 20)
                for call in case['dit_calls']:
                    native_counts(call['counts'], 1)
                    fastpack(call['zero_sf_checks'])
                    require(all(len(call[m+'_sha256']) == 64 for m in MODALITIES), 'Missing raw DiT receipts')
                last = {m: frozen[m+'_latents'] for m in MODALITIES}
                for row in case['steps']:
                    step, modality = row['step'], row['modality']
                    payload = load(row)
                    for key in ('latents_before', 'noise_pred', 'latents_after'):
                        finite_tensor(payload[key], SHAPES[modality], torch.bfloat16)
                    require(io.byte_equal(payload['latents_before'], last[modality]), 'Initial/adjacent latent mismatch')
                    for field in ('timestep', 'sigma'):
                        finite_tensor(payload[field], (), torch.float32)
                        require(io.byte_equal(payload[field], history[pid][(step, modality)][field]), 'Historical schedule scalar differs')
                    require(float(payload['timestep']) == check['schedule_cpu'][modality][step], 'CPU schedule mismatch')
                    require(io.signature(payload['latents_after'])['sha256'] == row['latent_sha256'] and
                            io.signature(payload['noise_pred'])['sha256'] == row['velocity_sha256'], 'Step tensor SHA mismatch')
                    last[modality] = payload['latents_after']
                final_record = dict(file=case['final_latents'], sha256=case['final_latents_sha256'])
                final = load(final_record)
                require(set(final) == {'video_latents', 'audio_latents'}, 'Unexpected terminal state fields')
                for modality in MODALITIES:
                    require(io.byte_equal(final[modality+'_latents'], last[modality]), 'Terminal state differs from final step')
                    require(case[modality+'_timesteps'] == check['schedule_cpu'][modality], 'Reported schedule mismatch')
                require(media['final_latents'] == final_record and media['settings'] == settings and
                        media['encoding_environment'] == encoding, 'Decode terminal/settings binding mismatch')
                video_record = dict(file=media['video'], sha256=media['video_sha256'])
                video_path = verify(video_record)
                with av.open(str(video_path)) as container:
                    require(len(container.streams.video) == len(container.streams.audio) == 1, 'Media stream count mismatch')
                    vs, aus = container.streams.video[0], container.streams.audio[0]
                    require(vs.frames == 124 and float(vs.average_rate) == 24 and vs.width == 1024 and vs.height == 576,
                            'Declared video layout mismatch')
                    require(vs.codec_context.name == 'h264' and aus.codec_context.name == 'aac' and
                            aus.codec_context.sample_rate == 32000 and aus.codec_context.channels == 2, 'Media codecs/audio layout mismatch')
                    frames = 0
                    for frame in container.decode(video=0):
                        require((frame.width, frame.height) == (1024, 576), 'Decoded frame dimensions mismatch')
                        frames += 1
                    require(frames == 124, 'Decoded frame count mismatch')
                expected_media = dict(frames=124, width=1024, height=576, fps=24., audio_streams=1, audio_sample_rate=32000)
                require(media['media'] == expected_media and media['codecs'] == dict(video='h264', audio='aac', audio_channels=2),
                        'Reported media layout mismatch')
                pcm = load(media['audio_pcm_artifact'])
                require(media['audio_pcm_file'] == media['audio_pcm_artifact']['file'], 'PCM path differs')
                finite_tensor(pcm['waveform'], bf_pcm[pid]['shape'], torch.float32)
                require(pcm['sample_rate'] == 32000 and media['audio_samples_before_aac'] == 165600 and
                        io.signature(pcm['waveform'])['sha256'] == media['audio_pcm_sha256'], 'PCM signature/layout mismatch')
                sidecar = json.loads(video_path.with_suffix('.json').read_text())
                require(sidecar == media, 'Media sidecar differs from completed report')
                arm_rows.append(dict(prompt_id=pid, seed=seed, step_files=40, initial_and_adjacent_states_byte_exact=True,
                    schedule_matches_original_bf16=True, terminal_byte_exact=True,
                    video=result['files'][str(video_path)], media=expected_media, pcm=io.signature(pcm['waveform'])))
            result['arms'][arm] = dict(replay_byte_exact=True, cases=arm_rows,
                complete_dit_calls=41, rollout_dit_calls=40, native_calls=8200, activation_packs=8200,
                sdpa_calls=4182, scheduler_updates=80, video_vae_calls=2, audio_vae_calls=2)
        data = Path('/data1/models/svdquant-wjq/research/20261004/E068')
        total_bytes = sum(p.stat().st_size for p in data.rglob('*') if p.is_file())
        require(total_bytes <= 10*1024**3 and not torch.cuda.is_initialized(), 'Data budget or CPU-only contract violated')
        result.update(status='complete', seconds=time.monotonic()-started, cuda_initialized=False, data_bytes=total_bytes,
            checked_new_step_files=160, checked_historical_step_files=80, checked_new_videos=4, decoded_video_frames=496,
            total_model_calls_in_verified_reports=82, total_native_calls_in_verified_reports=16400,
            total_sdpa_calls_in_verified_reports=8364, full_model_replay_byte_exact=True,
            scope=['Fresh file SHA and CPU tensor identity/finite/shape checks for saved artifacts.',
                   'All saved initial, adjacent and terminal states agree; original timestep/sigma verified against E010.',
                   'All four compressed videos decoded frame by frame; PCM identity and media metadata checked.'],
            limitations=['No DiT, VAE, native kernel or scheduler arithmetic re-execution.',
                         'Unsaved raw rollout tensors are represented only by original runner hash receipts.',
                         'Selected weight identity inherits frozen preflight and E065b selection checks; no repeat full weight hash.',
                         'Historical BF16 codec byte reproduction, perceptual quality and human preferences are not established.',
                         'Private blind mapping was hashed without parsing or exposing its contents.'])
    except BaseException as exc:
        result.update(status='failed_stop', seconds=time.monotonic()-started, error=repr(exc),
                      cuda_initialized=torch.cuda.is_initialized())
        raise
    finally:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        with DEST.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'checked_new_step_files',
                                           'checked_new_videos', 'cuda_initialized')}))


if __name__ == '__main__':
    main()
