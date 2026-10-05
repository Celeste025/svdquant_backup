#!/usr/bin/env python3
"""Independent CPU audit of E010 paired free trajectories and optional media.

Never imports or calls the generation runner, model loader, or CUDA kernels.
NMSE is trajectory disagreement relative to BF16, not local quantization error
or a perceptual/quality metric. Existing complete outputs are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E010')
MANIFEST = ROOT/'research_state/06_experiments/E010_h3_heldout_manifest.json'
PTQ_PYTHON = Path('/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python')
ARMS, MODALITIES = ('bf16', 'native'), ('video', 'audio')
FRAME_INDICES = (0, 25, 49, 74, 98, 123)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(tensor):
    if tensor.device.type != 'cpu':
        raise RuntimeError('CPU tensor required')
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_tensor_file(path, expected_sha):
    require(digest(path) == expected_sha, f'File SHA mismatch: {path}')
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def finite(tensor):
    require(tensor.device.type == 'cpu' and bool(torch.isfinite(tensor).all()), 'Nonfinite/non-CPU tensor')


def nmse(got, reference):
    require(got.shape == reference.shape and got.dtype == reference.dtype, 'Paired tensor layout mismatch')
    finite(got)
    finite(reference)
    a, b = got.reshape(-1), reference.reshape(-1)
    error = energy = maximum = 0.
    for begin in range(0, a.numel(), 1 << 20):
        x, y = a[begin:begin+(1 << 20)].double(), b[begin:begin+(1 << 20)].double()
        difference = x-y
        error += difference.square().sum().item()
        energy += y.square().sum().item()
        maximum = max(maximum, difference.abs().max().item())
    return {'nmse': error/energy if energy else (0. if error == 0 else None),
            'squared_error': error, 'bf16_reference_energy': energy,
            'max_abs_difference': maximum, 'numel': a.numel()}


def raw_dit_rows(noise_pred, modality):
    """Invert model_fn's unpack + negation for unconditional T2VA (no refs).

    The frozen pipeline returns -unpatchify(v_video_rows), -unpack(v_audio_rows).
    We independently reshape/permute -noise_pred back into those raw rows.
    """
    value = -noise_pred
    if modality == 'audio':
        channels, dim, steps = value.shape
        require((channels, dim) == (2, 32), 'Unexpected audio latent contract')
        return value.permute(0, 2, 1).reshape(channels*steps, dim).contiguous()
    batch, channels, frames, height, width = value.shape
    require(batch == 1 and channels == 24 and height % 2 == width % 2 == 0,
            'Unexpected video patch layout')
    return value.reshape(batch, channels, frames, height//2, 2, width//2, 2).permute(
        0, 2, 3, 5, 1, 4, 6).reshape(batch*frames*(height//2)*(width//2), channels*4).contiguous()


def source_identity(sources):
    return {path: (value['sha256'], value['bytes']) for path, value in sources.items()}


def verify_report_binding(report, prep, prep_path, manifest):
    require(report['status'] == 'complete', 'Incomplete report')
    require(report['settings'] == manifest['settings'], 'Generation settings drift')
    require(source_identity(report['sources']) == source_identity(prep['sources']), 'Source mismatch across phases')
    for path, row in report['sources'].items():
        require(digest(path) == row['sha256'], f'Current source drift: {path}')
        require(digest(row['snapshot']) == row['sha256'], f'Source snapshot drift: {path}')
    if report is not prep:
        require(report['prepared_reference']['file'] == str(prep_path), 'Prepare reference path mismatch')
        require(report['prepared_reference']['sha256'] == digest(prep_path), 'Prepare report binding drift')
    require(report['export_manifest'] == prep['export_manifest'], 'Export manifest mismatch')
    require(digest(report['export_manifest']['file']) == report['export_manifest']['sha256'], 'Export SHA drift')
    require(report['torch'] == prep['torch'] and report['sdpa_enabled'] == prep['sdpa_enabled'], 'Backend drift')


def verify_case(record, cached, cached_info, arm, check_schedule):
    require(record['status'] == 'complete' and record['variant'] == arm, 'Incomplete/wrong arm case')
    require(record['embedding_sha256'] == cached_info['embedding_sha256'], 'Embedding was not shared')
    require(record['initial_noise_sha256'] == cached_info['initial_noise_sha256'], 'Initial noise not shared')
    calls = record['dit_calls']
    require(len(calls) == 20 and [v['step'] for v in calls] == list(range(20)), 'Expected 20 distinct DiT calls')
    expected = {'sdpa_calls': 102, 'scaled_mm_calls': 200 if arm == 'native' else 0, 'disk_loads': 0}
    affected = []
    for row in calls:
        require(row['counts'] == expected, 'Actual kernel/disk counts violate contract')
        checks = row['zero_sf_checks']
        if arm == 'native':
            require(checks['checked_calls'] == 200 and checks['invalid_calls'] == 0, 'Invalid/incomplete fastpack checks')
            indices = checks['affected_call_indices_zero_based']
            require(len(indices) == len(set(indices)) == checks['calls_with_nonzero_input_zero_sf']
                    and all(0 <= i < 200 for i in indices), 'Invalid zero-SF index bookkeeping')
            affected.append({'step': row['step'], 'count': len(indices), 'indices': indices})
        else:
            require(checks is None, 'Unexpected BF16 fastpack checks')
    audit = record['runtime_audit']
    require(all(audit[key] == value*20 for key, value in expected.items()), 'Aggregate runtime counts differ')
    require(audit['qkv_dtypes'] == [['torch.bfloat16']*3], 'Attention was not BF16 QKV')
    require(len(record['steps']) == 40, 'Expected 40 scheduler records')
    steps = {(row['step'], row['modality']): row for row in record['steps']}
    require(set(steps) == {(i, m) for i in range(20) for m in MODALITIES}, 'Duplicate/missing scheduler record')
    checked_steps = []
    last_hashes = dict(cached_info['initial_noise_sha256'])
    finals = load_tensor_file(record['final_latents'], record['final_latents_sha256'])
    final_hashes = {}
    for index in range(20):
        for modality in MODALITIES:
            row = steps[index, modality]
            payload = load_tensor_file(row['file'], row['sha256'])
            require(set(payload) == {'latents_before', 'noise_pred', 'latents_after', 'timestep', 'sigma'},
                    'Unexpected scheduler artifact')
            for tensor in payload.values():
                finite(tensor)
            key = modality+'_latents'
            before_sha = tensor_digest(payload['latents_before'])
            after_sha = tensor_digest(payload['latents_after'])
            velocity_sha = tensor_digest(payload['noise_pred'])
            require(before_sha == last_hashes[key], 'Trajectory chain discontinuity')
            require(after_sha == row['latent_sha256'], 'Per-step updated latent SHA mismatch')
            require(tensor_digest(raw_dit_rows(payload['noise_pred'], modality)) == calls[index][modality+'_sha256'],
                    'pack(-noise_pred) differs from raw DiT output SHA')
            require(float(payload['timestep']) == record[modality+'_timesteps'][index]
                    == check_schedule[modality][index], 'Sampling schedule drift')
            require(0 <= float(payload['sigma']) <= 1, 'Invalid sigma')
            last_hashes[key] = after_sha
            checked_steps.append({**row, 'before_sha256': before_sha, 'velocity_sha256': velocity_sha,
                                  'raw_dit_output_sha_verified_after_inverse_pack_and_negation': True,
                                  'timestep': float(payload['timestep']), 'sigma': float(payload['sigma']),
                                  'shape': list(payload['latents_after'].shape), 'finite': True})
            if index == 19:
                finite(finals[key])
                final_hashes[key] = tensor_digest(finals[key])
                require(final_hashes[key] == after_sha, 'Final latent differs from last recurrent state')
    return {'steps': checked_steps, 'final_latents_file': record['final_latents'],
            'final_latents_file_sha256': record['final_latents_sha256'], 'final_tensor_sha256': final_hashes,
            'runtime_audit': audit, 'all_20_calls_exact_counts': True,
            'all_40_steps_file_sha_and_tensor_chain_verified': True, 'zero_sf_affected_calls_by_step': affected}


def media_audit(row):
    import av
    import numpy as np
    from PIL import Image
    path = Path(row['video'])
    require(digest(path) == row['video_sha256'], 'Encoded video SHA drift')
    frames, selected = [], {}
    with av.open(str(path)) as container:
        require(len(container.streams.video) == len(container.streams.audio) == 1, 'Expected one A/V stream')
        stream = container.streams.video[0]
        require(float(stream.average_rate) == 24. and (stream.width, stream.height) == (1024, 576), 'Video metadata mismatch')
        for index, frame in enumerate(container.decode(video=0)):
            rgb = frame.to_ndarray(format='rgb24')
            require(rgb.shape == (576, 1024, 3), 'Decoded RGB shape changed')
            frames.append({'index': index, 'pts': frame.pts, 'time_seconds': float(frame.time),
                           'rgb24_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
            if index in FRAME_INDICES:
                selected[index] = Image.fromarray(rgb)
    require(len(frames) == 124 and set(selected) == set(FRAME_INDICES), 'Missing decoded frames')
    samples = audio_frames = 0
    audio_hash = hashlib.sha256()
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        require(stream.codec_context.sample_rate == 32000, 'Encoded audio rate mismatch')
        for frame in container.decode(audio=0):
            require(len(frame.layout.channels) == 2, 'Encoded audio not stereo')
            array = frame.to_ndarray()
            require(bool(np.isfinite(array).all()), 'Nonfinite decoded AAC waveform')
            audio_hash.update(array.tobytes())
            samples += frame.samples
            audio_frames += 1
    pcm = torch.load(row['audio_pcm_file'], map_location='cpu', weights_only=True, mmap=True)
    finite(pcm['waveform'])
    require(pcm['sample_rate'] == 32000 and pcm['waveform'].shape[0] == 2, 'PCM format mismatch')
    require(pcm['waveform'].shape[-1] == row['audio_samples_before_aac'], 'PCM length mismatch')
    require(tensor_digest(pcm['waveform']) == row['audio_pcm_sha256'], 'Pre-AAC PCM SHA drift')
    return {'video_file': str(path), 'video_sha256': row['video_sha256'], 'decoded_video_frames': frames,
            'decoded_frame_count': len(frames), 'selected_indices': list(FRAME_INDICES),
            'decoded_audio_frames': audio_frames, 'decoded_audio_samples_per_channel': samples,
            'decoded_audio_sha256': audio_hash.hexdigest(), 'audio_sample_rate': 32000,
            'audio_pcm_file': row['audio_pcm_file'], 'audio_pcm_file_sha256': digest(row['audio_pcm_file']),
            'audio_pcm_tensor_sha256': row['audio_pcm_sha256'], 'pre_aac_samples_per_channel': row['audio_samples_before_aac'],
            'note': 'AAC decoded samples may include encoder padding; no audio quality metric computed.'}, selected


def contact_sheet(images, prompt_id, output):
    from PIL import Image, ImageDraw
    tile_width, tile_height = 320, 180
    canvas = Image.new('RGB', (6*tile_width, 2*(tile_height+28)+36), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f'E010 prompt {prompt_id}: paired free trajectories; six fixed frame indices', fill='black')
    for ri, arm in enumerate(ARMS):
        top = 36+ri*(tile_height+28)
        for ci, index in enumerate(FRAME_INDICES):
            canvas.paste(images[arm][index].resize((tile_width, tile_height), Image.Resampling.LANCZOS), (ci*tile_width, top+24))
            draw.text((ci*tile_width+5, top+5), f'{arm}  frame {index}  {index/24:.2f}s', fill='black')
    canvas.save(output)


def render_plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for row, modality in enumerate(MODALITIES):
        for case in data['cases']:
            points = case['trajectory_comparison'][modality]
            for col, kind in enumerate(('latents_after', 'velocity')):
                axes[row, col].plot([v['step']+1 for v in points], [v[kind]['nmse'] for v in points],
                                    marker='.', label=f"p{case['prompt_id']}")
                axes[row, col].set_title(f'{modality}: {kind.replace("_", " ")}')
                axes[row, col].set_ylabel('NMSE relative to BF16 trajectory')
                axes[row, col].grid(alpha=.25)
                axes[row, col].legend()
    for axis in axes[-1]:
        axis.set_xlabel('Denoising step (1–20)')
    fig.suptitle('E010 independent free trajectories — disagreement, not a quality score')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def execute(args, output):
    report_paths = {phase: args.report_dir/f'E010_h3_{phase}.json'
                    for phase in ('prepare', 'check', 'denoise_bf16', 'denoise_native', 'decode')}
    reports = {name: json.loads(path.read_text()) if path.exists() else None for name, path in report_paths.items()}
    required = ['prepare', 'check', 'denoise_bf16', 'denoise_native'] + (['decode'] if args.require_media else [])
    missing = {name: 'missing' if reports[name] is None else reports[name]['status'] for name in required
               if reports[name] is None or reports[name]['status'] != 'complete'}
    if missing:
        output.update(status='pending', pending_prerequisites=missing)
        return
    manifest = json.loads(MANIFEST.read_text())
    output['manifest'] = {'path': str(MANIFEST), 'sha256': digest(MANIFEST)}
    require([r['prompt_id'] for r in manifest['cases']] == [30, 36], 'Unexpected heldout cases')
    source = Path(manifest['source_prompt_file'])
    require(digest(source) == manifest['source_prompt_file_sha256'], 'Original prompt source SHA changed')
    original = {int(v['prompt_id']): v for v in map(json.loads, source.open())}
    prep = reports['prepare']
    for phase in required:
        report = reports[phase]
        if phase != 'check':
            verify_report_binding(report, prep, report_paths['prepare'], manifest)
        output['reports'][phase] = {'path': str(report_paths[phase]), 'sha256': digest(report_paths[phase])}
    require([r['prompt_id'] for r in prep['cases']] == [30, 36], 'Prepare case order mismatch')
    for index, expected_case in enumerate(manifest['cases']):
        pid = expected_case['prompt_id']
        require(pid not in manifest['excluded_calibration_prompt_ids'], 'PTQ overlap')
        require(original[pid]['prompt'] == expected_case['prompt'] and int(original[pid]['seed']) == expected_case['seed'],
                'Prompt/seed mismatch')
        require(hashlib.sha256(expected_case['prompt'].encode()).hexdigest() == expected_case['prompt_sha256'], 'Prompt text SHA mismatch')
        prepared = prep['cases'][index]
        cached = load_tensor_file(prepared['file'], prepared['sha256'])
        require(cached['prompt'] == expected_case['prompt'] and cached['seed'] == expected_case['seed'], 'Cached provenance mismatch')
        finite(cached['embedding'])
        require(tensor_digest(cached['embedding']) == prepared['embedding_sha256'], 'Cached embedding SHA mismatch')
        require(cached['text_token_tags'].unique().tolist() == [1], 'Unexpected text tags')
        for modality in MODALITIES:
            key = modality+'_latents'
            finite(cached[key])
            require(tensor_digest(cached[key]) == prepared['initial_noise_sha256'][key], 'Initial noise payload SHA mismatch')
            require(prepared['initial_noise_sha256'][key] == reports['check']['cases'][index]['initial_noise'][key]['sha256'],
                    'Noise differs from independent CPU preflight')
        case = {'prompt_id': pid, 'seed': expected_case['seed'], 'prompt_sha256': expected_case['prompt_sha256'],
                'shared_embedding_sha256': prepared['embedding_sha256'], 'shared_noise_sha256': prepared['initial_noise_sha256'],
                'prepare_file': {'path': prepared['file'], 'sha256': prepared['sha256']}, 'arms': {}, 'trajectory_comparison': {}}
        for arm in ARMS:
            record = reports['denoise_'+arm]['cases'][index]
            require(record['prompt_id'] == pid and record['seed'] == expected_case['seed']
                    and record['prompt'] == expected_case['prompt'], 'Denoise case mismatch')
            case['arms'][arm] = verify_case(record, cached, prepared, arm, reports['check']['schedule_cpu'])
        for modality in MODALITIES:
            points = []
            for step in range(20):
                pair = []
                for arm in ARMS:
                    row = next(r for r in case['arms'][arm]['steps'] if r['step'] == step and r['modality'] == modality)
                    pair.append(torch.load(row['file'], map_location='cpu', weights_only=True, mmap=True))
                ref, got = pair
                require(torch.equal(ref['timestep'], got['timestep']) and torch.equal(ref['sigma'], got['sigma']), 'Paired schedule differs')
                points.append({'step': step, 'timestep': float(ref['timestep']), 'sigma': float(ref['sigma']),
                               'latents_before': nmse(got['latents_before'], ref['latents_before']),
                               'velocity': nmse(got['noise_pred'], ref['noise_pred']),
                               'latents_after': nmse(got['latents_after'], ref['latents_after'])})
            case['trajectory_comparison'][modality] = points
        output['cases'].append(case)
        print(f'CPU trajectory audit complete: p{pid}', flush=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    media = reports['decode']
    if media is not None and media['status'] == 'complete':
        verify_report_binding(media, prep, report_paths['prepare'], manifest)
        require(len(media['cases']) == 4, 'Expected four media records')
        output['reports']['decode'] = {'path': str(report_paths['decode']), 'sha256': digest(report_paths['decode'])}
        for arm in ARMS:
            require(media['denoiser_reports'][arm]['sha256'] == digest(report_paths['denoise_'+arm]), 'Decoder denoiser binding mismatch')
        rows = {(row['prompt_id'], row['variant']): row for row in media['cases']}
        require(set(rows) == {(pid, arm) for pid in (30, 36) for arm in ARMS}, 'Duplicate/missing decoded arm')
        for case in output['cases']:
            images, case['media'] = {}, {}
            for arm in ARMS:
                case['media'][arm], images[arm] = media_audit(rows[case['prompt_id'], arm])
            path = args.artifact_dir/f"E010_p{case['prompt_id']:03d}_paired_contact.png"
            require(not path.exists(), f'Refusing to overwrite {path}')
            contact_sheet(images, case['prompt_id'], path)
            case['contact_sheet'] = {'path': str(path), 'sha256': digest(path)}
        output['media_status'] = 'complete_independent_pyav_audit'
    else:
        output['media_status'] = 'pending_not_audited'
    plot_input = args.artifact_dir/(args.output.stem+'.plot_input.json')
    require(not plot_input.exists(), f'Refusing to overwrite {plot_input}')
    plot_input.write_text(json.dumps({'cases': output['cases']}))
    plot_path = args.artifact_dir/(args.output.stem+'.trajectories.png')
    require(not plot_path.exists(), f'Refusing to overwrite {plot_path}')
    subprocess.run([str(PTQ_PYTHON), str(Path(__file__).resolve()), '--plot-data', str(plot_input),
                    '--plot-output', str(plot_path)], check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    output['trajectory_figure'] = {'path': str(plot_path), 'sha256': digest(plot_path),
                                  'data_path': str(plot_input), 'data_sha256': digest(plot_input)}
    require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly initialized CUDA')
    output.update(status='complete', cuda_initialized=False,
                  completed_scope='trajectory_and_media' if output['media_status'].startswith('complete') else 'trajectory_only_media_pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/research')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E010_summary.json')
    parser.add_argument('--artifact-dir', type=Path, default=DATA/'independent_summary')
    parser.add_argument('--require-media', action='store_true')
    parser.add_argument('--plot-data', type=Path)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.plot_data:
        render_plot(json.loads(args.plot_data.read_text()), args.plot_output)
        return
    if args.output.exists():
        require(json.loads(args.output.read_text()).get('status') == 'pending', f'Preserve existing output: {args.output}')
    output = {'experiment': 'E010_independent_cpu_audit', 'status': 'running',
              'source': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__))},
              'metric_definition': 'sum((native-BF16)^2)/sum(BF16^2), FP64 CPU reductions, separately for video/audio, latent-before/velocity/latent-after. Independent free trajectories from identical initial noise and text embedding.',
              'limitations': ['After step0, arms evaluate different recurrent inputs. Velocity NMSE is not local one-layer/one-call quantization error.',
                              'Two fixed PTQ-heldout prompts, not a quality benchmark or proof of perceptual/audio equivalence.',
                              'Generation wall times include instrumentation and persistence; this script makes no speed claim.',
                              'Actual kernel and disk counts are independently checked for consistency with frozen runtime audit records; no new GPU observation is performed.',
                              'Pre-AAC PCM and decoded AAC hashes have different meanings; no subjective audio quality score is inferred.'],
              'reports': {}, 'cases': []}
    try:
        execute(args, output)
    except BaseException:
        output.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n')
        temporary.replace(args.output)
        print(json.dumps({'status': output['status'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
