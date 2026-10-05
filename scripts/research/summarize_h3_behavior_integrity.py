#!/usr/bin/env python3
"""Independent CPU integrity audit of E013 first-seed BF16 cases only.

Does not assess behavior and imports no model/runner code. Intermediate hashes
verify the recorded chain, not fresh arithmetic: intermediate tensors were
intentionally not retained. Runtime counts are frozen observations.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E013')
FILES = {}
MODALITIES = ('video', 'audio')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verified(path, expected=None):
    path = Path(path).resolve()
    record = {'file': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
    if expected is not None:
        require(record['sha256'] == expected, f'File SHA mismatch: {path}')
    FILES[str(path)] = record
    return record


def artifact(row):
    record = verified(row['file'], row['sha256'])
    if 'bytes' in row:
        require(record['bytes'] == row['bytes'], 'Artifact size mismatch')
    return Path(record['file'])


def read_json(path):
    verified(path)
    return json.loads(Path(path).read_text())


def complete(path):
    result = read_json(path)
    require(result['status'] == 'complete', f'Incomplete report: {path}')
    return result


def tensor_record(value):
    require(value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'Tensor not finite CPU')
    data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def load_tensors(row):
    return torch.load(artifact(row), map_location='cpu', weights_only=True, mmap=True)


def sources(report):
    for path, row in report['sources'].items():
        verified(path, row['sha256'])
        if 'snapshot' in row:
            verified(row['snapshot'], row['sha256'])
    return {p: (r['sha256'], r['bytes']) for p, r in report['sources'].items()}


def binding(report, prep, check, manifest, prep_path):
    require(sources(report) == sources(prep) == sources(check), 'Source identity differs across phases')
    require(report['manifest'] == manifest and report['settings'] == manifest['settings'], 'Manifest/settings drift')
    require(report['manifest_file'] == check['manifest_file'] and report['plan'] == check['plan'], 'Plan/manifest binding drift')
    require(report['cpu_check_reference'] == prep['cpu_check_reference'], 'Different preflight reference')
    for key in ('export_manifest', 'torch', 'sdpa_enabled', 'assets', 'inherited_asset_sha_reference'):
        require(report[key] == prep[key], f'Inherited contract differs: {key}')
    if report is not prep:
        require(report['prepared_reference'] == verified(prep_path), 'Different prepare reference')


def media_check(report):
    path = artifact(report['video_file'])
    require(str(path) == report['video'], 'Video path differs')
    frames, rgb_hash, timestamps = 0, hashlib.sha256(), []
    with av.open(str(path)) as container:
        require(len(container.streams.video) == len(container.streams.audio) == 1, 'Stream counts differ')
        vs, aus = container.streams.video[0], container.streams.audio[0]
        require((vs.width, vs.height, vs.average_rate) == (1024, 576, Fraction(24)), 'Video format differs')
        require(aus.codec_context.sample_rate == 32000 and len(aus.codec_context.layout.channels) == 2, 'Audio format differs')
        for frame in container.decode(video=0):
            require((frame.width, frame.height) == (1024, 576), 'Frame size differs')
            rgb_hash.update(frame.to_ndarray(format='rgb24').tobytes())
            timestamps.append(Fraction(frame.pts) * frame.time_base)
            frames += 1
    require(frames == 124 and all(b-a == Fraction(1, 24) for a, b in zip(timestamps, timestamps[1:])), 'Frame count/timing differs')
    samples, decoded_audio_hash = 0, hashlib.sha256()
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            require(frame.sample_rate == 32000 and len(frame.layout.channels) == 2, 'Decoded audio is not 32 kHz stereo')
            array = frame.to_ndarray()
            require(bool(np.isfinite(array).all()), 'Decoded AAC nonfinite')
            decoded_audio_hash.update(array.tobytes())
            samples += frame.samples
    pcm = load_tensors(report['audio_pcm_file'])
    require(pcm['sample_rate'] == 32000 and pcm['waveform'].shape[0] == 2, 'PCM layout differs')
    require(tensor_record(pcm['waveform']) == report['audio_pcm'], 'PCM tensor hash mismatch')
    require(pcm['waveform'].shape[-1] == report['audio_samples_before_aac'], 'PCM sample count mismatch')
    actual = {'frames': frames, 'width': 1024, 'height': 576, 'fps': 24., 'audio_streams': 1, 'audio_sample_rate': 32000}
    require(report['media'] == actual, 'Actual media metadata differs from decoder report')
    sidecar = read_json(path.with_suffix('.json'))
    require(sidecar['video_sha256'] == report['video_file']['sha256'] and sidecar['video'] == str(path), 'Sidecar video binding mismatch')
    require(all(sidecar[k] == v for k, v in report['case'].items()) and sidecar['variant'] == 'bf16', 'Sidecar case differs')
    return {'video': str(path), **actual, 'audio_channels': 2, 'decoded_rgb_sha256': rgb_hash.hexdigest(),
            'decoded_aac_sha256': decoded_audio_hash.hexdigest(), 'decoded_aac_samples_per_channel': samples,
            'pcm_samples_per_channel': pcm['waveform'].shape[-1],
            'note': 'AAC sample count can include encoder padding; no media quality score.'}


def audit(args):
    rd = args.report_dir
    freeze = complete(rd/'frozen_contract.json')
    for path, sha in freeze['files'].items():
        verified(path, sha)
    manifest = read_json(ROOT/'research_state/06_experiments/E013_h3_behavior_manifest.json')
    check, prep = complete(rd/'E013_h3_check.json'), complete(rd/'E013_h3_prepare.json')
    require(check['cuda_initialized'] is False and check['settings'] == manifest['settings'], 'CPU preflight contract differs')
    require(prep['unique_text_encoder_calls'] == 2, 'TE call count differs')
    prep_path = rd/'E013_h3_prepare.json'
    binding(prep, prep, check, manifest, prep_path)
    asset_ref = complete(artifact(prep['inherited_asset_sha_reference']))
    require(asset_ref['assets'] == prep['assets'], 'Full-SHA inventory differs')
    for path, info in prep['assets'].items():
        st = Path(path).stat()
        require(st.st_size == info['bytes'] and st.st_mtime_ns == info['mtime_ns'], 'Asset changed since existing full SHA')
    require([c['prompt_id'] for c in prep['cases']] == [1, 2, 3, 4], 'Prepared four-case table differs')
    prepared, summaries = {}, []
    for case, row, cpu in zip(manifest['cases'], prep['cases'], check['cases'], strict=True):
        require(case['prompt_id'] == row['prompt_id'] == cpu['prompt_id'] and case['seed'] == row['seed'] == cpu['seed'], 'Prepared case identity differs')
        cached = load_tensors(row)
        require(all(cached[k] == v for k, v in case.items()), 'Prepared prompt/seed metadata differs')
        require(hashlib.sha256(case['prompt'].encode()).hexdigest() == case['prompt_sha256'], 'Prompt SHA differs')
        require(tensor_record(cached['embedding']) == row['embedding'] and row['embedding']['shape'] == [cpu['text_tokens'], 5120], 'TE tensor differs')
        require(tensor_record(cached['text_token_tags']) == row['tags'] == cpu['tags'] and cached['text_token_tags'].unique().tolist() == [1], 'Tags differ')
        require({k: tensor_record(cached[k]) for k in ('video_latents', 'audio_latents')} == row['initial_noise'] == cpu['initial_noise'], 'Prepared noise differs')
        prepared[case['prompt_id']] = row
    for a, b in ((1, 2), (3, 4)):
        require(prepared[a]['initial_noise'] == prepared[b]['initial_noise'], 'Same-seed directions did not share noise')
    for a, b in ((1, 3), (2, 4)):
        require(prepared[a]['embedding'] == prepared[b]['embedding'] and prepared[a]['tags'] == prepared[b]['tags'], 'Same prompt embedding differs across seeds')
    for cid in (1, 2):
        denoise_path = rd/f'p{cid:03d}'/'E013_h3_denoise_bf16.json'
        decode_path = rd/f'p{cid:03d}'/'E013_h3_decode_bf16.json'
        d, media = complete(denoise_path), complete(decode_path)
        for stage in (d, media):
            binding(stage, prep, check, manifest, prep_path)
            require(stage['case'] == manifest['cases'][cid-1] and stage['arm'] == 'bf16', 'Per-case identity differs')
        require(d['prepared_case'] == prepared[cid], 'Consumed prepared input differs')
        require(media['denoiser_reference'] == verified(denoise_path), 'Decoder consumed different denoiser output')
        require(len(d['dit_calls']) == 50 and [c['step'] for c in d['dit_calls']] == list(range(50)), 'DiT call grid differs')
        per_call = {'sdpa_calls': 102, 'scaled_mm_calls': 0, 'disk_loads': 0}
        for call in d['dit_calls']:
            require(call['counts'] == per_call and call['zero_sf_checks'] is None, 'BF16 execution path differs')
            for m, shape in (('video', [21312, 96]), ('audio', [414, 32])):
                require(call[m]['shape'] == shape and call[m]['dtype'] == 'torch.bfloat16'
                        and re.fullmatch('[a-f0-9]{64}', call[m]['sha256']), 'Raw DiT record layout/hash invalid')
        require(all(d['runtime_audit'][k] == v*50 for k, v in per_call.items())
                and d['runtime_audit']['qkv_dtypes'] == [['torch.bfloat16']*3], 'Aggregate runtime observations differ')
        require(len(d['steps']) == 100, 'Scheduler call count differs')
        steps = {(r['step'], r['modality']): r for r in d['steps']}
        require(set(steps) == {(s, m) for s in range(50) for m in MODALITIES}, 'Scheduler grid duplicate/missing')
        finals = load_tensors(d['final_latents'])
        final_records = {k: tensor_record(v) for k, v in finals.items()}
        require(final_records == d['final_latent_tensors'], 'Final tensor hashes differ')
        require(d['schedule_cpu'] == check['schedule_cpu'], 'Recorded CPU schedule differs')
        for modality, shift in (('video', 12.), ('audio', 3.)):
            base = torch.linspace(1., 0., 51, dtype=torch.float32)[:-1]
            sigmas = shift*base/(1+(shift-1)*base)
            times = sigmas*1000
            require(times.tolist() == check['schedule_cpu'][modality], 'Independently rebuilt schedule differs')
            previous = prepared[cid]['initial_noise'][modality+'_latents']
            for index in range(50):
                row = steps[index, modality]
                require(row['latents_before'] == previous, 'Recorded latent SHA chain broken')
                for key in ('latents_after', 'noise_pred'):
                    value = row[key]
                    require(value['shape'] == previous['shape'] and value['dtype'] == previous['dtype']
                            and re.fullmatch('[a-f0-9]{64}', value['sha256']), 'Scheduler tensor record malformed')
                require(row['timestep'] == float(times[index]) and row['sigma'] == float(sigmas[index]), 'Per-step schedule scalar differs')
                # Runner records null at the implicit final sigma=0 boundary.
                require(row['next_sigma'] == (float(sigmas[index+1]) if index < 49 else None), 'Next-sigma record differs')
                previous = row['latents_after']
            require(previous == final_records[modality+'_latents'], 'Final tensor differs from terminal SHA')
        summaries.append({'prompt_id': cid, 'seed': d['case']['seed'], 'direction': d['case']['direction'],
            'dit_calls': 50, 'scheduler_calls': 100, 'runtime_audit': d['runtime_audit'],
            'initial_noise': prepared[cid]['initial_noise'], 'final_latents': d['final_latents'],
            'final_tensors': final_records, 'media': media_check(media)})
    missing = []
    for cid in (1, 2, 3, 4):
        for arm in ('bf16', 'native'):
            if arm == 'bf16' and cid in (1, 2):
                continue
            for phase in ('denoise', 'decode'):
                for path in (rd/f'p{cid:03d}'/f'E013_h3_{phase}_{arm}.json', DATA/f'p{cid:03d}'/f'{phase}_{arm}'):
                    require(not path.exists(), f'Unexpected later-stage artifact: {path}; use a new summary version')
                    missing.append(str(path))
    launcher = complete(rd/'launcher_teacher1.json')
    require(len(launcher['stages']) == 4 and {(r['phase'], r['case'], r['arm']) for r in launcher['stages']}
            == {(p, c, 'bf16') for p in ('denoise', 'decode') for c in (1, 2)}, 'Launcher stage table differs')
    for row in launcher['stages']:
        verified(row['report'], row['report_sha256'])
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return {'experiment': 'E013', 'status': 'complete', 'stage': 'first_seed_bf16_only',
        'scope': 'Independent CPU contract and media integrity; no behavior verdict', 'cuda_initialized': False,
        'prepared_cases': 4, 'unique_text_encoder_calls': 2, 'cases': summaries,
        'totals': {'dit_calls': 100, 'scheduler_calls': 200, 'bf16_sdpa_calls': 10200, 'fp4_gemm_calls': 0, 'disk_loads': 0},
        'later_stage_absence_verified': missing,
        'limits': ['Step tensor bytes were not saved: this validates recorded SHA linkage and persisted initial/final bytes, not independent re-execution of Euler updates.',
                   'Raw DiT versus unpacked negative velocity SHA cannot be independently inverted from hashes alone.',
                   'Kernel counts and finite checks are frozen runner observations; no GPU is executed by this audit.',
                   'Existing full model hashes are reused with current size/mtime verification.'],
        'source': verified(__file__), 'verified_files': FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/research/E013')
    args = parser.parse_args()
    output = args.report_dir/'integrity_summary.json'
    require(not output.exists(), 'Do not overwrite a completed integrity summary')
    torch.set_num_threads(6)
    result = audit(args)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'stage': result['stage'], 'totals': result['totals'], 'output': str(output)}))


if __name__ == '__main__':
    main()
