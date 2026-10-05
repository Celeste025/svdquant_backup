#!/usr/bin/env python3
"""E015 CPU-only exact H3 scheduler replay and four-corner state construction.

Model inputs are ONLY embedding, text_token_tags, and state. The separate
teacher_reference is the historical next-step BF16 record, never a model input
or the sample to use when integrating a perturbed corner's velocity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_h3_plain_baseline as base
import torch

ROOT = HERE.parents[1]
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E015')
REPORTS = ROOT/'results/research/E015'
MANIFEST = ROOT/'research_state/06_experiments/E015_h3_crossmodal_propagation_manifest.json'
MANIFEST_SHA = '54b543d679393150751f2600ceadf4e7286165c2b54cbc746e65ea7338c3344e'
MODALITIES = ('video', 'audio')
ARMS = ('bf16', 'svd', 'plain')
CORNERS = ('BB', 'QB', 'BQ', 'QQ')
CASE_IDS = ('e010_p030_s05', 'e010_p036_s14')
STATE_KEYS = {'latents_before', 'timestep', 'sigma'}
SCHEMA = 'E015_h3_crossmodal_state_v1'

file_record, tensor_record, tree_signature = base.file_record, base.tensor_record, base.tree_signature
sha256, save = base.sha256, base.save


def configure_schedule(pipe, settings):
    """Initialize the unchanged E010 scheduler on an empty CPU or resident pipe."""
    if (settings['num_inference_steps'] != 20 or settings['cfg_scale'] != 1.
            or settings['video_shift'] != 12. or settings['audio_shift'] != 3.):
        raise RuntimeError('E015 only uses the original 20-step / CFG1 / shifts12,3 contract')
    pipe.scheduler.set_timesteps(settings['num_inference_steps'], shift=settings['video_shift'])
    pipe.scheduler_audio.set_timesteps(settings['num_inference_steps'], shift=settings['audio_shift'])
    return pipe


def make_cpu_pipeline(settings):
    pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    configure_schedule(pipe, settings)
    if any(getattr(pipe, name) is not None for name in ('dit', 'text_encoder', 'video_vae', 'audio_vae', 'controlnet')):
        raise RuntimeError('CPU state construction must not load model components')
    return pipe


def assert_bytes_equal(a, b, label):
    if tensor_record(a) != tensor_record(b):
        raise RuntimeError(f'Byte-exact replay failed: {label}')


def assert_clock(scheduler, state, step):
    if not 0 <= step < len(scheduler.timesteps):
        raise ValueError('Step outside the original schedule')
    for key, expected in (('timestep', scheduler.timesteps[step]), ('sigma', scheduler.sigmas[step])):
        value = state[key]
        if value.ndim != 0 or value.dtype != torch.float32:
            raise RuntimeError(f'{key} must remain original 0D FP32')
        assert_bytes_equal(value, expected, f'step{step} {key}')


@torch.inference_mode()
def advance(pipe, state, velocities, step):
    """Return both next latents using each modality's OWN sample and scheduler.

    No CPU/GPU transfer is imposed on the latents or velocities. In particular,
    this calls pipe.step verbatim rather than z_B + h*(v_Q-v_B), addcmul, or a
    higher-precision approximation. It never reads teacher_reference.
    """
    if set(state) != set(MODALITIES) or set(velocities) != set(MODALITIES):
        raise ValueError('Expected video/audio state and velocity pair')
    result = {}
    for modality, scheduler in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio)):
        row, velocity = state[modality], velocities[modality]
        sample = row['latents_before']
        assert_clock(scheduler, row, step)
        if (sample.dtype != torch.bfloat16 or velocity.dtype != torch.bfloat16
                or sample.shape != velocity.shape or sample.device != velocity.device):
            raise RuntimeError(f'{modality}: original BF16 sample/velocity shape/device contract changed')
        if not bool(torch.isfinite(sample).all()) or not bool(torch.isfinite(velocity).all()):
            raise RuntimeError('Nonfinite scheduler operand')
        updated = pipe.step(scheduler, sample, step, velocity)
        if updated.dtype != torch.bfloat16 or updated.shape != sample.shape or not bool(torch.isfinite(updated).all()):
            raise RuntimeError('Invalid original scheduler result')
        result[modality] = updated
    return result


def model_value(payload):
    """Select only the E014-compatible actual model inputs from an E015 payload."""
    return {name: payload[name] for name in ('embedding', 'text_token_tags', 'state')}


def load_prepared_case(row):
    path = base.verify_file(row['artifact'])
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload['schema'] != SCHEMA:
        raise RuntimeError('Unsupported E015 payload schema')
    for key in ('id', 'arm', 'corner', 'step', 'next_step'):
        if payload['metadata'][key] != row[key]:
            raise RuntimeError(f'Prepared artifact identity mismatch: {key}')
    if tree_signature(model_value(payload)) != row['input_signature']:
        raise RuntimeError('Prepared tensor input signature changed')
    return payload


def read_record(row, *, complete=False):
    path = base.verify_file(row)
    value = json.loads(path.read_text())
    if complete and value.get('status') != 'complete':
        raise RuntimeError(f'Incomplete input report: {path}')
    return value


def load_tensor_file(row, *, historical_sample=False):
    path = base.verify_file(row)
    return torch.load(path, map_location='cpu', weights_only=not historical_sample, mmap=True)


def prerequisites(args, report):
    if sha256(args.manifest) != MANIFEST_SHA:
        raise RuntimeError('E015 manifest differs from the preregistered revision')
    manifest = json.loads(args.manifest.read_text())
    if (tuple(c['id'] for c in manifest['cases']) != CASE_IDS or manifest['arms'] != list(ARMS)
            or manifest['corners'] != list(CORNERS) or manifest['corner_coordinate_order'] != list(MODALITIES)):
        raise RuntimeError('Fixed case/arm/corner contract changed')
    base.verify_file(manifest['plan'])
    e014 = read_record(manifest['e014_manifest'])
    checked = read_record(manifest['e014_check'], complete=True)
    read_record(manifest['e014_independent_summary'], complete=True)
    if checked.get('cuda_initialized') is not False:
        raise RuntimeError('Original E014 CPU check is not CPU-only')
    base.check_sources(checked['sources'])
    evaluations = {arm: read_record(manifest['e014_evaluations'][arm], complete=True) for arm in ARMS}
    for arm, evaluation in evaluations.items():
        if evaluation['arm'] != arm or evaluation['sources'] != checked['sources']:
            raise RuntimeError('E014 arm/source identity differs from completed CPU check')
        base.check_sources(evaluation['sources'])
    for path, row in checked['asset_binding']['assets'].items():
        st = Path(path).stat()
        if st.st_size != row['bytes'] or st.st_mtime_ns != row['mtime_ns']:
            raise RuntimeError(f'Original fully hashed asset changed: {path}')
    paths = {Path(__file__), args.manifest, Path(manifest['plan']['file'])}
    sources = dict(checked['sources'])
    sources.update({str(p.resolve()): file_record(p) for p in paths})
    report.update(sources=sources, manifest=file_record(args.manifest), settings=manifest['settings'],
        original_e014_sources=checked['sources'], e014_evaluations=manifest['e014_evaluations'],
        e014_independent_summary=manifest['e014_independent_summary'], cases=[], construction=[])
    for case in manifest['cases']:
        old = next(c for c in e014['cases'] if c['id'] == case['id'])
        if (case['source_step'] != old['step'] or case['next_step'] != case['source_step']+1
                or case['prepared'] != old['prepared'] or case['teacher_state'] != old['state']
                or case['bf16_trajectory_reference'] != old['reference']):
            raise RuntimeError('E015 teacher source differs from the frozen E014 case')
        trajectory_report = read_record(case['bf16_trajectory_reference'], complete=True)
        trajectory = next(r for r in trajectory_report['cases'] if r['prompt_id'] == case['prompt_id'])
        for step, field in ((case['source_step'], 'teacher_state'), (case['next_step'], 'teacher_next_state')):
            for modality, row in case[field].items():
                actual = next(r for r in trajectory['steps'] if r['step'] == step and r['modality'] == modality)
                if row['file'] != actual['file'] or row['sha256'] != actual['sha256']:
                    raise RuntimeError('Historical trajectory row binding failed')
        call = next(r for r in trajectory['dit_calls'] if r['step'] == case['next_step'])
        if case['teacher_next_raw_output_sha256'] != {m: call[m+'_sha256'] for m in MODALITIES}:
            raise RuntimeError('Next-step raw BF16 reference changed')
        for arm in ARMS:
            eval_row = next(r for r in evaluations[arm]['cases'] if r['id'] == case['id'])
            if eval_row['artifact'] != case['e014_outputs'][arm]:
                raise RuntimeError('E014 velocity artifact is not the completed evaluation artifact')
    return manifest, checked, evaluations


def validate_reference_state(state):
    if set(state) != set(MODALITIES):
        raise RuntimeError('Missing modality')
    for modality, row in state.items():
        if set(row) != {'latents_before', 'noise_pred', 'latents_after', 'timestep', 'sigma'}:
            raise RuntimeError(f'Historical state keys changed: {modality}')
        for name in ('latents_before', 'noise_pred', 'latents_after'):
            value = row[name]
            if value.dtype != torch.bfloat16 or value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
                raise RuntimeError('Expected finite original CPU BF16 latent/velocity')
        if row['latents_before'].shape != row['noise_pred'].shape or row['latents_after'].shape != row['noise_pred'].shape:
            raise RuntimeError('Historical latent/velocity shapes differ')


@torch.inference_mode()
def construct_case(case, settings, checked, evaluations):
    """Build one case in CPU memory, with no saved artifacts or model forwards."""
    pipe = make_cpu_pipeline(settings)
    prepared = load_tensor_file(case['prepared'])
    embedding, tags = prepared['embedding'], prepared['text_token_tags']
    if (embedding.ndim != 2 or embedding.shape[1] != 5120 or embedding.dtype != torch.bfloat16
            or tags.ndim != 1 or tags.numel() != embedding.shape[0] or not bool(torch.isfinite(embedding).all())):
        raise RuntimeError('Real 2D embedding/tag contract failed')
    state = {m: load_tensor_file(case['teacher_state'][m]) for m in MODALITIES}
    next_state = {m: load_tensor_file(case['teacher_next_state'][m]) for m in MODALITIES}
    validate_reference_state(state)
    validate_reference_state(next_state)
    original = {'embedding': embedding, 'text_token_tags': tags, 'state': state}
    old_signature = base.validate_case({'kind': 'model_fn'}, original, pipe)
    if old_signature != checked['inputs'][case['id']]:
        raise RuntimeError('Actual source state/embedding differs from E014 CPU check')
    source_outputs, updates = {}, {}
    for arm in ARMS:
        payload = load_tensor_file(case['e014_outputs'][arm])
        row = next(r for r in evaluations[arm]['cases'] if r['id'] == case['id'])
        if (payload['case_id'] != case['id'] or payload['arm'] != arm
                or payload['input_signature'] != old_signature
                or payload['actual_dit_inputs'] != row['actual_dit_inputs']):
            raise RuntimeError('E014 payload identity/common-input binding failed')
        if payload['velocities'] is None or set(payload['velocities']) != set(MODALITIES):
            raise RuntimeError('E015 requires the scheduler-layout velocities, not raw DiT outputs')
        if ({m: tensor_record(payload['velocities'][m]) for m in MODALITIES} != row['velocities']
                or {m: tensor_record(payload['raw_outputs'][m]) for m in MODALITIES} != row['raw_outputs']):
            raise RuntimeError('E014 tensor SHA/shape/dtype changed')
        source_outputs[arm] = payload['velocities']
        updates[arm] = advance(pipe, state, source_outputs[arm], case['source_step'])
    exact = {}
    for modality, scheduler in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio)):
        assert_bytes_equal(source_outputs['bf16'][modality], state[modality]['noise_pred'], f'{modality} source BF16 velocity')
        assert_bytes_equal(updates['bf16'][modality], state[modality]['latents_after'], f'{modality} source latents_after')
        assert_bytes_equal(updates['bf16'][modality], next_state[modality]['latents_before'], f'{modality} next latents_before')
        assert_clock(scheduler, next_state[modality], case['next_step'])
        exact[modality] = {'e014_velocity_matches_historical': True,
            'source_latents_after_exact': True, 'next_latents_before_exact': True}
    teacher_endpoint = advance(pipe, next_state, {m: next_state[m]['noise_pred'] for m in MODALITIES}, case['next_step'])
    for modality in MODALITIES:
        assert_bytes_equal(teacher_endpoint[modality], next_state[modality]['latents_after'], f'{modality} reference next-step update')
        exact[modality]['next_latents_after_exact'] = True
    baseline_value = {'embedding': embedding, 'text_token_tags': tags,
        'state': {m: {key: next_state[m][key] for key in STATE_KEYS} for m in MODALITIES}}
    baseline_signature = tree_signature(baseline_value)
    packed = base.make_packed(pipe, embedding, tags, baseline_value['state'])
    packed_signature = tree_signature(packed)
    if 50*int((packed['cu_seqlens'][1:] > packed['cu_seqlens'][:-1]).sum())+2 != 102:
        raise RuntimeError('Prepared layout does not imply 102 original BF16 SDPA calls')
    outputs = []
    for arm in ARMS:
        for corner in (('BB',) if arm == 'bf16' else CORNERS):
            corner_state = {}
            for modality, selection in zip(MODALITIES, corner, strict=True):
                chosen = updates['bf16' if selection == 'B' else arm][modality]
                corner_state[modality] = {'latents_before': chosen,
                    'timestep': next_state[modality]['timestep'], 'sigma': next_state[modality]['sigma']}
                if set(corner_state[modality]) != STATE_KEYS:
                    raise RuntimeError('Reference-only fields leaked into actual model state')
            metadata = {'id': case['id'], 'prompt_id': case['prompt_id'], 'arm': arm, 'corner': corner,
                        'step': case['source_step'], 'source_step': case['source_step'], 'next_step': case['next_step'],
                        'corner_coordinate_order': list(MODALITIES)}
            value = {'embedding': embedding, 'text_token_tags': tags, 'state': corner_state}
            signature = tree_signature(value)
            if corner == 'BB' and signature != baseline_signature:
                raise RuntimeError('All arm BB inputs must be byte-identical')
            # Shape/clock/packing remain fixed even when latent VALUES change.
            corner_packed = tree_signature(base.make_packed(pipe, embedding, tags, corner_state))
            if corner_packed != packed_signature:
                raise RuntimeError('Corner unexpectedly changed packed/RoPE metadata')
            artifact = {'schema': SCHEMA, 'metadata': metadata, **value,
                'teacher_reference': next_state,
                'teacher_reference_semantics': 'Historical BF16 call at next_step; noise_pred and latents_after are comparison targets only. Integrate each corner from its own state.latents_before.',
                'teacher_next_raw_output_sha256': case['teacher_next_raw_output_sha256'],
                'source_velocity_artifact': case['e014_outputs'][arm],
                'source_teacher_state': case['teacher_state'],
                'teacher_next_state': case['teacher_next_state'],
                'input_signature': signature, 'packed_signature': packed_signature}
            row = {**metadata, 'input_signature': signature, 'packed_signature': packed_signature,
                'teacher_reference_signature': tree_signature(next_state),
                'teacher_next_raw_output_sha256': case['teacher_next_raw_output_sha256'],
                'source_velocity_artifact': case['e014_outputs'][arm]}
            outputs.append((row, artifact))
    construction = {'id': case['id'], 'replay_exact': exact,
        'source_velocity_signature': {arm: tree_signature(v) for arm, v in source_outputs.items()},
        'updated_latent_signature': {arm: tree_signature(v) for arm, v in updates.items()},
        'baseline_input_signature': baseline_signature,
        'bf16_unwritten_corner_aliases': {'QB': 'BB', 'BQ': 'BB', 'QQ': 'BB'},
        'bf16_alias_reason': 'B update was verified byte-exact with the BF16 Q update, so all four inputs coincide',
        'expected_next_sdpa_calls': 102}
    return outputs, construction


def run(args, report):
    manifest, checked, evaluations = prerequisites(args, report)
    if args.phase == 'prepare':
        previous = json.loads(args.check_report.read_text())
        if (previous.get('status') != 'complete' or previous.get('cuda_initialized') is not False
                or previous['sources'] != report['sources']):
            raise RuntimeError('Formal prepare requires a complete, same-source CPU check')
        base.check_sources(previous['sources'])
        report['check_reference'] = file_record(args.check_report)
        directory = args.data_dir/'prepare'
        directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6)
    for case in manifest['cases']:
        outputs, construction = construct_case(case, manifest['settings'], checked, evaluations)
        report['construction'].append(construction)
        for row, payload in outputs:
            if args.phase == 'prepare':
                path = directory/case['id']/f'{row["arm"]}_{row["corner"]}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, path)
                row['artifact'] = file_record(path)
                loaded = load_prepared_case(row)
                if (tree_signature(loaded['teacher_reference']) != row['teacher_reference_signature']
                        or loaded['packed_signature'] != row['packed_signature']):
                    raise RuntimeError('Saved reference/layout roundtrip failed')
            report['cases'].append(row)
        print(f'E015 {args.phase}: {case["id"]}, BF16 source and next update byte-exact, 9 corners valid', flush=True)
        save(report, args.output)
    if len(report['cases']) != 18 or torch.cuda.is_initialized():
        raise RuntimeError('Expected 18 CPU-only prepared input records')
    if args.phase == 'prepare':
        strip_artifact = lambda rows: [{k: v for k, v in row.items() if k != 'artifact'} for row in rows]
        if strip_artifact(report['cases']) != previous['cases'] or report['construction'] != previous['construction']:
            raise RuntimeError('Prepare differs from its complete CPU check')
    report.update(status='complete', cuda_initialized=False, complete_dit_calls=0,
                  prepared_payload_count=18 if args.phase == 'prepare' else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'prepare'), required=True)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=REPORTS)
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.check_report = args.check_report or args.report_dir/'E015_prepare_check.json'
    args.output = args.output or args.report_dir/('E015_prepare_check.json' if args.phase == 'check' else 'E015_prepare.json')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite prior report: {args.output}')
    report = {'experiment': 'E015', 'phase': args.phase, 'schema': SCHEMA, 'status': 'running',
        'scope': 'CPU-only original scheduler replay and one-step counterfactual states; no model forwards'}
    try:
        run(args, report)
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc())
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
