#!/usr/bin/env python3
"""Independent, CPU-only E012 audit from saved inputs and raw velocities.

Imports no experiment runner or runner statistics. The observed runtime counts
are checked against the frozen reports, not independently measured GPU events.
A successful audit is a numeric response diagnostic, never semantic quality.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = '7ae18f19a78419171c2efa8837c1ce34e5ca3ed9c2a00d791c144694ba7e0f44'
MANIFEST_SHA = '1de2a37b2e2be9f98777039b11fbf63986fb0250faf4a24cbe4688c3b9c9a34d'
MODALITIES = ('video', 'audio')
VERIFIED = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def verify_file(record):
    path = Path(record['file']).resolve()
    found = file_record(path)
    require(found['sha256'] == record['sha256'], f'File hash mismatch: {path}')
    if 'bytes' in record:
        require(found['bytes'] == record['bytes'], f'File size mismatch: {path}')
    VERIFIED[str(path)] = found
    return path


def verify_sources(sources):
    for path, metadata in sources.items():
        verify_file({'file': path, **metadata})


def read_json(path):
    return json.loads(Path(path).read_text())


def tensor_record(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def signature(value):
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, dict):
        return {key: signature(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [signature(val) for val in value]
    return value


def load_tensor_file(record):
    return torch.load(verify_file(record), weights_only=True, map_location='cpu', mmap=True)


def vector(value):
    return value.to(dtype=torch.float64, device='cpu').reshape(-1)


def squared(value):
    return float(torch.sum(value.double().square()))


def close_tree(actual, expected, path='root'):
    """Compare independent FP64 reductions with documented rounding tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f'Keys: {path}')
        for key in expected:
            close_tree(actual[key], expected[key], path + '.' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'Length: {path}')
        for index, (a, b) in enumerate(zip(actual, expected)):
            close_tree(a, b, path + f'[{index}]')
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), f'Number: {path}: {actual} != {expected}')
    else:
        require(actual == expected, f'Value: {path}: {actual} != {expected}')


def verify_provenance(report, manifest, plan, manifest_file):
    require(report['experiment'] == 'E012', 'Wrong experiment')
    require(report['manifest'] == manifest, 'Report manifest differs')
    require(report['plan'] == file_record(plan), 'Plan reference differs')
    require(report['manifest_file'] == file_record(manifest_file), 'Manifest reference differs')
    verify_sources(report['sources'])
    for key in ('e010_prepare', 'e010_bf16'):
        verify_file(report[key])


def expected_states(e010_case, steps):
    result, refs = {}, []
    for step in steps:
        center = {}
        for modality in MODALITIES:
            rows = [r for r in e010_case['steps'] if r['step'] == step and r['modality'] == modality]
            require(len(rows) == 1, 'Ambiguous E010 teacher state')
            center[modality] = load_tensor_file(rows[0])
            refs.append(VERIFIED[str(Path(rows[0]['file']).resolve())])
        result[(step, 'center')] = center
        for variant, parity_sign in (('plus', 1), ('minus', -1)):
            changed = {}
            for modality, row in center.items():
                latent = row['latents_before']
                require(latent.dtype == torch.bfloat16 and bool(torch.isfinite(latent).all()), 'Bad latent dtype/domain')
                direction = torch.full((latent.numel(),), float('inf'), dtype=latent.dtype)
                direction[1::2] = -float('inf')
                direction *= parity_sign
                perturbed = torch.nextafter(latent, direction.reshape(latent.shape))
                require(bool(torch.isfinite(perturbed).all()) and bool((perturbed != latent).all()), 'Invalid BF16 ULP perturbation')
                changed[modality] = {**row, 'latents_before': perturbed}
            result[(step, variant)] = changed
    return result, refs


def verify_state_artifacts(report, expected, steps):
    require([r['step'] for r in report['states']] == steps, 'State table steps/count')
    for row in report['states']:
        payload = load_tensor_file(row['artifact'])
        require(set(payload['variants']) == {'center', 'plus', 'minus'}, 'State variants changed')
        for variant, state in payload['variants'].items():
            require(signature(state) == signature(expected[(row['step'], variant)]), 'Saved state differs from independently reconstructed E010 state/ULP')
        for modality in MODALITIES:
            record = row['perturbations'][modality]
            original = expected[(row['step'], 'center')][modality]['latents_before']
            require(record['original'] == tensor_record(original), 'Original perturbation SHA')
            for variant in ('plus', 'minus'):
                value = expected[(row['step'], variant)][modality]['latents_before']
                e = squared(vector(value) - vector(original))
                calculated = {**tensor_record(value), 'changed_elements': value.numel(), 'elements': value.numel(),
                              'delta_energy': e, 'relative_delta_energy': e / squared(original)}
                close_tree(record['perturbations'][variant], calculated, 'perturbation')
        require(payload['perturbation'] == row['perturbations'], 'Raw/report perturbation mismatch')
        require(payload['source_files'] == row['source_files'], 'Raw/report source-state mismatch')
        for source in row['source_files'].values():
            verify_file(source)


def verify_calls(report, manifest, prepared, states, selected=None, teacher_calls=None):
    arm = report['phase']
    ids = [row['id'] for row in manifest['conditions']] if arm == 'bf16' else ['original', 'direction', selected]
    variants = ['center', 'plus', 'minus'] if arm == 'bf16' else ['center']
    expected = set(itertools.product(manifest['steps'], ids, variants))
    if arm == 'bf16':
        expected |= {(step, 'original', 'repeat') for step in manifest['steps']}
    require(len(report['calls']) == len(expected) == (26 if arm == 'bf16' else 6), 'Forward count differs')
    outputs, rows, common = {}, {}, {}
    for row in report['calls']:
        key = row['step'], row['condition_id'], row['control_id']
        require(key in expected and key not in outputs, 'Missing/unexpected/duplicate call grid cell')
        require(row['id'] == f's{key[0]:02d}_{key[1]}_{key[2]}' and row['status'] == 'complete' and row['arm'] == arm, 'Call identity/status')
        variant = 'center' if key[2] == 'repeat' else key[2]
        state = states[(key[0], variant)]
        expected_input = {m: {k: tensor_record(state[m][k]) for k in ('latents_before', 'timestep', 'sigma')} for m in MODALITIES}
        require(row['input'] == expected_input, 'Actual call input differs from E010/ULP reconstruction')
        require(row['embedding'] == tensor_record(prepared[key[1]]['embedding']), 'Call embedding differs')
        require(row['runtime_audit'] == {'sdpa_calls': 102, 'scaled_mm_calls': 0 if arm == 'bf16' else 200, 'disk_loads': 0}, 'Wrong runtime path counts')
        checks = row['zero_sf_checks']
        if arm == 'bf16':
            require(checks is None, 'BF16 unexpected quantization checks')
        else:
            require(checks['checked_calls'] == 200 and checks['invalid_calls'] == 0, 'Invalid/incomplete native checks')
            require(len(checks['affected_call_indices_zero_based']) == checks['calls_with_nonzero_input_zero_sf'], 'SF0 count inconsistent')
        payload = load_tensor_file(row['artifact'])
        require(payload['id'] == row['id'], 'Raw call identity')
        for field in ('input', 'packed', 'actual_dit'):
            require(payload[field] == row[field], f'Raw/report {field} mismatch')
        require(set(payload['velocities']) == set(MODALITIES), 'Missing/extra modality')
        for modality, velocity in payload['velocities'].items():
            require(bool(torch.isfinite(velocity).all()) and velocity.dtype == torch.bfloat16, 'Bad raw velocity')
            require(tensor_record(velocity) == row['output'][modality], 'Velocity record mismatch')
            require(velocity.shape == state[modality]['noise_pred'].shape, 'Velocity shape changed')
            if arm == 'bf16' and key[1:] == ('original', 'center'):
                require(tensor_record(velocity) == tensor_record(state[modality]['noise_pred']), 'Original teacher no longer exact E010')
        layout = {field: row[field] for field in ('packed', 'input', 'actual_dit')}
        common_key = key[0], variant
        if common_key in common:
            require(common[common_key] == layout, 'Condition/repeat changed actual x/t/positions/tags')
        else:
            common[common_key] = layout
        if teacher_calls is not None:
            for field in ('packed', 'input', 'actual_dit', 'embedding'):
                require(row[field] == teacher_calls[key][field], 'Native and teacher actual inputs differ')
        outputs[key], rows[key] = payload['velocities'], row
    require(set(outputs) == expected, 'Incomplete call grid')
    return outputs, rows


def recompute_teacher(manifest, outputs):
    base, target = manifest['original_condition_id'], manifest['target_condition_id']
    controls, stats = manifest['control_candidate_ids'], {}
    for step in manifest['steps']:
        stats[str(step)] = {}
        for modality in MODALITIES:
            b0 = vector(outputs[(step, base, 'center')][modality])
            repeat = squared(vector(outputs[(step, base, 'repeat')][modality]) - b0)
            edits = {}
            for cid in [target] + controls:
                b1 = vector(outputs[(step, cid, 'center')][modality])
                difference = b1 - b0
                amplitude, backgrounds, single = squared(difference), {}, {}
                for variant in ('plus', 'minus'):
                    v0 = vector(outputs[(step, base, variant)][modality])
                    v1 = vector(outputs[(step, cid, variant)][modality])
                    backgrounds[variant] = squared((v1 - v0) - difference)
                    single[variant] = {'original': squared(v0 - b0), 'edited': squared(v1 - b1)}
                floor = max(backgrounds['plus'], backgrounds['minus'], 4 * repeat)
                edits[cid] = {'response_energy': amplitude, 'difference_perturbation_energy': backgrounds,
                    'single_condition_perturbation_energy': single, 'repeat_energy_times4': 4 * repeat,
                    'background_energy': floor, 'response_to_background_energy': amplitude / floor if floor else ('infinite' if amplitude else None),
                    'snr_pass': amplitude > 0 and amplitude >= manifest['teacher_gate']['min_response_to_difference_background_energy'] * floor}
            stats[str(step)][modality] = {'original_repeat_energy': repeat, 'edits': edits}
    primary = manifest['primary_modality']
    target_ok = all(stats[str(s)][primary]['edits'][target]['snr_pass'] for s in manifest['steps'])
    low, high = manifest['teacher_gate']['control_teacher_norm_ratio_range']
    candidates, chosen = [], None
    for cid in controls:
        rows = []
        for step in manifest['steps']:
            edits = stats[str(step)][primary]['edits']
            ratio = math.sqrt(edits[cid]['response_energy'] / edits[target]['response_energy']) if edits[target]['response_energy'] else None
            rows.append({'step': step, 'norm_ratio': ratio, 'snr_pass': edits[cid]['snr_pass'],
                         'norm_match': ratio is not None and low <= ratio <= high})
        eligible = target_ok and all(row['snr_pass'] and row['norm_match'] for row in rows)
        candidates.append({'condition_id': cid, 'steps': rows, 'eligible': eligible})
        if chosen is None and eligible:
            chosen = cid
    return {'statistics': stats, 'primary_modality': primary, 'target_stable': target_ok,
        'control_eligibility_fixed_order': candidates, 'selected_control_id': chosen,
        'pass': target_ok and chosen is not None,
        'status': 'teacher_gate_passed' if target_ok and chosen is not None else 'stopped_teacher_gate'}


def pair_metrics(b0, b1, q0, q1):
    b0, b1, q0, q1 = map(vector, (b0, b1, q0, q1))
    db, dq, e0, e1 = b1-b0, q1-q0, q0-b0, q1-b1
    reference, quantized = squared(db), squared(dq)
    difference_error, mean_error = squared(e1-e0), squared((e0+e1)*.5)
    endpoint_errors = [squared(e0), squared(e1)]
    inner = float(torch.sum(db*dq))
    pair_residual = sum(endpoint_errors) - 2*mean_error - .5*difference_error
    require(abs(pair_residual) <= 1e-10 * max(1., sum(endpoint_errors)), 'Pairwise identity failed')
    result = {'elements': db.numel(), 'teacher_response_energy': reference,
        'teacher_endpoint_energy': [squared(b0), squared(b1)], 'quantized_response_energy': quantized,
        'response_dot': inner, 'response_error_energy': difference_error,
        'mean_error_energy': mean_error, 'endpoint_error_energy': endpoint_errors,
        'endpoint_error_cross': float(torch.sum(e0*e1)), 'pair_identity_residual': pair_residual}
    if reference:
        gain, orthogonal = inner/reference, squared(dq-(inner/reference)*db)/reference
        nmse = difference_error/reference
        residual = nmse - ((gain-1)**2 + orthogonal)
        require(abs(residual) <= 1e-10 * max(1., nmse), 'Response decomposition failed')
        result.update(gain=gain, norm_ratio_squared=quantized/reference,
                      orthogonal_error_ratio_squared=orthogonal, response_nmse=nmse, decomposition_residual=residual)
    else:
        result.update({k: None for k in ('gain', 'norm_ratio_squared', 'orthogonal_error_ratio_squared', 'response_nmse', 'decomposition_residual')})
    return result


def recompute_native(manifest, teacher, native, selected):
    original, target = manifest['original_condition_id'], manifest['target_condition_id']
    limits, stats, candidates = manifest['candidate_gate'], {}, []
    for step in manifest['steps']:
        row = {}
        for modality in MODALITIES:
            endpoint, responses = {}, {}
            for cid in (original, target, selected):
                b, q = teacher[(step, cid, 'center')][modality], native[(step, cid, 'center')][modality]
                error, ref = squared(vector(q)-vector(b)), squared(b)
                endpoint[cid] = {'bf16_energy': ref, 'native_energy': squared(q), 'error_energy': error, 'nmse': error/ref if ref else None}
            for cid in (target, selected):
                responses[cid] = pair_metrics(teacher[(step, original, 'center')][modality], teacher[(step, cid, 'center')][modality],
                                             native[(step, original, 'center')][modality], native[(step, cid, 'center')][modality])
            row[modality] = {'endpoints': endpoint, 'responses': responses}
        video = row[manifest['primary_modality']]
        t, c = video['responses'][target], video['responses'][selected]
        checks = {'target_gain': t['gain'] is not None and t['gain'] < limits['target_gain_lt'],
            'target_energy_ratio': t['norm_ratio_squared'] is not None and t['norm_ratio_squared'] < limits['target_energy_ratio_lt'],
            'control_gain': c['gain'] is not None and limits['control_gain_range'][0] <= c['gain'] <= limits['control_gain_range'][1],
            'control_difference_nmse': c['response_nmse'] is not None and c['response_nmse'] <= limits['control_difference_nmse_le'],
            'per_condition_nmse': all(v['nmse'] is not None and v['nmse'] <= limits['all_per_condition_video_nmse_le'] for v in video['endpoints'].values())}
        row['candidate_gate_checks'] = checks
        candidates.append(all(checks.values()))
        stats[str(step)] = row
    return {'steps': stats, 'decision': 'candidate_requires_replication' if all(candidates) else 'stop_current_selective_attenuation_route'}


def summarize(args):
    manifest_path = ROOT/'research_state/06_experiments/E012_h3_conditional_response_manifest.json'
    plan = ROOT/'research_state/06_experiments/E012_h3_conditional_response_plan.md'
    verify_file({'file': str(manifest_path), 'sha256': MANIFEST_SHA})
    verify_file({'file': str(plan), 'sha256': PLAN_SHA})
    manifest = read_json(manifest_path)
    verify_sources(manifest['source_files'])
    paths = {phase: args.report_dir/f'E012_h3_{phase}.json' for phase in ('check', 'bf16', 'native')}
    for phase in ('check', 'bf16'):
        require(paths[phase].exists(), f'Pending {phase}: no summary emitted')
    teacher_metadata = read_json(paths['bf16'])
    require(teacher_metadata['status'] in ('complete', 'stopped_teacher_gate'), 'Teacher incomplete: no summary emitted')
    paths['prepare'] = verify_file(teacher_metadata['prepare_reference'])
    check, prepare, teacher = (read_json(paths[p]) for p in ('check', 'prepare', 'bf16'))
    require(check['status'] == prepare['status'] == 'complete', 'Prerequisite phase incomplete')
    require(teacher['status'] in ('complete', 'stopped_teacher_gate'), 'Teacher incomplete: no summary emitted')
    for report in (check, prepare, teacher):
        verify_provenance(report, manifest, plan, manifest_path)
    require(check['cuda_initialized'] is False and check['same_packed_layout'] is True, 'CPU check changed')
    require(len(check['calls']) == len(prepare['calls']) == 0, 'Check/prepare unexpectedly ran DiT')
    for report in (prepare, teacher):
        verify_file(report['cpu_check_reference'])
        require(report['cpu_conditions'] == check['cpu_conditions'], 'Token table drift')
    require(teacher['cpu_check_reference'] == file_record(paths['check']), 'Teacher used different CPU preflight')
    verify_file(teacher['prepare_reference'])
    require(teacher['prepare_reference'] == file_record(paths['prepare']), 'Teacher used different embeddings stage')
    original_cache = load_tensor_file(teacher['original_embedding'])
    condition_ids = [c['id'] for c in manifest['conditions']]
    require([c['id'] for c in prepare['conditions']] == condition_ids, 'Prepared condition table count/order')
    require([c['id'] for c in check['cpu_conditions']] == condition_ids, 'Token condition table count/order')
    prepared = {}
    for condition, row, checked in zip(manifest['conditions'], prepare['conditions'], check['cpu_conditions']):
        payload = load_tensor_file(row['artifact'])
        require(payload['condition'] == condition, 'Prepared prompt/edit differs')
        require(hashlib.sha256(condition['prompt'].encode()).hexdigest() == condition['prompt_sha256'], 'Prompt hash')
        require(payload['token_ids'] == checked['token_ids'], 'Token IDs differ from preflight')
        require(checked['tokens'] == condition['expected_text_tokens'] == 813, 'Token layout changed')
        for key, record_key in (('embedding', 'embedding'), ('text_token_tags', 'tags')):
            require(tensor_record(payload[key]) == row[record_key], 'Prepared tensor hash')
            require(bool(torch.isfinite(payload[key]).all()), 'Prepared tensor nonfinite')
        require(row['tags'] == checked['tags'], 'Prepared tags differ')
        if condition['id'] == 'original':
            require(tensor_record(payload['embedding']) == tensor_record(original_cache['embedding']), 'Original embedding no longer E010 exact')
            require(tensor_record(payload['text_token_tags']) == tensor_record(original_cache['text_token_tags']), 'Original text tags no longer exact')
        prepared[condition['id']] = payload
    e010 = read_json(verify_file(teacher['e010_bf16']))
    e010_case = next(c for c in e010['cases'] if c['prompt_id'] == manifest['original_prompt_id'])
    require(teacher['teacher_trajectory'] == e010_case, 'E010 trajectory reference differs')
    states, state_refs = expected_states(e010_case, manifest['steps'])
    verify_state_artifacts(teacher, states, manifest['steps'])
    b, teacher_calls = verify_calls(teacher, manifest, prepared, states)
    gate = recompute_teacher(manifest, b)
    close_tree(teacher['teacher_gate'], gate, 'teacher_gate')
    frozen_selection = read_json(verify_file(teacher['selection']))
    close_tree(frozen_selection['teacher_gate'], gate, 'frozen_selection')
    require(frozen_selection['raw_teacher_outputs'] == [row['artifact'] for row in teacher['calls']], 'Selection raw teacher table changed')
    require(frozen_selection['sources'] == teacher['sources'] and frozen_selection['manifest_file'] == teacher['manifest_file']
            and frozen_selection['plan'] == teacher['plan'], 'Selection source binding')
    summary = {'experiment': 'E012', 'status': 'complete', 'scope': 'Independent CPU audit of fixed-state numeric conditional response; not semantic controllability or quality',
        'teacher_gate': gate, 'teacher_calls': len(b), 'native_calls': 0, 'e010_teacher_state_files': state_refs,
        'selection': teacher['selection'], 'metric_comparison_tolerance': {'relative': 1e-10, 'absolute': 1e-9},
        'limitations': ['One previously viewed prompt and two correlated original-teacher trajectory states.',
                       'ULP background is a local numerical floor, not all diffusion noise sensitivity.',
                       'Runtime call counts are cross-checked frozen observations; this CPU audit executes no GPU kernels.']}
    if 'scalar_hash_fix_provenance' in teacher:
        provenance = teacher['scalar_hash_fix_provenance']
        for key in ('v1_runner', 'v1_failed_bf16', 'reused_prepare_report'):
            verify_file(provenance[key])
        failure = read_json(provenance['v1_failed_bf16']['file'])
        require(failure['status'] == 'failed_stop' and not failure['calls'], 'Previous failure unexpectedly consumed DiT calls')
        require(provenance['reused_prepare_report'] == teacher['prepare_reference'], 'v2 changed reused TE artifact')
        summary['preserved_zero_call_failure'] = provenance
    if not gate['pass']:
        require(teacher['status'] == 'stopped_teacher_gate', 'Teacher stop state inconsistent')
        require(not paths['native'].exists(), 'Native report present after forbidden teacher gate')
        require(not (Path(teacher['selection']['file']).parent/'native').exists(), 'Native data directory present after teacher stop')
        summary.update(final_state='stopped_teacher_gate', native_absence_verified=True)
    else:
        require(teacher['status'] == 'complete' and paths['native'].exists(), 'Native result pending: no summary emitted')
        native = read_json(paths['native'])
        require(native['status'] == 'complete', 'Native incomplete: no summary emitted')
        verify_provenance(native, manifest, plan, manifest_path)
        require(native['selection'] == teacher['selection'] and native['teacher_reference'] == file_record(paths['bf16']), 'Native not bound to frozen BF16 selection/report')
        require(native['prepare_reference'] == teacher['prepare_reference'] and native['cpu_conditions'] == teacher['cpu_conditions'], 'Native embedding/token provenance')
        verify_file(native['selection'])
        verify_file(native['teacher_reference'])
        close_tree(native['teacher_gate'], gate, 'native.teacher_gate')
        verify_state_artifacts(native, states, manifest['steps'])
        q, _ = verify_calls(native, manifest, prepared, states, gate['selected_control_id'], teacher_calls)
        stats = recompute_native(manifest, b, q, gate['selected_control_id'])
        close_tree(native['statistics'], stats, 'native.statistics')
        require(native['decision'] == stats['decision'], 'Final decision mismatch')
        # File timing is supporting consistency evidence, not a substitute for
        # source-reviewed stage ordering and immutable teacher/report hashes.
        frozen_ns = Path(teacher['selection']['file']).stat().st_mtime_ns
        require(all(frozen_ns <= Path(c['artifact']['file']).stat().st_mtime_ns for c in native['calls']), 'Native artifact predates frozen selection')
        summary.update(native_calls=len(q), native_statistics=stats, final_state=stats['decision'], selection_precedes_native_files=True)
    require(not torch.cuda.is_initialized(), 'CPU-only audit initialized CUDA')
    for phase, path in paths.items():
        if path.exists():
            VERIFIED[str(path.resolve())] = file_record(path)
    summary.update(cuda_initialized=False, raw_sources=VERIFIED, summary_source=file_record(__file__),
                   verified_file_count=len(VERIFIED))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/research/E012_v2')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E012_summary.json')
    args = parser.parse_args()
    require(not args.output.exists(), f'Preserve existing summary: {args.output}')
    torch.set_num_threads(6)
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': summary['status'], 'final_state': summary['final_state'],
                      'teacher_calls': summary['teacher_calls'], 'native_calls': summary['native_calls'],
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
