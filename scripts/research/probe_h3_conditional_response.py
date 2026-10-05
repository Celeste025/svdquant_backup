#!/usr/bin/env python3
"""E012 fixed-state conditional-response diagnostic; no sampling or decoding."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'] = 'torch'
os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'True'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent))
FROZEN_E010 = Path(__file__).with_name('run_h3_native_paired_video.py')
FROZEN_E010_SHA = '41e6e4eb1d9df554e5ea109b9bad3d1df83e9b64788137511262a82a0f34431f'
if hashlib.sha256(FROZEN_E010.read_bytes()).hexdigest() != FROZEN_E010_SHA:
    raise RuntimeError('E010 source differs from the frozen reviewed contract')
import run_h3_native_paired_video as inherited
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder, model_fn_minimax_h3

sha256, tensor_sha, save = inherited.sha256, inherited.tensor_sha, inherited.save
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E012')
E010_PREPARE = ROOT/'results/research/E010_h3_prepare.json'
E010_BF16 = ROOT/'results/research/E010_h3_denoise_bf16.json'
PLAN_SHA = '7ae18f19a78419171c2efa8837c1ce34e5ca3ed9c2a00d791c144694ba7e0f44'
MANIFEST_SHA = '1de2a37b2e2be9f98777039b11fbf63986fb0250faf4a24cbe4688c3b9c9a34d'


def report_path(args, phase):
    return args.report_dir/f'E012_h3_{phase}.json'


def file_record(path):
    path = Path(path)
    return {'file': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def tensor_record(value):
    return {'sha256': tensor_sha(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def tree_signature(value):
    if torch.is_tensor(value): return tensor_record(value)
    if isinstance(value, dict): return {k: tree_signature(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)): return [tree_signature(v) for v in value]
    return value


def check_sources(records):
    for path, info in records.items():
        if sha256(path) != info['sha256']:
            raise RuntimeError(f'Frozen source changed: {path}')


def prerequisites(args, report):
    if sha256(args.plan) != PLAN_SHA or sha256(args.manifest) != MANIFEST_SHA:
        raise RuntimeError('E012 plan/manifest differ from the preregistered contract')
    manifest = json.loads(args.manifest.read_text())
    check_sources(manifest['source_files'])
    inherited_report = {}
    inherited_args = types.SimpleNamespace(arm='native', export_dir=args.export_dir)
    _, reference = inherited.prerequisites(inherited_args, inherited_report)
    prep = inherited.load_complete(E010_PREPARE)
    bf16 = inherited.load_complete(E010_BF16)
    check_sources(prep['sources'])
    check_sources(bf16['sources'])
    for path, info in prep['assets'].items():
        st = Path(path).stat()
        if st.st_size != info['bytes'] or st.st_mtime_ns != info['mtime_ns']:
            raise RuntimeError(f'Asset changed after E010 full SHA: {path}')
    report.update(inherited_contract=inherited_report, inherited_asset_hashes=prep['assets'],
                  e010_prepare=file_record(E010_PREPARE), e010_bf16=file_record(E010_BF16),
                  manifest=manifest, manifest_file=file_record(args.manifest), plan=file_record(args.plan))
    report['sources'] = inherited_report['sources'] | {
        str(p.resolve()): {'sha256': sha256(p), 'bytes': p.stat().st_size}
        for p in (Path(__file__), args.manifest, args.plan)}
    report['original_embedding'] = next(c for c in prep['cases'] if c['prompt_id'] == 36)
    report['teacher_trajectory'] = next(c for c in bf16['cases'] if c['prompt_id'] == 36)
    if sha256(report['original_embedding']['file']) != report['original_embedding']['sha256']:
        raise RuntimeError('E010 original embedding cache changed')
    report['calls'] = []
    return manifest, reference


def original_payload(report):
    return torch.load(report['original_embedding']['file'], map_location='cpu', weights_only=True, mmap=True)


def load_teacher_state(report, step):
    result, files = {}, {}
    for modality in ('video', 'audio'):
        item = next(r for r in report['teacher_trajectory']['steps'] if r['step'] == step and r['modality'] == modality)
        if sha256(item['file']) != item['sha256']:
            raise RuntimeError('E010 teacher trajectory file changed')
        result[modality] = torch.load(item['file'], map_location='cpu', weights_only=True, mmap=True)
        files[modality] = file_record(item['file'])
    return result, files


def make_packed(pipe, embedding, tags, state):
    return MiniMaxH3Unit_PackedSequenceBuilder().process(pipe,
        prompt_embeds=embedding, text_token_tags=tags,
        video_latents=state['video']['latents_before'], audio_latents=state['audio']['latents_before'])['packed']


def cpu_check(manifest, report):
    from transformers import AutoProcessor
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va
    processor = AutoProcessor.from_pretrained(str(inherited.MODEL_ROOT/'FL2VA/processor'), local_files_only=True)
    original = original_payload(report)
    base = next(c for c in manifest['conditions'] if c['id'] == manifest['original_condition_id'])
    if base['prompt'] != original['prompt']:
        raise RuntimeError('Original condition differs from E010 p36')
    pipe = inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    state, state_files = load_teacher_state(report, manifest['steps'][0])
    condition_ids, token_rows, packed_rows = set(), [], {}
    for condition in manifest['conditions']:
        cid = condition['id']
        if cid in condition_ids: raise RuntimeError('Duplicate condition id')
        condition_ids.add(cid)
        if hashlib.sha256(condition['prompt'].encode()).hexdigest() != condition['prompt_sha256']:
            raise RuntimeError(f'Condition text SHA mismatch: {cid}')
        ids, tags = presentation_t2va(processor.tokenizer, condition['prompt'])
        if ids.numel() != condition['expected_text_tokens']:
            raise RuntimeError(f'Actual token count changed: {cid}')
        edit = condition['edit']
        if edit['count']:
            if (base['prompt'].count(edit['from']) != edit['count'] or
                    base['prompt'].replace(edit['from'], edit['to']) != condition['prompt']):
                raise RuntimeError(f'Condition edit contract changed: {cid}')
        if tags.unique().tolist() != [1]: raise RuntimeError('Expected ordinary T2VA tags')
        dummy = torch.empty((ids.numel(), original['embedding'].shape[-1]), dtype=torch.bfloat16)
        packed = make_packed(pipe, dummy, tags, state)
        packed_rows[cid] = tree_signature(packed)
        token_rows.append({'id': cid, 'tokens': ids.numel(), 'token_ids': ids.tolist(),
                          'tags': tensor_record(tags), 'packed': packed_rows[cid]})
    report['cpu_conditions'] = token_rows
    report['cpu_state'] = state_files
    first = packed_rows[manifest['original_condition_id']]
    if any(p != first for p in packed_rows.values()):
        raise RuntimeError('Conditions change actual packed/RoPE layout')
    report['same_packed_layout'] = True
    if torch.cuda.is_initialized(): raise RuntimeError('CPU contract check initialized CUDA')
    report['cuda_initialized'] = False
    return packed_rows


def start_stage(args, report, phase):
    directory = args.data_dir/phase
    directory.mkdir(parents=True, exist_ok=False)
    inherited.snapshot_sources(report, directory/'sources')
    return directory


@torch.inference_mode()
def prepare(args, manifest, report):
    directory = start_stage(args, report, 'prepare')
    original = original_payload(report)
    shards = sorted((inherited.MODEL_ROOT/'FL2VA/text_encoder').glob('model*.safetensors'))
    pipe = inherited.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[inherited.ModelConfig(path=[str(p) for p in shards], **inherited.disk_config())],
        processor_config=inherited.ModelConfig(path=str(inherited.MODEL_ROOT/'FL2VA/processor')), vram_limit=30.)
    if pipe.dit is not None or pipe.video_vae is not None or pipe.audio_vae is not None:
        raise RuntimeError('Text-only preparation contract violated')
    pipe.text_encoder.eval()
    unit = inherited.MiniMaxH3Unit_PromptEmbedder()
    report['conditions'] = []
    for condition in manifest['conditions']:
        print(f'E012 prepare {condition["id"]}', flush=True)
        embedded = unit.process(pipe, prompt=condition['prompt'], height=576, width=1024)
        embedding, tags = embedded['prompt_embeds'].detach().cpu(), embedded['text_token_tags'].detach().cpu()
        if (not bool(torch.isfinite(embedding).all()) or tags.unique().tolist() != [1]
                or embedding.shape != original['embedding'].shape or tags.shape != original['text_token_tags'].shape):
            raise RuntimeError('Invalid text embedding')
        row = {'id': condition['id'], 'embedding': tensor_record(embedding), 'tags': tensor_record(tags)}
        if condition['id'] == manifest['original_condition_id']:
            row['e010_embedding_exact'] = tensor_sha(embedding) == tensor_sha(original['embedding'])
            row['e010_tags_exact'] = tensor_sha(tags) == tensor_sha(original['text_token_tags'])
            if not row['e010_embedding_exact'] or not row['e010_tags_exact']:
                raise RuntimeError('Original p36 text-encoder replay differs from E010')
        path = directory/(condition['id']+'.pt')
        tokens = next(c for c in report['cpu_conditions'] if c['id'] == condition['id'])
        torch.save({'condition': condition, 'embedding': embedding, 'text_token_tags': tags,
                    'token_ids': tokens['token_ids']}, path)
        row['artifact'] = file_record(path)
        report['conditions'].append(row)
        save(report, args.output)
        inherited.memory_guard()
    report['status'] = 'complete'


def cpu64(value):
    return value.detach().to(device='cpu', dtype=torch.float64).reshape(-1)


def energy(value):
    value = cpu64(value)
    return float(torch.dot(value, value))


def response_statistics(b0, b1, q0=None, q1=None):
    """All energies/dot products use actual tensors and FP64 CPU accumulation."""
    b0, b1 = cpu64(b0), cpu64(b1)
    db = b1-b0
    eb = float(torch.dot(db, db))
    result = {'elements': db.numel(), 'teacher_response_energy': eb,
              'teacher_endpoint_energy': [float(torch.dot(b0, b0)), float(torch.dot(b1, b1))]}
    if q0 is None: return result
    q0, q1 = cpu64(q0), cpu64(q1)
    dq = q1-q0
    e0, e1 = q0-b0, q1-b1
    delta, mean = dq-db, (e0+e1)/2
    eq = float(torch.dot(dq, dq))
    dot = float(torch.dot(dq, db))
    ed = float(torch.dot(delta, delta))
    em = float(torch.dot(mean, mean))
    endpoint_e = [float(torch.dot(e0, e0)), float(torch.dot(e1, e1))]
    result.update(quantized_response_energy=eq, response_dot=dot,
        response_error_energy=ed, mean_error_energy=em, endpoint_error_energy=endpoint_e,
        endpoint_error_cross=float(torch.dot(e0, e1)),
        pair_identity_residual=endpoint_e[0]+endpoint_e[1]-2*em-.5*ed)
    if eb > 0:
        gain = dot/eb
        orthogonal = dq-gain*db
        orthogonal_energy = float(torch.dot(orthogonal, orthogonal))
        result.update(gain=gain, norm_ratio_squared=eq/eb,
            orthogonal_error_ratio_squared=orthogonal_energy/eb,
            response_nmse=ed/eb, decomposition_residual=ed/eb-((gain-1)**2+orthogonal_energy/eb))
    else:
        result.update(gain=None, norm_ratio_squared=None, orthogonal_error_ratio_squared=None,
                      response_nmse=None, decomposition_residual=None)
    return result


def load_prepared(args, report):
    path = report_path(args, 'prepare')
    prepared = inherited.load_complete(path)
    check_sources(prepared['sources'])
    conditions = {}
    for row in prepared['conditions']:
        artifact = row['artifact']
        if sha256(artifact['file']) != artifact['sha256']:
            raise RuntimeError('Prepared embedding file changed')
        conditions[row['id']] = torch.load(artifact['file'], map_location='cpu', weights_only=True, mmap=True)
    report['prepare_reference'] = file_record(path)
    return conditions


@torch.inference_mode()
def evaluate_call(pipe, audit, arm, step, condition_id, control_id, condition, state,
                  directory, report, output):
    """Call the unchanged original model_fn, with identical common x/t per pair."""
    call_id = f's{step:02d}_{condition_id}_{control_id}'
    if any(r['id'] == call_id for r in report['calls']):
        raise RuntimeError(f'Duplicate call: {call_id}')
    embedding = condition['embedding'].cuda()
    tags = condition['text_token_tags'].cuda()
    gpu_state = {m: {k: value.cuda() if torch.is_tensor(value) else value for k, value in row.items()}
                 for m, row in state.items()}
    packed = make_packed(pipe, embedding, tags, gpu_state)
    expected_sdpa = 50*int((packed['cu_seqlens'][1:] > packed['cu_seqlens'][:-1]).sum())+2
    if expected_sdpa != 102:
        raise RuntimeError('Preregistered H3 layout requires 102 SDPA calls')
    info = {'id': call_id, 'step': step, 'condition_id': condition_id, 'control_id': control_id,
        'arm': arm, 'packed': tree_signature(packed), 'embedding': tensor_record(embedding),
        'input': {m: {k: tensor_record(gpu_state[m][k]) for k in ('latents_before', 'timestep', 'sigma')}
                  for m in ('video', 'audio')}}
    report['calls'].append(info)
    save(report, output)
    audit.phase = call_id
    before = dict(audit.row())
    captures = []
    def observe(_module, pos, kw):
        if pos or kw.get('control_hints') is not None:
            raise RuntimeError('Unexpected positional DiT input/ControlNet')
        captures.append({k: tree_signature(kw[k]) for k in (
            'x', 'audio_x', 'unique_timesteps', 'inverse_indices', 'token_tags',
            'img_position_ids', 'img_pos_info', 'audio_pos_info', 'text_pos_info',
            'packed_seq_params', 'refiner_packed_seq_params')})
    handle = pipe.dit.register_forward_pre_hook(observe, with_kwargs=True)
    context = inherited.collect_fastpack_checks() if arm == 'native' else nullcontext(None)
    started = time.monotonic()
    print(f'E012 {arm} {call_id}', flush=True)
    try:
        with context as checks:
            result = model_fn_minimax_h3(dit=pipe.dit,
                video_latents=gpu_state['video']['latents_before'],
                audio_latents=gpu_state['audio']['latents_before'], packed=packed, prompt_embeds=embedding,
                timestep_video=gpu_state['video']['timestep'].reshape(1),
                timestep_audio=gpu_state['audio']['timestep'].reshape(1))
    finally:
        handle.remove()
    after = audit.row()
    counts = {k: after[k]-before[k] for k in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')}
    if counts != {'sdpa_calls': expected_sdpa, 'scaled_mm_calls': 200 if arm == 'native' else 0, 'disk_loads': 0}:
        raise RuntimeError(f'Unexpected actual execution path: {counts}')
    if len(captures) != 1 or (arm == 'native' and checks.summary['checked_calls'] != 200):
        raise RuntimeError('Incomplete actual call/quantization guard capture')
    velocities = {m: value.detach().cpu() for m, value in zip(('video', 'audio'), result, strict=True)}
    if any(not bool(torch.isfinite(v).all()) for v in velocities.values()):
        raise RuntimeError('Nonfinite conditional velocity')
    path = directory/(call_id+'.pt')
    torch.save({'id': call_id, 'velocities': velocities, 'input': info['input'],
                'packed': info['packed'], 'actual_dit': captures[0]}, path)
    info.update(status='complete', output={m: tensor_record(v) for m, v in velocities.items()},
                artifact=file_record(path), actual_dit=captures[0], runtime_audit=counts,
                zero_sf_checks=checks.summary if checks is not None else None,
                seconds_including_diagnostics=time.monotonic()-started)
    save(report, output)
    inherited.memory_guard()
    return velocities


def perturb_states(original):
    variants = {'center': original, 'plus': {}, 'minus': {}}
    records = {}
    for name, source in original.items():
        x = source['latents_before']
        if x.dtype != torch.bfloat16 or x.device.type != 'cpu' or not bool(torch.isfinite(x).all()):
            raise RuntimeError('1-ULP control requires finite CPU BF16 latent')
        destination = torch.empty_like(x).reshape(-1)
        destination[0::2], destination[1::2] = float('inf'), float('-inf')
        records[name] = {'original': tensor_record(x), 'perturbations': {}}
        for tag, target in (('plus', destination), ('minus', -destination)):
            changed = torch.nextafter(x, target.reshape(x.shape))
            if not bool(torch.isfinite(changed).all()):
                raise RuntimeError('Nonfinite 1-ULP perturbation')
            variants[tag][name] = {**source, 'latents_before': changed}
            delta = cpu64(changed)-cpu64(x)
            e = float(torch.dot(delta, delta))
            records[name]['perturbations'][tag] = {**tensor_record(changed),
                'changed_elements': int((changed != x).sum()), 'elements': x.numel(),
                'delta_energy': e, 'relative_delta_energy': e/energy(x)}
    return variants, records


def teacher_statistics(manifest, outputs):
    original, target = manifest['original_condition_id'], manifest['target_condition_id']
    candidates = manifest['control_candidate_ids']
    minimum = manifest['teacher_gate']['min_response_to_difference_background_energy']
    stats = {}
    for step in manifest['steps']:
        step_stats = {}
        for modality in ('video', 'audio'):
            base = cpu64(outputs[(step, original, 'center')][modality])
            repeated = cpu64(outputs[(step, original, 'repeat')][modality])
            repeat_energy = energy(repeated-base)
            edits = {}
            for cid in [target, *candidates]:
                b = cpu64(outputs[(step, cid, 'center')][modality])
                d = b-base
                ed = energy(d)
                backgrounds, single = {}, {}
                for variant in ('plus', 'minus'):
                    changed_base = cpu64(outputs[(step, original, variant)][modality])
                    changed_b = cpu64(outputs[(step, cid, variant)][modality])
                    backgrounds[variant] = energy((changed_b-changed_base)-d)
                    single[variant] = {'original': energy(changed_base-base), 'edited': energy(changed_b-b)}
                background = max(*backgrounds.values(), 4*repeat_energy)
                edits[cid] = {'response_energy': ed, 'difference_perturbation_energy': backgrounds,
                    'single_condition_perturbation_energy': single, 'repeat_energy_times4': 4*repeat_energy,
                    'background_energy': background, 'response_to_background_energy':
                        ed/background if background > 0 else ('infinite' if ed > 0 else None),
                    'snr_pass': ed > 0 and ed >= minimum*background}
            step_stats[modality] = {'original_repeat_energy': repeat_energy, 'edits': edits}
        stats[str(step)] = step_stats
    primary = manifest['primary_modality']
    target_pass = all(stats[str(s)][primary]['edits'][target]['snr_pass'] for s in manifest['steps'])
    low, high = manifest['teacher_gate']['control_teacher_norm_ratio_range']
    eligibility = []
    selected = None
    for cid in candidates:
        rows = []
        for step in manifest['steps']:
            edits = stats[str(step)][primary]['edits']
            ratio = (edits[cid]['response_energy']/edits[target]['response_energy'])**.5 if edits[target]['response_energy'] > 0 else None
            rows.append({'step': step, 'norm_ratio': ratio, 'snr_pass': edits[cid]['snr_pass'],
                         'norm_match': ratio is not None and low <= ratio <= high})
        eligible = target_pass and all(r['snr_pass'] and r['norm_match'] for r in rows)
        eligibility.append({'condition_id': cid, 'steps': rows, 'eligible': eligible})
        if eligible and selected is None: selected = cid
    return {'statistics': stats, 'primary_modality': primary, 'target_stable': target_pass,
        'control_eligibility_fixed_order': eligibility, 'selected_control_id': selected,
        'pass': target_pass and selected is not None,
        'status': 'teacher_gate_passed' if target_pass and selected is not None else 'stopped_teacher_gate'}


def load_raw_outputs(stage_report):
    result = {}
    for row in stage_report['calls']:
        if row.get('status') != 'complete': raise RuntimeError('Incomplete teacher call')
        artifact = row['artifact']
        if sha256(artifact['file']) != artifact['sha256']: raise RuntimeError('Raw velocity artifact changed')
        payload = torch.load(artifact['file'], map_location='cpu', weights_only=True, mmap=True)
        for name, velocity in payload['velocities'].items():
            if tensor_sha(velocity) != row['output'][name]['sha256']:
                raise RuntimeError('Velocity tensor SHA changed')
        result[(row['step'], row['condition_id'], row['control_id'])] = payload['velocities']
    return result


def native_statistics(manifest, teacher, quantized, selected):
    original, target = manifest['original_condition_id'], manifest['target_condition_id']
    thresholds, stats, passing = manifest['candidate_gate'], {}, []
    for step in manifest['steps']:
        row = {}
        for modality in ('video', 'audio'):
            endpoint = {}
            for cid in (original, target, selected):
                b, q = teacher[(step, cid, 'center')][modality], quantized[(step, cid, 'center')][modality]
                denominator = energy(b)
                error = energy(cpu64(q)-cpu64(b))
                endpoint[cid] = {'bf16_energy': denominator, 'native_energy': energy(q),
                    'error_energy': error, 'nmse': error/denominator if denominator > 0 else None}
            pairs = {cid: response_statistics(teacher[(step, original, 'center')][modality],
                teacher[(step, cid, 'center')][modality], quantized[(step, original, 'center')][modality],
                quantized[(step, cid, 'center')][modality]) for cid in (target, selected)}
            row[modality] = {'endpoints': endpoint, 'responses': pairs}
        primary = row[manifest['primary_modality']]
        t, c = primary['responses'][target], primary['responses'][selected]
        checks = {'target_gain': t['gain'] is not None and t['gain'] < thresholds['target_gain_lt'],
            'target_energy_ratio': t['norm_ratio_squared'] is not None and t['norm_ratio_squared'] < thresholds['target_energy_ratio_lt'],
            'control_gain': c['gain'] is not None and thresholds['control_gain_range'][0] <= c['gain'] <= thresholds['control_gain_range'][1],
            'control_difference_nmse': c['response_nmse'] is not None and c['response_nmse'] <= thresholds['control_difference_nmse_le'],
            'per_condition_nmse': all(v['nmse'] is not None and v['nmse'] <= thresholds['all_per_condition_video_nmse_le'] for v in primary['endpoints'].values())}
        row['candidate_gate_checks'] = checks
        passing.append(all(checks.values()))
        stats[str(step)] = row
    return {'steps': stats, 'decision': 'candidate_requires_replication' if all(passing) else 'stop_current_selective_attenuation_route'}


@torch.inference_mode()
def run_model_stage(args, manifest, report):
    arm = args.phase
    conditions = load_prepared(args, report)
    teacher_report, teacher_outputs, selection = None, None, None
    if arm == 'native':
        teacher_path = report_path(args, 'bf16')
        teacher_report = inherited.load_complete(teacher_path)
        check_sources(teacher_report['sources'])
        if not teacher_report['teacher_gate']['pass']:
            raise RuntimeError('Teacher gate did not pass; native is forbidden')
        selection_ref = teacher_report['selection']
        if sha256(selection_ref['file']) != selection_ref['sha256']:
            raise RuntimeError('Frozen teacher selection changed')
        selection = json.loads(Path(selection_ref['file']).read_text())
        if selection['teacher_gate'] != teacher_report['teacher_gate']:
            raise RuntimeError('Selection and teacher report disagree')
        teacher_outputs = load_raw_outputs(teacher_report)
        if teacher_statistics(manifest, teacher_outputs) != selection['teacher_gate']:
            raise RuntimeError('Independent reload did not reproduce teacher gate')
        report['teacher_reference'] = file_record(teacher_path)
        report['selection'] = selection_ref
        report['teacher_gate'] = selection['teacher_gate']
    directory = start_stage(args, report, arm)
    pipe = inherited.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    if pipe.model_fn is not model_fn_minimax_h3:
        raise RuntimeError('Original model_fn identity changed')
    report['resident_conversion'] = inherited.make_h3_resident(pipe.dit)
    if arm == 'native':
        report['native_installation'] = inherited.install_native_h3(pipe.dit, args.export_dir,
            activation_packer=inherited.pack_activation_fast, chunk_rows=1024)
    gc.collect()
    torch.cuda.empty_cache()
    outputs, report['states'] = {}, []
    original = manifest['original_condition_id']
    selected_ids = [c['id'] for c in manifest['conditions']] if arm == 'bf16' else [
        original, manifest['target_condition_id'], report['teacher_gate']['selected_control_id']]
    with inherited.RuntimeAudit().installed() as audit:
        for step in manifest['steps']:
            state, source_files = load_teacher_state(report, step)
            variants, perturbation = perturb_states(state)
            input_path = directory/f's{step:02d}_inputs.pt'
            torch.save({'variants': variants, 'perturbation': perturbation, 'source_files': source_files}, input_path)
            report['states'].append({'step': step, 'artifact': file_record(input_path),
                'perturbations': perturbation, 'source_files': source_files})
            variant_ids = manifest['perturbation']['variants'] if arm == 'bf16' else ['center']
            for variant in variant_ids:
                pair_signature = None
                for cid in selected_ids:
                    values = evaluate_call(pipe, audit, arm, step, cid, variant, conditions[cid], variants[variant],
                                           directory, report, args.output)
                    call = report['calls'][-1]
                    signature = {k: call[k] for k in ('packed', 'input', 'actual_dit')}
                    if pair_signature is None: pair_signature = signature
                    elif signature != pair_signature:
                        raise RuntimeError('Conditions changed actual x/t/packed/RoPE metadata')
                    if arm == 'bf16' and cid == original and variant == 'center':
                        call['e010_velocity_exact'] = {m: tensor_sha(values[m]) == tensor_sha(state[m]['noise_pred']) for m in values}
                        if not all(call['e010_velocity_exact'].values()):
                            raise RuntimeError('Original BF16 velocity differs from E010; no teacher-state substitution')
                    if arm == 'native':
                        teacher_call = next(c for c in teacher_report['calls'] if c['step'] == step and c['condition_id'] == cid and c['control_id'] == variant)
                        if any(call[k] != teacher_call[k] for k in ('packed', 'input', 'actual_dit', 'embedding')):
                            raise RuntimeError('Native and teacher did not consume identical actual inputs')
                    outputs[(step, cid, variant)] = values
                    save(report, args.output)
            if arm == 'bf16':
                values = evaluate_call(pipe, audit, arm, step, original, 'repeat', conditions[original], variants['center'],
                                       directory, report, args.output)
                outputs[(step, original, 'repeat')] = values
                original_call = next(c for c in report['calls'] if c['step'] == step and c['condition_id'] == original and c['control_id'] == 'center')
                if any(report['calls'][-1][k] != original_call[k] for k in ('packed', 'input', 'actual_dit', 'embedding')):
                    raise RuntimeError('Repeat input differs from original center')
    expected_calls = manifest['budgets']['bf16_calls'] if arm == 'bf16' else manifest['budgets']['native_calls_max']
    if len(report['calls']) != expected_calls:
        raise RuntimeError(f'Unexpected forward count: {len(report["calls"])} != {expected_calls}')
    if arm == 'bf16':
        gate = teacher_statistics(manifest, outputs)
        selection_path = args.data_dir/'teacher_selection.json'
        if selection_path.exists(): raise FileExistsError('Never replace a teacher selection after observing native')
        frozen = {'experiment': 'E012', 'teacher_gate': gate, 'manifest_file': report['manifest_file'],
            'plan': report['plan'], 'sources': report['sources'],
            'raw_teacher_outputs': [c['artifact'] for c in report['calls']]}
        save(frozen, selection_path)
        report['selection'] = file_record(selection_path)
        report['teacher_gate'] = gate
        report['status'] = 'complete' if gate['pass'] else 'stopped_teacher_gate'
        print(json.dumps({'teacher_gate': gate['status'], 'selected_control_id': gate['selected_control_id']}), flush=True)
    else:
        report['statistics'] = native_statistics(manifest, teacher_outputs, outputs, report['teacher_gate']['selected_control_id'])
        report['decision'] = report['statistics']['decision']
        report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--phase', choices=('prepare', 'bf16', 'native'))
    parser.add_argument('--manifest', type=Path, default=ROOT/'research_state/06_experiments/E012_h3_conditional_response_manifest.json')
    parser.add_argument('--plan', type=Path, default=ROOT/'research_state/06_experiments/E012_h3_conditional_response_plan.md')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/research')
    parser.add_argument('--export-dir', type=Path, default=Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export'))
    args = parser.parse_args()
    if args.check_only: args.phase = 'check'
    args.output = report_path(args, args.phase)
    if args.output.exists(): raise FileExistsError(f'Preserve existing report: {args.output}')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'experiment': 'E012', 'phase': args.phase, 'status': 'running',
              'scope': 'Fixed-state numeric diagnostic; no semantic-quality or novel-loss claim'}
    started = time.monotonic()
    try:
        manifest, _reference = prerequisites(args, report)
        if args.check_only:
            cpu_check(manifest, report)
            report['status'] = 'complete'
        else:
            check_path = report_path(args, 'check')
            checked = inherited.load_complete(check_path)
            check_sources(checked['sources'])
            if checked['cuda_initialized'] is not False: raise RuntimeError('CPU-only preflight missing')
            report['cpu_check_reference'] = file_record(check_path)
            report['cpu_conditions'] = checked['cpu_conditions']
            report['same_packed_layout'] = checked['same_packed_layout']
            if args.phase == 'prepare': prepare(args, manifest, report)
            else: run_model_stage(args, manifest, report)
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        if torch.cuda.is_initialized(): report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        print(json.dumps({'status': report['status'], 'report': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
