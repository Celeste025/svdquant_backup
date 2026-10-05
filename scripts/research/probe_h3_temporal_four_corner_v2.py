#!/usr/bin/env python3
"""E057 v2: reuse one successful replay; evaluate the four frozen QQ states.

Temporal QB here means quantized step n followed by BF16 step n+1; E015's
prepared QQ instead denotes video/audio input coordinates. No native forwards,
training, decoding, or research statistics are performed by this runner.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe_h3_plain_baseline as base
import probe_h3_crossmodal_propagation as old
import prepare_h3_crossmodal_states as builder
import torch

DATA = Path('/data1/models/svdquant-wjq/research/20261004/E057')
REPORTS = ROOT/'results/research/E057'
PLAN = ROOT/'research_state/06_experiments/E057_h3_temporal_four_corner_plan.md'
MODALITIES = ('video', 'audio')
CASE_IDS = ('e010_p030_s05', 'e010_p036_s14')
MAX_CALLS = 5
MAX_NEW_CALLS = 4
ORIGINAL_DEADLINE = 1791054484.3786852
save, file_record, tensor_record = base.save, base.file_record, base.tensor_record
signature = base.tree_signature
require = old.require


def key(item):
    return f"{item['prepared']['id']}/{item['source_arm']}"


def prerequisites(report):
    require(PLAN.is_file(), 'E057 plan must exist before check/evaluate')
    binding = {}
    old_args = types.SimpleNamespace(manifest=old.MANIFEST, plan=old.PLAN,
        data_dir=old.DATA, prepare_report=old.REPORTS/'E015_prepare.json')
    _, manifest, prepared, e014_manifest = old.prerequisites(old_args, binding)
    old_check = old.load_complete(old.REPORTS/'check.json')
    require(old_check['sources'] == binding['sources'] and
            old_check['prepared_reference'] == binding['prepared_reference'] and
            old_check['cuda_initialized'] is False, 'E015 CPU/source binding changed')
    evaluations = {}
    for arm, count in (('bf16', 2), ('plain', 8), ('svd', 8)):
        path = old.REPORTS/f'evaluate_{arm}.json'
        value = old.load_complete(path)
        require(value['arm'] == arm and value['sources'] == binding['sources'] and
                value['prepared_reference'] == binding['prepared_reference'] and
                value['complete_dit_calls'] == count, 'E015 evaluation binding changed')
        evaluations[arm] = value
    selected = [(CASE_IDS[0], 'bf16', 'BB')]
    selected += [(cid, arm, 'QQ') for cid in CASE_IDS for arm in ('plain', 'svd')]
    items = []
    for cid, arm, corner in selected:
        row = next(r for r in prepared['cases'] if (r['id'], r['arm'], r['corner']) == (cid, arm, corner))
        history = next(r for r in evaluations[arm]['cases'] if (r['id'], r['corner']) == (cid, corner))
        require(history['prepared_artifact'] == row['artifact'] and
                history['next_step'] == row['next_step'], 'E015 actual call used a different prepared record')
        base.verify_file(history['artifact'])
        items.append({'source_arm': arm, 'prepared': row, 'history': history,
                      'case': next(c for c in manifest['cases'] if c['id'] == cid)})
    sources = dict(binding['sources'])
    sources.update({str(p.resolve()): file_record(p) for p in (Path(__file__), PLAN)})
    report.update(sources=sources, plan=file_record(PLAN), settings=manifest['settings'],
        asset_binding=binding['asset_binding'], e015_manifest=binding['manifest'],
        e015_prepare=binding['prepared_reference'], e015_check=file_record(old.REPORTS/'check.json'),
        e015_evaluations={arm: file_record(old.REPORTS/f'evaluate_{arm}.json') for arm in evaluations},
        e014_check=manifest['e014_check'])
    previous_path = REPORTS/'evaluate.json'
    previous = json.loads(previous_path.read_text())
    require(previous['status'] == 'failed_stop' and previous['complete_dit_calls'] == 1
            and previous['attempted_dit_calls'] == 2 and previous['cases'] == []
            and previous['deadline_unix'] == ORIGINAL_DEADLINE,
            'Unexpected original E057 failure/count/deadline contract')
    require('Actual BF16 DiT input differs' in previous['error'], 'Original failure is not the input prehook guard')
    base.check_sources(previous['sources'])
    require(previous['plan'] == report['plan'] and previous['e015_evaluations'] == report['e015_evaluations'],
            'Original replay provenance differs from current inputs')
    replay = previous['teacher_replay']
    require(all(flag for group in replay['historical_bf16_replay'].values() for flag in group.values()),
            'Historical teacher replay did not pass every comparison')
    artifact = torch.load(base.verify_file(replay['artifact']), map_location='cpu', weights_only=True, mmap=True)
    require(signature(artifact['actual_dit_inputs']) == replay['actual_dit_input_signature'], 'Replay input hash changed')
    for field in ('raw_outputs', 'velocities', 'next_endpoints'):
        require(signature(artifact[field]) == replay[field], f'Replay {field} hashes changed')
    report['sources'].update(previous['sources'])
    report.update(prior_failed_report=file_record(previous_path), prior_teacher_replay=replay,
        prior_complete_dit_calls=1, prior_attempted_dit_calls=2,
        prior_failure_scope='Second attempt stopped in input prehook before the model forward; no new control completed',
        revision='v2: blocking GPU-to-CPU input snapshot before hashing; four remaining BF16 controls only')
    return manifest, e014_manifest, items


def input_signature(value, item, pipe):
    return old.validate_value(value, item['prepared'], item['case'], pipe, builder)


def capture_cpu_inputs(pipe, value):
    """Run only original input packing; stop at a non-model DiT sentinel."""
    captured = []
    class PackedInputCaptured(Exception):
        pass
    def capture(*positional, **keyword):
        captured.append(signature({'args': positional, 'kwargs': keyword}))
        raise PackedInputCaptured
    selected = builder.model_value(value)
    state = selected['state']
    packed = base.make_packed(pipe, selected['embedding'], selected['text_token_tags'], state)
    try:
        base.model_fn_minimax_h3(dit=capture,
            video_latents=state['video']['latents_before'], audio_latents=state['audio']['latents_before'],
            packed=packed, prompt_embeds=selected['embedding'],
            timestep_video=state['video']['timestep'].reshape(1),
            timestep_audio=state['audio']['timestep'].reshape(1))
    except PackedInputCaptured:
        pass
    require(len(captured) == 1, 'CPU packing must reach exactly one sentinel, with no model forward')
    return captured[0]


@torch.inference_mode()
def cpu_check(report, manifest, items):
    require(not torch.cuda.is_initialized(), 'CPU check already initialized CUDA')
    torch.set_num_threads(6)
    pipe = builder.make_cpu_pipeline(manifest['settings'])
    report['inputs'] = {}
    for item in items:
        value = builder.load_prepared_case(item['prepared'])
        validated = input_signature(value, item, pipe)
        history = item['history']
        require(validated == history['input_signature'], 'Prepared input differs from E015 evaluation')
        payload = torch.load(base.verify_file(history['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(payload['input_signature'] == validated, 'Historical artifact input binding changed')
        expected = signature(payload['actual_dit_inputs'])
        require(expected == history['actual_dit_input_signature'], 'Historical actual-input tensor hashes changed')
        actual = capture_cpu_inputs(pipe, value)
        require(actual == expected, 'Prepared state does not reproduce E015 actual DiT inputs')
        report['inputs'][key(item)] = {'prepared_artifact': item['prepared']['artifact'],
            'historical_artifact': history['artifact'], 'input_signature': validated,
            'actual_dit_input_signature': actual, 'byte_exact_to_E015_actual_input': True}
        print(f'E057 CPU input valid: {key(item)}', flush=True)
        del payload, value
    require(not torch.cuda.is_initialized(), 'CPU packing unexpectedly initialized CUDA')
    report.update(status='complete', cuda_initialized=False, prepared_QQ_rows=4,
                  teacher_replay_rows=1, complete_dit_calls=0)


def signature_differences(actual, expected, prefix=''):
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = []
        for name in sorted(set(actual) | set(expected)):
            path = f'{prefix}.{name}' if prefix else name
            if name not in actual or name not in expected:
                differences.append({'path': path, 'actual': actual.get(name), 'expected': expected.get(name)})
            else:
                differences.extend(signature_differences(actual[name], expected[name], path))
        return differences
    return [] if actual == expected else [{'path': prefix, 'actual': actual, 'expected': expected}]


def guard(args, report, *, before_forward=False):
    require(time.time() < args.deadline_unix, 'E057 absolute deadline reached')
    if before_forward:
        require(report['attempted_dit_calls'] < MAX_NEW_CALLS, 'Four remaining-forward budget exhausted')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E057 allocated peak reached 60 GiB')


@torch.inference_mode()
def evaluate(args, report, manifest, e014_manifest, items):
    checked = old.load_complete(REPORTS/'check_v2.json')
    require(checked['sources'] == report['sources'] and checked['cuda_initialized'] is False and
            checked['e015_evaluations'] == report['e015_evaluations'] and
            checked['e015_prepare'] == report['e015_prepare'], 'E057 checked provenance changed')
    report['cpu_check_reference'] = file_record(REPORTS/'check_v2.json')
    require(checked['prior_failed_report'] == report['prior_failed_report'], 'Prior replay/failure binding changed')
    report.update(complete_dit_calls=1, new_complete_dit_calls=0)
    setup_path = DATA/'setup_bf16_v2.json'
    require(not setup_path.exists(), f'Refusing existing setup {setup_path}')
    old_check = old.load_complete(manifest['e014_check']['file'])
    setup = {'experiment': 'E057_inherited_E014_setup', 'status': 'running',
             'complete_dit_calls': 0, 'sources': old_check['sources']}
    setup_args = types.SimpleNamespace(arm='bf16', deadline_unix=args.deadline_unix,
        check_report=Path(manifest['e014_check']['file']), output=setup_path)
    guard(args, report)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    report['resident_model_storage'] = setup['resident_model_storage']
    builder.configure_schedule(pipe, manifest['settings'])
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    report['cases'] = []
    for item in items[1:]:
        guard(args, report, before_forward=True)
        value = builder.load_prepared_case(item['prepared'])
        checked_input = checked['inputs'][key(item)]
        validated = input_signature(value, item, cpu_pipe)
        require(validated == checked_input['input_signature'], 'Prepared input changed after CPU check')
        replay = item['source_arm'] == 'bf16'
        require(not replay, 'v2 must not rerun the successful teacher replay')
        output_path = DATA/'evaluate_v2'/('teacher_replay.pt' if replay else
            f"{item['prepared']['id']}_{item['source_arm']}_QB.pt")
        require(not output_path.exists(), f'Refusing existing tensor artifact {output_path}')
        forward = base.gpu_call(pipe, {'kind': 'model_fn'}, builder.model_value(value))
        actual, raw_capture = [], []
        def capture_input(_module, positional, keyword):
            # tree_device(..., 'cpu') is non_blocking=True: hashing it immediately races D2H.
            captured = base.inherited.tree_cpu({'args': positional, 'kwargs': keyword})
            actual_signature = signature(captured)
            expected_signature = checked_input['actual_dit_input_signature']
            if actual_signature != expected_signature:
                report['failed_input'] = {'id': item['prepared']['id'], 'source_arm': item['source_arm'],
                    'actual_signature': actual_signature, 'expected_signature': expected_signature,
                    'differences': signature_differences(actual_signature, expected_signature)}
                save(report, args.output)
                raise RuntimeError('Blocking input snapshot differs from historical E015 actual input')
            actual.append(captured)
        def capture_output(_module, positional, output):
            raw_capture.append(output)
        pre = pipe.dit.register_forward_pre_hook(capture_input, with_kwargs=True)
        post = pipe.dit.register_forward_hook(capture_output)
        audit = base.RuntimeAudit()
        audit.phase = 'resident_bf16'
        report['attempted_dit_calls'] += 1
        save(report, args.output)
        print(f'E057 BF16 {key(item)} step={item["prepared"]["next_step"]}', flush=True)
        try:
            with audit.installed():
                result = forward()
                torch.cuda.synchronize()
            report['complete_dit_calls'] += 1
            report['new_complete_dit_calls'] += 1
        finally:
            pre.remove()
            post.remove()
        base.audit_contract(audit.row(), 'bf16')
        require(len(actual) == len(raw_capture) == 1, 'Expected exactly one BF16 DiT invocation')
        guard(args, report)
        raw, raw_records = base.cpu_outputs(raw_capture[0])
        velocities, velocity_records = base.cpu_outputs(result)
        gpu_state = base.tree_device(value['state'], 'cuda')
        updated = builder.advance(pipe, gpu_state, dict(zip(MODALITIES, result, strict=True)),
                                  item['prepared']['next_step'])
        torch.cuda.synchronize()
        endpoints = {m: updated[m].detach().cpu() for m in MODALITIES}
        endpoint_records = {m: tensor_record(x) for m, x in endpoints.items()}
        replay_checks = None
        if replay:
            replay_checks = {'raw_exact': {m: raw_records[m]['sha256'] == item['case']['teacher_next_raw_output_sha256'][m] for m in MODALITIES},
                'velocity_exact': {m: velocity_records[m] == tensor_record(value['teacher_reference'][m]['noise_pred']) for m in MODALITIES},
                'endpoint_exact': {m: endpoint_records[m] == tensor_record(value['teacher_reference'][m]['latents_after']) for m in MODALITIES}}
            require(all(flag for group in replay_checks.values() for flag in group.values()), 'Historical BF16 replay failed; stop before new controls')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'id': item['prepared']['id'], 'source_arm': item['source_arm'], 'model_arm': 'bf16',
            'temporal_corner': 'BB' if replay else 'QB', 'prepared_modal_corner': item['prepared']['corner'],
            'source_step': item['case']['source_step'], 'next_step': item['prepared']['next_step'],
            'prepared_artifact': item['prepared']['artifact'], 'historical_artifact': item['history']['artifact'],
            'input_signature': validated, 'actual_dit_inputs': actual[0], 'raw_outputs': raw,
            'velocities': velocities, 'next_endpoints': endpoints}
        with output_path.open('xb') as stream:
            torch.save(payload, stream)
        row = {k: payload[k] for k in ('id', 'source_arm', 'model_arm', 'temporal_corner',
            'prepared_modal_corner', 'source_step', 'next_step', 'prepared_artifact', 'historical_artifact', 'input_signature')}
        row.update(status='complete', actual_dit_input_signature=signature(actual[0]),
            raw_outputs=raw_records, velocities=velocity_records, next_endpoints=endpoint_records,
            runtime_audit=audit.row(), artifact=file_record(output_path))
        if replay:
            row['historical_bf16_replay'] = replay_checks
            report['teacher_replay'] = row
        else:
            report['cases'].append(row)
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save(report, args.output)
        del forward, value, result, raw_capture, actual, raw, velocities, gpu_state, updated, endpoints, payload
        gc.collect()
        guard(args, report)
    require(report['attempted_dit_calls'] == report['new_complete_dit_calls'] == MAX_NEW_CALLS
            and report['complete_dit_calls'] == MAX_CALLS and len(report['cases']) == 4,
            'Incomplete E057 cumulative/remaining call allocation')
    report.update(status='complete', peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate'), required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or REPORTS/f'{args.phase}_v2.json'
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    started = time.time()
    if args.phase == 'evaluate':
        require(args.deadline_unix is not None, 'Evaluate requires root-assigned absolute --deadline-unix')
        require(args.deadline_unix == ORIGINAL_DEADLINE, 'v2 must retain the original absolute deadline')
        require(args.deadline_unix > started, 'Expired E057 deadline')
    report = {'experiment': 'E057', 'phase': args.phase, 'status': 'running',
        'start_epoch': started, 'deadline_unix': args.deadline_unix, 'wall_budget_seconds': 1200,
        'max_dit_calls': MAX_CALLS, 'max_new_dit_calls': MAX_NEW_CALLS,
        'attempted_dit_calls': 0, 'complete_dit_calls': 0, 'new_complete_dit_calls': 0,
        'scope': 'Four temporal QB controls and one historical BF16 replay; no quality or performance claim'}
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E057 1200-second absolute wall deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, e014_manifest, items = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, manifest, items)
        else:
            evaluate(args, report, manifest, e014_manifest, items)
        report['elapsed_seconds'] = time.time()-started
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', elapsed_seconds=time.time()-started, error=traceback.format_exc())
        save(report, args.output)
        raise
    finally:
        if args.phase == 'evaluate':
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
