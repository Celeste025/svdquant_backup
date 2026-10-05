#!/usr/bin/env python3
"""E061: two fixed-smooth LR initialization targets, native full-H3 evaluation.

Baseline investment screen only. No recalibration, timestep rollout or decoder.
The one absolute deadline covers both streaming exports and at most 13 calls.
"""
from __future__ import annotations
import argparse
import gc
import inspect
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
import probe_h3_activation_residual_oracle as inherited
import probe_h3_plain_baseline as base
import probe_h3_crossmodal_propagation as old
import prepare_h3_crossmodal_states as builder
import export_h3_native_nvfp4 as exporter
from h3_native_nvfp4 import NativeH3Linear, TARGET_NAMES, FORMAT, RECIPE, pack_activation_legacy
from minimax_h3_svdquant_common import H3_DIT_PATH, LowRankBranch, nvfp4_qdq
from safetensors import safe_open
import torch

DATA = Path('/data1/models/svdquant-wjq/research/20261004/E061')
REPORTS = ROOT/'results/research/E061'
PLAN = ROOT/'research_state/06_experiments/E061_h3_lowrank_initialization_plan.md'
AUDIT = ROOT/'results/research/E060/audit.json'
ARMS = ('error_init', 'weight_init')
MAX_CALLS = 13
MAX_EXPORT_BYTES = 26*1024**3
CHUNK_ROWS = 1024
save, file_record, signature = base.save, base.file_record, base.tree_signature
require = old.require
key = inherited.key


def reference(row, experiment):
    return dict(artifact=row['artifact'], experiment=experiment,
        actual_input_signature=row['actual_dit_inputs'] if experiment == 'E014' else row['actual_dit_input_signature'],
        raw_outputs=row['raw_outputs'], velocities=row['velocities'])


def prerequisites(report):
    require(PLAN.is_file(), 'E061 plan must exist before check/evaluate')
    manifest, e014_manifest, items = inherited.prerequisites(report)
    e059_check = old.load_complete(inherited.REPORTS/'check.json')
    require(e059_check['sources'] == report['sources'] and e059_check['cuda_initialized'] is False,
            'E059 six-input CPU/source contract changed')
    zeros = json.loads((inherited.REPORTS/'evaluate.json').read_text())
    require(zeros['sources'] == report['sources'] and zeros['complete_by_mode']['zero'] == 6,
            'E059 zero provenance/count changed')
    zero_rows = [r for r in zeros['cases'] if r['mode'] == 'zero']
    require(len(zero_rows) == 6 and all(r['historical_replay_exact'] for r in zero_rows), 'Six E059 exact replays required')
    audit = old.load_complete(AUDIT)
    require(audit['cuda_initialized'] is False and audit['new_model_forwards'] == 0 and
            audit['verification_scope']['matched_layers'] == 200 and
            audit['verification_scope']['matched_factor_fields'] == 600 and
            audit['state_binding']['full_state_hash_verified'], 'E060 complete factor/source audit required')
    base.check_sources(audit['source_audit']['current_source_records'])
    legacy_path = Path(e014_manifest['svd_export_dir'])/'manifest.json'
    require(file_record(legacy_path) == audit['manifest'], 'E060 legacy manifest binding changed')
    legacy = old.load_complete(legacy_path)
    state_record = audit['state_binding']['observed']
    base.verify_file(state_record)
    require(legacy['sources'][state_record['file']]['sha256'] == state_record['sha256'], 'Legacy state hash differs')
    reports = dict(e014_bf16=old.load_complete(base.verify_file(manifest['e014_evaluations']['bf16'])),
        e014_plain=old.load_complete(base.verify_file(manifest['e014_evaluations']['plain'])),
        e015_bf16=old.load_complete(old.REPORTS/'evaluate_bf16.json'),
        e015_plain=old.load_complete(old.REPORTS/'evaluate_plain.json'),
        e057=old.load_complete(ROOT/'results/research/E057/evaluate_v2.json'))
    refs = {}
    for item in items:
        cid, role = item['id'], item['input_role']
        zero = next(r for r in zero_rows if (r['case_id'], r['position']) == (cid, role))
        require(zero['historical_artifact'] == item['history']['artifact'], 'E059 zero history differs')
        entry = {'legacy_selected': reference(zero, 'E059')}
        if role == 'source_teacher':
            for arm in ('bf16', 'plain'):
                entry[arm] = reference(next(r for r in reports['e014_'+arm]['cases'] if r['id'] == cid), 'E014')
        elif role == 'next_teacher':
            for arm in ('bf16', 'plain'):
                entry[arm] = reference(next(r for r in reports['e015_'+arm]['cases'] if
                    (r['id'], r['corner']) == (cid, 'BB')), 'E015')
        else:
            entry['bf16'] = reference(next(r for r in reports['e057']['cases'] if
                (r['id'], r['source_arm']) == (cid, 'svd')), 'E057')
            entry['plain'] = None  # E015 plain-QQ is a different input.
        for value in entry.values():
            if value is not None:
                base.verify_file(value['artifact'])
        refs[key(item)] = entry
    paths = (Path(__file__), PLAN, HERE/'export_h3_native_nvfp4.py',
             Path(inspect.getfile(LowRankBranch)), AUDIT, Path(audit['script']['file']))
    report['sources'].update({str(p.resolve()): file_record(p) for p in paths})
    report.update(plan=file_record(PLAN), e060_audit=file_record(AUDIT), legacy_manifest=file_record(legacy_path),
        state=state_record, e059_check=file_record(inherited.REPORTS/'check.json'),
        e059_evaluation=file_record(inherited.REPORTS/'evaluate.json'), references=refs,
        reference_reports={
            'e014_bf16': manifest['e014_evaluations']['bf16'], 'e014_plain': manifest['e014_evaluations']['plain'],
            'e015_bf16': file_record(old.REPORTS/'evaluate_bf16.json'),
            'e015_plain': file_record(old.REPORTS/'evaluate_plain.json'),
            'e057': file_record(ROOT/'results/research/E057/evaluate_v2.json')})
    return manifest, e014_manifest, items, legacy


def load_reference(record, expected):
    payload = torch.load(base.verify_file(record['artifact']), map_location='cpu', weights_only=True, mmap=True)
    actual = payload['actual_dit_inputs'] if record['experiment'] == 'E014' else signature(payload['actual_dit_inputs'])
    require(actual == record['actual_input_signature'] == expected, 'Reference is not the same actual input')
    require(signature(payload['raw_outputs']) == record['raw_outputs'] and
            signature(payload['velocities']) == record['velocities'], 'Reference output hashes changed')
    return payload


def output_metrics(velocities, teacher):
    result = {}
    for modality in inherited.MODALITIES:
        ref = teacher[modality].double()
        error = velocities[modality].double()-ref
        require(bool(torch.isfinite(error).all()), 'Nonfinite velocity error')
        centered = error-error.mean(tuple(range(2, error.ndim)), keepdim=True)
        energy = float(ref.square().sum())
        sse = float(error.square().sum())
        result[modality] = dict(sse=sse, reference_energy=energy, nmse=sse/max(energy, 1e-30),
            channel_centered_sse=float(centered.square().sum()), elements=error.numel())
    return result


@torch.inference_mode()
def cpu_check(report, manifest, items, legacy):
    inherited.cpu_check(report, manifest, items)
    expected = old.load_complete(inherited.REPORTS/'check.json')['inputs']
    require(report['inputs'] == expected, 'E061 six-input validation differs from E059')
    report['reference_metrics'] = {}
    for item in items:
        k = key(item)
        actual = report['inputs'][k]['actual_dit_input_signature']
        refs = {arm: load_reference(rec, actual) for arm, rec in report['references'][k].items() if rec is not None}
        report['reference_metrics'][k] = {arm: output_metrics(payload['velocities'], refs['bf16']['velocities'])
            for arm, payload in refs.items() if arm != 'bf16'}
    with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as source:
        keys = set(source.keys())
        for row in legacy['layers']:
            require(source.get_slice(row['name']+'.weight').get_shape() == row['shape'], 'Original weight shape changed')
            require(row['name']+'.bias' not in keys and row['bias'] is None, 'Main target bias contract changed')
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, checked_weight_headers=200, complete_dit_calls=0)


def guard(args, report, *, before_forward=False):
    require(time.time() < args.deadline_unix, 'E061 shared export/evaluate deadline reached')
    if before_forward:
        require(report['attempted_dit_calls'] < MAX_CALLS, 'E061 thirteen-forward budget exhausted')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E061 peak allocation reached 60 GiB')
    require(report['export_bytes'] <= MAX_EXPORT_BYTES, 'E061 exports exceeded 26 GiB')


@torch.inference_mode()
def export_initializations(args, report, legacy):
    state = torch.load(report['state']['file'], map_location='cpu', weights_only=True, mmap=True)
    require(set(state['layers']) == set(TARGET_NAMES) and state['config'] == legacy['state_config'], 'Legacy state layout changed')
    exports = {}
    for arm in ARMS:
        directory = DATA/arm
        require(not directory.exists(), f'Refusing existing export directory {directory}')
        directory.mkdir(parents=True)
        exports[arm] = dict(format=FORMAT, recipe=RECIPE, status='exporting', arm=arm,
            target_count=0, exact_roundtrip_count=0, layers=[], sources=report['sources'],
            legacy_manifest=report['legacy_manifest'], state=report['state'],
            initialization='SVD_r(BF16(Ws-Q0))' if arm == 'error_init' else 'SVD_r(Ws)',
            svd=dict(rank=32, q=40, niter=2, dtype='float32', common_seed='310000+1000*block+50*local_index'),
            export_policy='Original BF16 Ws and B@A residual; independent legacy QDQ roundtrip; no recalibration')
    report['export_progress'] = {arm: 0 for arm in ARMS}
    old_rows = {r['name']: r for r in legacy['layers']}
    with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as source:
        for index, name in enumerate(TARGET_NAMES):
            guard(args, report)
            started = time.monotonic()
            row = old_rows[name]
            w_cpu = source.get_tensor(name+'.weight').to(torch.bfloat16).contiguous()
            source_sha = base.tensor_record(w_cpu)['sha256']
            require(source_sha == row['source_weight_sha256'], f'{name}: original BF16 weight hash mismatch')
            w = w_cpu.to('cuda')
            smooth = state['layers'][name]['smooth'].to(device='cuda', dtype=torch.bfloat16)
            require(smooth.shape == (w.shape[1],) and bool(torch.isfinite(smooth).all()) and bool((smooth > 0).all()),
                    f'{name}: invalid frozen smoothing')
            ws = w*smooth
            q0 = nvfp4_qdq(ws, element_size=CHUNK_ROWS)
            error_target = ws-q0
            seed = 310000+1000*(index//4)+50*(index%4)
            del w_cpu, w, q0
            for arm in ARMS:
                guard(args, report)
                torch.manual_seed(seed)
                target = error_target if arm == 'error_init' else ws
                branch = LowRankBranch(ws.shape[1], ws.shape[0], rank=32, weight=target)
                a, b = branch.a.weight.detach(), branch.b.weight.detach()
                require(a.dtype == b.dtype == ws.dtype == torch.bfloat16 and
                        tuple(a.shape) == (32, ws.shape[1]) and tuple(b.shape) == (ws.shape[0], 32),
                        f'{name}: LR shape/dtype changed')
                residual = ws-b@a
                packet = pack_activation_legacy(residual, chunk_rows=CHUNK_ROWS)
                independent = nvfp4_qdq(residual, element_size=CHUNK_ROWS)
                decoded = packet.decode(dtype=torch.bfloat16, chunk_rows=CHUNK_ROWS)
                require(bool(torch.isfinite(independent).all()) and torch.equal(decoded, independent),
                        f'{name}/{arm}: independent packed weight roundtrip failed')
                module = NativeH3Linear(packet, smooth, a, b, chunk_rows=CHUNK_ROWS)
                payload = exporter.export_payload(module, name)
                path = DATA/arm/'layers'/(name+'.pt')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    torch.save(payload, stream)
                rec = file_record(path)
                report['export_bytes'] += rec['bytes']
                item = dict(name=name, shape=list(ws.shape), seed=seed, file=str(path.relative_to(DATA/arm)),
                    file_sha256=rec['sha256'], file_bytes=rec['bytes'], source_weight_sha256=source_sha,
                    smooth=base.tensor_record(payload['tensors']['smooth']),
                    lr_a=base.tensor_record(payload['tensors']['lr_a']), lr_b=base.tensor_record(payload['tensors']['lr_b']),
                    roundtrip=dict(exact=True, changed_elements=0, elements=residual.numel()),
                    residual_dtype=str(residual.dtype), bias=None, lr_multiplier=1.0)
                exports[arm]['layers'].append(item)
                report['export_progress'][arm] += 1
                del branch, a, b, target, residual, packet, independent, decoded, module, payload
                guard(args, report)
            report['active_export'] = dict(layer=name, seed=seed, seconds=time.monotonic()-started)
            save(report, args.output)
            print(f'E061 exported {index+1}/200 {name}', flush=True)
            del ws, error_target, smooth
    report['exports'] = {}
    for arm, result in exports.items():
        require(len(result['layers']) == 200, 'Incomplete new export')
        result.update(status='complete', target_count=200, exact_roundtrip_count=200)
        path = DATA/arm/'manifest.json'
        require(not path.exists(), f'Refusing manifest {path}')
        save(result, path)
        report['export_bytes'] += path.stat().st_size
        report['exports'][arm] = file_record(path)
    del state
    gc.collect()
    torch.cuda.empty_cache()
    guard(args, report)
    save(report, args.output)


def install_export(pipe, arm, args, report):
    guard(args, report)
    manifest = old.load_complete(base.verify_file(report['exports'][arm]))
    require(manifest['format'] == FORMAT and manifest['recipe'] == RECIPE and
            len(manifest['layers']) == manifest['target_count'] == manifest['exact_roundtrip_count'] == 200,
            'New export contract changed')
    before = base.non_target_identity(pipe.dit)
    for row in manifest['layers']:
        name = row['name']
        previous = pipe.dit.get_submodule(name)
        require(type(previous) is NativeH3Linear, f'{name}: expected existing frozen native layer')
        path = DATA/arm/row['file']
        require(file_record(path)['sha256'] == row['file_sha256'], f'{name}: new export hash changed')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        require(payload['name'] == name and payload['shape'] == [previous.out_features, previous.in_features], 'Replacement dimensions changed')
        require(torch.equal(payload['tensors']['smooth'], previous.smooth.cpu()), 'Replacement changed smoothing')
        replacement = NativeH3Linear.from_export(payload, device='cuda', activation_packer=base.pack_activation_fast, chunk_rows=1024)
        parent, attr = name.rsplit('.', 1)
        setattr(pipe.dit.get_submodule(parent), attr, replacement)
        del previous, replacement, payload
        guard(args, report)
    require(base.non_target_identity(pipe.dit) == before, 'Replacement changed non-target tensors')
    report['installed_arm'] = arm
    gc.collect()
    guard(args, report)


@torch.inference_mode()
def run_case(pipe, item, arm, checked, cpu_pipe, args, report):
    guard(args, report, before_forward=True)
    k = key(item)
    value = inherited.load_input(item)
    validated = inherited.input_signature(value, item, cpu_pipe)
    expected = checked['inputs'][k]
    require(validated == expected['input_signature'], 'Input changed after CPU check')
    path = DATA/'evaluate'/arm/(k.replace('/', '_')+'.pt')
    require(not path.exists(), f'Refusing output {path}')
    forward = base.gpu_call(pipe, {'kind': 'model_fn'}, builder.model_value(value))
    actual, raw_capture = [], []
    def capture_input(_module, positional, keyword):
        captured = base.inherited.tree_cpu({'args': positional, 'kwargs': keyword})
        require(signature(captured) == expected['actual_dit_input_signature'], 'Actual DiT input differs from frozen history')
        actual.append(captured)
    def capture_output(_module, positional, output):
        raw_capture.append(output)
    pre = pipe.dit.register_forward_pre_hook(capture_input, with_kwargs=True)
    post = pipe.dit.register_forward_hook(capture_output)
    audit = base.RuntimeAudit()
    audit.phase = 'native'
    report['attempted_dit_calls'] += 1
    report['active_call'] = dict(arm=arm, case_id=item['id'], position=item['input_role'])
    save(report, args.output)
    print(f'E061 {arm} {k}', flush=True)
    try:
        with audit.installed(), base.collect_fastpack_checks() as fastpack:
            result = forward()
            torch.cuda.synchronize()
        report['complete_dit_calls'] += 1
    finally:
        pre.remove()
        post.remove()
    base.audit_contract(audit.row(), 'svd')
    require(fastpack.summary['checked_calls'] == 200 and len(actual) == len(raw_capture) == 1, 'Native full-model call contract failed')
    raw, raw_records = base.cpu_outputs(raw_capture[0])
    velocities, velocity_records = base.cpu_outputs(result)
    replay = None
    if arm == 'legacy_selected':
        history = load_reference(report['references'][k][arm], expected['actual_dit_input_signature'])
        replay = all(torch.equal(outputs[m], history[field][m]) and signature(outputs[m]) == report['references'][k][arm][field][m]
            for field, outputs in (('raw_outputs', raw), ('velocities', velocities)) for m in inherited.MODALITIES)
        require(replay, 'Fresh legacy replay differs from six validated historical zero outputs')
    teacher = load_reference(report['references'][k]['bf16'], expected['actual_dit_input_signature'])
    metrics = output_metrics(velocities, teacher['velocities'])
    payload = dict(case_id=item['id'], position=item['input_role'], step=item['step'], arm=arm,
        input_signature=validated, actual_dit_inputs=actual[0], raw_outputs=raw, velocities=velocities,
        reference_artifacts=report['references'][k], metrics=metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    row = dict(case_id=item['id'], position=item['input_role'], step=item['step'], arm=arm, status='complete',
        artifact=file_record(path), actual_dit_input_signature=signature(actual[0]), raw_outputs=raw_records,
        velocities=velocity_records, metrics=metrics, legacy_replay_exact=replay,
        runtime_audit=audit.row(), fastpack_checks=fastpack.summary)
    report['cases'].append(row)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save(report, args.output)
    del forward, value, result, actual, raw_capture, raw, velocities, teacher, payload
    gc.collect()
    guard(args, report)


def teacher_gate(report, items):
    rows = []
    for item in items:
        if item['input_role'] == 'next_shifted':
            continue
        k = key(item)
        energy = {'legacy_selected': report['reference_metrics'][k]['legacy_selected']['video']['sse']}
        for arm in ARMS:
            match = next(r for r in report['cases'] if
                (r['case_id'], r['position'], r['arm']) == (item['id'], item['input_role'], arm))
            energy[arm] = match['metrics']['video']['sse']
        passed = all(energy['weight_init'] <= 0.8*energy[baseline] for baseline in ('legacy_selected', 'error_init'))
        rows.append(dict(case_id=item['id'], position=item['input_role'], video_sse=energy,
            weight_over_legacy=energy['weight_init']/energy['legacy_selected'] if energy['legacy_selected'] else None,
            weight_over_error_init=energy['weight_init']/energy['error_init'] if energy['error_init'] else None,
            passed=passed))
    require(len(rows) == 4, 'Four fixed teacher gate rows required')
    return dict(metric='video velocity FP64 SSE, uncentered', threshold_ratio=0.8,
        rows=rows, passed=all(r['passed'] for r in rows))


@torch.inference_mode()
def evaluate(args, report, manifest, e014_manifest, items, legacy):
    checked = old.load_complete(REPORTS/'check.json')
    for field in ('sources', 'e060_audit', 'legacy_manifest', 'state', 'references', 'reference_reports', 'e059_evaluation'):
        require(checked[field] == report[field], f'E061 CPU binding changed: {field}')
    require(checked['cuda_initialized'] is False, 'E061 check was not CPU-only')
    report['cpu_check_reference'] = file_record(REPORTS/'check.json')
    report['reference_metrics'] = checked['reference_metrics']
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    require(torch.cuda.get_device_capability() == (12, 0), 'E061 requires native SM120')
    export_initializations(args, report, legacy)
    setup_path = DATA/'setup_legacy.json'
    require(not setup_path.exists(), f'Refusing setup {setup_path}')
    old_check = old.load_complete(manifest['e014_check']['file'])
    setup = dict(experiment='E061_inherited_E014_setup', status='running', complete_dit_calls=0, sources=old_check['sources'])
    setup_args = types.SimpleNamespace(arm='svd', deadline_unix=args.deadline_unix,
        check_report=Path(manifest['e014_check']['file']), output=setup_path)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    builder.configure_schedule(pipe, manifest['settings'])
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    first = next(i for i in items if (i['id'], i['input_role']) == ('e010_p030_s05', 'source_teacher'))
    run_case(pipe, first, 'legacy_selected', checked, cpu_pipe, args, report)
    teachers = [i for i in items if i['input_role'] != 'next_shifted']
    for arm in ARMS:
        install_export(pipe, arm, args, report)
        for item in teachers:
            run_case(pipe, item, arm, checked, cpu_pipe, args, report)
    require(report['complete_dit_calls'] == report['attempted_dit_calls'] == 9, 'Required nine-call phase incomplete')
    report['teacher_gate'] = teacher_gate(report, items)
    save(report, args.output)
    if report['teacher_gate']['passed']:
        # weight_init is already installed; one extra installation serves error_init.
        for arm in reversed(ARMS):
            if report['installed_arm'] != arm:
                install_export(pipe, arm, args, report)
            for item in items:
                if item['input_role'] == 'next_shifted':
                    run_case(pipe, item, arm, checked, cpu_pipe, args, report)
    expected_calls = 13 if report['teacher_gate']['passed'] else 9
    require(report['complete_dit_calls'] == report['attempted_dit_calls'] == expected_calls, 'Final conditional call count changed')
    report.update(status='complete', active_call=None,
        shifted_evaluation='complete' if expected_calls == 13 else 'not_run_teacher_gate_failed',
        decision='Prepare formal matched PTQ, no method claim' if expected_calls == 13 else
                 'Stop fixed-smooth initialization comparison; full official recipe remains untested')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate'), required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or REPORTS/f'{args.phase}.json'
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    started = time.time()
    if args.phase == 'evaluate':
        require(args.deadline_unix is not None and 0 < args.deadline_unix-started <= 1800,
                'One root-assigned absolute deadline, at most 1800 seconds, must cover export and evaluate')
    report = dict(experiment='E061', phase=args.phase, status='running', start_epoch=started,
        deadline_unix=args.deadline_unix, wall_budget_seconds=1800, max_dit_calls=MAX_CALLS,
        export_byte_limit=MAX_EXPORT_BYTES, export_bytes=0, attempted_dit_calls=0, complete_dit_calls=0, cases=[],
        scope='Fixed legacy smooth, rank32, matched approximate SVD; baseline investment screen only')
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E061 export/evaluate absolute wall deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, e014_manifest, items, legacy = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, manifest, items, legacy)
        else:
            evaluate(args, report, manifest, e014_manifest, items, legacy)
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
