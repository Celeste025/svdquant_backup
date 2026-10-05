#!/usr/bin/env python3
"""E065b: separately budgeted continuation of deadline-stopped E065 v2.

The immutable complete calibration and contiguous completed layer prefix are
read-only. Every remaining layer repeats the original eight-by-two search.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
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
import probe_h3_carry_q_v2 as prior
import torch
import torch.nn.functional as F

base, old, inherited, builder, exporter, e061 = prior.base, prior.old, prior.inherited, prior.builder, prior.exporter, prior.e061
NativeH3Linear, TARGET_NAMES, FORMAT, RECIPE = prior.NativeH3Linear, prior.TARGET_NAMES, prior.FORMAT, prior.RECIPE
LowRankBranch, nvfp4_qdq, pack_activation_legacy = prior.LowRankBranch, prior.nvfp4_qdq, prior.pack_activation_legacy
safe_open, H3_DIT_PATH = prior.safe_open, prior.H3_DIT_PATH
save, file_record, signature, require, key = prior.save, prior.file_record, prior.signature, prior.require, prior.key
packet_signature, packet_from_payload = prior.packet_signature, prior.packet_from_payload
ARMS, N_CANDIDATES, CHUNK_ROWS = prior.ARMS, prior.N_CANDIDATES, prior.CHUNK_ROWS
WITNESS_ROWS, WITNESS_COLS = prior.WITNESS_ROWS, prior.WITNESS_COLS
MAX_LOCAL_CANDIDATE_CALLS = prior.MAX_LOCAL_CANDIDATE_CALLS
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E065b')
REPORTS = ROOT/'results/research/E065b'
CHECK_REPORT = REPORTS/'check.json'
PLAN = ROOT/'research_state/06_experiments/E065b_h3_carry_q_continuation_plan.md'
SOURCE_REPORT = prior.REPORTS/'evaluate.json'
SOURCE_WORKER_PID = 16932
MAX_CALLS, MAX_DATA_BYTES, MAX_CUMULATIVE_BYTES, WALL_SECONDS = 9, 128*1024**3, 256*1024**3, 3600


def source_worker_exited():
    require(not Path(f'/proc/{SOURCE_WORKER_PID}').exists(),
            f'E065 source worker PID {SOURCE_WORKER_PID} still exists; continuation forbidden')


def accounting(report):
    source = report.get('source_runtime_counts', {})
    n = report.get('reused_prefix_layers', 0)
    report['completed_candidate_native_calls'] = n*128+report['candidate_native_calls']
    report['cumulative_actual_calls'] = dict(
        complete_dit_calls=source.get('complete_dit_calls', 0)+report['complete_dit_calls'],
        attempted_dit_calls=source.get('attempted_dit_calls', 0)+report['attempted_dit_calls'],
        candidate_native_calls=source.get('candidate_native_calls', 0)+report['candidate_native_calls'],
        shape_native_calls=source.get('shape_native_calls', 0)+report['shape_native_calls'])
    report['cumulative_data_bytes'] = report.get('source_data_actual_bytes', 0)+report['data_bytes']


def guard(args, report, *, before_forward=False):
    accounting(report)
    require(time.time() < args.deadline_unix, 'E065b independent 3600-second deadline reached')
    require(report['attempted_dit_calls'] <= MAX_CALLS and
            (not before_forward or report['attempted_dit_calls'] < MAX_CALLS), 'E065b nine-full-call budget exceeded')
    require(report['candidate_native_calls'] <= report['remaining_candidate_call_limit'], 'E065b candidate budget exceeded')
    require(report['calibration_bf16_calls'] == report['calibration_activation_packs'] == report['shape_native_calls'] == 0,
            'E065b must not rerun calibration or shape controls')
    require(report['data_bytes'] < MAX_DATA_BYTES and report['cumulative_data_bytes'] <= MAX_CUMULATIVE_BYTES,
            'E065b new/cumulative data limit reached')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E065b peak reached 60 GiB')


def clone_helper(function, namespace, *, inference=False):
    original = inspect.unwrap(function)
    copied = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    copied.__kwdefaults__ = original.__kwdefaults__
    return torch.inference_mode()(copied) if inference else copied


# The score and artifact implementations are unchanged, with only new globals.
score_candidate = clone_helper(prior.score_candidate, globals())
save_tensor = clone_helper(prior.save_tensor, globals())
save_json_artifact = clone_helper(prior.save_json_artifact, globals())
_E061_GLOBALS = dict(e061.__dict__)
_E061_GLOBALS.update(__file__=__file__, DATA=DATA, REPORTS=REPORTS, PLAN=PLAN, guard=guard, MAX_CALLS=MAX_CALLS)
_run_case = clone_helper(e061.run_case, _E061_GLOBALS, inference=True)
install_export = clone_helper(e061.install_export, _E061_GLOBALS)


def export_source_changes():
    return (
        ('layers=[], target_count=0, exact_roundtrip_count=0, sources=report[\'sources\'],',
         'layers=[dict(row) for row in report[\'prefix_export_rows\'][arm]], target_count=0, exact_roundtrip_count=0, sources=report[\'sources\'],'),
        ("report['export_progress'] = {arm: 0 for arm in ARMS}",
         "report['export_progress'] = {arm: report['reused_prefix_layers'] for arm in ARMS}"),
        ('for index, name in enumerate(TARGET_NAMES):',
         "for index, name in list(enumerate(TARGET_NAMES))[report['reused_prefix_layers']:] :"),
        ("report['candidate_native_calls'] == MAX_LOCAL_CANDIDATE_CALLS",
         "report['candidate_native_calls'] == report['remaining_candidate_call_limit']"),
    )


def audit_expanded_source():
    original = inspect.getsource(prior.export_candidates)
    expected = original
    for before, after in export_source_changes():
        require(expected.count(before) == 1, f'Original export source patch is not unique: {before}')
        expected = expected.replace(before, after)
    observed = inspect.getsource(export_candidates)
    require(observed == expected, 'Expanded continuation export differs beyond the four allowed source edits')
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return dict(original_function_sha256=digest(original), expanded_function_sha256=digest(observed),
                original_file=file_record(prior.__file__), expanded_file=file_record(__file__),
                allowed_replacements=[dict(before=a, after=b) for a, b in export_source_changes()])


def prerequisites(report):
    require(PLAN.is_file(), 'E065b continuation plan must exist')
    source_worker_exited()
    parent = json.loads(SOURCE_REPORT.read_text())
    require(parent['status'] == 'failed_stop' and
            ('absolute wall deadline reached' in parent.get('error', '') or
             'absolute deadline reached' in parent.get('error', '')) and
            time.time() >= parent['deadline_unix'], 'Source must have stopped at its original deadline')
    require(parent['complete_dit_calls'] == parent['attempted_dit_calls'] == 9 and
            parent['calibration_bf16_calls'] == 8 and parent['calibration_activation_packs'] == 1600 and
            parent['shape_native_calls'] == 8, 'Source did not finish exactly calibration eight plus legacy one')
    require(len(parent['cases']) == 1 and parent['cases'][0]['arm'] == 'legacy_selected' and
            parent['cases'][0]['legacy_replay_exact'] is True, 'Source legacy replay missing or source diagnostics already started')
    prefix = parent['search_layers']
    require(0 <= len(prefix) < 200 and [r['layer'] for r in prefix] == list(TARGET_NAMES[:len(prefix)]),
            'Only a complete contiguous source prefix below 200 may be reused')
    manifest, e014_manifest, items, legacy = prior.prerequisites(report)
    require(report['sources'] == parent['sources'], 'Original source/plan graph changed after E065')
    old_checked = old.load_complete(base.verify_file(parent['cpu_check_reference']))
    require(old_checked['sources'] == parent['sources'] and old_checked['cuda_initialized'] is False,
            'Source CPU check does not bind the same source graph')
    for field in ('calibration_inputs', 'references', 'reference_reports', 'search_contract', 'state', 'legacy_manifest'):
        require(report[field] == parent[field] == old_checked[field], f'Inherited binding changed: {field}')
    expected_cases = [c['id'] for c in report['calibration_inputs']]
    require([c['case_id'] for c in parent['calibration_cases']] == expected_cases and
            len(parent['calibration_cases']) == 8 and len(parent['selection_shape_controls']) == 4,
            'Source calibration/shape-control coverage incomplete')
    for case in parent['calibration_cases']:
        require(set(case['layers']) == set(TARGET_NAMES), 'Source calibration layer coverage incomplete')
    report['source_sources'] = dict(parent['sources'])
    report['sources'].update({str(p.resolve()): file_record(p) for p in (Path(__file__), PLAN, SOURCE_REPORT)})
    report.update(plan=file_record(PLAN), source_report=file_record(SOURCE_REPORT),
        source_worker_pid=SOURCE_WORKER_PID, source_worker_exit_verified=True,
        source_cpu_check=parent['cpu_check_reference'], source_plan=parent['plan'],
        source_runtime_counts={k: parent[k] for k in ('complete_dit_calls', 'attempted_dit_calls',
            'candidate_native_calls', 'calibration_bf16_calls', 'calibration_activation_packs', 'shape_native_calls')},
        source_elapsed_seconds=parent['elapsed_seconds'], source_deadline_unix=parent['deadline_unix'],
        source_partial_candidate_native_calls=parent['candidate_native_calls']-128*len(prefix),
        reused_prefix_layers=len(prefix), remaining_candidate_call_limit=(200-len(prefix))*128,
        reused_calibration_bf16_calls=8, reused_calibration_activation_packs=1600, reused_shape_native_calls=8,
        calibration_cases=parent['calibration_cases'], reused_selection_shape_controls=parent['selection_shape_controls'],
        source_legacy_replay=parent['cases'][0], search_layers=list(prefix),
        inputs=old_checked['inputs'], calibration_checks=old_checked['calibration_checks'],
        reference_metrics=old_checked['reference_metrics'], source_assembly=audit_expanded_source())
    require(0 <= report['source_partial_candidate_native_calls'] <= 128, 'Unexpected source partial-layer accounting')
    return manifest, e014_manifest, items, legacy, parent


@torch.inference_mode()
def cpu_check(report, legacy, parent):
    require(not torch.cuda.is_initialized(), 'Continuation CPU check initialized CUDA')
    verified = {}
    def verify(record):
        path = record['file']
        if path in verified:
            require(verified[path] == record, f'Conflicting reused file records: {path}')
        else:
            base.verify_file(record)
            verified[path] = record
        return Path(path)
    for record in parent['sources'].values():
        verify(record)
    verify(report['source_report'])
    verify(report['source_cpu_check'])
    for case in report['calibration_inputs']:
        verify(case['artifact'])
    for case in report['calibration_cases']:
        verify(case['artifact'])
        for row in case['layers'].values():
            verify(row['artifact'])
    for row in report['reused_selection_shape_controls']:
        verify(row['artifact'])
    verify(report['source_legacy_replay']['artifact'])
    for references in report['references'].values():
        for reference in references.values():
            if reference is not None:
                verify(reference['artifact'])
    prefix_rows = {arm: [] for arm in ARMS}
    original_rows = {r['name']: r for r in legacy['layers']}
    for index, entry in enumerate(report['search_layers']):
        name = TARGET_NAMES[index]
        metadata = json.loads(verify(entry['metadata']).read_text())
        require(metadata['layer'] == entry['layer'] == name and metadata['candidate_zero_exact'] is True and
                set(metadata['arms']) == set(ARMS), 'Invalid completed prefix metadata')
        verify(metadata['recurrence_witness'])
        first_candidates = []
        for arm in ARMS:
            row = metadata['arms'][arm]
            candidates = row['candidates']
            require([r['k'] for r in candidates] == list(range(8)), 'Prefix candidate coverage is incomplete')
            chosen = min(range(8), key=lambda k: candidates[k]['score'])
            require(chosen == row['selected_k'] and candidates[chosen]['score'] == row['selected_score'],
                    'Prefix selected candidate is not first argmin')
            first_candidates.append(candidates[0])
            best = candidates[chosen]
            payload = torch.load(verify(row['export']), map_location='cpu', weights_only=True, mmap=True)
            require(payload['format'] == FORMAT and payload['recipe'] == RECIPE and payload['name'] == name and
                    payload['shape'] == original_rows[name]['shape'], 'Prefix export ABI differs')
            tensors = payload['tensors']
            packet = signature(dict(codes=tensors['weight_packed'], scales=tensors['weight_scales_swizzled'],
                                    global_scale=tensors['weight_global'], original_shape=payload['shape']))
            require(packet == best['packet'] and base.tensor_record(tensors['lr_a']) == best['lr_a'] and
                    base.tensor_record(tensors['lr_b']) == best['lr_b'], 'Prefix Q/A/B are not the same selected candidate')
            verify(row['selected_local_outputs'])
            prefix_rows[arm].append(dict(name=name, shape=payload['shape'], file=row['export']['file'],
                file_sha256=row['export']['sha256'], file_bytes=row['export']['bytes'],
                source_weight_sha256=metadata['source_weight_sha256'], smooth=base.tensor_record(tensors['smooth']),
                lr_a=best['lr_a'], lr_b=best['lr_b'],
                roundtrip=dict(exact=True, changed_elements=0, elements=payload['shape'][0]*payload['shape'][1]),
                bias=None, residual_dtype='torch.bfloat16', lr_multiplier=1.0, selected_k=chosen,
                selected_score=row['selected_score'], selected_decoded_q=best['decoded_q'], selected_packet=best['packet'],
                selected_local_outputs=row['selected_local_outputs'], reused_from=report['source_report']))
            del payload, tensors
        require(first_candidates[0] == first_candidates[1], 'Prefix candidate zero arms differ')
    actual_bytes = sum(path.stat().st_size for path in prior.DATA.rglob('*') if path.is_file())
    require(actual_bytes <= 128*1024**3, 'Original data exceeds its unchanged stage budget')
    require(not torch.cuda.is_initialized(), 'Continuation CPU check initialized CUDA')
    report.update(prefix_export_rows=prefix_rows, verified_reused_files=verified,
        verified_reused_file_count=len(verified), source_data_actual_bytes=actual_bytes,
        source_data_accounting='Includes all source files, including abandoned partial-layer artifacts; read-only',
        status='complete', cuda_initialized=False, new_model_forwards=0)
    accounting(report)


@torch.inference_mode()
def export_candidates(args, report, legacy, state):
    exports = {}
    for arm in ARMS:
        directory = DATA/arm
        require(not directory.exists(), f'Refusing existing export directory {directory}')
        directory.mkdir(parents=True)
        exports[arm] = dict(format=FORMAT, recipe=RECIPE, status='exporting', arm=arm,
            layers=[dict(row) for row in report['prefix_export_rows'][arm]], target_count=0, exact_roundtrip_count=0, sources=report['sources'],
            legacy_manifest=report['legacy_manifest'], state=report['state'], search_contract=report['search_contract'])
    report['export_progress'] = {arm: report['reused_prefix_layers'] for arm in ARMS}
    old_rows = {r['name']: r for r in legacy['layers']}
    with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as source:
        for index, name in list(enumerate(TARGET_NAMES))[report['reused_prefix_layers']:] :
            guard(args, report)
            started = time.monotonic()
            w_cpu = source.get_tensor(name+'.weight').to(torch.bfloat16).contiguous()
            source_sha = base.tensor_record(w_cpu)['sha256']
            require(source_sha == old_rows[name]['source_weight_sha256'], 'Original BF16 weight hash mismatch')
            smooth = state['layers'][name]['smooth'].to(device='cuda', dtype=torch.bfloat16)
            require(bool(torch.isfinite(smooth).all()) and bool((smooth > 0).all()), 'Invalid frozen smooth')
            ws = w_cpu.cuda()*smooth
            del w_cpu
            q0 = nvfp4_qdq(ws, element_size=CHUNK_ROWS)
            q0_sig = base.tensor_record(q0)
            wr = torch.linspace(0, ws.shape[0]-1, WITNESS_ROWS, dtype=torch.float64).long().cuda()
            wc = torch.linspace(0, ws.shape[1]-1, WITNESS_COLS, dtype=torch.float64).long().cuda()
            def witness(tensor):
                return tensor.index_select(0, wr).index_select(1, wc).cpu()
            samples = []
            for case in report['calibration_cases']:
                record = case['layers'][name]['artifact']
                sample = torch.load(base.verify_file(record), map_location='cpu', weights_only=True, mmap=True)
                require(sample['layer'] == name and sample['case_id'] == case['case_id'] and
                        base.tensor_record(sample['xs']) == sample['source_signature'] and
                        base.tensor_record(sample['target']) == sample['target_signature'], 'Saved calibration row hashes changed')
                require(sample['smooth_signature'] == base.tensor_record(smooth), 'Calibration smoothing changed')
                samples.append(dict(case_id=case['case_id'], artifact=record,
                    packet=packet_from_payload(sample['packet'], 'cuda'),
                    xs=sample['xs'].cuda(), target=sample['target'].cuda()))
            histories, first, witness_values = {}, None, []
            for arm in ARMS:
                q_previous, previous_sig = q0, q0_sig
                best_score, best_payload, best_outputs, best_k = None, None, None, None
                history = []
                for k in range(N_CANDIDATES):
                    guard(args, report)
                    seed = 310000+1000*(index//4)+50*(index%4)+k
                    parent_q = q0 if arm == 'restart' else q_previous
                    parent_sig = q0_sig if arm == 'restart' else previous_sig
                    target = ws-parent_q
                    torch.manual_seed(seed)
                    branch = LowRankBranch(ws.shape[1], ws.shape[0], rank=32, weight=target)
                    a, b = branch.a.weight.detach(), branch.b.weight.detach()
                    require(a.dtype == b.dtype == torch.bfloat16 and
                            tuple(a.shape) == (32, ws.shape[1]) and tuple(b.shape) == (ws.shape[0], 32), 'LR ABI changed')
                    lowrank = b@a
                    residual = ws-lowrank
                    packet = pack_activation_legacy(residual, chunk_rows=CHUNK_ROWS)
                    independent = nvfp4_qdq(residual, element_size=CHUNK_ROWS)
                    decoded = packet.decode(dtype=torch.bfloat16, chunk_rows=CHUNK_ROWS)
                    require(bool(torch.isfinite(independent).all()) and torch.equal(decoded, independent),
                            f'{name}/{arm}/{k}: independent packed weight roundtrip failed')
                    module = NativeH3Linear(packet, smooth, a, b, chunk_rows=CHUNK_ROWS)
                    total, scores, outputs = score_candidate(module, samples, args, report)
                    candidate = dict(k=k, seed=seed, score=total, case_ids=[s['case_id'] for s in samples],
                        per_case_sse=scores, parent_q=parent_sig, target=base.tensor_record(target),
                        lr_a=base.tensor_record(a), lr_b=base.tensor_record(b),
                        residual=base.tensor_record(residual), decoded_q=base.tensor_record(decoded),
                        actual_q=base.tensor_record(independent),
                        packet=packet_signature(packet), roundtrip_exact=True,
                        output_signatures=[base.tensor_record(o) for o in outputs])
                    if k == 0:
                        common = {field: candidate[field] for field in
                            ('seed', 'parent_q', 'target', 'lr_a', 'lr_b', 'residual', 'decoded_q', 'actual_q',
                             'packet', 'per_case_sse', 'score', 'output_signatures')}
                        if arm == 'restart':
                            first = common
                        else:
                            require(common == first, f'{name}: candidate zero arms differ')
                    witness_values.append(dict(arm=arm, k=k, ws=witness(ws), parent_q=witness(parent_q),
                        target=witness(target), lowrank=witness(lowrank), residual=witness(residual),
                        decoded_q=witness(decoded), actual_q=witness(independent)))
                    history.append(candidate)
                    if best_score is None or total < best_score:
                        best_score, best_k = total, k
                        best_payload, best_outputs = exporter.export_payload(module, name), outputs
                    # Latest candidate, not argmin: this is the carry-Q recurrence.
                    q_previous, previous_sig = independent, candidate['actual_q']
                    del branch, a, b, lowrank, residual, target, packet, decoded, module, outputs, independent
                    guard(args, report)
                require(best_k == min(range(8), key=lambda k: history[k]['score']), 'First-tie best selection differs')
                path = DATA/arm/'layers'/(name+'.pt')
                record = save_tensor(best_payload, path, report)
                report['export_bytes'] += record['bytes']
                best = history[best_k]
                require(base.tensor_record(best_payload['tensors']['lr_a']) == best['lr_a'] and
                        base.tensor_record(best_payload['tensors']['lr_b']) == best['lr_b'], 'Export factors not from best candidate')
                exported_packet = signature(dict(codes=best_payload['tensors']['weight_packed'],
                    scales=best_payload['tensors']['weight_scales_swizzled'],
                    global_scale=best_payload['tensors']['weight_global'], original_shape=best_payload['shape']))
                require(exported_packet == best['packet'], 'Export packet not from the same selected candidate')
                selected_outputs = save_tensor(dict(layer=name, arm=arm, selected_k=best_k,
                    case_ids=[s['case_id'] for s in samples], outputs=best_outputs,
                    output_signatures=best['output_signatures']), DATA/'selected_local_outputs'/arm/(name+'.pt'), report)
                item = dict(name=name, shape=list(ws.shape), file=str(path.relative_to(DATA/arm)),
                    file_sha256=record['sha256'], file_bytes=record['bytes'], source_weight_sha256=source_sha,
                    smooth=base.tensor_record(best_payload['tensors']['smooth']), lr_a=best['lr_a'], lr_b=best['lr_b'],
                    roundtrip=dict(exact=True, changed_elements=0, elements=ws.numel()), bias=None,
                    residual_dtype='torch.bfloat16', lr_multiplier=1.0, selected_k=best_k,
                    selected_score=best_score, selected_decoded_q=best['decoded_q'], selected_packet=best['packet'],
                    selected_local_outputs=selected_outputs)
                exports[arm]['layers'].append(item)
                report['export_progress'][arm] += 1
                histories[arm] = dict(selected_k=best_k, selected_score=best_score, candidates=history,
                                     export=record, selected_local_outputs=selected_outputs)
                del q_previous, best_payload, best_outputs
            witness_record = save_tensor(dict(layer=name, row_indices=wr.cpu(), column_indices=wc.cpu(),
                candidates=witness_values, scope='Fixed 8x8 coordinate arithmetic witness, not full SVD reproduction'),
                DATA/'candidate_witnesses'/(name+'.pt'), report)
            search_record = save_json_artifact(dict(layer=name, source_weight_sha256=source_sha,
                q0=q0_sig, candidate_zero_exact=True, arms=histories, recurrence_witness=witness_record),
                DATA/'candidate_metadata'/(name+'.json'), report)
            report['search_layers'].append(dict(layer=name, metadata=search_record, candidate_zero_exact=True,
                selected={arm: dict(k=histories[arm]['selected_k'], score=histories[arm]['selected_score'])
                          for arm in ARMS}, recurrence_witness=witness_record))
            report['active_export'] = dict(layer=name, seconds=time.monotonic()-started)
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            save(report, args.output)
            print(f'E065 searched {index+1}/200 {name}', flush=True)
            del samples, ws, smooth, q0, witness_values, sample
            gc.collect()
    report['exports'] = {}
    for arm, result in exports.items():
        require(len(result['layers']) == 200, 'Incomplete selected export')
        result.update(status='complete', target_count=200, exact_roundtrip_count=200)
        path = DATA/arm/'manifest.json'
        require(not path.exists(), f'Refusing manifest {path}')
        save(result, path)
        record = file_record(path)
        report['exports'][arm] = record
        report['data_bytes'] += record['bytes']
        report['export_bytes'] += record['bytes']
    require(report['candidate_native_calls'] == report['remaining_candidate_call_limit'], 'Incomplete candidate evaluation count')
    torch.cuda.empty_cache()
    guard(args, report)
    save(report, args.output)


def run_case(pipe, item, arm, checked, cpu_pipe, args, report):
    _run_case(pipe, item, arm, checked, cpu_pipe, args, report)
    report['diagnostic_complete_dit_calls'] += 1
    report['data_bytes'] += report['cases'][-1]['artifact']['bytes']
    guard(args, report)
    save(report, args.output)


@torch.inference_mode()
def evaluate(args, report, manifest, e014_manifest, items, legacy):
    checked = old.load_complete(CHECK_REPORT)
    for field in ('sources', 'source_report', 'source_cpu_check', 'source_plan', 'source_sources',
                  'source_runtime_counts', 'reused_prefix_layers', 'remaining_candidate_call_limit',
                  'calibration_cases', 'calibration_inputs', 'calibration_checks', 'inputs',
                  'references', 'reference_metrics', 'search_contract', 'source_assembly'):
        require(report[field] == checked[field], f'Continuation CPU binding changed: {field}')
    require(checked['cuda_initialized'] is False, 'E065b check was not CPU-only')
    source_worker_exited()
    for field in ('prefix_export_rows', 'source_data_actual_bytes', 'verified_reused_file_count'):
        report[field] = checked[field]
    require(not DATA.exists(), f'Refusing existing E065b data directory {DATA}')
    DATA.mkdir(parents=True)
    report['cpu_check_reference'] = file_record(CHECK_REPORT)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    require(torch.cuda.get_device_capability() == (12, 0), 'E065b requires SM120')
    setup_path = DATA/'setup_legacy.json'
    old_check = old.load_complete(manifest['e014_check']['file'])
    setup = dict(experiment='E065b_inherited_E014_setup', status='running', complete_dit_calls=0, sources=old_check['sources'])
    setup_args = types.SimpleNamespace(arm='svd', deadline_unix=args.deadline_unix,
        check_report=Path(manifest['e014_check']['file']), output=setup_path)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    report['data_bytes'] += report['model_setup']['bytes']
    builder.configure_schedule(pipe, manifest['settings'])
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    first = next(i for i in items if (i['id'], i['input_role']) == ('e010_p030_s05', 'source_teacher'))
    run_case(pipe, first, 'legacy_selected', checked, cpu_pipe, args, report)
    state = torch.load(base.verify_file(report['state']), map_location='cpu', weights_only=True, mmap=True)
    require(set(state['layers']) == set(TARGET_NAMES) and state['config'] == legacy['state_config'], 'Inherited state layout differs')
    export_candidates(args, report, legacy, state)
    del state
    teachers = [i for i in items if i['input_role'] != 'next_shifted']
    require(len(teachers) == 4, 'Four unchanged diagnostic teachers required')
    for arm in ARMS:
        install_export(pipe, arm, args, report)
        for item in teachers:
            run_case(pipe, item, arm, checked, cpu_pipe, args, report)
    guard(args, report)
    require(report['complete_dit_calls'] == report['attempted_dit_calls'] == MAX_CALLS and len(report['cases']) == 9 and
            report['completed_candidate_native_calls'] == 25600 and
            report['cumulative_actual_calls']['complete_dit_calls'] == 18, 'Continuation final accounting differs')
    report.update(status='complete', active_call=None, diagnostic_readout=prior.diagnostic_readout(report, items),
        decision='Separately budgeted continuation; no claim of completion within original 17-call/1800-second budget')


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
        require(args.deadline_unix is not None and 0 < args.deadline_unix-started <= WALL_SECONDS,
                'E065b needs its own root-assigned deadline, no more than 3600 seconds')
    report = dict(experiment='E065b', phase=args.phase, status='running', start_epoch=started,
        deadline_unix=args.deadline_unix, wall_budget_seconds=WALL_SECONDS, max_dit_calls=MAX_CALLS,
        data_byte_limit=MAX_DATA_BYTES, cumulative_data_byte_limit=MAX_CUMULATIVE_BYTES,
        data_bytes=0, export_bytes=0, attempted_dit_calls=0, complete_dit_calls=0,
        calibration_bf16_calls=0, calibration_activation_packs=0, shape_native_calls=0,
        candidate_native_calls=0, diagnostic_complete_dit_calls=0, peak_allocated_bytes=0,
        cases=[], search_layers=[], scope='Original algorithm, readonly complete prefix, separate continuation budget')
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E065b independent continuation deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, e014_manifest, items, legacy, parent = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, legacy, parent)
        else:
            evaluate(args, report, manifest, e014_manifest, items, legacy)
        accounting(report)
        report['elapsed_seconds'] = time.time()-started
        save(report, args.output)
    except Exception:
        accounting(report)
        report.update(status='failed_stop', elapsed_seconds=time.time()-started, error=traceback.format_exc())
        save(report, args.output)
        raise
    finally:
        if args.phase == 'evaluate':
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
