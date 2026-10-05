#!/usr/bin/env python3
"""E066: frozen carry/restart local transfer on four BF16 teacher inputs.

Four unchanged model_fn forwards; full-M and M512 comparisons share one
full-input activation pack and its byte-preserving row subset. No fitting.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe_h3_carry_q_v2 as packets
import probe_h3_lowrank_initialization as e061
import torch
import torch.nn.functional as F

base, old, inherited, builder = e061.base, e061.old, e061.inherited, e061.builder
NativeH3Linear, TARGET_NAMES, FORMAT, RECIPE = e061.NativeH3Linear, e061.TARGET_NAMES, e061.FORMAT, e061.RECIPE
save, file_record, signature, require, key = e061.save, e061.file_record, e061.signature, e061.require, e061.key
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E066')
REPORTS = ROOT/'results/research/E066'
PLAN = ROOT/'research_state/06_experiments/E066_h3_carry_q_local_transfer_plan.md'
SOURCE_REPORTS = ROOT/'results/research/E065b'
CHECK_REPORT = REPORTS/'check.json'
ARMS = ('restart', 'carry')
DOMAINS = ('video', 'audio', 'text', 'joint')
MAX_CALLS, MAX_PACKS, MAX_NATIVE_CALLS = 4, 800, 3200
MAX_DATA_BYTES, WALL_SECONDS, REDUCE_ROWS, CHUNK_ROWS = 96*1024**3, 1800, 2048, 1024


# Rebind only source/plan identity in a private namespace. The inherited source
# modules remain immutable and their model/input helpers keep their own globals.
_PREREQUISITE_GLOBALS = dict(e061.__dict__)
_PREREQUISITE_GLOBALS.update(__file__=__file__, PLAN=PLAN)
_prerequisites = types.FunctionType(e061.prerequisites.__code__, _PREREQUISITE_GLOBALS,
                                  e061.prerequisites.__name__, e061.prerequisites.__defaults__,
                                  e061.prerequisites.__closure__)


def guard(args, report, *, before_forward=False):
    require(time.time() < args.deadline_unix, 'E066 absolute 1800-second deadline reached')
    require(report['attempted_dit_calls'] <= MAX_CALLS and
            (not before_forward or report['attempted_dit_calls'] < MAX_CALLS), 'E066 four-forward budget exceeded')
    require(report['activation_packs'] <= MAX_PACKS and
            report['attempted_local_native_calls'] <= MAX_NATIVE_CALLS and
            report['full_native_calls'] <= 1600 and report['subset_native_calls'] <= 1600,
            'E066 local call budget exceeded')
    require(report['data_bytes'] <= MAX_DATA_BYTES, 'E066 96-GiB new-data budget exceeded')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E066 peak allocation reached 60 GiB')


def save_tensor(value, path, report):
    require(not path.exists(), f'Refusing existing artifact {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(value, stream)
    record = file_record(path)
    report['data_bytes'] += record['bytes']
    require(report['data_bytes'] <= MAX_DATA_BYTES, 'E066 96-GiB new-data budget exceeded')
    return record


def prerequisites(report):
    require(PLAN.is_file(), 'E066 frozen plan must exist')
    manifest, e014_manifest, all_items, legacy = _prerequisites(report)
    items = [item for item in all_items if item['input_role'] != 'next_shifted']
    require([(i['id'], i['input_role'], i['step']) for i in items] == [
        ('e010_p030_s05', 'source_teacher', 5), ('e010_p030_s05', 'next_teacher', 6),
        ('e010_p036_s14', 'source_teacher', 14), ('e010_p036_s14', 'next_teacher', 15)],
        'Four fixed diagnostic teacher inputs changed')
    source_record = file_record(SOURCE_REPORTS/'evaluate.json')
    source = old.load_complete(base.verify_file(source_record))
    require(source['experiment'] == 'E065b' and source['complete_dit_calls'] == 9 and
            source['completed_candidate_native_calls'] == 25600 and len(source['search_layers']) == 200,
            'E065b complete search and diagnostics required')
    base.check_sources(source['sources'])
    for path, record in report['sources'].items():
        if path in (str(Path(__file__).resolve()), str(PLAN.resolve())):
            continue
        require(source['sources'].get(path) == record, f'Inherited source differs from E065b: {path}')
    source_checks = {name: file_record(SOURCE_REPORTS/f'independent_{name}.json')
                     for name in ('outputs', 'selection')}
    for name, record in source_checks.items():
        checked = old.load_complete(base.verify_file(record))
        require(checked['evaluation_sha256'] == source_record['sha256'] and
                checked['cuda_initialized'] is False and checked['new_model_forwards'] == 0,
                'E065b independent checker binding differs')
        script = HERE/f'check_h3_carry_q_{name}.py'
        require(file_record(script)['sha256'] == checked['script_sha256'], 'Executed independent checker source changed')
        report['sources'][str(script.resolve())] = file_record(script)
        if name == 'selection':
            require(checked['final_manifest_selection_bindings'] == 400 and
                    checked['candidate_zero_exact_layers'] == 200, 'Incomplete independent selection verification')
        else:
            require(checked['legacy_replay_byte_exact'] is True, 'E065b legacy replay was not exact')
    exit_record = file_record(SOURCE_REPORTS/'process_exit.json')
    process_exit = json.loads(base.verify_file(exit_record).read_text())
    require(process_exit['exit_code'] == 0 and process_exit['timeout_exit'] is False,
            'E065b worker did not finish with exit code zero')
    require(source['state'] == report['state'] and source['legacy_manifest'] == report['legacy_manifest'],
            'Selected exports do not share the inherited frozen state')
    expected_keys = {key(item) for item in items}
    report['references'] = {k: v for k, v in report['references'].items() if k in expected_keys}
    for item in items:
        k = key(item)
        require(report['references'][k] == source['references'][k], 'Diagnostic references changed')
        for arm in ARMS:
            match = [r for r in source['cases'] if (r['case_id'], r['position'], r['arm']) ==
                     (item['id'], item['input_role'], arm)]
            require(len(match) == 1 and match[0]['status'] == 'complete', 'E065b diagnostic coverage incomplete')
    exports, selected, details = {}, {}, {}
    for arm in ARMS:
        exports[arm] = source['exports'][arm]
        path = base.verify_file(exports[arm])
        exported = old.load_complete(path)
        require(exported['format'] == FORMAT and exported['recipe'] == RECIPE and
                exported['target_count'] == exported['exact_roundtrip_count'] == 200 and
                [r['name'] for r in exported['layers']] == list(TARGET_NAMES), 'Selected export manifest coverage/ABI differs')
        selected[arm], details[arm] = {}, {}
        for row in exported['layers']:
            # Path / absolute-file correctly retains the read-only E065 prefix.
            selected[arm][row['name']] = dict(file=str((path.parent/row['file']).resolve()),
                sha256=row['file_sha256'], bytes=row['file_bytes'])
            details[arm][row['name']] = row
    report['sources'].update(source['sources'])
    for path in (Path(__file__), PLAN, Path(packets.__file__), Path(e061.__file__),
                 SOURCE_REPORTS/'evaluate.json', SOURCE_REPORTS/'process_exit.json',
                 SOURCE_REPORTS/'independent_outputs.json', SOURCE_REPORTS/'independent_selection.json'):
        report['sources'][str(path.resolve())] = file_record(path)
    report.update(plan=file_record(PLAN), source_evaluation=source_record, source_checks=source_checks,
        source_process_exit=exit_record, export_manifests=exports, selected_exports=selected,
        selected_export_details=details, source_input_checks={key(i): source['inputs'][key(i)] for i in items},
        reuse=dict(input_helpers=['E059 load_input/input_signature', 'E061 prerequisites/load_reference'],
                   packet_helpers=['E065 select_rows/subset_packet/packet_payload'],
                   model_entry='base.gpu_call(kind=model_fn), preserving raw AND velocity',
                   old_module_globals_modified=False))
    return manifest, e014_manifest, items, legacy


def capture_cpu_tree(pipe, value):
    captured = []
    class Captured(Exception):
        pass
    def capture(*args, **kwargs):
        captured.append(dict(args=args, kwargs=kwargs))
        raise Captured
    state = value['state']
    packed = base.make_packed(pipe, value['embedding'], value['text_token_tags'], state)
    try:
        base.model_fn_minimax_h3(dit=capture,
            video_latents=state['video']['latents_before'], audio_latents=state['audio']['latents_before'],
            packed=packed, prompt_embeds=value['embedding'],
            timestep_video=state['video']['timestep'].reshape(1),
            timestep_audio=state['audio']['timestep'].reshape(1))
    except Captured:
        pass
    require(len(captured) == 1, 'CPU input packing must reach one sentinel, zero model forwards')
    return captured[0]


def row_selection(actual):
    selected = packets.select_rows(actual)
    domains = {name: actual['kwargs'][field]['position_ids'].reshape(-1).long().cpu()
               for name, field in (('video', 'img_pos_info'), ('audio', 'audio_pos_info'), ('text', 'text_pos_info'))}
    domains['joint'] = torch.cat(list(domains.values())).sort().values
    cu = actual['kwargs']['packed_seq_params']['cu_seqlens_q']
    require(cu.ndim == 1 and cu.dtype in (torch.int32, torch.int64) and int(cu[0]) == 0 and
            int(cu[-1]) == selected['full_rows'] and bool((cu[1:] >= cu[:-1]).all()) and
            50*int((cu[1:] > cu[:-1]).sum())+2 == 102, 'Diagnostic SDPA segment contract differs')
    return selected, domains


@torch.inference_mode()
def cpu_check(report, manifest, items, legacy):
    require(not torch.cuda.is_initialized(), 'E066 CPU check already initialized CUDA')
    inherited.cpu_check(report, manifest, items)
    require(report['inputs'] == report['source_input_checks'], 'Four diagnostic input identities changed')
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    report['row_checks'] = {}
    for item in items:
        k = key(item)
        value = inherited.load_input(item)
        actual = capture_cpu_tree(cpu_pipe, value)
        require(signature(actual) == report['inputs'][k]['actual_dit_input_signature'], 'CPU actual input mismatch')
        selected, domains = row_selection(actual)
        e061.load_reference(report['references'][k]['bf16'], signature(actual))
        report['row_checks'][k] = dict(full_rows=selected['full_rows'],
            modality_indices={name: rows.tolist() for name, rows in domains.items()},
            sample_indices=selected['indices'].tolist(), modality_counts=selected['modality_counts'],
            selection_signature=signature(selected))
        del value, actual
    state = torch.load(base.verify_file(report['state']), map_location='cpu', weights_only=True, mmap=True)
    require(set(state['layers']) == set(TARGET_NAMES) and state['config'] == legacy['state_config'], 'Frozen state layout changed')
    old_rows = {row['name']: row for row in legacy['layers']}
    for name in TARGET_NAMES:
        smooth = state['layers'][name]['smooth']
        require(smooth.dtype == torch.bfloat16 and bool(torch.isfinite(smooth).all()) and
                bool((smooth > 0).all()), 'Invalid frozen smooth')
        for arm in ARMS:
            record, detail = report['selected_exports'][arm][name], report['selected_export_details'][arm][name]
            payload = torch.load(base.verify_file(record), map_location='cpu', weights_only=True, mmap=True)
            tensors = payload['tensors']
            require(payload['name'] == name and payload['format'] == FORMAT and payload['recipe'] == RECIPE and
                    payload['shape'] == detail['shape'] == old_rows[name]['shape'] and tensors['bias'] is None,
                    'Selected payload shape/name/format/bias differs')
            require(torch.equal(tensors['smooth'], smooth) and base.tensor_record(tensors['smooth']) == detail['smooth'],
                    'Selected arm changed frozen smoothing')
            require(tensors['lr_a'].dtype == tensors['lr_b'].dtype == torch.bfloat16 and
                    list(tensors['lr_a'].shape) == [32, payload['shape'][1]] and
                    list(tensors['lr_b'].shape) == [payload['shape'][0], 32] and
                    base.tensor_record(tensors['lr_a']) == detail['lr_a'] and
                    base.tensor_record(tensors['lr_b']) == detail['lr_b'], 'Selected LR identity differs')
            packet_sig = signature(dict(codes=tensors['weight_packed'], scales=tensors['weight_scales_swizzled'],
                                        global_scale=tensors['weight_global'], original_shape=payload['shape']))
            require(packet_sig == detail['selected_packet'] and detail['source_weight_sha256'] ==
                    old_rows[name]['source_weight_sha256'], 'Selected weight packet/source identity differs')
            del payload, tensors
    with e061.safe_open(str(e061.H3_DIT_PATH), framework='pt', device='cpu') as source:
        keys = set(source.keys())
        for name in TARGET_NAMES:
            require(source.get_slice(name+'.weight').get_shape() == old_rows[name]['shape'] and
                    name+'.bias' not in keys, 'Original BF16 weight header changed')
    require(not torch.cuda.is_initialized(), 'E066 CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, input_count=4, checked_selected_exports=400,
                  checked_weight_headers=200, new_model_forwards=0, complete_dit_calls=0)


def per_row_energy(output, target, args, report):
    rows = torch.empty(output.shape[0], dtype=torch.float64, device='cpu')
    for start in range(0, output.shape[0], REDUCE_ROWS):
        guard(args, report)
        stop = min(output.shape[0], start+REDUCE_ROWS)
        values = output[start:stop].double()
        if target is not None:
            values -= target[start:stop].double()
        require(bool(torch.isfinite(values).all()), 'Nonfinite full-row output/difference')
        rows[start:stop] = values.square().sum(dim=1, dtype=torch.float64).cpu()
        del values
    return rows


def domain_metrics(energies, domains):
    metrics = {}
    for name, indices in domains.items():
        values = {field: float(vector.index_select(0, indices).sum(dtype=torch.float64))
                  for field, vector in energies.items()}
        metrics[name] = dict(rows=indices.numel(), teacher_energy=values['teacher'],
            restart_sse=values['restart'], carry_sse=values['carry'],
            carry_over_restart=values['carry']/values['restart'] if values['restart'] > 0 else None)
    return metrics


@torch.inference_mode()
def run_case(pipe, item, state, checked, cpu_pipe, args, report):
    guard(args, report, before_forward=True)
    k = key(item)
    value = inherited.load_input(item)
    validated = inherited.input_signature(value, item, cpu_pipe)
    expected = checked['inputs'][k]
    require(validated == expected['input_signature'], 'Diagnostic input changed after CPU check')
    forward = base.gpu_call(pipe, {'kind': 'model_fn'}, builder.model_value(value))
    actual, raw_capture, pending, layers, handles = [], [], {}, {}, []
    domain_cpu = {name: torch.tensor(rows, dtype=torch.long) for name, rows in checked['row_checks'][k]['modality_indices'].items()}
    sample_cpu = torch.tensor(checked['row_checks'][k]['sample_indices'], dtype=torch.long)
    indices = sample_cpu.cuda()
    full_rows = checked['row_checks'][k]['full_rows']
    sample_domains = {name: sample_cpu[torch.isin(sample_cpu, rows)] for name, rows in domain_cpu.items()}
    subset_domains = {name: torch.nonzero(torch.isin(sample_cpu, rows), as_tuple=False).flatten()
                      for name, rows in domain_cpu.items()}
    def capture_input(_module, positional, keyword):
        captured = base.inherited.tree_cpu(dict(args=positional, kwargs=keyword))
        require(signature(captured) == expected['actual_dit_input_signature'], 'Actual BF16 DiT input changed')
        selected, domains = row_selection(captured)
        require(signature(selected) == checked['row_checks'][k]['selection_signature'] and
                all(torch.equal(domains[name], domain_cpu[name]) for name in DOMAINS), 'Actual row domains changed')
        actual.append(captured)
    def capture_output(_module, positional, output):
        raw_capture.append(output)
    handles.extend((pipe.dit.register_forward_pre_hook(capture_input, with_kwargs=True),
                    pipe.dit.register_forward_hook(capture_output)))
    for name in TARGET_NAMES:
        def pre_hook(module, positional, name=name):
            guard(args, report)
            require(name not in pending and name not in layers, 'Repeated BF16 target linear')
            x = positional[0]
            require(x.ndim == 2 and x.shape[0] == full_rows and x.dtype == torch.bfloat16,
                    'BF16 target linear input shape/dtype differs')
            smooth = state['layers'][name]['smooth'].cuda()
            xs = x/smooth
            full = base.pack_activation_fast(xs, chunk_rows=CHUNK_ROWS)
            report['activation_packs'] += 1
            subset, logical_rows = packets.subset_packet(full, indices)
            xs_sample = xs.index_select(0, indices).cpu()
            detail = dict(case_id=item['id'], position=item['input_role'], layer=name,
                full_rows=full_rows, modality_indices=domain_cpu, sample_indices=sample_cpu,
                xs=xs_sample, packet=packets.packet_payload(subset), logical_scale_rows=logical_rows.cpu(),
                full_packet_global_signature=base.tensor_record(full.global_scale), full_packet_shape=list(full.original_shape),
                full_matrix_hash_policy='not_retained_not_hashed', smooth_signature=base.tensor_record(smooth),
                source_signature=base.tensor_record(xs_sample), packet_subset_bytes_verified=True,
                export_files={arm: report['selected_exports'][arm][name] for arm in ARMS})
            pending[name] = (xs, full, subset, detail)
            del smooth, logical_rows, xs_sample
        def post_hook(module, positional, output, name=name):
            xs, full, subset, detail = pending.pop(name)
            require(output.ndim == 2 and output.shape[0] == full_rows and output.dtype == torch.bfloat16,
                    'BF16 target linear output shape/dtype differs')
            detail.update(output_features=output.shape[1], target=output.index_select(0, indices).cpu())
            detail['target_signature'] = base.tensor_record(detail['target'])
            energies = {'teacher': per_row_energy(output, None, args, report)}
            sampled, output_signatures = {}, {}
            subset_outputs, subset_signatures = {}, {}
            subset_energies = {'teacher': energies['teacher'].index_select(0, sample_cpu)}
            subset_xs = xs.index_select(0, indices)
            subset_target = output.index_select(0, indices)
            for arm in ARMS:
                guard(args, report)
                record = report['selected_exports'][arm][name]
                payload = torch.load(base.verify_file(record), map_location='cpu', weights_only=True, mmap=True)
                native = NativeH3Linear.from_export(payload, device='cuda', activation_packer=base.pack_activation_fast,
                                                   chunk_rows=CHUNK_ROWS)
                require(native.in_features == xs.shape[1] and native.out_features == output.shape[1] and
                        base.tensor_record(native.smooth) == detail['smooth_signature'], 'Local selected module ABI/smooth differs')
                report['attempted_local_native_calls'] += 1
                guard(args, report)
                main = native.main_from_packet(full, include_bias=True)
                report['local_native_calls'] += 1
                report['full_native_calls'] += 1
                branch = 1.0*F.linear(F.linear(xs, native.lr_a), native.lr_b)
                result = main+branch
                del main, branch
                require(result.dtype == torch.bfloat16 and result.shape == output.shape, 'Local output contract changed')
                energies[arm] = per_row_energy(result, output, args, report)
                sampled[arm] = result.index_select(0, indices).cpu()
                output_signatures[arm] = base.tensor_record(sampled[arm])
                del result
                guard(args, report)
                report['attempted_local_native_calls'] += 1
                guard(args, report)
                main = native.main_from_packet(subset, include_bias=True)
                report['local_native_calls'] += 1
                report['subset_native_calls'] += 1
                branch = 1.0*F.linear(F.linear(subset_xs, native.lr_a), native.lr_b)
                result = main+branch
                del main, branch
                require(result.dtype == torch.bfloat16 and result.shape == subset_target.shape,
                        'M512 native output contract changed')
                subset_energies[arm] = per_row_energy(result, subset_target, args, report)
                subset_outputs[arm] = result.cpu()
                subset_signatures[arm] = base.tensor_record(subset_outputs[arm])
                del result, native, payload
            detail.update(outputs=sampled, output_signatures=output_signatures, row_energy=energies,
                metrics=domain_metrics(energies, domain_cpu), sample_metrics=domain_metrics(energies, sample_domains),
                subset_outputs=subset_outputs, subset_output_signatures=subset_signatures,
                subset_metrics=domain_metrics(subset_energies, subset_domains))
            artifact = save_tensor(detail, DATA/'layers'/k.replace('/', '_')/(name+'.pt'), report)
            layers[name] = dict(artifact=artifact, metrics=detail['metrics'], sample_metrics=detail['sample_metrics'],
                               subset_metrics=detail['subset_metrics'])
            report['active_call']['completed_layers'] = len(layers)
            report['active_layer_artifacts'] = layers
            if len(layers) % 20 == 0:
                report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                save(report, args.output)
                print(f'E066 {k}: {len(layers)}/200 paired full-M layers', flush=True)
            del xs, full, subset, detail, energies, sampled, subset_xs, subset_target, subset_outputs, subset_energies
            guard(args, report)
            # No replacement: the original BF16 output proceeds unchanged.
        module = pipe.dit.get_submodule(name)
        handles.extend((module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)))
    audit = base.RuntimeAudit()
    audit.phase = 'resident_bf16'
    report['attempted_dit_calls'] += 1
    report['active_call'] = dict(case_id=item['id'], position=item['input_role'], completed_layers=0)
    before_counts = (report['activation_packs'], report['full_native_calls'], report['subset_native_calls'])
    save(report, args.output)
    try:
        with audit.installed(), base.collect_fastpack_checks() as fastpack:
            result = forward()
            torch.cuda.synchronize()
        report['complete_dit_calls'] += 1
    finally:
        for handle in handles:
            handle.remove()
    require(len(actual) == len(raw_capture) == 1 and not pending and list(layers) == list(TARGET_NAMES),
            'BF16 full call did not visit each target exactly once')
    row = audit.row()
    require({name: row[name] for name in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')} ==
            dict(sdpa_calls=102, scaled_mm_calls=800, disk_loads=0), 'Actual runtime call contract differs')
    require(fastpack.summary['checked_calls'] == 200 and fastpack.summary['invalid_calls'] == 0 and
            (report['activation_packs']-before_counts[0], report['full_native_calls']-before_counts[1],
             report['subset_native_calls']-before_counts[2]) == (200, 400, 400),
            'Full-input pack or paired native count differs')
    raw, raw_records = base.cpu_outputs(raw_capture[0])
    velocities, velocity_records = base.cpu_outputs(result)
    teacher = e061.load_reference(report['references'][k]['bf16'], expected['actual_dit_input_signature'])
    replay = all(torch.equal(values[modality], teacher[field][modality]) and
                 signature(values[modality]) == report['references'][k]['bf16'][field][modality]
                 for field, values in (('raw_outputs', raw), ('velocities', velocities)) for modality in inherited.MODALITIES)
    require(replay, 'BF16 raw/velocity output did not byte-exactly replay historical teacher')
    payload = dict(case_id=item['id'], position=item['input_role'], step=item['step'], input_signature=validated,
        actual_dit_inputs=actual[0], raw_outputs=raw, velocities=velocities, modality_indices=domain_cpu,
        sample_indices=sample_cpu, reference_artifacts=report['references'][k])
    artifact = save_tensor(payload, DATA/'cases'/(k.replace('/', '_')+'.pt'), report)
    report['cases'].append(dict(case_id=item['id'], position=item['input_role'], step=item['step'], key=k,
        artifact=artifact, actual_dit_input_signature=signature(actual[0]), raw_outputs=raw_records,
        velocities=velocity_records, historical_replay_exact=replay, runtime_audit=row,
        fastpack_checks=fastpack.summary, layers=layers))
    for field in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads'):
        report['runtime_totals'][field] += row[field]
    report['runtime_totals']['pack_calls'] += fastpack.summary['checked_calls']
    report['active_layer_artifacts'] = {}
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save(report, args.output)
    del forward, value, result, raw_capture, actual, teacher, payload, raw, velocities
    gc.collect()
    guard(args, report)


def summarize(report):
    summary = []
    for case in report['cases']:
        result = dict(case_id=case['case_id'], position=case['position'], groups={})
        for group in ('all', 'attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2'):
            names = [name for name in TARGET_NAMES if group == 'all' or name.endswith('.'+group)]
            require(bool(names), f'No layers in summary group {group}')
            result['groups'][group] = {}
            for modality in DOMAINS:
                records = [case['layers'][name]['metrics'][modality] for name in names]
                ratios = [r['carry_over_restart'] for r in records if r['carry_over_restart'] is not None]
                result['groups'][group][modality] = dict(layer_count=len(records),
                    improved=sum(r['carry_sse'] < r['restart_sse'] for r in records),
                    reversed=sum(r['carry_sse'] > r['restart_sse'] for r in records),
                    equal=sum(r['carry_sse'] == r['restart_sse'] for r in records),
                    zero_restart=sum(r['restart_sse'] == 0 for r in records),
                    ratio_median=statistics.median(ratios) if ratios else None,
                    ratio_min=min(ratios) if ratios else None, ratio_max=max(ratios) if ratios else None)
        summary.append(result)
    return dict(cases=summary, no_cross_layer_raw_sse_sum=True,
        scope='Local transfer diagnostic only; no attribution of full-model reversal or video quality claim')


@torch.inference_mode()
def evaluate(args, report, manifest, e014_manifest, items):
    checked = old.load_complete(CHECK_REPORT)
    for field in ('sources', 'source_evaluation', 'source_checks', 'source_process_exit', 'export_manifests',
                  'selected_exports', 'selected_export_details', 'state', 'references', 'source_input_checks'):
        require(report[field] == checked[field], f'E066 CPU binding changed: {field}')
    require(checked['cuda_initialized'] is False and checked['checked_selected_exports'] == 400,
            'E066 complete CPU-only check required')
    report.update(inputs=checked['inputs'], row_checks=checked['row_checks'], cpu_check_reference=file_record(CHECK_REPORT))
    require(not DATA.exists(), f'Refusing existing E066 data directory {DATA}')
    DATA.mkdir(parents=True)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    setup_path = DATA/'setup_bf16.json'
    old_check = old.load_complete(base.verify_file(manifest['e014_check']))
    setup = dict(experiment='E066_inherited_E014_setup', status='running', complete_dit_calls=0, sources=old_check['sources'])
    setup_args = types.SimpleNamespace(arm='bf16', deadline_unix=args.deadline_unix,
        check_report=Path(manifest['e014_check']['file']), output=setup_path)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    report['data_bytes'] += report['model_setup']['bytes']
    builder.configure_schedule(pipe, manifest['settings'])
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    state = torch.load(base.verify_file(report['state']), map_location='cpu', weights_only=True, mmap=True)
    for item in items:
        run_case(pipe, item, state, checked, cpu_pipe, args, report)
    require(report['attempted_dit_calls'] == report['complete_dit_calls'] == 4 and
            report['activation_packs'] == 800 and
            report['attempted_local_native_calls'] == report['local_native_calls'] == 3200 and
            report['full_native_calls'] == report['subset_native_calls'] == 1600 and
            report['runtime_totals'] == dict(sdpa_calls=408, scaled_mm_calls=3200, disk_loads=0, pack_calls=800),
            'E066 final execution counts differ')
    guard(args, report)
    report.update(status='complete', active_call=None, summary=summarize(report),
        limitation='Independent checker can recompute saved 512-row channel SSE and all row reductions, not unsaved channels or native kernels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate'), required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or REPORTS/f'{args.phase}.json'
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    started = time.time()
    report = dict(experiment='E066', phase=args.phase, status='running', start_epoch=started,
        deadline_unix=args.deadline_unix, wall_budget_seconds=WALL_SECONDS, max_dit_calls=MAX_CALLS,
        data_byte_limit=MAX_DATA_BYTES, data_bytes=0, attempted_dit_calls=0, complete_dit_calls=0,
        activation_packs=0, attempted_local_native_calls=0, local_native_calls=0,
        full_native_calls=0, subset_native_calls=0, peak_allocated_bytes=0,
        cases=[], active_layer_artifacts={}, runtime_totals=dict(sdpa_calls=0, scaled_mm_calls=0, disk_loads=0, pack_calls=0),
        scope='Four BF16 teacher replays plus selected full-M/M512 comparisons sharing full-input packs; no fitting')
    try:
        if args.phase == 'evaluate':
            require(args.deadline_unix is not None and 0 < args.deadline_unix-started <= WALL_SECONDS,
                    'E066 needs one root-assigned deadline no more than 1800 seconds')
            def timeout(_signal, _frame):
                raise TimeoutError('E066 absolute wall deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, e014_manifest, items, legacy = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, manifest, items, legacy)
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
