#!/usr/bin/env python3
"""E015 thin execution layer: prepared cross-modal states, one native update.

No research statistics or performance measurements are computed here. The
frozen E014 loader/model_fn/RuntimeAudit and original pipeline.step are reused.
CPU validation precedes the externally scheduled 2 + 8 + 8 GPU forwards.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe_h3_plain_baseline as base
import torch

DATA = Path('/data1/models/svdquant-wjq/research/20261002/E015')
REPORTS = ROOT/'results/research/E015'
MANIFEST = ROOT/'research_state/06_experiments/E015_h3_crossmodal_propagation_manifest.json'
PLAN = MANIFEST.with_name('E015_h3_crossmodal_propagation_plan.md')
MODALITIES = ('video', 'audio')
ARMS = ('bf16', 'svd', 'plain')
CORNERS = ('BB', 'QB', 'BQ', 'QQ')
CALLS = {'bf16': 2, 'svd': 8, 'plain': 8}
save, file_record, verify_file = base.save, base.file_record, base.verify_file
tensor_record, tree_signature = base.tensor_record, base.tree_signature


def require(value, message):
    if not value:
        raise RuntimeError(message)


def row_key(row):
    return f"{row['id']}/{row['arm']}/{row['corner']}"


def load_complete(path):
    value = json.loads(Path(path).read_text())
    require(value['status'] == 'complete', f'Incomplete prerequisite: {path}')
    return value


def deadline(args, report):
    require(args.deadline_unix is not None, 'GPU execution requires original shared --deadline-unix')
    require(time.time() < args.deadline_unix, 'Shared E015 deadline reached')
    require(report['complete_dit_calls'] < CALLS[args.arm], 'Per-arm call budget exhausted')
    require(report.get('prior_complete_dit_calls', 0)+report['complete_dit_calls'] < 20, 'Global call budget exhausted')
    base.inherited.memory_guard()


def prerequisites(args, report):
    import prepare_h3_crossmodal_states as builder
    manifest = json.loads(args.manifest.read_text())
    require(manifest['experiment'] == 'E015' and manifest['arms'] == list(ARMS)
            and manifest['corners'] == list(CORNERS)
            and manifest['corner_coordinate_order'] == list(MODALITIES), 'Protocol layout changed')
    require([(r['id'], r['source_step'], r['next_step']) for r in manifest['cases']] ==
            [('e010_p030_s05', 5, 6), ('e010_p036_s14', 14, 15)], 'Fixed case list changed')
    require(manifest['budget']['planned_complete_dit_calls'] == 18
            and manifest['budget']['max_complete_dit_calls'] == 20, 'Call allocation changed')
    require(file_record(args.plan) == manifest['plan'], 'Protocol plan binding changed')
    require(args.data_dir.resolve() == Path(manifest['data_dir']).resolve(), 'Wrong E015 data directory')
    prepared = load_complete(args.prepare_report)
    require(prepared.get('schema') == builder.SCHEMA and prepared.get('cuda_initialized') is False
            and prepared['manifest'] == file_record(args.manifest), 'Prepared source/CPU contract changed')
    base.check_sources(prepared['sources'])
    inherited_check = load_complete(verify_file(manifest['e014_check']))
    base.check_sources(inherited_check['sources'])
    e014_manifest = json.loads(verify_file(manifest['e014_manifest']).read_text())
    require(e014_manifest['svd_export_dir'] == manifest['svd_export_dir']
            and e014_manifest['plain_export_dir'] == manifest['plain_export_dir'], 'Export directories changed')
    for arm in ARMS:
        prior = load_complete(verify_file(manifest['e014_evaluations'][arm]))
        base.check_sources(prior['sources'])
        require(prior['arm'] == arm, 'E014 reference arm mismatch')
    source_paths = set(Path(p) for p in inherited_check['sources']) | {
        Path(__file__), HERE/'prepare_h3_crossmodal_states.py', args.manifest, args.plan,
    }
    report.update(sources={str(p.resolve()): file_record(p) for p in sorted(source_paths)},
                  manifest=file_record(args.manifest), plan=file_record(args.plan),
                  prepared_reference=file_record(args.prepare_report),
                  e014_check_reference=manifest['e014_check'])
    rows = prepared['cases']
    expected = {(case['id'], arm, corner) for case in manifest['cases'] for arm in ARMS
                for corner in (('BB',) if arm == 'bf16' else CORNERS)}
    require(len(rows) == 18 and {(r['id'], r['arm'], r['corner']) for r in rows} == expected,
            'Prepared table must contain exactly the prescribed 18 cases')
    for row in rows:
        verify_file(row['artifact'])
        case = next(c for c in manifest['cases'] if c['id'] == row['id'])
        require(row['next_step'] == case['next_step'], 'Prepared next step changed')
    for case in manifest['cases']:
        verify_file(case['prepared'])
        for name in ('teacher_state', 'teacher_next_state', 'e014_outputs'):
            for ref in case[name].values():
                verify_file(ref)
        verify_file(case['bf16_trajectory_reference'])
    # Reuse full model hashes from E010; only current stat metadata is read.
    asset = inherited_check['asset_binding']
    for path, record in asset['assets'].items():
        st = Path(path).stat()
        require(st.st_size == record['bytes'] and st.st_mtime_ns == record['mtime_ns'],
                f'Model asset changed since prior full hash: {path}')
    report['asset_binding'] = asset
    return builder, manifest, prepared, e014_manifest


def validate_value(value, row, case, pipe, builder):
    embedding, tags, state = value['embedding'], value['text_token_tags'], value['state']
    require(embedding.ndim == 2 and embedding.shape[1] == 5120 and embedding.dtype == torch.bfloat16,
            'Embedding must be actual BF16 [tokens,5120]')
    require(tags.ndim == 1 and tags.numel() == embedding.shape[0], 'Text tag layout changed')
    require(bool(torch.isfinite(embedding).all()), 'Nonfinite embedding')
    original_prompt = torch.load(case['prepared']['file'], map_location='cpu', weights_only=True, mmap=True)
    require(tensor_record(embedding) == tensor_record(original_prompt['embedding']) and
            tensor_record(tags) == tensor_record(original_prompt['text_token_tags']), 'Prepared condition differs from original E010')
    require(set(state) == set(MODALITIES), 'Unexpected modalities')
    teacher = value['teacher_reference']
    for modality in MODALITIES:
        source = torch.load(case['teacher_next_state'][modality]['file'], map_location='cpu', weights_only=True, mmap=True)
        require(tree_signature(teacher[modality]) == tree_signature(source), 'Teacher-next reference differs from original bytes')
        current = state[modality]
        require(set(current) == {'latents_before', 'timestep', 'sigma'}, 'Prepared state has unexpected fields')
        require(current['latents_before'].dtype == torch.bfloat16 and
                current['latents_before'].shape == source['latents_before'].shape, 'Latent layout changed')
        for key in ('timestep', 'sigma'):
            scalar = current[key]
            require(scalar.ndim == 0 and scalar.dtype == torch.float32 and tensor_record(scalar) == tensor_record(source[key]),
                    f'{modality} {key}: original next-step scalar changed')
        require(all(bool(torch.isfinite(t).all()) and t.device.type == 'cpu' for t in current.values()), 'Invalid prepared CPU state')
        coordinate = row['corner'][MODALITIES.index(modality)]
        if coordinate == 'B':
            require(tensor_record(current['latents_before']) == tensor_record(source['latents_before']), 'B coordinate is not teacher-next bytes')
    packed = base.make_packed(pipe, embedding, tags, state)
    require(50*int((packed['cu_seqlens'][1:] > packed['cu_seqlens'][:-1]).sum())+2 == 102,
            'Packed attention segment count changed')
    if row['arm'] == 'bf16':
        replay = builder.advance(pipe, state, {m: teacher[m]['noise_pred'] for m in MODALITIES}, row['next_step'])
        require(all(tensor_record(replay[m]) == tensor_record(teacher[m]['latents_after']) for m in MODALITIES),
                'CPU original second scheduler update does not replay teacher bytes')
    return {'embedding': tensor_record(embedding), 'text_token_tags': tensor_record(tags),
            'state': tree_signature(state), 'packed': tree_signature(packed)}


def cpu_check(args, report, builder, manifest, prepared):
    require(not torch.cuda.is_initialized(), 'CPU check already initialized CUDA')
    torch.set_num_threads(6)
    pipe = builder.make_cpu_pipeline(manifest['settings'])
    report['inputs'] = {}
    for row in prepared['cases']:
        value = builder.load_prepared_case(row)
        case = next(c for c in manifest['cases'] if c['id'] == row['id'])
        report['inputs'][row_key(row)] = validate_value(value, row, case, pipe, builder)
    for case in manifest['cases']:
        teacher = report['inputs'][f"{case['id']}/bf16/BB"]
        for arm in ('svd', 'plain'):
            signatures = {corner: report['inputs'][f"{case['id']}/{arm}/{corner}"] for corner in CORNERS}
            for modality, index in (('video', 0), ('audio', 1)):
                for coordinate in 'BQ':
                    choices = [signatures[c]['state'][modality] for c in CORNERS if c[index] == coordinate]
                    require(choices[0] == choices[1], 'Corner coordinate does not share identical input state')
            for corner, sig in signatures.items():
                require(sig['embedding'] == teacher['embedding'] and sig['text_token_tags'] == teacher['text_token_tags']
                        and sig['packed'] == teacher['packed'], 'Condition or packed metadata changed between corners')
            require(signatures['BB'] == teacher, 'BB is not identical across model arms')
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, prepared_rows=18, complete_dit_calls=0)


def setup_model(args, report, manifest, e014_manifest):
    deadline(args, report)
    # Frozen E014 setup validates its own existing CPU contract. Keep those
    # fields separate from the new E015 CPU contract; no monkeypatch is needed.
    old_check = load_complete(manifest['e014_check']['file'])
    setup_path = args.data_dir/'setup'/f'{args.arm}.json'
    require(not setup_path.exists(), f'Refusing old setup metadata {setup_path}')
    setup = {'experiment': 'E015_inherited_E014_model_setup', 'status': 'running',
             'complete_dit_calls': 0, 'sources': old_check['sources']}
    setup_args = types.SimpleNamespace(arm=args.arm, deadline_unix=args.deadline_unix,
        check_report=Path(manifest['e014_check']['file']), output=setup_path)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    report['resident_model_storage'] = setup['resident_model_storage']
    return pipe


@torch.inference_mode()
def verify_first_gpu_updates(args, report, builder, manifest, prepared, pipe):
    """Check actual quantized velocities under original GPU scheduler arithmetic."""
    report['first_update_gpu_replay'] = []
    for case in manifest['cases']:
        deadline(args, report)
        source_state = {m: torch.load(case['teacher_state'][m]['file'], map_location='cpu',
                                      weights_only=True, mmap=True) for m in MODALITIES}
        velocity_source = torch.load(case['e014_outputs'][args.arm]['file'], map_location='cpu',
                                     weights_only=True, mmap=True)
        velocities = velocity_source['velocities']
        corner = 'BB' if args.arm == 'bf16' else 'QQ'
        row = next(r for r in prepared['cases'] if r['id'] == case['id'] and
                   r['arm'] == args.arm and r['corner'] == corner)
        target = builder.load_prepared_case(row)
        updated = builder.advance(pipe, base.tree_device(source_state, 'cuda'),
                                  base.tree_device(velocities, 'cuda'), case['source_step'])
        torch.cuda.synchronize()
        actual = {m: tensor_record(updated[m]) for m in MODALITIES}
        expected = {m: tensor_record(target['state'][m]['latents_before']) for m in MODALITIES}
        proof = {'id': case['id'], 'arm': args.arm, 'source_step': case['source_step'],
                 'source_state_files': case['teacher_state'], 'velocity_source': case['e014_outputs'][args.arm],
                 'source_state_signature': tree_signature(source_state), 'velocity_signature': tree_signature(velocities),
                 'prepared_target': row['artifact'], 'gpu_updated': actual, 'cpu_prepared': expected,
                 'byte_exact': {m: actual[m] == expected[m] for m in MODALITIES},
                 'additional_dit_calls': 0}
        report['first_update_gpu_replay'].append(proof)
        save(report, args.output)
        require(all(proof['byte_exact'].values()), 'GPU original first update differs from prepared CPU B/Q state')
        del source_state, velocity_source, velocities, target, updated


@torch.inference_mode()
def evaluate(args, report, builder, manifest, prepared, e014_manifest):
    checked = load_complete(args.check_report)
    require(checked['sources'] == report['sources'] and checked['prepared_reference'] == report['prepared_reference']
            and checked['cuda_initialized'] is False, 'E015 CPU/source binding changed')
    report['cpu_check_reference'] = file_record(args.check_report)
    report['prior_complete_dit_calls'] = 0
    report['prior_arms'] = {}
    for arm in ARMS[:ARMS.index(args.arm)]:
        path = args.report_dir/f'evaluate_{arm}.json'
        prior = load_complete(path)
        require(prior['sources'] == report['sources'] and prior['prepared_reference'] == report['prepared_reference']
                and prior['complete_dit_calls'] == CALLS[arm], 'Prior arm source/count mismatch')
        report['prior_complete_dit_calls'] += prior['complete_dit_calls']
        report['prior_arms'][arm] = file_record(path)
    pipe = setup_model(args, report, manifest, e014_manifest)
    builder.configure_schedule(pipe, manifest['settings'])
    verify_first_gpu_updates(args, report, builder, manifest, prepared, pipe)
    report['cases'] = []
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    rows = [r for r in prepared['cases'] if r['arm'] == args.arm]
    for row in rows:
        deadline(args, report)
        value = builder.load_prepared_case(row)
        case = next(c for c in manifest['cases'] if c['id'] == row['id'])
        signature = validate_value(value, row, case, cpu_pipe, builder)
        require(signature == checked['inputs'][row_key(row)], 'Actual corner differs from CPU-checked input')
        forward = base.gpu_call(pipe, {'kind': 'model_fn'}, builder.model_value(value))
        actual, raw_capture = [], []
        def capture_input(_module, positional, keyword):
            actual.append(base.tree_device({'args': positional, 'kwargs': keyword}, 'cpu'))
        def capture_output(_module, positional, output):
            raw_capture.append(output)
        pre = pipe.dit.register_forward_pre_hook(capture_input, with_kwargs=True)
        post = pipe.dit.register_forward_hook(capture_output)
        audit = base.RuntimeAudit()
        audit.phase = 'resident_bf16' if args.arm == 'bf16' else 'native'
        context = base.collect_fastpack_checks() if args.arm != 'bf16' else nullcontext(None)
        print(f'E015 {row_key(row)} next_step={row["next_step"]}', flush=True)
        try:
            with audit.installed(), context as checks:
                result = forward()
                torch.cuda.synchronize()
            report['complete_dit_calls'] += 1
        finally:
            pre.remove()
            post.remove()
        base.audit_contract(audit.row(), args.arm)
        require(len(actual) == len(raw_capture) == 1, 'Expected exactly one actual DiT call')
        require(checks is None or checks.summary['checked_calls'] == 200, 'Incomplete quantization checks')
        raw, raw_records = base.cpu_outputs(raw_capture[0])
        velocities, velocity_records = base.cpu_outputs(result)
        gpu_state = base.tree_device(value['state'], 'cuda')
        # Crucially, each corner supplies its OWN latents as scheduler sample.
        updated = builder.advance(pipe, gpu_state, dict(zip(MODALITIES, result, strict=True)), row['next_step'])
        torch.cuda.synchronize()
        endpoints = {m: updated[m].detach().cpu() for m in MODALITIES}
        require(all(bool(torch.isfinite(x).all()) for x in endpoints.values()), 'Nonfinite updated endpoint')
        endpoint_records = {m: tensor_record(x) for m, x in endpoints.items()}
        replay = None
        if args.arm == 'bf16':
            replay = {'raw_exact': {m: raw_records[m]['sha256'] == case['teacher_next_raw_output_sha256'][m] for m in MODALITIES},
                      'velocity_exact': {m: velocity_records[m] == tensor_record(value['teacher_reference'][m]['noise_pred']) for m in MODALITIES},
                      'endpoint_exact': {m: endpoint_records[m] == tensor_record(value['teacher_reference'][m]['latents_after']) for m in MODALITIES}}
            require(all(flag for group in replay.values() for flag in group.values()), 'Historical teacher-next replay failed')
        path = args.data_dir/'evaluate'/args.arm/row['id']/(row['corner']+'.pt')
        require(not path.exists(), f'Refusing existing output {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'id': row['id'], 'arm': args.arm, 'corner': row['corner'], 'next_step': row['next_step'],
                    'prepared_artifact': row['artifact'], 'input_signature': signature,
                    'actual_dit_inputs': actual[0], 'raw_outputs': raw, 'velocities': velocities,
                    'next_endpoints': endpoints}, path)
        report['cases'].append({'id': row['id'], 'arm': args.arm, 'corner': row['corner'],
            'next_step': row['next_step'], 'status': 'complete', 'prepared_artifact': row['artifact'],
            'input_signature': signature, 'actual_dit_input_signature': tree_signature(actual[0]),
            'raw_outputs': raw_records, 'velocities': velocity_records, 'next_endpoints': endpoint_records,
            'historical_bf16_replay': replay, 'runtime_audit': audit.row(),
            'fastpack_checks': checks.summary if checks is not None else None, 'artifact': file_record(path)})
        save(report, args.output)
        del forward, value, result, raw_capture, actual, raw, velocities, gpu_state, updated, endpoints
        gc.collect()
        base.inherited.memory_guard()
        require(time.time() < args.deadline_unix, 'Shared E015 deadline reached after forward')
    require(report['complete_dit_calls'] == CALLS[args.arm], 'Incomplete prescribed arm calls')
    report.update(status='complete', peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate'), required=True)
    parser.add_argument('--arm', choices=ARMS, default='bf16')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--prepare-report', type=Path, default=REPORTS/'E015_prepare.json')
    parser.add_argument('--check-report', type=Path, default=REPORTS/'check.json')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=REPORTS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.output = args.output or args.report_dir/('check.json' if args.phase == 'check' else f'evaluate_{args.arm}.json')
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    report = {'experiment': 'E015', 'phase': args.phase, 'arm': args.arm, 'status': 'running',
              'deadline_unix': args.deadline_unix, 'complete_dit_calls': 0,
              'scope': 'Cross-modal one-step propagation raw outputs; no GPU-side research metrics or timing claim'}
    try:
        builder, manifest, prepared, inherited_manifest = prerequisites(args, report)
        if args.phase == 'check':
            cpu_check(args, report, builder, manifest, prepared)
        else:
            evaluate(args, report, builder, manifest, prepared, inherited_manifest)
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc())
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
