#!/usr/bin/env python3
"""E015 CPU-only tensor audit and FP64 four-corner reduction.

Reuses the independent E014 numerical utilities, not the experiment runners.
Original scheduler code is loaded alone, with CUDA hidden. No weight re-audit,
model loading, quality claim, or threshold-based mechanism decision is made.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import summarize_h3_plain_baseline as prior

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E015'
MANIFEST = ROOT/'research_state/06_experiments/E015_h3_crossmodal_propagation_manifest.json'
FLOW = Path('/home/wjq/workspace/DiffSynth-Studio/diffsynth/diffusion/flow_match.py')
ARMS, MODS, CORNERS = ('bf16', 'svd', 'plain'), ('video', 'audio'), ('BB', 'QB', 'BQ', 'QQ')
require, record, signature = prior.require, prior.tensor_record, prior.signature
metrics, ratio = prior.metrics, prior.ratio


def load_tensor(row, files):
    files.record(row)
    return prior.load_tensor_file(row['file'])


def same(a, b, label):
    require(record(a) == record(b), f'Byte mismatch: {label}')


def schedules(files):
    files.verify(FLOW)
    spec = importlib.util.spec_from_file_location('e015_original_flow_match', FLOW)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for modality, shift in [('video', 12.), ('audio', 3.)]:
        result[modality] = module.FlowMatchScheduler('MiniMax-H3')
        result[modality].set_timesteps(20, shift=shift)
    return result


def step(scheduler, state, velocity, index):
    same(state['timestep'], scheduler.timesteps[index], 'timestep')
    same(state['sigma'], scheduler.sigmas[index], 'sigma')
    require(state['latents_before'].dtype == velocity.dtype == torch.bfloat16, 'Original BF16 arithmetic required')
    # Original pipeline.step delegates to this exact method without masks here.
    return scheduler.step(velocity, state['timestep'], state['latents_before'])


def actual_input_check(actual, payload, baseline_metadata):
    require(actual['args'] == (), 'Unexpected positional DiT input')
    kw, state = actual['kwargs'], payload['state']
    text_len = payload['embedding'].shape[0]
    v, a = prior.video_rows(state['video']['latents_before']), prior.audio_rows(state['audio']['latents_before'])
    used, seq = text_len+len(a)+len(v), kw['x'].shape[1]
    require(seq == ((used+63)//64)*64, 'Wrong packed length')
    positions = dict(audio=torch.arange(text_len, text_len+len(a)), video=torch.arange(text_len+len(a), used))
    x = torch.zeros_like(kw['x']); ax = torch.zeros_like(kw['audio_x'])
    x[0, positions['video']] = v; ax[0, positions['audio']] = a
    same(kw['x'], x, 'actual video latent packing')
    same(kw['audio_x'], ax, 'actual audio latent packing')
    same(kw['prompt_embeds'], payload['embedding'], 'actual embedding')
    same(kw['img_pos_info']['position_ids'], positions['video'], 'video positions')
    same(kw['audio_pos_info']['position_ids'], positions['audio'], 'audio positions')
    same(kw['packed_seq_params']['cu_seqlens_q'], torch.tensor([0, used, seq], dtype=torch.int32), 'cu_seqlens')
    ts = torch.full((seq,), 1.-float(state['video']['timestep'])/1000, dtype=torch.float32)
    ts[positions['audio']] = 1.-float(state['audio']['timestep'])/1000
    unique, inverse = torch.unique(ts, sorted=True, return_inverse=True)
    same(kw['unique_timesteps'], unique, 'actual unique next timesteps')
    same(kw['inverse_indices'], inverse, 'actual next time indices')
    metadata = signature({k: v for k, v in kw.items() if k not in ('x', 'audio_x')})
    require(baseline_metadata is None or metadata == baseline_metadata, 'Non-latent actual metadata differs by corner/arm')
    return metadata


def four_corner(values, reference, restore_corner):
    y = {key: value.double().reshape(-1) for key, value in values.items()}
    ref = reference.double().reshape(-1)
    terms = dict(base_error=y['BB']-ref, video_input_delta=y['QB']-y['BB'],
                 audio_input_delta=y['BQ']-y['BB'], interaction=y['QQ']-y['QB']-y['BQ']+y['BB'])
    eqq = y['QQ']-ref
    identity_max = float((sum(terms.values())-eqq).abs().max())
    require(identity_max <= 1e-12*max(1., float(eqq.abs().max())), 'Four-corner vector identity failed')
    energies = {key: float(v.square().sum()) for key, v in terms.items()}
    crosses = {f'{a}__{b}': 2*float((terms[a]*terms[b]).sum()) for a, b in itertools.combinations(terms, 2)}
    eqq2 = float(eqq.square().sum())
    energy_sum = sum(energies.values())+sum(crosses.values())
    require(math.isclose(energy_sum, eqq2, rel_tol=1e-10, abs_tol=1e-12), 'Four-corner energy identity failed')
    restored = y[restore_corner]-ref
    delta = y['QQ']-y[restore_corner]
    er2, delta2 = float(restored.square().sum()), float(delta.square().sum())
    cross = 2*float((restored*delta).sum())
    net = eqq2-er2
    require(math.isclose(net, delta2+cross, rel_tol=1e-10, abs_tol=1e-12), 'Oracle net-error identity failed')
    return dict(term_energies=energies, twice_signed_cross_terms=crosses,
                qq_error_energy=eqq2, sum_energy_with_cross_terms=energy_sum,
                vector_identity_max_abs=identity_max,
                interaction_over_qq_error_energy=ratio(energies['interaction'], eqq2),
                oracle_restore=dict(from_corner='QQ', to_corner=restore_corner,
                    qq_error_energy=eqq2, restored_error_energy=er2,
                    net_error_energy_reduction=net, relative_error_energy_reduction=ratio(net, eqq2),
                    restored_over_qq_error_energy=ratio(er2, eqq2), output_delta_energy=delta2,
                    twice_restored_error_dot_delta=cross,
                    restored_error_delta_cosine=ratio(cross/2, math.sqrt(er2*delta2)),
                    positive_means='Oracle replacement reduces net endpoint error; negative means it removes cancellation'))


def inherited_and_chain(args, files):
    manifest, _ = files.json(MANIFEST, complete=False)
    files.record(manifest['plan'])
    files.record(manifest['e014_independent_summary'])
    inherited, _ = files.json(manifest['e014_independent_summary']['file'])
    require(inherited['export']['target_count'] == 200, 'Missing E014 export audit')
    frozen, frozen_rec = files.json(args.report_dir/'frozen_contract.json')
    for path, sha in frozen['files'].items():
        files.verify(path, sha)
    prepared, prepared_rec = files.json(args.report_dir/'E015_prepare.json')
    checked, checked_rec = files.json(args.report_dir/'check.json')
    require(prepared['cuda_initialized'] is checked['cuda_initialized'] is False, 'CPU preparations initialized CUDA')
    require(len(prepared['cases']) == len(checked['inputs']) == 18, 'Prepared call count changed')
    files.sources(prepared['sources']); files.sources(checked['sources'])
    launcher, launcher_rec = files.json(args.report_dir/'launcher.json')
    require(launcher['freeze_sha256'] == frozen_rec['sha256'] and launcher['planned_dit_calls'] == 18
            and launcher['seconds_total'] <= 1200 and launcher['deadline_epoch']-launcher['start_epoch'] == 1200, 'Launcher budget/freeze differs')
    reports, pids, gpu_ids, last_end = {}, set(), set(), launcher['start_epoch']
    require([s['name'] for s in launcher['stages']] == list(ARMS), 'Unexpected arm ordering')
    for stage, arm, count in zip(launcher['stages'], ARMS, (2, 8, 8), strict=True):
        require(stage['status'] == 'complete' and stage['returncode'] == 0 and stage['planned_dit_calls'] == count, 'Stage failed/count changed')
        require(stage['pid'] not in pids and stage['start_epoch'] >= last_end
                and stage['end_epoch'] <= launcher['deadline_epoch'], 'Processes overlap/reuse PID/deadline exceeded')
        pids.add(stage['pid']); last_end = stage['end_epoch']; gpu_ids.add(stage['gpu_before']['uuid'])
        require(stage['gpu_before']['index'] == 5 and stage['gpu_before']['memory_mib'] <= 64
                and stage['gpu_before']['utilization_percent'] <= 1, 'GPU allocation not idle')
        files.verify(stage['report'], stage['report_sha256'])
        report, _ = files.json(stage['report'])
        require(report['arm'] == arm and report['complete_dit_calls'] == len(report['cases']) == count
                and report['deadline_unix'] == launcher['deadline_epoch'], 'Runtime count/deadline differs')
        require(report['sources'] == checked['sources'] and report['prepared_reference'] == checked['prepared_reference'], 'Arm source/input drift')
        files.sources(report['sources']); files.record(report['model_setup']); files.record(report['cpu_check_reference'])
        require(report['peak_allocated_bytes'] <= 60*1024**3, 'Allocated peak exceeded')
        reports[arm] = report
    require(len(gpu_ids) == 1, 'GPU changed across arms')
    return manifest, prepared, checked, reports, dict(frozen=frozen_rec, prepare=prepared_rec, check=checked_rec,
        launcher=launcher_rec, e014_full_weight_audit=manifest['e014_independent_summary'],
        source_files='Frozen source hashes checked; E014 200-weight audit inherited without rehashing weights',
        actual_dit_calls=18, gpu_uuid=next(iter(gpu_ids)), pids=sorted(pids))


def analyze(args, result):
    files = prior.Files()
    manifest, prepared, checked, reports, provenance = inherited_and_chain(args, files)
    sched = schedules(files)
    pindex = {(r['id'], r['arm'], r['corner']): r for r in prepared['cases']}
    rindex = {(r['id'], a, r['corner']): r for a, report in reports.items() for r in report['cases']}
    require(len(pindex) == len(rindex) == 18 and pindex.keys() == rindex.keys(), 'Case/corner set changed')
    cases, table = [], []
    for case in manifest['cases']:
        cid, n, next_step = case['id'], case['source_step'], case['next_step']
        old = {m: load_tensor(case['teacher_state'][m], files) for m in MODS}
        teacher = {m: load_tensor(case['teacher_next_state'][m], files) for m in MODS}
        embedding = load_tensor(case['prepared'], files)
        files.record(case['bf16_trajectory_reference'])
        historical, _ = files.json(case['bf16_trajectory_reference']['file'])
        trajectory = next(c for c in historical['cases'] if c['prompt_id'] == case['prompt_id'])
        next_call = next(c for c in trajectory['dit_calls'] if c['step'] == next_step)
        require({m: next_call[m+'_sha256'] for m in MODS} == case['teacher_next_raw_output_sha256'], 'Historical raw reference changed')
        updates, source_velocities = {}, {}
        for arm in ARMS:
            source = load_tensor(case['e014_outputs'][arm], files)
            require(source['case_id'] == cid and source['arm'] == arm, 'Wrong E014 velocity source')
            source_velocities[arm] = source['velocities']
            updates[arm] = {m: step(sched[m], old[m], source['velocities'][m], n) for m in MODS}
        for m in MODS:
            same(source_velocities['bf16'][m], old[m]['noise_pred'], 'E014 BF16 velocity')
            same(updates['bf16'][m], old[m]['latents_after'], 'First CPU step old after')
            same(updates['bf16'][m], teacher[m]['latents_before'], 'First CPU step next before')
            same(step(sched[m], teacher[m], teacher[m]['noise_pred'], next_step), teacher[m]['latents_after'], 'Second BF16 CPU step')
        case_result = dict(case_id=cid, source_step=n, next_step=next_step, arms={}, rankings={})
        metadata, outputs = None, {}
        for arm in ARMS:
            proof = next(p for p in reports[arm]['first_update_gpu_replay'] if p['id'] == cid)
            require(proof['additional_dit_calls'] == 0 and proof['source_state_files'] == case['teacher_state']
                    and proof['velocity_source'] == case['e014_outputs'][arm], 'First GPU update provenance changed')
            require(proof['source_state_signature'] == signature(old)
                    and proof['velocity_signature'] == signature(source_velocities[arm]), 'Wrong GPU scheduler operands')
            expected_update = {m: record(updates[arm][m]) for m in MODS}
            require(proof['gpu_updated'] == proof['cpu_prepared'] == expected_update
                    and all(proof['byte_exact'].values()), 'First GPU step differs from independent CPU step')
            outputs[arm], arm_metrics = {}, {}
            for corner in (('BB',) if arm == 'bf16' else CORNERS):
                key = (cid, arm, corner); prep, row = pindex[key], rindex[key]
                payload, out = load_tensor(prep['artifact'], files), load_tensor(row['artifact'], files)
                require(row['prepared_artifact'] == prep['artifact'] == out['prepared_artifact'], 'Prepared binding changed')
                require(out['id'] == cid and out['arm'] == arm and out['corner'] == corner and out['next_step'] == next_step, 'Output identity changed')
                for field in ('embedding', 'text_token_tags'):
                    same(payload[field], embedding[field], field)
                require(signature(payload['teacher_reference']) == signature(teacher), 'Teacher reference changed')
                for m, choice in zip(MODS, corner, strict=True):
                    expected = updates['bf16' if choice == 'B' else arm][m]
                    same(payload['state'][m]['latents_before'], expected, 'Corner selected wrong B/Q state')
                    for field in ('timestep', 'sigma'):
                        same(payload['state'][m][field], teacher[m][field], 'Wrong next-step clock')
                sig = signature({k: payload[k] for k in ('embedding', 'text_token_tags', 'state')})
                require(sig == prep['input_signature'] == payload['input_signature'], 'Prepared input signature differs')
                checked_sig = checked['inputs']['/'.join(key)]
                require({**sig, 'packed': prep['packed_signature']} == checked_sig == row['input_signature'] == out['input_signature'], 'CPU/runtime input signatures differ')
                require(signature(out['actual_dit_inputs']) == row['actual_dit_input_signature'], 'Actual DiT input hash mismatch')
                metadata = actual_input_check(out['actual_dit_inputs'], payload, metadata)
                prior.execution_contract(row['runtime_audit'], row['fastpack_checks'], arm)
                for field in ('raw_outputs', 'velocities', 'next_endpoints'):
                    require(signature(out[field]) == row[field], f'Output tensor hashes differ: {field}')
                arm_metrics[corner] = {}
                for m, pack in [('video', prior.video_rows), ('audio', prior.audio_rows)]:
                    same(pack(-out['velocities'][m]), out['raw_outputs'][m], 'Raw/velocity sign or layout differs')
                    own_step = step(sched[m], payload['state'][m], out['velocities'][m], next_step)
                    same(own_step, out['next_endpoints'][m], 'Endpoint did not update own corner sample')
                    if arm == 'bf16':
                        require(record(out['raw_outputs'][m])['sha256'] == case['teacher_next_raw_output_sha256'][m], 'BF16 raw replay failed')
                        same(out['velocities'][m], teacher[m]['noise_pred'], 'BF16 velocity replay')
                        same(out['next_endpoints'][m], teacher[m]['latents_after'], 'BF16 endpoint replay')
                    arm_metrics[corner][m] = dict(endpoint=metrics(out['next_endpoints'][m], teacher[m]['latents_after']),
                        velocity=metrics(out['velocities'][m], teacher[m]['noise_pred']),
                        input_perturbation=metrics(payload['state'][m]['latents_before'], teacher[m]['latents_before']))
                outputs[arm][corner] = out
            case_result['arms'][arm] = dict(corners=arm_metrics, first_update_gpu_cpu_exact=True)
            if arm != 'bf16':
                decomposition = {}
                for m, restore in [('video', 'QB'), ('audio', 'BQ')]:
                    values = {c: outputs[arm][c]['next_endpoints'][m] for c in CORNERS}
                    decomposition[m] = four_corner(values, teacher[m]['latents_after'], restore)
                    row = dict(case_id=cid, arm=arm, modality=m, restored_corner=restore,
                        endpoint_nmse={c: arm_metrics[c][m]['endpoint']['nmse'] for c in CORNERS},
                        velocity_nmse={c: arm_metrics[c][m]['velocity']['nmse'] for c in CORNERS},
                        qq_input_nmse={mod: arm_metrics['QQ'][mod]['input_perturbation']['nmse'] for mod in MODS},
                        **decomposition[m]['oracle_restore'])
                    table.append(row)
                case_result['arms'][arm]['four_corner_decomposition'] = decomposition
        for m, restore in [('video', 'QB'), ('audio', 'BQ')]:
            differences = {c: case_result['arms']['plain']['corners'][c][m]['endpoint']['error_energy']-
                               case_result['arms']['svd']['corners'][c][m]['endpoint']['error_energy'] for c in CORNERS}
            winner = {c: 'svd' if d > 0 else 'plain' if d < 0 else 'tie' for c, d in differences.items()}
            case_result['rankings'][m] = dict(plain_minus_svd_error_energy=differences, lower_error_arm=winner,
                opposite_modality_restored_corner=restore, qq_to_restore_winner_changed=winner['QQ'] != winner[restore],
                strict_sign_flip=differences['QQ']*differences[restore] < 0,
                scope='Comparing whole recipes under their own actual one-step perturbations; unequal perturbations are not Jacobian normalization')
        cases.append(case_result)
        print(f'E015 independent CPU reduction complete: {cid}', flush=True)
    result.update(provenance=provenance, cases=cases, machine_table=table, files_verified=files.checked,
                  status='complete', cuda_initialized=torch.cuda.is_initialized(),
                  limitations=['No BF16 QB/BQ/QQ model calls: no claim of quantization-induced intrinsic sensitivity.',
                    'Oracle effects include attention, shared quantization scales and all downstream paths.',
                    'Unequal modality/recipe perturbation directions and scheduler step sizes; not equal-amplitude Jacobians.',
                    'Two-step latent fidelity, not free-rollout quality or deployable oracle restoration.'])
    require(result['cuda_initialized'] is False, 'CPU reducer initialized CUDA')


def self_test():
    r = torch.tensor([1., 2.], dtype=torch.float64)
    values = dict(BB=r+torch.tensor([.1, -.2]), QB=r+torch.tensor([.4, -.1]),
                  BQ=r+torch.tensor([.2, .3]), QQ=r+torch.tensor([.8, .6]))
    got = four_corner(values, r, 'QB')
    require(got['oracle_restore']['net_error_energy_reduction'] > 0, 'Restore sign test')
    bad = dict(BB=r, QB=r+2, BQ=r+1, QQ=r+.5)
    require(four_corner(bad, r, 'QB')['oracle_restore']['net_error_energy_reduction'] < 0, 'Cancellation destruction test')
    zero = four_corner({c: r for c in CORNERS}, r, 'QB')
    require(zero['oracle_restore']['relative_error_energy_reduction'] is None, 'Zero-error ratio must be undefined')
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    return dict(status='complete', cuda_initialized=False,
                tests=['full_vector_and_energy_identities', 'positive_oracle_effect', 'negative_oracle_effect', 'zero_error_undefined'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path, default=RD/'independent_summary.json')
    parser.add_argument('--table-output', type=Path, default=RD/'independent_table.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(6)
    source = {str(p): prior.file_sha(p) for p in (Path(__file__).resolve(), Path(prior.__file__).resolve())}
    if args.self_test:
        print(json.dumps({**self_test(), 'sources': source}, indent=2, allow_nan=False)); return
    require(not args.output.exists() and not args.table_output.exists(), 'Refusing to overwrite independent evidence')
    result = dict(experiment='E015', status='running', summary_sources=source,
                  numerical_contract='Independent FP64 CPU reductions of raw tensors; no runner metrics used')
    start = time.monotonic()
    try:
        analyze(args, result)
        args.table_output.write_text(json.dumps(dict(experiment='E015', source_summary=str(args.output), rows=result['machine_table'],
                                                     rankings={c['case_id']: c['rankings'] for c in result['cases']}), indent=2, allow_nan=False)+'\n')
    except Exception:
        result.update(status='failed', error=traceback.format_exc()); raise
    finally:
        result['seconds_total'] = time.monotonic()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output), seconds=result['seconds_total'])), flush=True)


if __name__ == '__main__':
    main()
