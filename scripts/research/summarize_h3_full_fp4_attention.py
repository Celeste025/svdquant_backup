#!/usr/bin/env python3
"""E016 independent CPU tensor/cost audit; imports no model or experiment runner.

Reuses frozen independent E014 numerical utilities. Its complete 200-weight
provenance audit is inherited rather than repeated. Only this run's three SVD
implementations enter speed ratios; tensor error is not generated-video quality.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import summarize_h3_plain_baseline as prior

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E016'
MANIFEST = ROOT/'research_state/06_experiments/E016_h3_full_attention_manifest.json'
ARMS = ('bf16_original', 'svd_bf16', 'svd_block_mean', 'svd_global_mean')
BENCH = ARMS[1:]
MODS = ('video', 'audio')
MODE = dict(zip(ARMS, ('bf16', 'bf16', 'block_mean', 'global_mean')))
require, record, metrics = prior.require, prior.tensor_record, prior.metrics


def load_tensor(row, files):
    files.record(row)
    return prior.load_tensor_file(row['file'])


def attention_check(row, case, arm, diagnostics):
    mode, low = MODE[arm], MODE[arm] != 'bf16'
    cu, refcu = case['main_cu_seqlens'], case['refiner_cu_seqlens']
    n, total, padded = cu[1], cu[2], case['flashinfer_padded_length']
    require(row['mode'] == mode and row['expected_cu'] == cu and row['expected_refiner_cu'] == refcu,
            'Attention mode/boundaries differ')
    require(row['main_helper_calls'] == 50 and row['refiner_helper_calls'] == 2
            and row['fp4_calls'] == (50 if low else 0)
            and row['original_bf16_segments'] == (52 if low else 102), 'Attention route coverage differs')
    require(row['diagnostics'] is diagnostics and row['finite'] is (True if diagnostics else None)
            and row['finite_flag_count'] == ((252 if low else 52) if diagnostics else 0), 'Finite diagnostics differ')
    require('All main modes use the same prebound CPU boundaries' in row['metadata_policy'], 'Unmatched metadata policy')
    require({r['kind'] for r in row['actual_cu_checks']} == {'main', 'refiner'}, 'Missing actual cu checks')
    for r in row['actual_cu_checks']:
        expected = cu if r['kind'] == 'main' else refcu
        require(r['exact'] is True and r['actual'] == r['expected'] == expected, 'Actual cu check failed')
    require([r['block'] for r in row['main_rows']] == list(range(50)), 'Main attention block set/order differs')
    for r in row['main_rows']:
        require(r['input_shape'] == [total, 56, 128] and r['valid_length'] == n
                and r['padding_length'] == total-n and r['softmax_scale'] == 128**-.5
                and r['mode'] == mode and r['per_block_mean'] is (mode == 'block_mean' if low else None),
                'Actual layer attention geometry/scale/mode differs')
        if low:
            correction = [1, 56, padded//128 if mode == 'block_mean' else 1, padded]
            require(r['official_input_shape'] == [1, 56, n, 128]
                    and r['official_output_shape'] == [1, 56, padded, 128]
                    and r['correction_shape'] == correction and r['correction_bytes'] == math.prod(correction)*4
                    and r['unpadded_k_len'] == n and r['causal'] is False and r['return_lse'] is False,
                    'Official attention crop/mask/correction contract differs')


def runtime_check(row, checks, arm):
    low = MODE[arm] != 'bf16'
    expected = dict(sdpa_calls=52 if low else 102,
                    scaled_mm_calls=0 if arm == 'bf16_original' else 200, disk_loads=0)
    require({k: row[k] for k in expected} == expected, 'Actual runtime counts differ')
    require(row['qkv_dtypes'] == [['torch.bfloat16']*3], 'BF16 attention QKV precision changed')
    prior.check_flags(checks, 'bf16' if arm == 'bf16_original' else 'svd')


def common_report(report, checked, manifest, files):
    require(report['sources'] == checked['sources'] and report['environment'] == checked['environment'], 'Runtime source/environment drift')
    require(report['manifest'] == checked['manifest'] and report['plan'] == checked['plan'], 'Protocol drift')
    binding = json.loads(json.dumps(report['e014_binding']))
    cpu_binding = json.loads(json.dumps(checked['e014_binding']))
    require(binding['inherited_binding']['environment'].pop('CUDA_VISIBLE_DEVICES') == '5'
            and cpu_binding['inherited_binding']['environment'].pop('CUDA_VISIBLE_DEVICES') == '', 'CPU/GPU visibility contract differs')
    require(binding == cpu_binding and report['e014_inventory_recheck'] == checked['e014_inventory_recheck'], 'Inherited E014 binding/recheck changed')
    require(report['checked_inputs'] == checked['inputs'] and report['attention_contracts'] == checked['attention_contracts'], 'CPU/runtime input contract differs')
    files.record(report['cpu_check_reference'])
    require(report['non_target_identity_preserved'] is True, 'Non-target tensor identity guard failed')
    require(report['device']['visible_devices'] == '5' and report['device']['capability'] == [12, 0], 'Unexpected GPU assignment')
    require(max(report['startup_peak_allocated_bytes'], report['final_memory']['peak_allocated_bytes']) <= 60*1024**3, 'Startup/final memory exceeded')
    install = report['attention_installation']
    arm = report['arm']
    require(install['mode'] == MODE[arm] and install['main_blocks'] == list(range(50))
            and install['refiner_blocks'] == 2 and install['cpu_fixture_only'] is False
            and install['official_per_block_mean'] is (None if MODE[arm] == 'bf16' else MODE[arm] == 'block_mean'),
            'Installed wrong attention recipe')
    for source in install['sources'].values():
        files.record(source)
    if arm != 'bf16_original':
        linear = report['installation']
        require(linear['target_count'] == linear['exact_roundtrip_count'] == 200
                and {r['name'] for r in linear['layers']} == prior.TARGETS
                and all(r['old_weight_parameter_cleared'] for r in linear['layers']), 'Incomplete SVD replacement')
        expected = checked['e014_binding']['inherited_binding']['export_manifest']['sha256']
        require(linear['manifest_sha256'] == expected, 'Different SVD export installed')
    required = [] if arm == 'bf16_original' else ['bf16_original']
    if MODE[arm] != 'bf16':
        required.append('svd_bf16')
    require(set(report['control_references']) == set(required), 'Missing two-level replay prerequisite')
    for source in report['control_references'].values():
        files.record(source)


def launcher_check(launcher, freeze_rec, files, manifest):
    names = [f'evaluate_{a}' for a in ARMS]+[f'bench_{a}' for a in BENCH]
    counts = [3]*4+[5]*3
    require(launcher['freeze_sha256'] == freeze_rec['sha256'] and launcher['planned_dit_calls'] == sum(counts) == 27,
            'Launcher freeze/allocation changed')
    require(launcher['max_complete_dit_calls'] == 30 and launcher['wall_budget_seconds'] == 1800
            and launcher['deadline_epoch']-launcher['start_epoch'] == 1800 and launcher['seconds_total'] <= 1800,
            'Shared wall/call budget exceeded')
    require([r['name'] for r in launcher['stages']] == names, 'Launcher stage order differs')
    pids, uuids, last = set(), set(), launcher['start_epoch']
    for stage, name, calls in zip(launcher['stages'], names, counts, strict=True):
        require(stage['status'] == 'complete' and stage['returncode'] == 0 and stage['planned_dit_calls'] == calls,
                'Incomplete stage/call count')
        require(stage['pid'] not in pids and stage['start_epoch'] >= last
                and stage['start_epoch'] < stage['end_epoch'] <= launcher['deadline_epoch'], 'Stage overlap/PID/deadline changed')
        pids.add(stage['pid']); last = stage['end_epoch']
        device = stage['gpu_before']; uuids.add(device['uuid'])
        require(device['index'] == 5 and device['memory_mib'] <= 64 and device['utilization_percent'] <= 1, 'GPU not idle before stage')
        cmd = stage['command']
        require(cmd[0] == manifest['python'] and float(cmd[cmd.index('--deadline-unix')+1]) == launcher['deadline_epoch'], 'Environment or deadline reset')
        files.verify(stage['report'], stage['report_sha256'])
    require(len(uuids) == 1 and len(pids) == 7, 'Different GPU or non-independent processes')
    return dict(actual_dit_calls=27, gpu_uuid=next(iter(uuids)), distinct_pids=sorted(pids), wall_seconds=launcher['seconds_total'])


def evaluate_audit(reports, old_reports, manifest, checked, files):
    output, table = [], []
    for case in manifest['cases']:
        cid = case['id']
        source = prior.source_case(case, checked['inputs'][cid], files)
        data, rows = {}, {}
        for arm in ARMS:
            row = next(r for r in reports[arm]['cases'] if r['id'] == cid); rows[arm] = row
            require(row['status'] == 'complete' and row['kind'] == case['kind'], 'Evaluation case incomplete')
            require(row['memory']['peak_allocated_bytes'] <= 60*1024**3, 'Evaluation peak exceeded')
            value = load_tensor(row['artifact'], files); data[arm] = value
            require(value['case_id'] == cid and value['arm'] == arm, 'Output artifact identity differs')
            for field in ('input_signature', 'actual_dit_inputs'):
                require(value[field] == row[field] == source[field], 'Actual/common original input differs')
            require(prior.signature(value['raw_outputs']) == row['raw_outputs'] and prior.signature(value['velocities']) == row['velocities'], 'Output tensor SHA differs')
            if case['kind'] == 'model_fn':
                for m, pack in [('video', prior.video_rows), ('audio', prior.audio_rows)]:
                    require(record(pack(-value['velocities'][m])) == row['raw_outputs'][m], 'Raw/velocity layout or sign differs')
            else:
                require(row['velocities'] is None, 'Unexpected raw-case velocity')
            runtime_check(row['runtime_audit'], row['fastpack_checks'], arm)
            attention_check(row['attention'], case, arm, True)
            if arm in ARMS[:2]:
                oldarm = 'bf16' if arm == 'bf16_original' else 'svd'
                oldrow = next(r for r in old_reports[oldarm]['cases'] if r['id'] == cid)
                old = load_tensor(case['e014_artifacts'][oldarm], files)
                require(case['e014_artifacts'][oldarm] == oldrow['artifact'], 'Historical artifact binding changed')
                for field in ('raw_outputs', 'velocities', 'input_signature', 'actual_dit_inputs'):
                    require(prior.signature(value[field]) == prior.signature(old[field]), 'Historical exact replay failed: '+field)
                replay = row['historical_replay']
                require(replay['exact'] is True and replay['raw_sha_exact'] is True and replay['tensor_equal'] is True
                        and replay['reference_arm'] == oldarm and replay['reference_artifact'] == oldrow['artifact']
                        and replay['velocity_sha_exact'] is (True if case['kind'] == 'model_fn' else None), 'Historical gate false')
        result = dict(case_id=cid, split=case['split'], input_equal=True, historical_bf16_and_svd_byte_exact=True,
                      attention={a: rows[a]['attention'] for a in ARMS}, modalities={})
        for m in MODS:
            teacher, baseline = data['bf16_original']['raw_outputs'][m], data['svd_bf16']['raw_outputs'][m]
            result['modalities'][m] = {}
            for arm in BENCH:
                got = data[arm]['raw_outputs'][m]
                values = dict(relative_to_bf16_original=metrics(got, teacher), relative_to_svd_bf16=metrics(got, baseline))
                if case['kind'] == 'model_fn':
                    vm = metrics(data[arm]['velocities'][m], data['bf16_original']['velocities'][m])
                    require(math.isclose(vm['error_energy'], values['relative_to_bf16_original']['error_energy'], rel_tol=1e-12)
                            and math.isclose(vm['reference_energy'], values['relative_to_bf16_original']['reference_energy'], rel_tol=1e-12), 'Velocity packing changes energy')
                    values['velocity_relative_to_bf16_original'] = vm
                result['modalities'][m][arm] = values
                table.append(dict(case_id=cid, modality=m, arm=arm, **{k: values['relative_to_bf16_original'][k]
                    for k in ('nmse', 'error_energy', 'reference_energy', 'error_l2', 'cosine', 'max_abs')},
                    nmse_relative_to_svd_bf16=values['relative_to_svd_bf16']['nmse']))
        output.append(result)
        print('CPU tensor audit complete: '+cid, flush=True)
    return output, table


def benchmark_audit(reports, evaluations, manifest, files):
    case, result = manifest['cases'][0], {}
    names = ('cuda_ms', 'host_forward_ms_excluding_checks', 'attention_validation_ms', 'flag_validation_ms', 'validation_ms', 'host_ms_including_checks')
    for arm in BENCH:
        report = reports[arm]
        files.record(report['evaluate_reference'])
        expected = evaluations[arm]['cases'][0]['raw_outputs']
        shas = {m: expected[m]['sha256'] for m in MODS}
        require(len(report['repeats']) == 3 and [r['repeat'] for r in report['repeats']] == list(range(3)), 'Wrong repeat allocation')
        for row in [report['warmup'], *report['repeats'], report['profile']]:
            require(row['raw_output_sha256'] == shas, 'Bench output differs from same-arm evaluation')
            prior.check_flags(row['fastpack_checks'], 'svd')
            attention_check(row['attention'], case, arm, row is report['profile'])
        for row in [report['warmup'], *report['repeats']]:
            require(all(math.isfinite(row[k]) and row[k] >= 0 for k in names), 'Invalid timing')
            require(math.isclose(row['attention_validation_ms']+row['flag_validation_ms'], row['validation_ms'], abs_tol=1e-7)
                    and math.isclose(row['host_forward_ms_excluding_checks']+row['validation_ms'], row['host_ms_including_checks'], abs_tol=1e-7), 'Timing components do not sum')
        latency = {}
        for name in names:
            v = [r[name] for r in report['repeats']]
            latency[name] = dict(values=v, median=statistics.median(v), min=min(v), max=max(v))
            require(latency[name] == report['latency_ms'][name], 'Runner latency reducer differs')
        profile = report['profile']
        runtime_check(profile['runtime_audit'], profile['fastpack_checks'], arm)
        files.record(profile['trace'])
        trace = json.loads(Path(profile['trace']['file']).read_text())
        kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
        linear = [e for e in kernels if all(s in e.get('name', '').lower() for s in ('sm120', 'e2m1', 'gemm'))]
        attn = [e for e in kernels if 'nvfp4_attention::attention_kernel_ws' in e.get('name', '') and 'float_e2m1' in e.get('name', '')]
        require(len(linear) == 200 and len(attn) == (50 if MODE[arm] != 'bf16' else 0), 'Actual GPU kernel counts differ')
        storage, mem = report['resident_model_storage'], report['steady_memory']
        require(sum(v['bytes'] for v in storage['categories'].values()) == storage['unique_cuda_storage_bytes'], 'Storage sum differs')
        require(mem['peak_reserved_bytes'] >= mem['peak_allocated_bytes'] >= report['steady_before']['allocated_bytes']
                and max(mem['peak_allocated_bytes'], report['startup_peak_allocated_bytes']) <= 60*1024**3, 'Memory accounting/budget differs')
        result[arm] = dict(latency_ms=latency, storage=storage, steady_memory=mem,
            startup_peak_allocated_bytes=report['startup_peak_allocated_bytes'], startup_seconds=report['startup_seconds'],
            final_memory_including_profile=report['final_memory'],
            actual_fp4_gemm_kernels=len(linear), actual_fp4_attention_kernels=len(attn), actual_kernel_count=len(kernels),
            correction_shape=report['profile']['attention']['main_rows'][0].get('correction_shape'),
            correction_bytes=report['profile']['attention']['main_rows'][0].get('correction_bytes'), raw_output_sha256=shas)
    baseline = result['svd_bf16']['latency_ms']
    for arm in BENCH[1:]:
        result[arm]['speedup_vs_same_run_svd_bf16'] = {name: baseline[name]['median']/result[arm]['latency_ms'][name]['median']
            for name in ('cuda_ms', 'host_forward_ms_excluding_checks', 'host_ms_including_checks')}
    return result


def run(args, result):
    files = prior.Files()
    manifest, manifest_rec = files.json(MANIFEST, complete=False)
    require(tuple(manifest['arms']) == ARMS and tuple(manifest['benchmark_arms']) == BENCH, 'Experiment arms changed')
    files.record(manifest['plan']); files.sources(manifest['official_sources'])
    inherited, _ = files.json(manifest['e014_independent_summary']['file']); files.record(manifest['e014_independent_summary'])
    require(inherited['export']['target_count'] == 200, 'Incomplete inherited E014 weight audit')
    e006, _ = files.json(manifest['e006_reference']['file']); files.record(manifest['e006_reference'])
    require(len(e006['cases']) == 12, 'E006 source scope changed')
    checked, checked_rec = files.json(args.report_dir/'E016_check_bf16_original.json')
    require(checked['cuda_initialized'] is False and checked['complete_dit_calls'] == checked['attempted_dit_calls'] == 0, 'Preflight was not CPU-only')
    require(checked['manifest']['sha256'] == manifest_rec['sha256'] and checked['plan'] == manifest['plan'], 'Preflight protocol changed')
    files.sources(checked['sources'])
    recheck = checked['e014_inventory_recheck']
    inherited_binding = checked['e014_binding']['inherited_binding']
    require(recheck['actual_sdpa_enabled'] == inherited_binding['sdpa_enabled']
            and recheck['torch'] == inherited_binding['torch'], 'Actual backend recheck differs')
    for path, rec in manifest['official_sources'].items():
        require(checked['sources'][path] == rec, 'Official source not bound by runtime')
    environment = checked['environment']
    require(environment['python'] == manifest['python'] and environment['attention_implementation'] == 'torch'
            and environment['packages']['torch'] == '2.11.0+cu128' and environment['torch_cuda'] == '12.8', 'Wrong precision environment')
    freeze, freeze_rec = files.json(args.report_dir/'frozen_contract.json')
    for path, digest in freeze['files'].items():
        files.verify(path, digest)
    launcher, launcher_rec = files.json(args.report_dir/'launcher.json')
    chain = launcher_check(launcher, freeze_rec, files, manifest)
    old = {a: files.json(row['file'])[0] for a, row in manifest['e014_evaluations'].items()}
    for rec in manifest['e014_evaluations'].values():
        files.record(rec)
    evals, benches = {}, {}
    for phase, arms, target, count in [('evaluate', ARMS, evals, 3), ('bench', BENCH, benches, 5)]:
        for arm in arms:
            report, _ = files.json(args.report_dir/f'E016_{phase}_{arm}.json')
            require(report['arm'] == arm and report['phase'] == phase
                    and report['complete_dit_calls'] == report['attempted_dit_calls'] == count
                    and report['deadline_unix'] == launcher['deadline_epoch'], 'Runtime call/deadline differs')
            common_report(report, checked, manifest, files)
            if phase == 'evaluate':
                require(tuple(r['id'] for r in report['cases']) == prior.CASES, 'Evaluation case set/order differs')
            target[arm] = report
    cases, table = evaluate_audit(evals, old, manifest, checked, files)
    costs = benchmark_audit(benches, evals, manifest, files)
    files.verify(__file__); files.verify(prior.__file__)
    require(not torch.cuda.is_initialized(), 'Independent audit initialized CUDA')
    result.update(status='complete', cuda_initialized=False, manifest=manifest_rec, cpu_check=checked_rec,
        frozen_contract=freeze_rec, launcher=launcher_rec, execution_chain=chain,
        inherited_weight_audit=manifest['e014_independent_summary'], prior_local_attention_reference=manifest['e006_reference'],
        weight_audit_scope='E014 200-weight audit inherited; no repeated original weight streaming or quantization',
        cases=cases, benchmark=costs, files_verified=files.checked)
    return dict(experiment='E016', status='complete', numeric_rows=table,
        costs={arm: dict(cuda_median_ms=costs[arm]['latency_ms']['cuda_ms']['median'],
            host_including_checks_median_ms=costs[arm]['latency_ms']['host_ms_including_checks']['median'],
            steady_peak_allocated_bytes=costs[arm]['steady_memory']['peak_allocated_bytes'],
            storage_bytes=costs[arm]['storage']['unique_cuda_storage_bytes']) for arm in BENCH},
        scope='Fixed-state numerical and same-run SVD implementation cost; not video quality or a novel method')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path, default=RD/'independent_summary.json')
    parser.add_argument('--table', type=Path, default=RD/'independent_table.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(6)
    require(not torch.cuda.is_initialized(), 'CPU-only startup required')
    if args.self_test:
        print(json.dumps(dict(**prior.self_test(), source_sha256=prior.file_sha(__file__),
            numerical_helper_sha256=prior.file_sha(prior.__file__)), indent=2, allow_nan=False))
        return
    require(not args.output.exists() and not args.table.exists(), 'Refusing to overwrite previous summary')
    result = dict(experiment='E016', status='running', audit_source=dict(file=str(Path(__file__).resolve()), sha256=prior.file_sha(__file__)),
        torch=torch.__version__, numerical_source='FP64 CPU reductions of raw tensor artifacts; imports no experiment/model runner',
        limitations=['Three previously inspected fixed states; no decoded quality or generalization claim.',
            'Modes are two official complete recipes, not a causal isolation of centering granularity.',
            'Timing compares only the same-run three SVD arms, including all online attention preprocessing.',
            'CPU metadata caching and deferred checks are common to every main mode; validation cost is separately included.',
            'Peak correction tensor size is not total-model memory capacity; no long-video/OOM extrapolation.'])
    start = time.monotonic()
    try:
        table = run(args, result)
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(json.dumps(table, indent=2, allow_nan=False)+'\n')
        result['table'] = dict(file=str(args.table), sha256=prior.file_sha(args.table))
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        result['seconds_total'] = time.monotonic()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output), seconds=result['seconds_total']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
