#!/usr/bin/env python3
"""E016 fixed-state H3: frozen native SVD linears and native FP4 attention.

No generation or fitting. Independent processes share one external deadline.
Original numerical helpers remain frozen; diagnostic validation is outside timers.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import importlib
import importlib.metadata
import json
import os
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

inherited = base.inherited
sha256, save = base.sha256, base.save
file_record, verify_file = base.file_record, base.verify_file
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E016')
REPORTS = ROOT/'results/research/E016'
MANIFEST = ROOT/'research_state/06_experiments/E016_h3_full_attention_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E016_h3_full_attention_plan.md'
ARMS = ('bf16_original', 'svd_bf16', 'svd_block_mean', 'svd_global_mean')
MODE = {'bf16_original': 'bf16', 'svd_bf16': 'bf16',
        'svd_block_mean': 'block_mean', 'svd_global_mean': 'global_mean'}
CONTROL = {'bf16_original': 'bf16', 'svd_bf16': 'svd'}


def environment_record():
    packages = {}
    for name in ('torch', 'triton', 'flashinfer-python'):
        packages[name] = importlib.metadata.version(name)
    return {'python': sys.executable, 'packages': packages,
            'torch_file': file_record(torch.__file__),
            'torch_cuda': torch.version.cuda,
            'attention_implementation': importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION}


def recheck_e014_inventory(checked, manifest_path, plan_path):
    # Only source-file discovery changes. Keep every frozen prerequisite check,
    # including headers, provenance proofs, export binding and backend flags.
    # The original discovery imports unused AV decoding modules; its already
    # verified E014 file inventory is sufficient to rehash those same sources.
    inventory = [Path(path) for path in checked['inherited_binding']['sources']]
    base.check_sources(checked['sources'])
    original_discovery = inherited.source_paths
    old_report = {}
    try:
        inherited.source_paths = lambda: inventory
        old_manifest = base.prerequisites(
            types.SimpleNamespace(manifest=manifest_path, plan=plan_path), old_report)
    finally:
        inherited.source_paths = original_discovery
    if old_report['sources'] != checked['sources']:
        raise RuntimeError('Original E014 source-file inventory changed')
    return old_manifest, old_report


def prerequisites(args, report):
    manifest = json.loads(args.manifest.read_text())
    for key, value in manifest['environment'].items():
        os.environ.setdefault(key, value)
        if os.environ[key] != value:
            raise RuntimeError(f'Pinned environment differs: {key}')
    if sys.executable != manifest['python']:
        raise RuntimeError('E016 requires the pinned native-attention Python environment')
    if verify_file(manifest['plan']).resolve() != args.plan.resolve():
        raise RuntimeError('Plan differs from the preregistered manifest')
    for row in manifest['official_sources'].values():
        verify_file(row)
    if tuple(manifest['arms']) != ARMS or manifest['attention_modes'] != MODE:
        raise RuntimeError('Fixed E016 arms changed')
    if manifest['benchmark']['warmup'] != 1 or manifest['benchmark']['repeats'] != 3:
        raise RuntimeError('Fixed E016 benchmark allocation changed')
    if manifest.get('source_files'):
        base.check_sources(manifest['source_files'])
    e014_checked = inherited.load_complete(verify_file(manifest['e014_check']))
    base.check_sources(e014_checked['sources'])
    old_manifest_path = verify_file(manifest['e014_manifest'])
    old_plan_path = verify_file(e014_checked['plan'])
    old_manifest, old_report = recheck_e014_inventory(e014_checked, old_manifest_path, old_plan_path)
    report['e014_inventory_recheck'] = {
        'scope': 'All frozen E014/E010 prerequisite checks execute unchanged. Only source-file discovery uses the already verified E014 inventory, avoiding imports of unrelated AV decoding dependencies; the temporary discovery override is restored before model setup.',
        'actual_sdpa_enabled': old_report['inherited_binding']['sdpa_enabled'],
        'torch': old_report['inherited_binding']['torch']}
    if e014_checked.get('cuda_initialized') is not False:
        raise RuntimeError('Original E014 CPU check was not CPU-only')
    if (tuple(c['id'] for c in manifest['cases']) != base.CASE_IDS or
            manifest['svd_export_dir'] != old_manifest['svd_export_dir']):
        raise RuntimeError('E016 must reuse exactly the three E014 cases and SVD export')
    for current, old in zip(manifest['cases'], old_manifest['cases'], strict=True):
        if any(current[key] != value for key, value in old.items()):
            raise RuntimeError('Original E014 case fields changed')
    references = {}
    for arm in ('bf16', 'svd'):
        reference = inherited.load_complete(verify_file(manifest['e014_evaluations'][arm]))
        base.check_sources(reference['sources'])
        if reference['sources'] != e014_checked['sources'] or reference['arm'] != arm:
            raise RuntimeError('Historical E014 evaluation source or arm mismatch')
        if tuple(c['id'] for c in reference['cases']) != base.CASE_IDS:
            raise RuntimeError('Historical E014 case coverage changed')
        for row, case in zip(reference['cases'], manifest['cases'], strict=True):
            verify_file(row['artifact'])
            if case['e014_artifacts'][arm] != row['artifact']:
                raise RuntimeError('Manifest/reference historical artifact binding differs')
        references[arm] = reference
    # Importing the adapter must not initialize CUDA or build/run a kernel.
    attention = importlib.import_module('h3_native_fp4_attention')
    backend = importlib.import_module('flashinfer.nvfp4_attention_sm120')
    paths = [Path(__file__), args.manifest, args.plan, Path(attention.__file__), Path(backend.__file__)]
    report['sources'] = (old_report['sources'] | manifest['official_sources'] |
                         {str(p.resolve()): file_record(p) for p in paths})
    report['environment'] = environment_record()
    report['manifest'] = file_record(args.manifest)
    report['plan'] = file_record(args.plan)
    report['e014_binding'] = {'check': manifest['e014_check'], 'manifest': manifest['e014_manifest'],
                             'evaluations': manifest['e014_evaluations'],
                             'inherited_binding': old_report['inherited_binding'],
                             'asset_binding': old_report['asset_binding']}
    return manifest, references


def cpu_pipe():
    return inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)


def attention_contract(case, value, pipe):
    if case['kind'] == 'raw_dit':
        kwargs = value['kwargs']
        main = kwargs['packed_seq_params']['cu_seqlens_q'].tolist()
        refiner = kwargs['refiner_packed_seq_params']['cu_seqlens_q'].tolist()
    else:
        packed = base.make_packed(pipe, value['embedding'], value['text_token_tags'], value['state'])
        main = packed['cu_seqlens'].tolist()
        length = value['embedding'].shape[0]
        refiner = [0, length, length]
    if len(main) != 3 or not (main[0] == 0 < main[1] < main[2]) or refiner[0] != 0 or refiner[1] != refiner[2]:
        raise RuntimeError('Expected one valid main segment, one nonempty padding segment, and one refiner segment')
    if (main != case['main_cu_seqlens'] or refiner != case['refiner_cu_seqlens'] or
            case['heads'] != 56 or case['head_dim'] != 128 or
            ((main[1]+127)//128)*128 != case['flashinfer_padded_length']):
        raise RuntimeError('Actual attention geometry differs from preregistered case')
    return {'expected_cu': main, 'expected_refiner_cu': refiner}


def reference_case(references, arm, case_id):
    return next(c for c in references[arm]['cases'] if c['id'] == case_id)


def cpu_check(args, report, manifest, references):
    pipe = cpu_pipe()
    original = inherited.load_complete(verify_file(manifest['e014_check']))
    fixture_path = args.report_dir/'attention_cpu_fixture.json'
    fixture = inherited.load_complete(fixture_path)
    if (fixture.get('cuda_initialized') is not False or
            fixture['source'] != file_record(HERE/'h3_native_fp4_attention.py')):
        raise RuntimeError('Attention routing fixture does not bind the final adapter source')
    report['attention_cpu_fixture'] = file_record(fixture_path)
    report['inputs'], report['attention_contracts'], report['historical_artifact_checks'] = {}, {}, {}
    for case in manifest['cases']:
        value = base.load_case(case)
        signature = base.validate_case(case, value, pipe)
        if signature != original['inputs'][case['id']]:
            raise RuntimeError('Actual input differs from the original E014 CPU check')
        report['inputs'][case['id']] = signature
        report['attention_contracts'][case['id']] = attention_contract(case, value, pipe)
        report['historical_artifact_checks'][case['id']] = {}
        for arm in ('bf16', 'svd'):
            row = reference_case(references, arm, case['id'])
            artifact = torch.load(row['artifact']['file'], map_location='cpu', weights_only=True, mmap=True)
            for name in ('raw_outputs', 'velocities'):
                tensors = artifact[name]
                records = None if tensors is None else {m: base.tensor_record(tensors[m]) for m in base.MODALITIES}
                if records != row[name]:
                    raise RuntimeError('Historical tensor bytes differ from the E014 report')
            if artifact['input_signature'] != signature or artifact['actual_dit_inputs'] != row['actual_dit_inputs']:
                raise RuntimeError('Historical actual/input signatures changed')
            report['historical_artifact_checks'][case['id']][arm] = True
        if reference_case(references, 'bf16', case['id'])['actual_dit_inputs'] != reference_case(references, 'svd', case['id'])['actual_dit_inputs']:
            raise RuntimeError('Historical BF16/SVD actual inputs differ')
        print(f'E016 CPU input valid: {case["id"]}', flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only validation initialized CUDA')
    report.update(status='complete', cuda_initialized=False, complete_dit_calls=0)


def check_budget(args, report):
    if args.deadline_unix is None or time.time() >= args.deadline_unix:
        raise RuntimeError('GPU phase requires an unexpired shared --deadline-unix; never reset its clock')
    maximum = 3 if args.phase == 'evaluate' else 5
    if report['attempted_dit_calls'] >= maximum:
        raise RuntimeError('Per-process preregistered DiT-call allocation exhausted')
    inherited.memory_guard()


def require_current_report(path, report, arm, phase='evaluate'):
    previous = inherited.load_complete(path)
    base.check_sources(previous['sources'])
    if (previous['sources'] != report['sources'] or previous['environment'] != report['environment'] or
            previous['arm'] != arm or previous['phase'] != phase):
        raise RuntimeError(f'Source/environment/arm mismatch in completed {path}')
    return previous


def setup_model(args, report, manifest):
    check_budget(args, report)
    checked = inherited.load_complete(args.check_report)
    base.check_sources(checked['sources'])
    if (checked['sources'] != report['sources'] or checked['environment'] != report['environment'] or
            checked.get('cuda_initialized') is not False):
        raise RuntimeError('E016 CPU source/environment check changed; rerun after final freeze')
    report['cpu_check_reference'] = file_record(args.check_report)
    report['checked_inputs'], report['attention_contracts'] = checked['inputs'], checked['attention_contracts']
    # Gate subsequent arms on the previous exact historical replays.
    required = [] if args.arm == 'bf16_original' else ['bf16_original']
    if MODE[args.arm] != 'bf16':
        required.append('svd_bf16')
    report['control_references'] = {}
    for arm in required:
        path = args.report_dir/f'E016_evaluate_{arm}.json'
        control = require_current_report(path, report, arm)
        if len(control['cases']) != 3 or not all(c['historical_replay']['exact'] is True for c in control['cases']):
            raise RuntimeError('Historical full-model replay gate has not passed')
        report['control_references'][arm] = file_record(path)
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    if report['environment']['attention_implementation'] != 'torch':
        raise RuntimeError('Original BF16 torch attention backend required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(manifest['benchmark']['gpu']):
        raise RuntimeError('Only the preregistered physical GPU may be used')
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('E016 binds the native SM120 path')
    report['device'] = {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'name': torch.cuda.get_device_name(), 'capability': [12, 0]}
    start = time.monotonic()
    pipe = inherited.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    report['resident_conversion'] = base.make_h3_resident(pipe.dit)
    before = base.non_target_identity(pipe.dit)
    if args.arm != 'bf16_original':
        report['installation'] = base.install_native_h3(pipe.dit, manifest['svd_export_dir'],
            activation_packer=base.pack_activation_fast, chunk_rows=1024)
        if report['installation']['target_count'] != 200:
            raise RuntimeError('Exactly 200 frozen SVD linears required')
    if base.non_target_identity(pipe.dit) != before:
        raise RuntimeError('SVD installation changed non-target tensor identities/versions')
    report['non_target_identity_preserved'] = True
    report['non_target_guard_scope'] = 'Object/storage identity, shape, dtype/device and available version counter; not value hashing'
    from diffsynth.core.vram.layers import AutoTorchModule, AutoWrappedLinear, AutoWrappedModule
    if any(isinstance(m, AutoTorchModule) for m in pipe.dit.modules()):
        raise RuntimeError('Offload wrapper survived resident conversion')
    def forbidden_disk(*_args, **_kwargs):
        raise RuntimeError('Disk weight loading forbidden in resident E016')
    AutoWrappedLinear.load_from_disk = AutoWrappedModule.load_from_disk = forbidden_disk
    from h3_native_fp4_attention import install_h3_fp4_attention
    router = install_h3_fp4_attention(pipe.dit, mode=MODE[args.arm])
    report['attention_installation'] = router.manifest
    gc.collect()
    torch.cuda.empty_cache()
    report['startup_seconds'] = time.monotonic()-start
    report['startup_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['resident_model_storage'] = base.resident_storage(pipe.dit)
    inherited.memory_guard()
    save(report, args.output)
    return pipe, router


def memory_record():
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def audit_contract(row, arm, attention):
    low = MODE[arm] != 'bf16'
    expected = {'sdpa_calls': 52 if low else 102,
                'scaled_mm_calls': 0 if arm == 'bf16_original' else 200, 'disk_loads': 0}
    if {key: row[key] for key in expected} != expected:
        raise RuntimeError(f'Actual execution differs: {row}')
    if attention['fp4_calls'] != (50 if low else 0) or attention['original_bf16_segments'] != expected['sdpa_calls']:
        raise RuntimeError(f'Attention route counts differ: {attention}')


def check_flags(checks, arm):
    if arm == 'bf16_original':
        if checks is not None:
            raise RuntimeError('BF16 arm unexpectedly installed a pack checker')
    elif checks.summary['checked_calls'] != 200:
        raise RuntimeError('Expected 200 activation-domain checks')


def replay_outputs(arm, case, raw, records, velocities, velocity_records, references):
    historical_arm = CONTROL.get(arm)
    if historical_arm is None:
        return {'reference_arm': None, 'exact': None}
    row = reference_case(references, historical_arm, case['id'])
    if records != row['raw_outputs'] or velocity_records != row['velocities']:
        raise RuntimeError(f'Historical E014 {historical_arm} raw/velocity SHA replay failed: {case["id"]}')
    artifact = torch.load(row['artifact']['file'], map_location='cpu', weights_only=True, mmap=True)
    for name, tensors in (('raw_outputs', raw), ('velocities', velocities)):
        if tensors is not None and not all(torch.equal(tensors[m], artifact[name][m]) for m in base.MODALITIES):
            raise RuntimeError('Historical SHA matched but byte-level tensor replay failed')
    return {'reference_arm': historical_arm, 'reference_artifact': row['artifact'],
            'raw_sha_exact': True, 'velocity_sha_exact': True if velocities is not None else None,
            'tensor_equal': True, 'exact': True}


@torch.inference_mode()
def evaluate(args, report, manifest, references):
    pipe, router = setup_model(args, report, manifest)
    report['cases'] = []
    for case in manifest['cases']:
        check_budget(args, report)
        value = base.load_case(case)
        signature = base.validate_case(case, value, cpu_pipe())
        if signature != report['checked_inputs'][case['id']]:
            raise RuntimeError('Input signature differs from CPU check')
        contract = attention_contract(case, value, cpu_pipe())
        if contract != report['attention_contracts'][case['id']]:
            raise RuntimeError('Packed segment contract changed')
        forward = base.gpu_call(pipe, case, value)
        actual_inputs, raw_capture = [], []
        def observe_input(_module, pos, kw):
            actual_inputs.append(base.tree_signature({'args': pos, 'kwargs': kw}))
        def observe_output(_module, pos, output):
            raw_capture.append(output)
        pre = pipe.dit.register_forward_pre_hook(observe_input, with_kwargs=True)
        post = pipe.dit.register_forward_hook(observe_output)
        audit = base.RuntimeAudit()
        audit.phase = 'resident_bf16' if args.arm == 'bf16_original' else 'native'
        context = nullcontext(None) if args.arm == 'bf16_original' else base.collect_fastpack_checks()
        print(f'E016 evaluate {args.arm} {case["id"]}', flush=True)
        try:
            with audit.installed(), context as checks, router.forward_context(**contract, diagnostics=True) as attn:
                report['attempted_dit_calls'] += 1
                result = forward()
                torch.cuda.synchronize()
            report['complete_dit_calls'] += 1
        finally:
            pre.remove()
            post.remove()
        audit_contract(audit.row(), args.arm, attn.summary)
        check_flags(checks, args.arm)
        if len(actual_inputs) != 1 or len(raw_capture) != 1:
            raise RuntimeError('Exactly one actual full DiT input/output capture required')
        if actual_inputs[0] != reference_case(references, 'bf16', case['id'])['actual_dit_inputs']:
            raise RuntimeError('Actual DiT inputs differ from E014 historical reference')
        raw, records = base.cpu_outputs(raw_capture[0])
        velocities, velocity_records = base.cpu_outputs(result) if case['kind'] == 'model_fn' else (None, None)
        replay = replay_outputs(args.arm, case, raw, records, velocities, velocity_records, references)
        artifact = args.data_dir/'evaluate'/args.arm/(case['id']+'.pt')
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists():
            raise FileExistsError(artifact)
        torch.save({'case_id': case['id'], 'arm': args.arm, 'raw_outputs': raw, 'velocities': velocities,
                    'input_signature': signature, 'actual_dit_inputs': actual_inputs[0]}, artifact)
        report['cases'].append({'id': case['id'], 'kind': case['kind'], 'status': 'complete',
            'input_signature': signature, 'actual_dit_inputs': actual_inputs[0], 'raw_outputs': records,
            'velocities': velocity_records, 'artifact': file_record(artifact), 'historical_replay': replay,
            'runtime_audit': audit.row(), 'attention': attn.summary,
            'fastpack_checks': checks.summary if checks is not None else None,
            'memory': memory_record()})
        save(report, args.output)
        del forward, result, raw_capture, raw, velocities, value
        gc.collect()
        inherited.memory_guard()
    router.close()
    report['final_memory'] = memory_record()
    inherited.memory_guard()
    report['status'] = 'complete'


def profile_kernel_proof(summary, arm):
    rows = summary['kernel_aggregate']
    linear = [r for r in rows if 'sm120' in r['name'].lower() and 'e2m1' in r['name'].lower() and 'gemm' in r['name'].lower()]
    from h3_native_fp4_attention import is_native_fp4_attention_kernel
    attention = [r for r in rows if is_native_fp4_attention_kernel(r['name'])]
    expected_attention = 50 if MODE[arm] != 'bf16' else 0
    if sum(r['calls'] for r in linear) != 200 or sum(r['calls'] for r in attention) != expected_attention:
        raise RuntimeError(f'Actual kernel counts differ: linear={linear}, attention={attention}')
    return {'sm120_e2m1_gemm': linear, 'native_fp4_attention': attention,
            'expected_gemm_calls': 200, 'expected_attention_calls': expected_attention}


@torch.inference_mode()
def benchmark(args, report, manifest):
    if args.arm == 'bf16_original' or not args.profile:
        raise RuntimeError('E016 benches exactly the three SVD arms with one independent profile')
    evaluated = require_current_report(args.evaluate_report, report, args.arm)
    report['evaluate_reference'] = file_record(args.evaluate_report)
    expected = next(r for r in evaluated['cases'] if r['id'] == base.CASE_IDS[0])
    verify_file(expected['artifact'])
    pipe, router = setup_model(args, report, manifest)
    case = manifest['cases'][0]
    value = base.load_case(case)
    if base.tree_signature(value) != expected['input_signature']:
        raise RuntimeError('Benchmark input differs from same-arm evaluation')
    contract = report['attention_contracts'][case['id']]
    forward = base.gpu_call(pipe, case, value)
    del value

    def checked_endpoint(result):
        cpu, records = base.cpu_outputs(result)
        if records != expected['raw_outputs']:
            raise RuntimeError('Benchmark output SHA differs from same-arm E016 evaluation')
        return {m: row['sha256'] for m, row in records.items()}

    def measured_forward():
        check_budget(args, report)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with base.collect_fastpack_checks() as checks:
            with router.forward_context(**contract, diagnostics=False) as attn:
                wall_start = time.perf_counter()
                start.record()
                report['attempted_dit_calls'] += 1
                result = forward()
                end.record()
                end.synchronize()
                forward_end = time.perf_counter()
            attention_checked = time.perf_counter()
        flags_checked = time.perf_counter()
        report['complete_dit_calls'] += 1
        check_flags(checks, args.arm)
        # No RuntimeAudit hooks or diagnostic finite reductions in timed passes.
        if attn.summary['fp4_calls'] != (50 if MODE[args.arm] != 'bf16' else 0):
            raise RuntimeError('Timed attention route coverage changed')
        timing = {'cuda_ms': start.elapsed_time(end),
            'host_forward_ms_excluding_checks': 1000*(forward_end-wall_start),
            'attention_validation_ms': 1000*(attention_checked-forward_end),
            'flag_validation_ms': 1000*(flags_checked-attention_checked),
            'validation_ms': 1000*(flags_checked-forward_end),
            'host_ms_including_checks': 1000*(flags_checked-wall_start),
            'fastpack_checks': checks.summary, 'attention': attn.summary}
        timing['raw_output_sha256'] = checked_endpoint(result)
        del result
        return timing

    report['warmup'] = measured_forward()
    gc.collect()
    report['steady_before'] = {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()}
    torch.cuda.reset_peak_memory_stats()
    report['repeats'] = []
    for index in range(3):
        row = measured_forward()
        row['repeat'] = index
        report['repeats'].append(row)
        save(report, args.output)
        print(f'E016 bench {args.arm} repeat{index}: {row["cuda_ms"]:.3f} ms', flush=True)
    report['steady_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'after_allocated_bytes': torch.cuda.memory_allocated(),
        'after_reserved_bytes': torch.cuda.memory_reserved(), 'scope': 'Three unprofiled forwards after warmup; excludes startup and profiler'}
    report['latency_ms'] = {key: base.distribution([r[key] for r in report['repeats']]) for key in
        ('cuda_ms', 'host_forward_ms_excluding_checks', 'attention_validation_ms', 'flag_validation_ms', 'validation_ms', 'host_ms_including_checks')}
    check_budget(args, report)
    audit = base.RuntimeAudit()
    audit.phase = 'native'
    with base.collect_fastpack_checks() as checks, router.forward_context(**contract, diagnostics=True) as attn:
        with audit.installed(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False, profile_memory=False) as prof:
            report['attempted_dit_calls'] += 1
            result = forward()
            torch.cuda.synchronize()
        checks_start = time.perf_counter()
    checks_ms = 1000*(time.perf_counter()-checks_start)
    report['complete_dit_calls'] += 1
    audit_contract(audit.row(), args.arm, attn.summary)
    check_flags(checks, args.arm)
    digests = checked_endpoint(result)
    del result
    path = args.data_dir/'bench'/f'{args.arm}.trace.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    prof.export_chrome_trace(str(path))
    summary = base.trace_summary(path)
    report['profile'] = {'trace': file_record(path), 'summary': summary, 'runtime_audit': audit.row(),
        'attention': attn.summary, 'raw_output_sha256': digests, 'fastpack_checks': checks.summary,
        'validation_ms_outside_trace': checks_ms, 'kernel_proof': profile_kernel_proof(summary, args.arm),
        'scope': 'Independent pass with runtime counters; not steady latency; no arm-specific semantic labels'}
    router.close()
    report['final_memory'] = memory_record()
    inherited.memory_guard()
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate', 'bench'), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=REPORTS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--evaluate-report', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output = args.output or args.report_dir/('E016_check_bf16_original.json' if args.phase == 'check' else f'E016_{args.phase}_{args.arm}.json')
    args.check_report = args.check_report or args.report_dir/'E016_check_bf16_original.json'
    args.evaluate_report = args.evaluate_report or args.report_dir/f'E016_evaluate_{args.arm}.json'
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite prior report: {args.output}')
    report = {'experiment': 'E016', 'phase': args.phase, 'arm': args.arm, 'status': 'running',
        'deadline_unix': args.deadline_unix, 'attempted_dit_calls': 0, 'complete_dit_calls': 0,
        'scope': 'Fixed-state native FP4-attention compatibility/cost test; no decoded quality or new method claim'}
    try:
        manifest, references = prerequisites(args, report)
        if args.phase == 'check':
            cpu_check(args, report, manifest, references)
        elif args.phase == 'evaluate':
            evaluate(args, report, manifest, references)
        else:
            benchmark(args, report, manifest)
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc())
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
