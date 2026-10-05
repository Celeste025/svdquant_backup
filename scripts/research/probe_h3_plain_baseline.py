#!/usr/bin/env python3
"""E014: three frozen full-H3 states, BF16 / legacy SVD / direct-W NVFP4.

No generation, fitting, input selection, or quality gate. GPU phases are run by
an external serial launcher with one absolute deadline shared across all arms.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import importlib
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
# This frozen import establishes the original offline, BF16-SDPA environment.
import run_h3_native_paired_video as inherited
import torch
from minimax_h3_svdquant_common import tree_device
from probe_h3_conditional_response import make_packed, model_fn_minimax_h3
from bench_h3_native_nvfp4 import TARGETS, RuntimeAudit, make_h3_resident
from profile_h3_native_nvfp4 import distribution, resident_storage, trace_summary
from h3_native_nvfp4 import install_native_h3
from h3_nvfp4_zero_sf_compat import collect_fastpack_checks, pack_activation_fast

sha256, save = inherited.sha256, inherited.save
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E014')
REPORTS = ROOT/'results/research/E014'
MANIFEST = ROOT/'research_state/06_experiments/E014_h3_plain_baseline_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E014_h3_plain_baseline_plan.md'
MODALITIES = ('video', 'audio')
CASE_IDS = ('e009_p001_s00', 'e010_p030_s05', 'e010_p036_s14')


def file_record(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def verify_file(row):
    path = Path(row['file'])
    if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
        raise RuntimeError(f'File provenance changed: {path}')
    return path


def tensor_record(value):
    # The historical helper cannot byte-view a scalar FP32 tensor directly.
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': inherited.tensor_sha(value.reshape(-1))}


def tree_signature(value):
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, dict):
        return {k: tree_signature(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [tree_signature(v) for v in value]
    return value


def check_sources(records):
    for path, row in records.items():
        if sha256(path) != row['sha256']:
            raise RuntimeError(f'Frozen source changed: {path}')


def source_records(args):
    paths = set(inherited.source_paths()) | {
        Path(__file__), args.manifest, args.plan,
        HERE/'probe_h3_conditional_response.py', HERE/'h3_plain_native_nvfp4.py',
    }
    # CPU input validation may precede the independent plain module/export.
    return {str(p.resolve()): file_record(p) for p in sorted(paths) if p.exists()}


def check_budget(args, report):
    if args.deadline_unix is None:
        raise RuntimeError('GPU phases require the shared --deadline-unix; never restart its clock')
    if time.time() >= args.deadline_unix:
        raise RuntimeError('Shared E014 wall-clock deadline reached')
    if report['complete_dit_calls'] >= 5:
        raise RuntimeError('This process would exceed its preregistered maximum of five DiT calls')
    inherited.memory_guard()


def prerequisites(args, report):
    manifest = json.loads(args.manifest.read_text())
    if tuple(c['id'] for c in manifest['cases']) != CASE_IDS or manifest['arms'] != ['bf16', 'svd', 'plain']:
        raise RuntimeError('The fixed E014 case/arm list changed')
    if manifest['benchmark']['warmup'] != 1 or manifest['benchmark']['repeats'] != 3:
        raise RuntimeError('E014 benchmark call allocation changed')
    if 'source_files' in manifest:
        check_sources(manifest['source_files'])
    inherited_report = {}
    inherited.prerequisites(types.SimpleNamespace(arm='native', export_dir=Path(manifest['svd_export_dir'])), inherited_report)
    report['inherited_binding'] = inherited_report
    asset = inherited.load_complete(verify_file(manifest['asset_reference']))
    check_sources(asset['sources'])
    for path, row in asset['assets'].items():
        st = Path(path).stat()
        if st.st_size != row['bytes'] or st.st_mtime_ns != row['mtime_ns']:
            raise RuntimeError(f'Asset changed since original full SHA verification: {path}')
    report['asset_binding'] = {'reference': manifest['asset_reference'], 'assets': asset['assets'],
        'current_check': 'Size and mtime equal the full-SHA-verified E010 prepare assets; no repeated full TE hash'}
    for case in manifest['cases']:
        reference = inherited.load_complete(verify_file(case['reference']))
        if 'sources' in reference:
            check_sources(reference['sources'])
        if case['kind'] == 'raw_dit':
            verify_file(case['sample'])
            expected = reference['warmup']['endpoints_sha256']
        elif case['kind'] == 'model_fn':
            verify_file(case['prepared'])
            for row in case['state'].values():
                verify_file(row)
            trajectory = next(r for r in reference['cases'] if r['prompt_id'] == case['prompt_id'])
            actual = next(r for r in trajectory['dit_calls'] if r['step'] == case['step'])
            expected = {m: actual[m+'_sha256'] for m in MODALITIES}
            for modality, record in case['state'].items():
                old = next(r for r in trajectory['steps'] if r['step'] == case['step'] and r['modality'] == modality)
                if record['file'] != old['file'] or record['sha256'] != old['sha256']:
                    raise RuntimeError('E010 teacher-state source changed')
        else:
            raise RuntimeError('Unsupported case kind')
        if case['bf16_raw_output_sha256'] != expected:
            raise RuntimeError('E014 expected output differs from the original BF16 reference')
    report['sources'] = source_records(args)
    report['manifest'] = file_record(args.manifest)
    report['plan'] = file_record(args.plan)
    return manifest


def load_case(case):
    if case['kind'] == 'raw_dit':
        value = torch.load(case['sample']['file'], map_location='cpu', weights_only=False, mmap=True)
        return {'args': value['input_args'], 'kwargs': value['input_kwargs']}
    prepared = torch.load(case['prepared']['file'], map_location='cpu', weights_only=True, mmap=True)
    state = {m: torch.load(case['state'][m]['file'], map_location='cpu', weights_only=True, mmap=True) for m in MODALITIES}
    return {'embedding': prepared['embedding'], 'text_token_tags': prepared['text_token_tags'], 'state': state}


def validate_case(case, value, pipe):
    if case['kind'] == 'raw_dit':
        if value['args'] or value['kwargs'].get('control_hints') is not None:
            raise RuntimeError('Unexpected raw DiT positional input or control hints')
        return tree_signature(value)
    embedding, tags, state = value['embedding'], value['text_token_tags'], value['state']
    if embedding.ndim != 2 or embedding.shape[1] != 5120 or embedding.dtype != torch.bfloat16:
        raise RuntimeError('Actual E010 embedding must be BF16 [text_tokens, 5120]')
    if tags.ndim != 1 or tags.numel() != embedding.shape[0] or not bool(torch.isfinite(embedding).all()):
        raise RuntimeError('Actual embedding/tag contract failed')
    for modality, row in state.items():
        if row['latents_before'].dtype != torch.bfloat16 or row['noise_pred'].dtype != torch.bfloat16:
            raise RuntimeError(f'{modality}: teacher latents and velocity must be BF16')
        for name in ('timestep', 'sigma'):
            if row[name].ndim != 0 or row[name].dtype != torch.float32:
                raise RuntimeError(f'{modality}: {name} must be the original 0D FP32 scalar')
        if any(not bool(torch.isfinite(v).all()) for v in row.values()):
            raise RuntimeError('Nonfinite saved teacher state')
    packed = make_packed(pipe, embedding, tags, state)
    if 50*int((packed['cu_seqlens'][1:] > packed['cu_seqlens'][:-1]).sum())+2 != 102:
        raise RuntimeError('Unexpected original packed segment count')
    return {'embedding': tensor_record(embedding), 'text_token_tags': tensor_record(tags),
            'state': tree_signature(state), 'packed': tree_signature(packed)}


def cpu_check(args, report, manifest):
    pipe = inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['inputs'] = {}
    for case in manifest['cases']:
        value = load_case(case)
        report['inputs'][case['id']] = validate_case(case, value, pipe)
        print(f'E014 CPU input valid: {case["id"]}', flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only validation initialized CUDA')
    report.update(status='complete', cuda_initialized=False, complete_dit_calls=0)


def non_target_identity(model):
    # Inference tensors have no version counter. This is an identity/storage
    # guard, not a byte-level claim that arbitrary in-place edits were excluded.
    return {n: (id(t), None if torch.is_inference(t) else t._version,
                t.untyped_storage().data_ptr(), tuple(t.shape), str(t.dtype), str(t.device))
            for n, t in list(model.named_parameters())+list(model.named_buffers())
            if not any(n.startswith(target+'.') for target in TARGETS)}


def setup_model(args, report, manifest):
    check_budget(args, report)
    checked = inherited.load_complete(args.check_report)
    check_sources(checked['sources'])
    if checked['sources'] != report['sources'] or checked.get('cuda_initialized') is not False:
        raise RuntimeError('CPU check source set changed; rerun CPU check after final freeze')
    report['cpu_check_reference'] = file_record(args.check_report)
    report['checked_inputs'] = checked['inputs']
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    if importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('Original torch SDPA backend required')
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('This experiment binds the SM120 native NVFP4 path')
    report['device'] = {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'name': torch.cuda.get_device_name(), 'capability': [12, 0]}
    start = time.monotonic()
    pipe = inherited.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    report['resident_conversion'] = make_h3_resident(pipe.dit)
    before = non_target_identity(pipe.dit)
    if args.arm == 'svd':
        report['installation'] = install_native_h3(pipe.dit, manifest['svd_export_dir'],
            activation_packer=pack_activation_fast, chunk_rows=1024)
    elif args.arm == 'plain':
        from h3_plain_native_nvfp4 import install_plain_h3
        report['installation'] = install_plain_h3(pipe.dit, manifest['plain_export_dir'],
            activation_packer=pack_activation_fast, chunk_rows=1024)
    if args.arm != 'bf16' and report['installation']['target_count'] != 200:
        raise RuntimeError('Quantized installation must cover exactly 200 linears')
    if non_target_identity(pipe.dit) != before:
        raise RuntimeError('Quantized installation changed non-target tensor identities/versions')
    report['non_target_identity_preserved'] = True
    report['non_target_guard_scope'] = 'Object/storage identity, shape, dtype/device and available version counter; not value hashing'
    from diffsynth.core.vram.layers import AutoTorchModule, AutoWrappedLinear, AutoWrappedModule
    if any(isinstance(m, AutoTorchModule) for m in pipe.dit.modules()):
        raise RuntimeError('Offload wrapper survived resident conversion')
    def forbidden_disk(*_args, **_kwargs):
        raise RuntimeError('Disk weight loading forbidden in resident E014')
    AutoWrappedLinear.load_from_disk = AutoWrappedModule.load_from_disk = forbidden_disk
    gc.collect()
    torch.cuda.empty_cache()
    report['startup_seconds'] = time.monotonic()-start
    report['startup_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['resident_model_storage'] = resident_storage(pipe.dit)
    inherited.memory_guard()
    save(report, args.output)
    return pipe


def gpu_call(pipe, case, value):
    gpu = tree_device(value, 'cuda')
    if case['kind'] == 'raw_dit':
        return lambda: pipe.dit(*gpu['args'], **gpu['kwargs'])
    state = gpu['state']
    packed = make_packed(pipe, gpu['embedding'], gpu['text_token_tags'], state)
    return lambda: model_fn_minimax_h3(dit=pipe.dit,
        video_latents=state['video']['latents_before'], audio_latents=state['audio']['latents_before'],
        packed=packed, prompt_embeds=gpu['embedding'],
        timestep_video=state['video']['timestep'].reshape(1), timestep_audio=state['audio']['timestep'].reshape(1))


def cpu_outputs(result):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError('Expected video/audio output pair')
    outputs = {m: x.detach().cpu() for m, x in zip(MODALITIES, result, strict=True)}
    if any(not bool(torch.isfinite(x).all()) for x in outputs.values()):
        raise RuntimeError('Nonfinite full-model output')
    return outputs, {m: tensor_record(x) for m, x in outputs.items()}


def audit_contract(row, arm):
    expected = {'sdpa_calls': 102, 'scaled_mm_calls': 0 if arm == 'bf16' else 200, 'disk_loads': 0}
    if {key: row[key] for key in expected} != expected:
        raise RuntimeError(f'Actual full-model execution differs: {row}')


@torch.inference_mode()
def evaluate(args, report, manifest):
    pipe = setup_model(args, report, manifest)
    report['cases'] = []
    for case in manifest['cases']:
        check_budget(args, report)
        value = load_case(case)
        signature = validate_case(case, value, inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16))
        if signature != report['checked_inputs'][case['id']]:
            raise RuntimeError('Input signature differs from the CPU check')
        forward = gpu_call(pipe, case, value)
        actual_inputs, raw_capture = [], []
        def observe_input(_module, pos, kw):
            actual_inputs.append(tree_signature({'args': pos, 'kwargs': kw}))
        def observe_output(_module, pos, output):
            raw_capture.append(output)
        pre = pipe.dit.register_forward_pre_hook(observe_input, with_kwargs=True)
        post = pipe.dit.register_forward_hook(observe_output)
        audit = RuntimeAudit()
        audit.phase = 'native' if args.arm != 'bf16' else 'resident_bf16'
        checks_context = collect_fastpack_checks() if args.arm != 'bf16' else nullcontext(None)
        print(f'E014 evaluate {args.arm} {case["id"]}', flush=True)
        try:
            with audit.installed(), checks_context as checks:
                result = forward()
                torch.cuda.synchronize()
            report['complete_dit_calls'] += 1
        finally:
            pre.remove()
            post.remove()
        audit_contract(audit.row(), args.arm)
        if len(raw_capture) != 1 or len(actual_inputs) != 1:
            raise RuntimeError('Must capture exactly one full DiT call')
        if checks is not None and checks.summary['checked_calls'] != 200:
            raise RuntimeError('Expected 200 activation-domain checks')
        raw, records = cpu_outputs(raw_capture[0])
        raw_sha = {m: row['sha256'] for m, row in records.items()}
        velocities, velocity_records = (None, None)
        replay = {'raw_sha_exact': None, 'velocity_exact': None}
        if case['kind'] == 'model_fn':
            velocities, velocity_records = cpu_outputs(result)
        if args.arm == 'bf16':
            replay['raw_sha_exact'] = raw_sha == case['bf16_raw_output_sha256']
            if not replay['raw_sha_exact']:
                raise RuntimeError(f'Historical BF16 raw endpoint replay failed: {case["id"]}, got {raw_sha}')
            if velocities is not None:
                replay['velocity_exact'] = {m: (torch.equal(velocities[m], value['state'][m]['noise_pred']) and
                    tensor_record(velocities[m]) == tensor_record(value['state'][m]['noise_pred'])) for m in MODALITIES}
                if not all(replay['velocity_exact'].values()):
                    raise RuntimeError('Historical E010 scheduler velocity replay failed')
        artifact = args.data_dir/'evaluate'/args.arm/(case['id']+'.pt')
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists():
            raise FileExistsError(artifact)
        torch.save({'case_id': case['id'], 'arm': args.arm, 'raw_outputs': raw,
                    'velocities': velocities, 'input_signature': signature,
                    'actual_dit_inputs': actual_inputs[0]}, artifact)
        report['cases'].append({'id': case['id'], 'kind': case['kind'], 'status': 'complete',
            'input_signature': signature, 'actual_dit_inputs': actual_inputs[0],
            'raw_outputs': records, 'velocities': velocity_records, 'artifact': file_record(artifact),
            'historical_bf16_replay': replay, 'runtime_audit': audit.row(),
            'fastpack_checks': checks.summary if checks is not None else None})
        save(report, args.output)
        del forward, result, raw_capture, raw, velocities, value
        gc.collect()
        inherited.memory_guard()
    report['status'] = 'complete'


@torch.inference_mode()
def benchmark(args, report, manifest):
    evaluated = inherited.load_complete(args.evaluate_report)
    if evaluated['arm'] != args.arm or evaluated['sources'] != report['sources']:
        raise RuntimeError('Bench must bind the same-arm, same-source completed E014 evaluation')
    check_sources(evaluated['sources'])
    report['evaluate_reference'] = file_record(args.evaluate_report)
    expected = next(r for r in evaluated['cases'] if r['id'] == CASE_IDS[0])
    verify_file(expected['artifact'])
    pipe = setup_model(args, report, manifest)
    case = manifest['cases'][0]
    value = load_case(case)
    if tree_signature(value) != expected['input_signature']:
        raise RuntimeError('Benchmark input differs from its evaluation')
    forward = gpu_call(pipe, case, value)
    del value

    def checked_endpoint(result):
        cpu, records = cpu_outputs(result)
        if records != expected['raw_outputs']:
            raise RuntimeError('Benchmark endpoint SHA differs from same-arm E014 evaluation')
        return {m: row['sha256'] for m, row in records.items()}

    def measured_forward():
        check_budget(args, report)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        context = collect_fastpack_checks() if args.arm != 'bf16' else nullcontext(None)
        with context as checks:
            wall_start = time.perf_counter()
            start.record()
            result = forward()
            end.record()
            end.synchronize()
            forward_end = time.perf_counter()
        checked = time.perf_counter()
        report['complete_dit_calls'] += 1
        if checks is not None and checks.summary['checked_calls'] != 200:
            raise RuntimeError('Expected 200 flags outside timing')
        timing = {'cuda_ms': start.elapsed_time(end),
                  'host_forward_ms_excluding_checks': 1000*(forward_end-wall_start),
                  'flag_validation_ms': 1000*(checked-forward_end),
                  'host_ms_including_checks': 1000*(checked-wall_start),
                  'fastpack_checks': checks.summary if checks is not None else None}
        # All output copies/hashing and domain validation occur after end sync.
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
        print(f'E014 bench {args.arm} repeat{index}: {row["cuda_ms"]:.3f} ms', flush=True)
    report['steady_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'after_allocated_bytes': torch.cuda.memory_allocated(),
        'after_reserved_bytes': torch.cuda.memory_reserved(), 'scope': 'Three unprofiled forwards after warmup; excludes startup and profiler'}
    report['latency_ms'] = {key: distribution([r[key] for r in report['repeats']]) for key in
        ('cuda_ms', 'host_forward_ms_excluding_checks', 'flag_validation_ms', 'host_ms_including_checks')}
    if args.profile:
        check_budget(args, report)
        audit = RuntimeAudit()
        audit.phase = 'native' if args.arm != 'bf16' else 'resident_bf16'
        context = collect_fastpack_checks() if args.arm != 'bf16' else nullcontext(None)
        with context as checks:
            with audit.installed(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False, profile_memory=False) as prof:
                result = forward()
                torch.cuda.synchronize()
        report['complete_dit_calls'] += 1
        audit_contract(audit.row(), args.arm)
        digests = checked_endpoint(result)
        del result
        if checks is not None and checks.summary['checked_calls'] != 200:
            raise RuntimeError('Profiler must cover 200 activation checks')
        path = args.data_dir/'bench'/f'{args.arm}.trace.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        prof.export_chrome_trace(str(path))
        summary = trace_summary(path)
        # No E009 semantic wrappers: identical raw profiler treatment for all arms.
        if args.arm != 'bf16' and not any('sm120' in row['name'] and 'e2m1' in row['name'] for row in summary['kernel_aggregate']):
            raise RuntimeError('Profiler has no actual SM120 E2M1 GEMM evidence')
        report['profile'] = {'trace': file_record(path), 'summary': summary, 'runtime_audit': audit.row(),
            'raw_output_sha256': digests, 'fastpack_checks': checks.summary if checks is not None else None,
            'scope': 'Independent pass with runtime counters; not steady latency; no arm-specific semantic labels'}
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate', 'bench'), required=True)
    parser.add_argument('--arm', choices=('bf16', 'svd', 'plain'), required=True)
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
    args.output = args.output or args.report_dir/('E014_check_bf16.json' if args.phase == 'check' else f'E014_{args.phase}_{args.arm}.json')
    args.check_report = args.check_report or args.report_dir/'E014_check_bf16.json'
    args.evaluate_report = args.evaluate_report or args.report_dir/f'E014_evaluate_{args.arm}.json'
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite prior report: {args.output}')
    report = {'experiment': 'E014', 'phase': args.phase, 'arm': args.arm, 'status': 'running',
              'deadline_unix': args.deadline_unix, 'complete_dit_calls': 0,
              'scope': 'Fixed-state numerical/cost baseline only; no decoded video or quality claim'}
    try:
        manifest = prerequisites(args, report)
        if args.phase == 'check':
            cpu_check(args, report, manifest)
        elif args.phase == 'evaluate':
            evaluate(args, report, manifest)
        else:
            benchmark(args, report, manifest)
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc())
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
