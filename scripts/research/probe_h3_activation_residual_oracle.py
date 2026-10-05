#!/usr/bin/env python3
"""E059: activation-residual source oracle atop the frozen SVD native path.

The FP32 correction is a diagnostic, not native W4A16 or a deployment method.
Six zero replays must match history before six oracle calls are permitted.
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
from h3_native_nvfp4 import NativeH3Linear, TARGET_NAMES
from wan_native_nvfp4 import PackedNVFP4, unswizzle_scales
import torch
import torch.nn.functional as F

DATA = Path('/data1/models/svdquant-wjq/research/20261004/E059')
REPORTS = ROOT/'results/research/E059'
PLAN = ROOT/'research_state/06_experiments/E059_h3_activation_residual_oracle_plan.md'
CASE_IDS = ('e010_p030_s05', 'e010_p036_s14')
MODALITIES = ('video', 'audio')
MAX_CALLS = 12
save, file_record, signature = base.save, base.file_record, base.tree_signature
require = old.require


def key(item):
    return f"{item['id']}/{item['input_role']}"


def prerequisites(report):
    require(PLAN.is_file(), 'E059 plan must exist before check/evaluate')
    binding = {}
    args = types.SimpleNamespace(manifest=old.MANIFEST, plan=old.PLAN,
        data_dir=old.DATA, prepare_report=old.REPORTS/'E015_prepare.json')
    _, manifest, prepared, e014_manifest = old.prerequisites(args, binding)
    old_check = old.load_complete(old.REPORTS/'check.json')
    require(old_check['sources'] == binding['sources'] and
            old_check['prepared_reference'] == binding['prepared_reference'] and
            old_check['cuda_initialized'] is False, 'E015 CPU/source binding changed')
    e014 = old.load_complete(base.verify_file(manifest['e014_evaluations']['svd']))
    e015 = old.load_complete(old.REPORTS/'evaluate_svd.json')
    require(e015['sources'] == binding['sources'] and
            e015['prepared_reference'] == binding['prepared_reference'] and
            e015['complete_dit_calls'] == 8 and e015['arm'] == 'svd', 'E015 SVD provenance changed')
    items = []
    for cid in CASE_IDS:
        case = next(c for c in manifest['cases'] if c['id'] == cid)
        source = next(c for c in e014_manifest['cases'] if c['id'] == cid)
        history = next(r for r in e014['cases'] if r['id'] == cid)
        items.append(dict(id=cid, input_role='source_teacher', step=case['source_step'],
            source_case=source, case=case, history=history, historical_experiment='E014'))
        for corner, role in (('BB', 'next_teacher'), ('QQ', 'next_shifted')):
            row = next(r for r in prepared['cases'] if (r['id'], r['arm'], r['corner']) == (cid, 'svd', corner))
            history = next(r for r in e015['cases'] if (r['id'], r['corner']) == (cid, corner))
            require(history['prepared_artifact'] == row['artifact'] and history['next_step'] == row['next_step'],
                    'E015 historical input differs from selected prepared input')
            items.append(dict(id=cid, input_role=role, step=row['next_step'], prepared=row,
                case=case, history=history, historical_experiment='E015'))
    for item in items:
        base.verify_file(item['history']['artifact'])
    sources = dict(binding['sources'])
    sources.update({str(p.resolve()): file_record(p) for p in
        (Path(__file__), PLAN, HERE/'h3_native_nvfp4.py', HERE/'wan_native_nvfp4.py')})
    report.update(sources=sources, plan=file_record(PLAN), settings=manifest['settings'],
        asset_binding=binding['asset_binding'], e015_manifest=binding['manifest'],
        e015_prepare=binding['prepared_reference'], e015_check=file_record(old.REPORTS/'check.json'),
        e014_evaluation=manifest['e014_evaluations']['svd'],
        e015_evaluation=file_record(old.REPORTS/'evaluate_svd.json'), e014_check=manifest['e014_check'])
    return manifest, e014_manifest, items


def load_input(item):
    return (base.load_case(item['source_case']) if item['historical_experiment'] == 'E014'
            else builder.load_prepared_case(item['prepared']))


def input_signature(value, item, pipe):
    if item['historical_experiment'] == 'E014':
        return base.validate_case(item['source_case'], value, pipe)
    return old.validate_value(value, item['prepared'], item['case'], pipe, builder)


def capture_cpu_inputs(pipe, value):
    captured = []
    class Captured(Exception):
        pass
    def capture(*args, **kwargs):
        captured.append(signature({'args': args, 'kwargs': kwargs}))
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
    require(len(captured) == 1, 'CPU packing must reach one sentinel and perform zero model forwards')
    return captured[0]


@torch.inference_mode()
def cpu_check(report, manifest, items):
    require(not torch.cuda.is_initialized(), 'CPU check already initialized CUDA')
    torch.set_num_threads(6)
    pipe = builder.make_cpu_pipeline(manifest['settings'])
    report['inputs'] = {}
    for item in items:
        value = load_input(item)
        validated = input_signature(value, item, pipe)
        history = item['history']
        payload = torch.load(base.verify_file(history['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(validated == history['input_signature'] == payload['input_signature'], 'Historical input binding changed')
        # E014 stored the actual input signature; E015 stored the actual tensor tree.
        if item['historical_experiment'] == 'E014':
            expected = payload['actual_dit_inputs']
            require(expected == history['actual_dit_inputs'], 'E014 actual-input signature changed')
        else:
            expected = signature(payload['actual_dit_inputs'])
            require(expected == history['actual_dit_input_signature'], 'E015 actual-input tensors changed')
        for field in ('raw_outputs', 'velocities'):
            require(signature(payload[field]) == history[field], f'Historical {field} hashes changed')
        actual = capture_cpu_inputs(pipe, value)
        require(actual == expected, 'Selected input does not reproduce the historical actual DiT input')
        report['inputs'][key(item)] = dict(input_signature=validated, actual_dit_input_signature=actual,
            historical_artifact=history['artifact'], historical_experiment=item['historical_experiment'],
            step=item['step'], byte_exact_historical_input=True)
        print(f'E059 CPU input valid: {key(item)}', flush=True)
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, input_count=6, complete_dit_calls=0)


def guard(args, report, *, before_forward=False):
    require(time.time() < args.deadline_unix, 'E059 absolute deadline reached')
    if before_forward:
        require(report['attempted_dit_calls'] < MAX_CALLS, 'Twelve-forward budget exhausted')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E059 allocated peak reached 60 GiB')


def without_global(packet, *, start=0, stop=None, logical_scales=None):
    rows, k2 = packet.packed.shape
    stop = rows if stop is None else stop
    scales = logical_scales
    if scales is None:
        scales = unswizzle_scales(packet.swizzled_scales, rows, k2*2//16)
    return PackedNVFP4(packet.packed[start:stop], scales[start:stop],
        torch.ones_like(packet.global_scale), packet.swizzled_scales,
        (stop-start, k2*2), 'E059_decode_codes_times_block_scales_global_one')


def decode_sample64(packed, scales):
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1).long()
    return lut[codes] * scales.double().repeat_interleave(16, -1)


def error64(value, reference):
    error = value.double()-reference
    rms = float(reference.square().mean().sqrt())
    rms_error = float(error.square().mean().sqrt())
    max_error = float(error.abs().max())
    denom = max(rms, 1e-30)
    return dict(reference_rms=rms, rms_error=rms_error, max_abs_error=max_error,
        relative_rms_error=rms_error/denom, max_error_over_reference_rms=max_error/denom,
        passed=(rms_error == 0 and max_error == 0 if rms == 0 else
                rms_error/denom <= 1e-3 and max_error/denom <= 1e-2) and bool(torch.isfinite(error).all()))


def sample_check(native, packet, scales, x_s, wbar, residual, delta, main, corrected):
    """Check actual first full-chunk GEMM outputs against independent CPU FP64."""
    wp = native.weight_packet()
    ws = unswizzle_scales(wp.swizzled_scales, native.out_features, native.in_features//16)
    sample = dict(activation_packed=packet.packed[:2].detach().cpu(),
        activation_scales=scales[:2].float().cpu(), activation_global=packet.global_scale.cpu(),
        weight_packed=wp.packed[:32].detach().cpu(), weight_scales=ws[:32].float().cpu(),
        weight_global=wp.global_scale.cpu(), x_s=x_s[:2].detach().cpu(),
        delta_after_gemm=delta[:2, :32].detach().cpu(), main_native=main[:2, :32].detach().cpu(),
        main_corrected=corrected[:2, :32].detach().cpu())
    # Alternative FP32 global position, same actual residual and packed weights.
    sample['delta_before_gemm'] = (residual[:2] @ (wbar[:32]*wp.global_scale).t()).cpu()
    abar64 = decode_sample64(sample['activation_packed'], sample['activation_scales'])
    wbar64 = decode_sample64(sample['weight_packed'], sample['weight_scales'])
    exact = ((sample['x_s'].double()-abar64*sample['activation_global'].double()) @ wbar64.t())
    exact *= sample['weight_global'].double()
    sample['delta_fp64_reference'] = exact
    errors = {name: error64(sample[name], exact) for name in ('delta_after_gemm', 'delta_before_gemm')}
    exact_local_add = sample['main_native'].double()+sample['delta_after_gemm'].double()
    sample['local_addback_residual_fp64'] = sample['main_corrected'].double()-exact_local_add
    errors['addback_rounding_rms'] = float(sample['local_addback_residual_fp64'].square().mean().sqrt())
    errors['addback_rounding_max'] = float(sample['local_addback_residual_fp64'].abs().max())
    reference_rms = errors['delta_after_gemm']['reference_rms']
    errors['addback_rounding_over_reference_rms'] = errors['addback_rounding_rms']/max(reference_rms, 1e-30)
    errors['addback_resolution_passed'] = (errors['addback_rounding_rms'] == 0 if reference_rms == 0
        else errors['addback_rounding_over_reference_rms'] < 0.1)
    return sample, errors


class ActivationResidualOracle(torch.nn.Module):
    """Wrap the existing module without changing any packed/bias/LR tensors."""
    def __init__(self, native, name, context):
        super().__init__()
        self.native, self.name, self.context = native, name, context

    @torch.inference_mode()
    def forward(self, x):
        ctx, native = self.context, self.native
        guard(ctx['args'], ctx['report'])
        require(x.dtype == torch.bfloat16 and native.execution_mode == 'native', 'Frozen native/BF16 path changed')
        x_s = x/native.smooth
        packet = native.pack_input(x_s)
        branch = 1.0 * F.linear(F.linear(x_s, native.lr_a), native.lr_b)
        main_native = native.main_from_packet(packet, mode='native', include_bias=True)
        flat_x = x_s.reshape(-1, native.in_features)
        flat_main = main_native.reshape(-1, native.out_features)
        corrected = torch.empty_like(flat_main)
        total = torch.zeros(4, dtype=torch.float64, device=x.device)
        oracle = ctx['mode'] == 'oracle'
        if oracle:
            wbar = without_global(native.weight_packet()).decode(dtype=torch.float32, chunk_rows=ctx['chunk_rows'])
            scales = unswizzle_scales(packet.swizzled_scales, flat_x.shape[0], native.in_features//16)
        for start in range(0, flat_x.shape[0], ctx['chunk_rows']):
            stop = min(start+ctx['chunk_rows'], flat_x.shape[0])
            main32 = flat_main[start:stop].float()
            if oracle:
                abar = without_global(packet, start=start, stop=stop, logical_scales=scales).decode(
                    dtype=torch.float32, chunk_rows=ctx['chunk_rows'])
                aq = abar*packet.global_scale
                residual = flat_x[start:stop].float()-aq
                delta = (residual @ wbar.t())*native.weight_global
            else:
                delta = torch.zeros_like(main32)
            added32 = main32+delta
            corrected[start:stop] = added32.to(torch.bfloat16)
            if oracle:
                rounding = corrected[start:stop].float()-added32
                applied = corrected[start:stop].float()-main32
                total += torch.stack([t.square().sum().double() for t in (delta, main32, rounding, applied)])
                if start == 0 and self.name.startswith('blocks.0.'):
                    sample, errors = sample_check(native, packet, scales, flat_x, wbar, residual,
                        delta, flat_main, corrected)
                    ctx['samples'][self.name] = sample
                    ctx['checks'][self.name] = errors
                    require(all(errors[k]['passed'] for k in ('delta_after_gemm', 'delta_before_gemm')),
                            f'{self.name}: FP32 correction failed the FP64 sample bound: {errors}')
                    require(errors['addback_resolution_passed'],
                            f'{self.name}: BF16 addback rounding reaches 10% of correction RMS; insufficient resolution: {errors}')
                del abar, aq, residual, delta, rounding, applied
            guard(ctx['args'], ctx['report'])
        count = flat_main.numel()
        stats = dict(name=self.name, mode=ctx['mode'], elements=count, activation_global=float(packet.global_scale),
            weight_global=float(native.weight_global), native_scaled_mm_calls=1,
            correction_gemm_chunks=(flat_x.shape[0]+ctx['chunk_rows']-1)//ctx['chunk_rows'] if oracle else 0)
        if oracle:
            values = (total/count).sqrt().cpu().tolist()
            stats.update(dict(zip(('delta_rms', 'main_native_rms', 'bf16_addback_rounding_rms', 'applied_change_rms'), values)))
            require(all(torch.isfinite(total)), f'{self.name}: nonfinite correction statistics')
            del wbar, scales
        else:
            require(torch.equal(corrected, flat_main), f'{self.name}: FP32 + zero changed BF16 native result')
            stats.update(delta_rms=0., bf16_addback_rounding_rms=0., applied_change_rms=0., zero_local_exact=True)
        ctx['layers'].append(stats)
        guard(ctx['args'], ctx['report'])
        return corrected.reshape_as(main_native)+branch


@torch.inference_mode()
def evaluate(args, report, manifest, e014_manifest, items):
    checked = old.load_complete(REPORTS/'check.json')
    require(checked['sources'] == report['sources'] and checked['cuda_initialized'] is False and
            checked['e015_evaluation'] == report['e015_evaluation'] and
            checked['e014_evaluation'] == report['e014_evaluation'] and
            checked['chunk_rows'] == args.chunk_rows, 'E059 checked provenance/chunk size changed')
    report['cpu_check_reference'] = file_record(REPORTS/'check.json')
    setup_path = DATA/'setup_svd.json'
    require(not setup_path.exists(), f'Refusing existing setup {setup_path}')
    old_check = old.load_complete(manifest['e014_check']['file'])
    setup = dict(experiment='E059_inherited_E014_setup', status='running', complete_dit_calls=0, sources=old_check['sources'])
    setup_args = types.SimpleNamespace(arm='svd', deadline_unix=args.deadline_unix,
        check_report=Path(manifest['e014_check']['file']), output=setup_path)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    builder.configure_schedule(pipe, manifest['settings'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    require(not torch.backends.cuda.matmul.allow_tf32, 'TF32 must be disabled')
    context = dict(args=args, report=report, chunk_rows=args.chunk_rows, mode='zero')
    non_target = base.non_target_identity(pipe.dit)
    for name in TARGET_NAMES:
        native = pipe.dit.get_submodule(name)
        require(type(native) is NativeH3Linear, f'{name}: not original SVD native module')
        parent, attr = name.rsplit('.', 1)
        setattr(pipe.dit.get_submodule(parent), attr, ActivationResidualOracle(native, name, context))
    require(base.non_target_identity(pipe.dit) == non_target, 'Wrapper installation changed non-target tensors')
    report['wrapper_count'] = len(TARGET_NAMES)
    report['tf32_enabled'] = torch.backends.cuda.matmul.allow_tf32
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    for mode in ('zero', 'oracle'):
        if mode == 'oracle':
            require(report['complete_by_mode']['zero'] == 6 and all(r['historical_replay_exact']
                    for r in report['cases'] if r['mode'] == 'zero'), 'All six zero replays must pass before oracle')
        for item in items:
            guard(args, report, before_forward=True)
            context.update(mode=mode, layers=[], samples={}, checks={})
            value = load_input(item)
            validated = input_signature(value, item, cpu_pipe)
            expected = checked['inputs'][key(item)]
            require(validated == expected['input_signature'], 'Selected input changed after CPU check')
            path = DATA/'evaluate'/mode/(key(item).replace('/', '_')+'.pt')
            require(not path.exists(), f'Refusing existing artifact {path}')
            forward = base.gpu_call(pipe, {'kind': 'model_fn'}, builder.model_value(value))
            actual, raw_capture = [], []
            def capture_input(_module, positional, keyword):
                captured = base.inherited.tree_cpu({'args': positional, 'kwargs': keyword})
                require(signature(captured) == expected['actual_dit_input_signature'],
                        'Blocking actual DiT input snapshot differs from historical input')
                actual.append(captured)
            def capture_output(_module, positional, output):
                raw_capture.append(output)
            pre = pipe.dit.register_forward_pre_hook(capture_input, with_kwargs=True)
            post = pipe.dit.register_forward_hook(capture_output)
            audit = base.RuntimeAudit()
            audit.phase = 'native'
            report['attempted_dit_calls'] += 1
            report['active_call'] = dict(mode=mode, id=item['id'], input_role=item['input_role'])
            save(report, args.output)
            print(f'E059 {mode} {key(item)} step={item["step"]}', flush=True)
            try:
                with audit.installed(), base.collect_fastpack_checks() as fastpack:
                    result = forward()
                    torch.cuda.synchronize()
                report['complete_dit_calls'] += 1
                report['complete_by_mode'][mode] += 1
            except Exception:
                report['failed_call_diagnostics'] = dict(layers=context['layers'], numerical_checks=context['checks'])
                if context['samples']:
                    failed_path = DATA/'failed_samples'/mode/(key(item).replace('/', '_')+'.pt')
                    failed_path.parent.mkdir(parents=True, exist_ok=True)
                    with failed_path.open('xb') as stream:
                        torch.save(dict(samples=context['samples'], numerical_checks=context['checks']), stream)
                    report['failed_call_diagnostics']['artifact'] = file_record(failed_path)
                raise
            finally:
                pre.remove()
                post.remove()
            base.audit_contract(audit.row(), 'svd')
            require(fastpack.summary['checked_calls'] == 200 and len(context['layers']) == 200,
                    'Must execute all 200 native linears and activation checks')
            require(len(actual) == len(raw_capture) == 1, 'Expected one full DiT call')
            require(mode == 'zero' or len(context['checks']) == 4, 'Missing block0 numerical checks')
            raw, raw_records = base.cpu_outputs(raw_capture[0])
            velocities, velocity_records = base.cpu_outputs(result)
            history = torch.load(base.verify_file(item['history']['artifact']), map_location='cpu', weights_only=True, mmap=True)
            exact = all(torch.equal(tensors[m], history[field][m]) and base.tensor_record(tensors[m]) == item['history'][field][m]
                for field, tensors in (('raw_outputs', raw), ('velocities', velocities)) for m in MODALITIES)
            require(mode != 'zero' or exact, 'Zero raw/velocity replay differs from history; stop before oracle')
            payload = dict(case_id=item['id'], position=item['input_role'], step=item['step'], mode=mode,
                input_signature=validated, actual_dit_inputs=actual[0], raw_outputs=raw, velocities=velocities,
                correction_samples=context['samples'], layer_statistics=context['layers'],
                numerical_checks=context['checks'], historical_artifact=item['history']['artifact'])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                torch.save(payload, stream)
            report['cases'].append(dict(case_id=item['id'], position=item['input_role'], step=item['step'], mode=mode,
                status='complete', artifact=file_record(path), historical_artifact=item['history']['artifact'],
                historical_replay_exact=exact, raw_outputs=raw_records, velocities=velocity_records,
                actual_dit_input_signature=signature(actual[0]),
                runtime_audit=audit.row(), fastpack_checks=fastpack.summary,
                layer_statistics=context['layers'], numerical_checks=context['checks']))
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            save(report, args.output)
            del forward, value, result, raw_capture, actual, raw, velocities, history, payload
            gc.collect()
            guard(args, report)
    require(report['attempted_dit_calls'] == report['complete_dit_calls'] == MAX_CALLS, 'Incomplete prescribed 12 forwards')
    report.update(status='complete', active_call=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate'), required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--chunk-rows', type=int, choices=(512, 1024), default=1024)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or REPORTS/f'{args.phase}.json'
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    started = time.time()
    if args.phase == 'evaluate':
        require(args.deadline_unix is not None and 0 < args.deadline_unix-started <= 1800,
                'Evaluate requires root-assigned absolute deadline no more than 1800 seconds away')
    report = dict(experiment='E059', phase=args.phase, status='running', start_epoch=started,
        deadline_unix=args.deadline_unix, wall_budget_seconds=1800, max_dit_calls=MAX_CALLS,
        attempted_dit_calls=0, complete_dit_calls=0, complete_by_mode={'zero': 0, 'oracle': 0}, cases=[],
        chunk_rows=args.chunk_rows, scope='Activation residual source oracle; no native W4A16, novelty or deployment claim')
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E059 absolute wall deadline reached')
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
