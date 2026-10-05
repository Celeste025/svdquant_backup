#!/usr/bin/env python3
"""E004: ordinary modality-weighted block reconstruction, fixed fc2 A/rank.

This is a cheap competing explanation, not a new method. All raw calls belonged
to the original PTQ calibration set; only this refit's train/eval prompts differ.
No gradient descent, rank increase, quantizer search, or full-network QAT.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
MODALITIES = ('video', 'audio', 'text')
REFITS = ('pooled_refit', 'normalized_refit', 'video4_refit')
TRAIN = ((20, 0), (20, 19), (25, 0), (25, 19))
EVAL = ((11, 0), (11, 19), (46, 0), (46, 19))
RIDGE = 1e-3


def loss_weights(mode, stats):
    if mode == 'pooled_refit':
        return {m: 1 / sum(stats[k]['ref2'] for k in MODALITIES) for m in MODALITIES}
    alpha = dict(video=4 if mode == 'video4_refit' else 1, audio=1, text=1)
    return {m: alpha[m] / sum(alpha.values()) / max(stats[m]['ref2'], 1e-30) for m in MODALITIES}


def solve_refit(mode, stats, old_b):
    weights = loss_weights(mode, stats)
    gram = sum(weights[m] * stats[m]['gram'] for m in MODALITIES)
    rhs = sum(weights[m] * stats[m]['rhs'] for m in MODALITIES)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1)
    floor = max(scale.mean().item() * 1e-6, 1e-30)
    ridge = RIDGE * scale.clamp_min(floor)
    regularized = gram + ridge[:, None, None] * torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    delta = torch.linalg.solve(regularized, rhs.unsqueeze(-1)).squeeze(-1)
    candidate = (old_b.double() + delta).to(old_b.dtype)
    if not torch.isfinite(candidate).all():
        raise RuntimeError(f'nonfinite fitted B: {mode}')
    # Account for serialization rounding in the predictive quadratic objective.
    actual_delta = candidate.double() - old_b.double()
    before = sum(weights[m] * stats[m]['err2'] for m in MODALITIES)
    after = before - 2 * (actual_delta * rhs).sum().item()
    after += torch.einsum('or,ors,os->', actual_delta, gram, actual_delta).item()
    return candidate, {'weights': weights, 'ridge_rule': RIDGE,
                       'ridge_min': ridge.min().item(), 'ridge_max': ridge.max().item(),
                       'predicted_before': before, 'predicted_after': after,
                       'delta_relative_norm': (actual_delta.norm() / old_b.double().norm()).item()}


def self_test():
    """CPU check against independently materialized gated weighted regression."""
    torch.manual_seed(20261002)
    r, out = 3, 5
    old = torch.randn(out, r, dtype=torch.float64)
    target_delta = torch.randn(out, r, dtype=torch.float64)
    stats = {}
    design = {}
    for m in MODALITIES:
        phi = torch.randn(17, r, dtype=torch.float64)
        gate = torch.randn(out, dtype=torch.float64)
        target = (phi @ target_delta.T) * gate
        stats[m] = {'gram': gate[:, None, None].square() * (phi.T @ phi)[None],
                    'rhs': gate[:, None] * (phi.T @ target).T,
                    'ref2': float(1 + torch.randn(17, out).square().sum()),
                    'err2': target.square().sum().item()}
        design[m] = (phi, gate, target)
    for mode in REFITS:
        b, info = solve_refit(mode, stats, old)
        delta = b - old
        weights = loss_weights(mode, stats)
        explicit = sum(weights[m] * (((design[m][0] @ delta.T) * design[m][1] - design[m][2]).square().sum().item())
                       for m in MODALITIES)
        if abs(explicit - info['predicted_after']) > 1e-10:
            raise AssertionError((mode, explicit, info['predicted_after']))
        if explicit > info['predicted_before'] * .01:
            raise AssertionError('gated fit did not recover known representable target')
    return {'passed': True, 'device': 'cpu', 'modes': list(REFITS),
            'boundary': 'closed-form gated sufficient statistics only; no H3 or GPU'}


@torch.inference_mode()
def execute(args, report):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from minimax_h3_svdquant_common import (H3_DIT_PATH, TARGET_SUFFIXES, load_h3_pipeline,
        raw_call_to_block0, tree_cpu, tree_device, nvfp4_qdq, install_runtime_hooks)
    from probe_h3_modality_pulse import sha256, tensor_sha, partitions, summarize, save
    start = time.time()
    def checkpoint():
        report['elapsed_seconds'] = time.time() - start
        save(report, args.output)
        if report['elapsed_seconds'] > args.max_seconds:
            raise TimeoutError('E004 preregistered wall-clock budget exhausted')
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    attention = importlib.import_module('diffsynth.core.attention.attention')
    if attention.ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('E004 requires DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch')
    sdpa_original = torch.nn.functional.scaled_dot_product_attention
    sdpa = {'calls': 0, 'dtypes': set()}
    def audited(q, k, v, *pos, **kw):
        sdpa['calls'] += 1
        sdpa['dtypes'].add(tuple(str(t.dtype) for t in (q, k, v)))
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            raise RuntimeError('non-BF16 attention input')
        return sdpa_original(q, k, v, *pos, **kw)
    torch.nn.functional.scaled_dot_product_attention = audited
    paths = [Path(__file__), ROOT/'scripts/minimax_h3_svdquant_common.py',
             ROOT/'scripts/research/probe_h3_modality_pulse.py', args.state, H3_DIT_PATH]
    paths += [args.cache/f'p{pid}'/f'sample_p{pid}_s{step:02d}.pt' for pid, step in TRAIN + EVAL]
    for name in ('diffsynth.models.minimax_h3_dit', 'diffsynth.models.minimax_h3_dit_comfy',
                 'diffsynth.core.attention.attention'):
        paths.append(Path(importlib.import_module(name).__file__))
    report['files'] = {str(p): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in paths}
    report['environment'] = {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'DIFFSYNTH_ROOT',
        'MINIMAX_H3_DIT_PATH', 'SVDQUANT_DATA_ROOT', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION')}
    report['torch'] = torch.__version__
    report['device'] = torch.cuda.get_device_name()
    checkpoint()
    pipe = load_h3_pipeline(full=False, vram_limit_gib=35.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    if len(pipe.dit.blocks) != 50:
        raise RuntimeError('expected 50 blocks')
    block = pipe.dit.blocks[0]
    state = torch.load(args.state, map_location='cpu', weights_only=False)
    linears, weights, plain, quant = {}, {}, {}, {}
    for suffix in TARGET_SUFFIXES:
        old = block.get_submodule(suffix)
        w, bias = old.load_from_disk(torch.bfloat16, 'cuda', assign=False)
        linear = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, device='cuda', dtype=torch.bfloat16)
        linear.weight.copy_(w)
        if bias is not None:
            linear.bias.copy_(bias)
        parent, attr = suffix.rsplit('.', 1)
        setattr(block.get_submodule(parent), attr, linear)
        linears[suffix] = linear
        weights[suffix] = w.clone()
        plain[suffix] = nvfp4_qdq(w)
        s = state['layers']['blocks.0.' + suffix]
        smooth, a, b = [s[k].to(w) for k in ('smooth', 'final_a', 'final_b')]
        quant[suffix] = (nvfp4_qdq(w*smooth-b@a), smooth, a, b)
    del state, old, w, bias, s, smooth, a, b
    fitted = {}
    def fixed_hashes():
        return {suffix: [tensor_sha(t) for t in quant[suffix][:3]] for suffix in TARGET_SUFFIXES}
    report['fixed_qweight_smooth_A_before'] = fixed_hashes()
    def restore():
        for suffix, linear in linears.items():
            linear.weight.copy_(weights[suffix])

    @contextmanager
    def configured(mode):
        runtimes = []
        try:
            for suffix, linear in linears.items():
                if mode == 'plain_w4a4':
                    linear.weight.copy_(plain[suffix])
                    runtimes.append(install_runtime_hooks(linear, None, None, None))
                else:
                    q, smooth, a, b = quant[suffix]
                    if suffix == 'mlp.fc2' and mode in fitted:
                        b = fitted[mode]
                    linear.weight.copy_(q)
                    runtimes.append(install_runtime_hooks(linear, smooth, a, b))
            yield runtimes
        finally:
            for runtime in runtimes:
                runtime.remove()
            restore()

    def prepare(pid, step):
        sample = torch.load(args.cache/f'p{pid}'/f'sample_p{pid}_s{step:02d}.pt', map_location='cpu', weights_only=False)
        if sample['input_kwargs'].get('control_hints') is not None:
            raise RuntimeError('control hints unsupported')
        restore()
        hidden, kw = raw_call_to_block0(pipe.dit, sample)
        ref = block(hidden.cuda(), **tree_device(kw, 'cuda')).cpu()
        replay = block(hidden.cuda(), **tree_device(kw, 'cuda')).cpu()
        if not torch.equal(ref, replay):
            raise RuntimeError('BF16 block replay mismatch')
        idx, desc = partitions(sample, ref.shape[0])
        return dict(prompt_id=pid, step=step, sample=sample, hidden=hidden, kwargs=kw,
                    ref=ref, indices=idx, partitions=desc)

    def run_block(case, mode, features=False):
        cap = {}
        with configured(mode) as runtimes:
            handles = []
            if features:
                handles.append(runtimes[-1].branch.a.register_forward_hook(
                    lambda _m, _inp, out: cap.__setitem__('phi', out.detach().cpu())))
                handles.append(block.adaln_proj.register_forward_hook(
                    lambda _m, _inp, out: cap.__setitem__('gates', out[-1].detach().cpu())))
            try:
                out = block(case['hidden'].cuda(), **tree_device(case['kwargs'], 'cuda')).cpu()
            finally:
                for handle in handles:
                    handle.remove()
            if any(runtime.act.calls != 1 for runtime in runtimes):
                raise RuntimeError('quantizer hook count mismatch')
        metrics = {'all': summarize(out, case['ref']),
                   **{m: summarize(out, case['ref'], case['indices'][m]) for m in MODALITIES}}
        return out, metrics, cap

    b0 = quant['mlp.fc2'][3]
    out_features, rank = b0.shape
    stats = {m: dict(gram=torch.zeros(out_features, rank, rank, device='cuda', dtype=torch.float64),
                     rhs=torch.zeros(out_features, rank, device='cuda', dtype=torch.float64),
                     ref2=0., err2=0.) for m in MODALITIES}
    train_cases = []
    for pid, step in TRAIN:
        checkpoint()
        print(f'train sufficient statistics p{pid} step{step}', flush=True)
        case = prepare(pid, step)
        base, metrics, cap = run_block(case, 'corrected_svdquant', features=True)
        case['baseline_metrics'] = metrics
        phi, gates = cap['phi'], cap['gates']
        combined = case['kwargs']['combined_indices'].long().cpu()
        for m in MODALITIES:
            stats[m]['ref2'] += metrics[m]['ref2']
            stats[m]['err2'] += metrics[m]['err2']
            indices = case['indices'][m]
            for group in combined[indices].unique().tolist():
                selected = indices[combined[indices] == group]
                cov = torch.zeros(rank, rank, dtype=torch.float64, device='cuda')
                cross = torch.zeros(rank, out_features, dtype=torch.float64, device='cuda')
                for begin in range(0, selected.numel(), 1024):
                    ix = selected[begin:begin+1024]
                    ph = phi[ix].cuda().double()
                    target = (case['ref'][ix].float() - base[ix].float()).cuda().double()
                    cov += ph.T @ ph
                    cross += ph.T @ target
                gate = gates[group].cuda().double()
                stats[m]['gram'] += gate[:, None, None].square() * cov[None]
                stats[m]['rhs'] += gate[:, None] * cross.T
        train_cases.append(case)
        del base, phi, gates, cap, cov, cross, ph, target, gate
    report['fit'] = {}
    for mode in REFITS:
        fitted[mode], report['fit'][mode] = solve_refit(mode, stats, b0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        artifact = args.output.parent/f'E004_{mode}_fc2_B.pt'
        torch.save({'B': fitted[mode].cpu(), 'train': TRAIN, 'mode': mode, 'ridge_rule': RIDGE,
                    'fixed_A_sha256': tensor_sha(quant['mlp.fc2'][2])}, artifact)
        report['fit'][mode]['artifact'] = str(artifact)
    report['train'] = []
    for case in train_cases:
        rec = dict(prompt_id=case['prompt_id'], step=case['step'], metrics={'corrected_svdquant': case['baseline_metrics']})
        for mode in REFITS:
            _, rec['metrics'][mode], _ = run_block(case, mode)
        report['train'].append(rec)
        checkpoint()
    for mode in REFITS:
        weights_m = loss_weights(mode, stats)
        actual = sum(weights_m[m] * sum(rec['metrics'][mode][m]['err2'] for rec in report['train']) for m in MODALITIES)
        info = report['fit'][mode]
        info['actual_training_objective'] = actual
        info['actual_over_original'] = actual / info['predicted_before']
        info['training_valid'] = info['actual_over_original'] <= 1.02
    del stats, train_cases, case
    gc.collect(); torch.cuda.empty_cache()
    modes = ('plain_w4a4', 'corrected_svdquant', *REFITS)
    eval_cases = []
    report['eval_block'] = []
    for pid, step in EVAL:
        print(f'isolated block evaluation p{pid} step{step}', flush=True)
        case = prepare(pid, step)
        rec = dict(prompt_id=pid, step=step, partitions=case['partitions'], metrics={})
        case['donors'] = {}
        for mode in modes:
            case['donors'][mode], rec['metrics'][mode], _ = run_block(case, mode)
        eval_cases.append(case)
        report['eval_block'].append(rec)
        checkpoint()
    report['block_gate'] = {}
    for mode in REFITS:
        rows = report['eval_block']
        ratio = sum(r['metrics'][mode]['video']['err2'] for r in rows) / sum(r['metrics']['corrected_svdquant']['video']['err2'] for r in rows)
        improved = sum(r['metrics'][mode]['video']['err2'] < r['metrics']['corrected_svdquant']['video']['err2'] for r in rows)
        report['block_gate'][mode] = dict(video_ratio=ratio, improved_cases=improved,
            passed=ratio <= .9 and improved >= 3 and report['fit'][mode]['training_valid'])
    any_training_gain = any(report['fit'][m]['actual_over_original'] <= .98 for m in REFITS)
    report['any_training_gain_at_least_2pct'] = any_training_gain
    any_gate = any_training_gain and any(report['block_gate'][m]['passed'] for m in ('normalized_refit', 'video4_refit'))
    full_cases = eval_cases if any_gate else eval_cases[:2]
    report['full_case_policy'] = '4 preregistered eval cases' if any_gate else 'block gate failed; only p11 step0/19 negative checks'
    report['full'] = []
    for case in full_cases:
        checkpoint()
        call = tree_device(case['sample'], 'cuda')
        restore()
        calls_before = sdpa['calls']
        teacher = tree_cpu(pipe.dit(*call['input_args'], **call['input_kwargs']))
        count = sdpa['calls'] - calls_before
        if count < 50 or not isinstance(teacher, tuple) or len(teacher) != 2:
            raise RuntimeError('teacher path not exercised')
        rec = dict(prompt_id=case['prompt_id'], step=case['step'], teacher_sdpa_calls=count,
                   teacher_sha256=[tensor_sha(t) for t in teacher], arms=[])
        report['full'].append(rec)
        for mode in ('bf16_zero_pulse', *modes):
            donor = case['ref'] if mode == 'bf16_zero_pulse' else case['donors'][mode]
            candidate, reference = donor.cuda(), case['ref'].cuda()
            calls = [0]
            def inject(_module, _inp, out):
                calls[0] += 1
                if not torch.equal(out, reference):
                    raise RuntimeError('block0 BF16 reference changed')
                return candidate.clone()
            handle = block.register_forward_hook(inject)
            before = sdpa['calls']
            try:
                output = tree_cpu(pipe.dit(*call['input_args'], **call['input_kwargs']))
            finally:
                handle.remove()
            row = dict(mode=mode, video=summarize(output[0], teacher[0]), audio=summarize(output[1], teacher[1]),
                       sdpa_calls=sdpa['calls'] - before)
            if calls != [1] or row['sdpa_calls'] != count:
                raise RuntimeError('full arm hook/attention count mismatch')
            if mode == 'bf16_zero_pulse' and (row['video']['err2'] != 0 or row['audio']['err2'] != 0):
                raise RuntimeError('full BF16 zero replay mismatch')
            rec['arms'].append(row)
            print(json.dumps({'prompt':case['prompt_id'], 'step':case['step'], **row}), flush=True)
            checkpoint()
            del output, candidate, reference
            gc.collect(); torch.cuda.empty_cache()
        del teacher, call
    report['fixed_qweight_smooth_A_after'] = fixed_hashes()
    if report['fixed_qweight_smooth_A_before'] != report['fixed_qweight_smooth_A_after']:
        raise RuntimeError('fixed quantization parameters changed')
    report['qkv_dtypes'] = [list(d) for d in sorted(sdpa['dtypes'])]
    report['peak_gpu_gib'] = torch.cuda.max_memory_allocated()/1024**3
    report['status'] = 'complete'
    checkpoint()
    torch.nn.functional.scaled_dot_product_attention = sdpa_original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--cache', type=Path, default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s')
    parser.add_argument('--state', type=Path, default=ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E004_h3_modality_reweight.json')
    parser.add_argument('--max-seconds', type=float, default=7200.)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    report = dict(experiment='E004', status='partial', train=TRAIN, eval=EVAL, ridge=RIDGE,
                  native_fp4=False, train_eval_scope='original PTQ calibration set; isolated only for this refit',
                  limitation='Only fc2 B refit in fixed A. Negative result cannot establish mechanism novelty or disprove general reweighting.',
                  modes=list(REFITS), self_test=self_test())
    try:
        execute(args, report)
    except Exception:
        report['status'] = 'failed'; report['error'] = traceback.format_exc()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
