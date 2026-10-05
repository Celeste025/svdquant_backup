#!/usr/bin/env python3
"""E006: fixed 12-case H3 projection x attention factorial, local diagnostics only.

All inputs are original PTQ calibration calls. Full BF16 upstream, native NVFP4
SVDQuant qkv only, original QK norm/RoPE, real sequence boundaries. No rollout,
weight search, quality claim, or performance claim.
"""
from __future__ import annotations
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = (0, 24, 48)
CALLS = ((1, 0), (1, 19), (20, 0), (20, 19))
ARMS = ('O00', 'O10', 'O01', 'O11', 'O1F')


def factorial(outputs, indices=None, chunk_rows=128):
    """Form FP32 interaction tensors before reducing energies in FP64.

    I = O11-O10-O01+O00; NMSE differences are never used as interactions.
    Chunking limits memory, without changing the stated per-element algebra.
    """
    if indices is None:
        indices = torch.arange(outputs['O00'].shape[0])
    totals = {k: 0. for k in ('reference', 'projection', 'attention', 'joint',
        'additive', 'interaction', 'projection_attention_cross',
        'additive_interaction_cross', 'fixed_joint', 'fixed_interaction',
        'dynamic_minus_fixed', 'fixed_dynamic_cross')}
    count = 0
    for begin in range(0, indices.numel(), chunk_rows):
        ix = indices[begin:begin + chunk_rows]
        o00, o10, o01, o11, o1f = [outputs[k][ix].float() for k in ARMS]
        p, a, j = o10-o00, o01-o00, o11-o00
        s, i = p+a, o11-o10-o01+o00
        f, fi, d = o1f-o00, o1f-o10-o01+o00, o11-o1f
        for name, value in [('reference', o00), ('projection', p), ('attention', a),
                ('joint', j), ('additive', s), ('interaction', i),
                ('fixed_joint', f), ('fixed_interaction', fi), ('dynamic_minus_fixed', d)]:
            totals[name] += value.double().square().sum().item()
        totals['projection_attention_cross'] += 2*(p.double()*a.double()).sum().item()
        totals['additive_interaction_cross'] += 2*(s.double()*i.double()).sum().item()
        totals['fixed_dynamic_cross'] += 2*(f.double()*d.double()).sum().item()
        count += o00.numel()
    e = totals
    def ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else None
    joint_residual = e['joint']-(e['projection']+e['attention']+e['projection_attention_cross']+
                                  e['interaction']+e['additive_interaction_cross'])
    fixed_residual = e['joint']-(e['fixed_joint']+e['dynamic_minus_fixed']+e['fixed_dynamic_cross'])
    identity_scale = max(sum(abs(v) for v in e.values()), 1e-30)
    if max(abs(joint_residual), abs(fixed_residual))/identity_scale > 1e-5:
        raise RuntimeError('factorial energy identity failed beyond FP32 tensor-rounding tolerance')
    return {'tokens': indices.numel(), 'numel': count, 'energy': totals,
            'joint_nmse': ratio(e['joint'], e['reference']),
            'projection_nmse': ratio(e['projection'], e['reference']),
            'attention_nmse': ratio(e['attention'], e['reference']),
            'interaction_over_joint': ratio(e['interaction'], e['joint']),
            'interaction_over_individual': ratio(e['interaction'], e['projection']+e['attention']),
            'fixed_joint_nmse': ratio(e['fixed_joint'], e['reference']),
            'fixed_interaction_over_joint': ratio(e['fixed_interaction'], e['fixed_joint']),
            'joint_identity_residual': joint_residual, 'fixed_identity_residual': fixed_residual}


def scopes(indices, tokens):
    valid = torch.cat([indices[m] for m in ('video', 'audio', 'text')]).sort().values
    return {'primary_valid_tokens': valid, **indices, 'all_including_padding': torch.arange(tokens)}


def self_test():
    # Cancellation: additive errors can have large energy but zero joint error.
    ref = torch.ones(3, 2, 2)
    p = torch.full_like(ref, 2)
    a = -p
    interaction = torch.full_like(ref, .5)
    outputs = dict(O00=ref, O10=ref+p, O01=ref+a,
                   O11=ref+p+a+interaction, O1F=ref+p+a)
    result = factorial(outputs, torch.tensor([0, 1]), chunk_rows=1)
    assert result['energy']['interaction'] == 2.
    assert result['energy']['projection_attention_cross'] == -64.
    assert result['interaction_over_joint'] == 1.
    assert result['joint_identity_residual'] == result['fixed_identity_residual'] == 0.
    idx = {'video': torch.tensor([0]), 'audio': torch.tensor([1]),
           'text': torch.tensor([], dtype=torch.long), 'pad_or_unassigned': torch.tensor([2])}
    outputs['O11'][2] = 10000  # Must not affect the primary statistic.
    primary = scopes(idx, 3)['primary_valid_tokens']
    assert factorial(outputs, primary) == result
    return {'passed': True, 'device': 'cpu', 'checks': [
        'explicit tensor interaction and cross terms', 'energy identities under cancellation',
        'primary valid-token scope excludes adversarial padding'],
        'boundary': 'metric algebra only; no model, CUDA, or native packing validated'}


@torch.inference_mode()
def execute(args, report):
    sys.path.insert(0, str(ROOT/'scripts'))
    from minimax_h3_svdquant_common import H3_DIT_PATH, load_h3_pipeline, tree_cpu, tree_device
    from probe_h3_modality_pulse import sha256, tensor_sha, partitions, summarize, save
    from probe_h3_native_contract import quantize_pack, native
    from nvfp4_attention_fixed_scale import pack_with_scales, compare_packs, logical_scales, k_permutation
    flash = importlib.import_module('flashinfer.nvfp4_attention_sm120')
    comfy = importlib.import_module('diffsynth.models.minimax_h3_dit_comfy')
    attention = importlib.import_module('diffsynth.core.attention.attention')
    if attention.ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('E006 requires explicit torch BF16 upstream attention')
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    start_time = time.time()
    def checkpoint():
        report['elapsed_seconds_not_benchmark'] = time.time()-start_time
        report['peak_gpu_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        if report['elapsed_seconds_not_benchmark'] > args.max_seconds:
            raise TimeoutError('E006 fixed wall-clock budget exceeded')

    sources = [Path(__file__), ROOT/'scripts/minimax_h3_svdquant_common.py',
        ROOT/'scripts/research/probe_h3_native_contract.py',
        ROOT/'scripts/research/probe_h3_modality_pulse.py',
        ROOT/'scripts/research/nvfp4_attention_fixed_scale.py',
        ROOT/'research_state/06_experiments/E006_attention_interface_plan.md',
        Path(importlib.import_module('diffsynth.models.minimax_h3_dit').__file__),
        Path(comfy.__file__), Path(attention.__file__), Path(flash.__file__), H3_DIT_PATH, args.state]
    sources += [args.cache/f'p{p}'/f'sample_p{p}_s{s:02d}.pt' for p, s in CALLS]
    print('Hashing immutable model, state, code, and four raw calibration calls', flush=True)
    report['files'] = {str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sources}
    report.update(torch=torch.__version__, torch_file=torch.__file__, cuda=torch.version.cuda,
        flashinfer=importlib.import_module('flashinfer').__version__, device=torch.cuda.get_device_name(),
        attention_implementation=attention.ATTENTION_IMPLEMENTATION,
        environment={k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME',
            'FLASHINFER_WORKSPACE_BASE', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'MINIMAX_H3_DIT_PATH']})
    state = torch.load(args.state, map_location='cpu', weights_only=False)
    report['state_config'] = state['config']
    original_sdpa, original_helper = F.scaled_dot_product_attention, comfy._sdpa_varlen_attention
    sdpa = {'calls': 0, 'qkv_dtypes': set()}
    def audited_sdpa(q, k, v, *pos, **kw):
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            raise RuntimeError('BF16 torch SDPA contract violated')
        sdpa['calls'] += 1
        sdpa['qkv_dtypes'].add(tuple(str(t.dtype) for t in (q, k, v)))
        return original_sdpa(q, k, v, *pos, **kw)
    F.scaled_dot_product_attention = audited_sdpa
    profiles = set()

    def profile_once(name, fn):
        if name in profiles:
            return fn()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            result = fn()
            torch.cuda.synchronize()
        kernels = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
        report.setdefault('native_profile_kernel_names', {})[name] = kernels
        expected = (lambda n: 'sm120' in n and 'e2m1' in n) if name == 'qkv_projection' else (
            lambda n: 'attention_kernel_ws' in n and 'float_e2m1' in n)
        if not any(expected(k) for k in kernels):
            raise RuntimeError(f'no native FP4 kernel evidence for {name}: {kernels}')
        profiles.add(name)
        return result

    def flash_forward(packed, n, scale):
        result = profile_once('attention', lambda: flash.nvfp4_attention_sm120_fwd(
            *packed, sm_scale=scale, causal=False, per_block_mean=True,
            out_dtype=torch.bfloat16, return_lse=False, unpadded_k_len=n))
        result = result[:, :, :n, :]
        if not torch.isfinite(result).all():
            raise RuntimeError('nonfinite native attention output')
        return result

    pipe = None
    try:
        pipe = load_h3_pipeline(full=False, vram_limit_gib=args.vram_limit_gib)
        pipe.load_models_to_device(['dit'])
        pipe.dit.eval()
        if len(pipe.dit.blocks) != 50:
            raise RuntimeError('expected exactly 50 H3 main blocks')
        checkpoint()
        for pid, step in CALLS:
            path = args.cache/f'p{pid}'/f'sample_p{pid}_s{step:02d}.pt'
            sample = torch.load(path, map_location='cpu', weights_only=False)
            if sample['input_kwargs'].get('control_hints') is not None:
                raise RuntimeError('control hints unsupported by isolated block contract')
            captured, handles = {}, []
            for bi in BLOCKS:
                captured[bi] = {}
                def pre(_module, inp, kw, bi=bi):
                    captured[bi]['hidden'] = inp[0].detach().cpu()
                    captured[bi]['kwargs'] = tree_cpu(kw)
                def post(_module, inp, out, bi=bi):
                    captured[bi]['endpoint'] = out.detach().cpu()
                handles.extend([pipe.dit.blocks[bi].register_forward_pre_hook(pre, with_kwargs=True),
                                pipe.dit.blocks[bi].register_forward_hook(post)])
            call = tree_device(sample, 'cuda')
            before = sdpa['calls']
            print(f'p{pid} step{step}: full BF16 teacher', flush=True)
            try:
                teacher = tree_cpu(pipe.dit(*call['input_args'], **call['input_kwargs']))
            finally:
                for h in handles:
                    h.remove()
            if not isinstance(teacher, tuple) or len(teacher) != 2 or sdpa['calls']-before < 50:
                raise RuntimeError('full BF16 teacher not verified')
            if any(not torch.isfinite(t).all() for t in teacher):
                raise RuntimeError('nonfinite BF16 teacher output')
            teacher_info = {'prompt_id': pid, 'step': step, 'path': str(path),
                'sdpa_calls': sdpa['calls']-before,
                'video_audio_output_shapes': [list(t.shape) for t in teacher],
                'video_audio_output_sha256': [tensor_sha(t) for t in teacher]}
            report['teachers'].append(teacher_info)
            del teacher, call
            for bi in BLOCKS:
                case_start = time.time()
                capture = captured.pop(bi)
                block = pipe.dit.blocks[bi]
                hidden = capture['hidden'].cuda()
                kw = tree_device(capture['kwargs'], 'cuda')
                idx, partdesc = partitions(sample, hidden.shape[0])
                scope = scopes(idx, hidden.shape[0])
                bounds = kw['cu_seqlens'].tolist()
                if bounds[0] != 0 or bounds[-1] != hidden.shape[0] or any(b < a for a, b in zip(bounds, bounds[1:])):
                    raise RuntimeError('invalid actual cu_seqlens')
                case = {'prompt_id': pid, 'step': step, 'block': bi, 'status': 'partial',
                    'hidden_shape': list(hidden.shape), 'partitions': partdesc, 'cu_seqlens': bounds,
                    'teacher_hidden_sha256': tensor_sha(capture['hidden']),
                    'teacher_endpoint_sha256': tensor_sha(capture['endpoint']),
                    'segments': [], 'replay_audit': {}, 'projection': {}}
                report['cases'].append(case)
                checkpoint()
                print(f'p{pid} step{step} block{bi}: five-arm local diagnostic', flush=True)
                qkv_module = block.attn.qkv_proj
                w, bias = qkv_module.load_from_disk(torch.bfloat16, 'cuda', assign=False)
                if bias is not None:
                    raise RuntimeError('expected H3 bias-free qkv projection')
                layer = state['layers'][f'blocks.{bi}.attn.qkv_proj']
                smooth, a, b = [layer[k].to(w) for k in ('smooth', 'final_a', 'final_b')]
                if a.shape[0] != 32 or b.shape[1] != 32:
                    raise RuntimeError('expected fixed original rank32 state')
                residual_w = w*smooth-b@a
                qw = quantize_pack(residual_w, args.chunk_rows)
                case['projection']['weight_quant_stats'] = qw['stats']
                case['projection']['recipe'] = 'native RNE residual W4A4 + BF16 low-rank branch on unquantized smoothed input'
                case['projection']['other_three_linears'] = 'unchanged BF16 disk-backed out_proj/fc1/fc2'
                del residual_w, w
                qkv_input, native_output = None, None
                qkv_pairs, attention_outputs, endpoints = {}, {}, {}
                current = {'arm': None, 'helper_calls': 0, 'qkv_hook_calls': 0}

                def qkv_hook(_module, inp, original_output):
                    nonlocal qkv_input, native_output
                    current['qkv_hook_calls'] += 1
                    x = inp[0]
                    if qkv_input is None:
                        qkv_input = x.detach().clone()
                    elif not torch.equal(x, qkv_input):
                        raise RuntimeError('qkv input changed between paired arms')
                    if native_output is None:
                        xs = x/smooth
                        qa = quantize_pack(xs, args.chunk_rows)
                        lowrank = F.linear(F.linear(xs, a), b)
                        main = profile_once('qkv_projection', lambda: native(qa, qw, 'rne'))
                        native_output = main+lowrank
                        if not torch.isfinite(native_output).all():
                            raise RuntimeError('nonfinite native qkv output')
                        case['projection']['activation_quant_stats'] = qa['stats']
                        case['projection']['native_vs_bf16_qkv'] = summarize(native_output, original_output)
                        case['projection']['native_output_sha256'] = tensor_sha(native_output)
                    return native_output

                def helper(q, k, v, cu_seqlens, softmax_scale):
                    current['helper_calls'] += 1
                    arm = current['arm']
                    if cu_seqlens.tolist() != bounds:
                        raise RuntimeError('sequence boundaries changed between arms')
                    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
                        raise RuntimeError('expected BF16 QKV after original norm/RoPE')
                    if any(not torch.isfinite(t).all() for t in (q, k, v)):
                        raise RuntimeError('nonfinite actual norm/RoPE QKV')
                    if arm in ('O00', 'O10'):
                        pair = 'T0' if arm == 'O00' else 'T1'
                        qkv_pairs[pair] = tuple(t.detach().cpu() for t in (q, k, v))
                        case['softmax_scale'] = float(softmax_scale)
                        before_sdpa = sdpa['calls']
                        out = original_helper(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=softmax_scale)
                        if not torch.isfinite(out).all():
                            raise RuntimeError('nonfinite torch attention output')
                        expected_calls = sum(stop > begin for begin, stop in zip(bounds, bounds[1:]))
                        if sdpa['calls']-before_sdpa != expected_calls:
                            raise RuntimeError('actual torch segment SDPA calls not verified')
                        attention_outputs[arm] = out.detach().cpu()
                        return out
                    pair = 'T0' if arm == 'O01' else 'T1'
                    if any(not torch.equal(t.cpu(), ref) for t, ref in zip((q, k, v), qkv_pairs[pair])):
                        raise RuntimeError('replay QKV changed from captured norm/RoPE tensors')
                    return attention_outputs[arm].to(q.device)

                def replay(arm):
                    current.update(arm=arm, helper_calls=0, qkv_hook_calls=0)
                    h = qkv_module.register_forward_hook(qkv_hook) if arm in ('O10', 'O11', 'O1F') else None
                    comfy._sdpa_varlen_attention = helper
                    before_sdpa = sdpa['calls']
                    try:
                        endpoint = block(hidden, **kw).cpu()
                    finally:
                        comfy._sdpa_varlen_attention = original_helper
                        if h is not None:
                            h.remove()
                    if current['helper_calls'] != 1 or current['qkv_hook_calls'] != int(h is not None):
                        raise RuntimeError('unexpected block helper/projection hook call count')
                    if not torch.isfinite(endpoint).all():
                        raise RuntimeError('nonfinite isolated block endpoint')
                    endpoints[arm] = endpoint
                    case['replay_audit'][arm] = {'helper_calls': current['helper_calls'],
                        'native_projection_hook_calls': current['qkv_hook_calls'],
                        'actual_torch_sdpa_calls': sdpa['calls']-before_sdpa,
                        'native_attention_precomputed_from_identical_captured_qkv': arm in ('O01', 'O11', 'O1F')}

                replay('O00')
                case['bf16_zero_replay'] = summarize(endpoints['O00'], capture['endpoint'])
                if case['bf16_zero_replay']['err2'] != 0:
                    raise RuntimeError('BF16 block replay differs from full teacher')
                replay('O10')
                for arm in ('O01', 'O11', 'O1F'):
                    attention_outputs[arm] = torch.empty_like(attention_outputs['O00'])
                case['qkv_after_norm_rope'] = {pair: {'shapes': [list(t.shape) for t in ts],
                    'sha256': [tensor_sha(t) for t in ts]} for pair, ts in qkv_pairs.items()}
                for segment, (begin, end) in enumerate(zip(bounds, bounds[1:])):
                    if end == begin:
                        continue
                    n = end-begin
                    segment_counts = {name: int(((ix >= begin)&(ix < end)).sum()) for name, ix in idx.items()}
                    valid_count = sum(segment_counts[m] for m in ('video', 'audio', 'text'))
                    diagnostic = {'segment': segment, 'start': begin, 'end': end, 'unpadded_k_len': n,
                        'tokens_per_modality': segment_counts, 'primary_valid_tokens': valid_count,
                        'primary_eligible': valid_count > 0,
                        'fixed_pack_counts_include_api_padding': True,
                        'clipped_input_energy_definition': 'energy of entire clipped input elements, not squared excess beyond 6*scale',
                        'own_scale_byte_mismatches': {}, 'scale_changes': {}}
                    case['segments'].append(diagnostic)
                    packs = {}
                    segment_qkv = {}
                    for pair in ('T0', 'T1'):
                        segment_qkv[pair] = tuple(t[begin:end].transpose(0, 1).unsqueeze(0).cuda().contiguous()
                                                  for t in qkv_pairs[pair])
                        if any(not torch.isfinite(t).all() for t in segment_qkv[pair]):
                            raise RuntimeError('nonfinite QKV before native quantizer')
                        packed = flash.nvfp4_attention_sm120_quantize_qkv(*segment_qkv[pair], per_block_mean=True)
                        own, own_stats = pack_with_scales(*segment_qkv[pair], packed, chunk_rows=args.chunk_rows)
                        mismatch = compare_packs(own, packed)
                        diagnostic['own_scale_byte_mismatches'][pair] = mismatch
                        diagnostic.setdefault('own_scale_pack_stats_including_api_padding', {})[pair] = own_stats
                        if any(mismatch.values()):
                            checkpoint()
                            raise RuntimeError(f'{pair} own-scale byte parity failed in segment {segment}')
                        del own
                        packs[pair] = packed
                        out = flash_forward(packed, n, case['softmax_scale'])
                        arm = 'O01' if pair == 'T0' else 'O11'
                        attention_outputs[arm][begin:end] = out.squeeze(0).transpose(0, 1).cpu()
                        del out
                    frozen, frozen_stats = pack_with_scales(*segment_qkv['T1'], packs['T0'], chunk_rows=args.chunk_rows)
                    diagnostic['fixed_pack_stats_including_api_padding'] = frozen_stats
                    correction_mismatch = int(torch.count_nonzero(frozen[6].contiguous().view(torch.uint8) !=
                                                       packs['T1'][6].contiguous().view(torch.uint8)).item())
                    diagnostic['fixed_correction_vs_T1_byte_mismatches'] = correction_mismatch
                    if correction_mismatch:
                        checkpoint()
                        raise RuntimeError('fixed scale pack changed current T1 correction')
                    for qi, name in enumerate(('q', 'k', 'v')):
                        s0, s1 = [logical_scales(packs[pair][3+qi]).view(torch.uint8) for pair in ('T0', 'T1')]
                        changed = s0 != s1
                        if name == 'q':
                            selected = changed[:, :, :n, :]
                        elif name == 'k':
                            rowmask = k_permutation(s0.shape[-2], s0.device) < n
                            selected = changed[:, :, rowmask, :]
                        else:
                            selected = changed[:, :, :, :(n+15)//16]
                        diagnostic['scale_changes'][name] = {'changed_groups': int(selected.sum()),
                            'groups': selected.numel(), 'fraction': selected.float().mean().item(),
                            'excludes_fully_api_padded_groups': True,
                            'v_last_partial_group_includes_padding': name == 'v' and n % 16 != 0}
                    out = flash_forward(frozen, n, case['softmax_scale'])
                    attention_outputs['O1F'][begin:end] = out.squeeze(0).transpose(0, 1).cpu()
                    del out, frozen, packs, segment_qkv, packed, s0, s1, changed, selected
                    local = {arm: values[begin:end] for arm, values in attention_outputs.items()}
                    seg_idx = {name: ix[(ix >= begin)&(ix < end)]-begin for name, ix in idx.items()}
                    diagnostic['attention_factorial'] = {name: factorial(local, ix)
                        for name, ix in scopes(seg_idx, n).items()}
                    checkpoint()
                for arm in ('O01', 'O11', 'O1F'):
                    replay(arm)
                case['attention_factorial'] = {name: factorial(attention_outputs, ix) for name, ix in scope.items()}
                case['block_endpoint_factorial_secondary'] = {name: factorial(endpoints, ix) for name, ix in scope.items()}
                case['attention_output_sha256'] = {arm: tensor_sha(value) for arm, value in attention_outputs.items()}
                case['block_endpoint_sha256'] = {arm: tensor_sha(value) for arm, value in endpoints.items()}
                case['elapsed_seconds_not_benchmark'] = time.time()-case_start
                case['status'] = 'complete'
                report['sdpa_audit'] = {'calls': sdpa['calls'], 'qkv_dtypes': [list(v) for v in sorted(sdpa['qkv_dtypes'])]}
                checkpoint()
                print(json.dumps({'prompt_id': pid, 'step': step, 'block': bi,
                    'primary': case['attention_factorial']['primary_valid_tokens'],
                    'seconds': case['elapsed_seconds_not_benchmark']}, ensure_ascii=False), flush=True)
                del capture, hidden, kw, smooth, a, b, qw, qkv_input, native_output
                del qkv_pairs, attention_outputs, endpoints
                gc.collect()
                torch.cuda.empty_cache()
            del sample, captured
        if len(report['cases']) != 12 or any(c['status'] != 'complete' for c in report['cases']):
            raise RuntimeError('fixed 12-case grid incomplete')
        # Every case contributes once; no padding segment enters this median.
        import statistics
        primary_rows = [c['attention_factorial']['video'] for c in report['cases']]
        report['primary_summary'] = {'cases': len(primary_rows), 'scope': 'video; no pad_or_unassigned',
            'median_interaction_over_joint': statistics.median(r['interaction_over_joint'] for r in primary_rows),
            'median_interaction_over_individual': statistics.median(r['interaction_over_individual'] for r in primary_rows),
            'median_joint_nmse': statistics.median(r['joint_nmse'] for r in primary_rows),
            'median_fixed_joint_nmse': statistics.median(r['fixed_joint_nmse'] for r in primary_rows)}
        report['status'] = 'complete'
        checkpoint()
    finally:
        F.scaled_dot_product_attention = original_sdpa
        comfy._sdpa_varlen_attention = original_helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--chunk-rows', type=int, default=128)
    p.add_argument('--vram-limit-gib', type=float, default=35.)
    p.add_argument('--max-seconds', type=float, default=3600.)
    p.add_argument('--cache', type=Path, default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s')
    p.add_argument('--state', type=Path, default=ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt')
    p.add_argument('--output', type=Path, default=ROOT/'results/research/E006_h3_attention_interface.json')
    args = p.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    report = {'experiment': 'E006', 'status': 'partial', 'teachers': [], 'cases': [],
        'metric_self_test': self_test(),
        'fixed_grid': {'blocks': list(BLOCKS), 'prompt_steps': [list(c) for c in CALLS]},
        'arms': {'O00': 'BF16 qkv projection / torch BF16 attention',
            'O10': 'native NVFP4 rank32 corrected SVDQuant qkv / torch BF16 attention',
            'O01': 'BF16 qkv projection / FlashInfer SM120 FP4 attention',
            'O11': 'native NVFP4 rank32 corrected SVDQuant qkv / FlashInfer SM120 FP4 attention',
            'O1F': 'T1 QKV + current T1 preprocessing/correction + frozen T0 attention scales'},
        'primary_output': 'attention T x H x D before BF16 out_proj',
        'secondary_output': 'isolated full block endpoint; never used to select cases',
        'interaction': 'FP32 tensor I=O11-O10-O01+O00, FP64 energy/cross-term accumulation',
        'limitations': ['all prompts/steps are original PTQ-calibration data; exploratory, no held-out claim',
            'local isolated block diagnostic; no continuation or free rollout',
            'software preparation with native FP4 kernels; no performance claim',
            'fixed pack clipping/underflow counts include API padding; not effective-token rates',
            'fixed scales are a causal diagnostic, not a proposed deployment method',
            'original global legacy state, no recalibration or rank/weight search']}
    try:
        execute(args, report)
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        for case in report['cases']:
            if case['status'] == 'partial':
                case['status'] = 'failed_no_causal_interpretation'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        raise


if __name__ == '__main__':
    main()
