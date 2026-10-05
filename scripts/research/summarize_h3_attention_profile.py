#!/usr/bin/env python3
"""CPU-only, exclusive attribution of the completed E016 Chrome traces.

This is an independent profiler pass, not a reconstruction of steady latency.
No model, torch, frozen runner, or adapter is imported.
"""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E016'
ARMS = ('svd_bf16', 'svd_block_mean', 'svd_global_mean')
PREPROCESS_PATTERN = (
    'aten::mean', 'aten::sub', 'aten::fill_', 'aten::copy_',
    'aten::fill_', 'aten::copy_', 'aten::fill_', 'aten::copy_',
    'aten::mean', 'aten::sub', 'aten::copy_', 'aten::copy_', 'aten::bmm')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    return dict(file=str(path), sha256=digest(path), bytes=path.stat().st_size)


def aggregate(rows):
    groups = collections.defaultdict(lambda: dict(calls=0, gpu_ms=0.0))
    for e in rows:
        g = groups[e['name']]
        g['calls'] += 1
        g['gpu_ms'] += e['dur'] / 1000
    return [dict(name=k, **v) for k, v in sorted(groups.items(), key=lambda p: -p[1]['gpu_ms'])]


def parse(arm):
    report_path = RD / f'E016_bench_{arm}.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' and report['complete_dit_calls'] == 5
    trace_rec = report['profile']['trace']
    assert digest(trace_rec['file']) == trace_rec['sha256']
    events = json.loads(Path(trace_rec['file']).read_text())['traceEvents']
    kernels = sorted((e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X'), key=lambda e: e['ts'])
    cpu = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X']
    ops = {e['args']['External id']: e for e in cpu}
    ancestry = {}
    threads = collections.defaultdict(list)
    for e in cpu:
        threads[e['pid'], e['tid']].append(e)
    for thread in threads.values():
        stack = []
        for e in sorted(thread, key=lambda e: (e['ts'], -e['dur'])):
            while stack and e['ts'] >= stack[-1]['ts'] + stack[-1]['dur']:
                stack.pop()
            ancestry[e['args']['External id']] = tuple(p['name'] for p in stack) + (e['name'],)
            stack.append(e)
    categories = [None] * len(kernels)

    def assign(index, category):
        assert categories[index] is None, (index, categories[index], category)
        categories[index] = category

    def op(index):
        return ops.get(kernels[index].get('args', {}).get('External id'), {}).get('name')

    low = arm != 'svd_bf16'
    counts = collections.Counter(e['name'] for e in cpu)
    expected_flags = 252 if low else 52
    assert counts['aten::isfinite'] == counts['aten::all'] == expected_flags
    qpack = [i for i, e in enumerate(kernels) if 'flashinfer::nvfp4_attention_sm120::scaled_fp4_quant_kernel<128u, 128u, false' in e['name']]
    assert len(qpack) == (50 if low else 0)
    if low:
        # All 50 observed regions must match the exact official source sequence.
        # No broad classification of generic copies/reductions by name alone.
        assert counts['aten::mean'] == 100 and counts['aten::bmm'] == 50
        for i in qpack:
            assert tuple(op(j) for j in range(i-13, i)) == PREPROCESS_PATTERN
            for j in range(i-13, i-3):
                assign(j, 'official_mean_center_and_pad')
            for j in range(i-3, i-1):
                assign(j, 'official_correction_fp32_casts')
            assign(i-1, 'official_correction_fp32_matmul')
    for i, e in enumerate(kernels):
        if categories[i] is not None:
            continue
        name = e['name']
        parents = ancestry.get(e.get('args', {}).get('External id'), ())
        if 'aten::isfinite' in parents or op(i) == 'aten::all':
            category = 'diagnostic_finite_checks_only'
        elif 'nvfp4_attention::attention_kernel_ws' in name and 'float_e2m1' in name:
            category = 'native_fp4_attention'
        elif 'flashinfer::nvfp4_attention_sm120::scaled_fp4_quant' in name:
            category = 'official_qkv_fp4_pack'
        elif 'pytorch_flash::flash_fwd_kernel' in name:
            category = 'bf16_flash_attention'
        elif all(s in name.lower() for s in ('sm120', 'e2m1', 'gemm')):
            category = 'native_fp4_linear'
        elif name in ('_amax_parts', '_global_scale', '_encode_groups'):
            category = 'h3_linear_activation_pack'
        else:
            category = 'other_model_and_routing'
        assign(i, category)
    groups = collections.defaultdict(list)
    for e, c in zip(kernels, categories):
        groups[c].append(e)
    groups = {k: dict(calls=len(v), gpu_ms=sum(e['dur'] for e in v)/1000,
                      kernels=aggregate(v)) for k, v in groups.items()}
    total = sum(e['dur'] for e in kernels)/1000
    assert abs(sum(v['gpu_ms'] for v in groups.values()) - total) < 1e-7
    assert groups['native_fp4_linear']['calls'] == 200
    assert groups['h3_linear_activation_pack']['calls'] == 600
    assert groups['bf16_flash_attention']['calls'] == (52 if low else 102)
    assert groups.get('native_fp4_attention', {}).get('calls', 0) == (50 if low else 0)
    assert groups.get('official_qkv_fp4_pack', {}).get('calls', 0) == (150 if low else 0)
    assert groups['diagnostic_finite_checks_only']['calls'] == 5*expected_flags
    for v in groups.values():
        v['fraction_of_profile_kernel_sum'] = v['gpu_ms']/total
    transfers = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset') and e.get('ph') == 'X']
    runtime = [e for e in events if e.get('cat') == 'cuda_runtime' and e.get('ph') == 'X']
    result = dict(report=record(report_path), trace=trace_rec,
        unprofiled_latency_ms=report['latency_ms'], steady_memory=report['steady_memory'],
        resident_model_storage=report['resident_model_storage'],
        independent_profile=dict(kernel_count=len(kernels), kernel_sum_ms=total, categories=groups,
            diagnostic_flag_count=expected_flags, kernel_streams=sorted({e['args']['stream'] for e in kernels}),
            official_preprocess_source_pattern_matches=len(qpack),
            gpu_transfers_and_memset=dict(calls=len(transfers), device_ms=sum(e['dur'] for e in transfers)/1000,
                bytes=sum(e.get('args', {}).get('bytes', 0) for e in transfers)),
            cuda_runtime_cpu=dict(calls=len(runtime), cpu_duration_sum_ms=sum(e['dur'] for e in runtime)/1000)))
    return result


def main():
    out = RD / 'E016_profile_attribution.json'
    md = RD / 'E016_profile_attribution.md'
    assert not out.exists() and not md.exists(), 'Refusing to overwrite completed attribution'
    rows = {a: parse(a) for a in ARMS}
    baseline = rows['svd_bf16']['unprofiled_latency_ms']['host_ms_including_checks']['median']
    for row in rows.values():
        row['steady_host_speedup_vs_svd_bf16'] = baseline / row['unprofiled_latency_ms']['host_ms_including_checks']['median']
    scope = [
        'Only cat=kernel, ph=X durations enter the mutually exclusive kernel sum; no CPU/GPU annotation double count.',
        'Each independent profile includes diagnostics (52/252 flags), unlike all steady repeats. The profile sum is not steady latency, critical path, or predicted speedup.',
        'Generic preprocessing ops are attributed only after all 50 contiguous 13-kernel sequences match the official source via External id CPU links. Other layout/copy/model/LR work remains other.',
        'FP32 correction GEMM/GEMV and casts are separated from mean/center/pad and official Q/K/V packing. Both modes center K and Q; they are complete official recipes, not a one-factor causal comparison.',
        'Memory is actual post-warmup three-repeat allocated/reserved peak and measured resident storage. No full-model memory estimate from single correction tensors.',
        'CUDA runtime CPU duration may include host synchronization waits; GPU transfers are separate, not an explanation by pure bandwidth. CPU runtime sums are not added to GPU sums.',
    ]
    result = dict(status='complete', experiment='E016', source=record(__file__), scope=scope, arms=rows)
    out.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# E016 profiler attribution', '',
        'All three completed reports and Chrome trace hashes were checked. No GPU was launched for this analysis.', '',
        '| Attention arm | Steady host median [min, max], ms, including checks | Speedup | Measured peak allocated / reserved, GiB |',
        '|---|---:|---:|---:|']
    for arm, row in rows.items():
        t = row['unprofiled_latency_ms']['host_ms_including_checks']; m = row['steady_memory']
        lines.append(f'| {arm} | {t["median"]:.3f} [{t["min"]:.3f}, {t["max"]:.3f}] | {row["steady_host_speedup_vs_svd_bf16"]:.4f}× | {m["peak_allocated_bytes"]/2**30:.5f} / {m["peak_reserved_bytes"]/2**30:.5f} |')
    lines += ['', 'The next table is **exclusive GPU kernel duration from one separate diagnostic profile**, not steady execution time. Finite checks are absent in the timed repeats.', '',
              '| Profile category, ms | BF16 | Block mean | Global mean |', '|---|---:|---:|---:|']
    cats = ['bf16_flash_attention', 'native_fp4_attention', 'official_mean_center_and_pad',
            'official_correction_fp32_casts', 'official_correction_fp32_matmul', 'official_qkv_fp4_pack',
            'native_fp4_linear', 'h3_linear_activation_pack', 'diagnostic_finite_checks_only', 'other_model_and_routing']
    for c in cats:
        v = [rows[a]['independent_profile']['categories'].get(c, {}).get('gpu_ms', 0) for a in ARMS]
        lines.append('| '+c+' | '+' | '.join(f'{n:.3f}' for n in v)+' |')
    lines += ['| Total kernel sum | '+' | '.join(f'{rows[a]["independent_profile"]["kernel_sum_ms"]:.3f}' for a in ARMS)+' |', '',
        'Each low-precision profile contains exactly 50 native attention kernels, 150 official Q/K/V pack kernels, 50 correction matmul kernels, 200 native linear GEMMs, and 600 linear activation-pack kernels. BF16 attention counts are 52 for the low-precision arms versus 102 for the BF16 arm.', '',
        'Correction uses `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` for block mean and `gemv2T_kernel_val` for global mean. Native attention is `nvfp4_attention::attention_kernel_ws<…float_e2m1…>`; Q/K packing uses `scaled_fp4_quant_kernel<…, false/true,…>` and V uses `scaled_fp4_quant_trans_kernel`.', '',
        *['- '+s for s in scope], '',
        'Source-linked per-category kernel names/counts and actual memory/timing records are in `E016_profile_attribution.json`. No quality or new-method claim follows from this profile.', '']
    md.write_text('\n'.join(lines))
    print(out)
    print(md)


if __name__ == '__main__':
    main()
