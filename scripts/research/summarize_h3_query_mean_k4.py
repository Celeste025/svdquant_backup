#!/usr/bin/env python3
"""E026 independent CPU tensor readout; fixed-mu experiment, not an optimum."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E026'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(6)
    report_path = RD / 'run.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' and report['attention_calls'] == 2 and report['complete_dit_calls'] == 0
    assert sha(report['sources']['runner']['file']) == report['sources']['runner']['sha256']
    oldref = report['source_reports']['probe_run.json']
    assert sha(oldref['file']) == oldref['sha256']
    previous = json.loads(Path(oldref['file']).read_text())
    case = next(c for c in previous['cases'] if c['block'] == 0)
    block0 = next(o for o in case['outputs'] if o['mode'] == 'block_mean' and o['shift'] == 0)
    n, heads, dim, padded = 22539, 56, 128, 22656
    checked = []

    def load(ref, expected_tensor=None):
        path = Path(ref['file'])
        assert path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'], path
        value = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        if expected_tensor is not None:
            assert list(value.shape) == expected_tensor['shape'] and str(value.dtype) == expected_tensor['dtype']
            assert tensor_sha(value) == expected_tensor['sha256']
        checked.append(ref)
        return value

    refs = report['inputs']['references']
    baseline = load(refs['bf16']['artifact'], refs['bf16']['tensor'])
    values = {name: load(rec['artifact'], rec['tensor']) for name, rec in refs.items() if name != 'bf16'}
    values.update({name: load(rec['artifact'], rec['tensor']) for name, rec in report['outputs'].items()})
    assert torch.equal(values['original'], values['block_mean'])
    metrics = {}
    for name, value in values.items():
        rows = []
        for head in range(heads):
            target, actual = baseline[:, head].double(), value[:, head].double()
            assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(target).all())
            energy = float(target.square().sum())
            error = float((actual - target).square().sum())
            rows.append(dict(head=head, error_energy=error, reference_energy=energy, nmse=error / energy))
        error = sum(r['error_energy'] for r in rows)
        energy = sum(r['reference_energy'] for r in rows)
        metrics[name] = dict(pooled_nmse=error / energy, error_energy=error, reference_energy=energy, per_head=rows)
        reported = (report['outputs'][name]['versus_bf16'] if name in report['outputs']
                    else report['existing_references'][name])
        assert math.isclose(metrics[name]['pooled_nmse'], reported['pooled']['nmse'], rel_tol=1e-12, abs_tol=1e-15)
        assert all(math.isclose(row['nmse'], other['nmse'], rel_tol=1e-12, abs_tol=1e-15)
                   for row, other in zip(rows, reported['per_head']))
    signs = {}
    for name in ('block_mean', 'global_mean'):
        delta = [a['nmse'] - b['nmse'] for a, b in zip(metrics['mean_times_k4']['per_head'], metrics[name]['per_head'])]
        signs[name] = dict(delta_nmse=delta, worse_heads=[h for h, d in enumerate(delta) if d > 0],
                          better_heads=[h for h, d in enumerate(delta) if d < 0], equal_heads=[h for h, d in enumerate(delta) if d == 0])
    assert report['inputs']['q_packets'] == block0['q_packet_artifact']
    assert report['inputs']['kv_packets'] == case['kv_packets']
    qp, kv = load(report['inputs']['q_packets']), load(report['inputs']['kv_packets'])
    tensor_refs = {**block0['actual_q_packets'], **case['kv_tensor_records']}
    packets = {}
    for name in ('q_fp4', 'k_fp4', 'v_fp4_t', 'q_scale', 'k_scale', 'v_scale_t'):
        value = (qp if name in qp else kv)[name]
        packets[name] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value))
        assert packets[name] == {k: tensor_refs[name][k] for k in ('shape', 'dtype', 'sha256')}
    # Independently decode the E2M1 nibbles and E4M3 scales, undoing both layouts.
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    r = torch.arange(padded)[:, None]
    c = torch.arange(dim // 16)[None, :]
    offset = (r // 64) * 64 * (dim // 16) + (c // 4) * 256 + (r % 16) * 16 + ((r % 64) // 16) * 4 + c % 4
    t = torch.arange(padded)
    u = t % 32
    inverse = torch.argsort((t // 32) * 32 + (u // 8) * 2 + ((u % 8) // 2) * 8 + u % 2)
    khash = hashlib.sha256()
    for head in range(heads):
        codes = kv['k_fp4'][0, head]
        nibbles = torch.stack((codes & 15, codes >> 4), -1).reshape(padded, dim)
        scales = kv['k_scale'][0, head].view(torch.uint8).reshape(-1)[offset.flatten()].view(torch.float8_e4m3fn).reshape(padded, dim // 16).float()
        decoded = (levels[nibbles.long()] * scales.repeat_interleave(16, -1)).index_select(0, inverse).contiguous()
        assert bool(decoded.isfinite().all())
        khash.update(decoded.view(torch.uint8).numpy())
    assert khash.hexdigest() == report['decoded_key']['sha256']
    # Only mmap Q from the previously audited capture. No re-hash of its unrelated tensors.
    raw = torch.load(report['inputs']['capture']['file'], map_location='cpu', mmap=True, weights_only=True)
    means = []
    for head in range(heads):
        query = torch.zeros((padded, dim), dtype=torch.bfloat16)
        query[:n] = raw['q'][:, head]
        means.append(query.reshape(padded // 128, 128, dim).mean(1))
    mu = torch.stack(means).unsqueeze(0)
    mean_hash = tensor_sha(mu)
    assert not torch.cuda.is_initialized()
    return dict(experiment='E026', status='complete', cpu_only=True, source_sha256=sha(__file__),
        run_reference=dict(file=str(report_path), sha256=sha(report_path)), checked_files=checked,
        metrics=metrics, fixed_mean_k4_minus_reference_by_head=signs, original_output_equals_saved_block=True,
        packets=dict(records=packets, all_six_equal_E018=True,
            evidence='Input files/tensor SHA independently checked; source binds the same six GPU tensor objects to both calls. No additional GPU post-call packet hashes were saved.'),
        semantic_checks=dict(decoded_K4_full_padded_sha256=khash.hexdigest(), decoded_K4_matches_record=True,
            mean_CPU_BF16_reduction_sha256=mean_hash, mean_CPU_matches_actual_GPU_record=mean_hash == report['mean']['sha256'],
            mean_definition='Actual padded BF16 Q, 177 groups of 128, BF16 mean retained without additional mu quantization.',
            key_definition='Decoded centered K4 with actual E4M3 scales, inverse scale swizzle and inverse K row permutation; no per-tensor global scale.',
            correction_definition='Only last packet replaced by FP32 matmul of the fixed actual BF16 mu and decoded centered K4. Original uses centered unquantized BF16 K. One common sm_scale is applied by the unchanged kernel.',
            correction_numerical_audit='Corrections themselves were not saved as tensor artifacts. Their GPU matmul arithmetic is source-bound, not independently byte-recomputed here.'),
        scope='One H3 p36/s14/block0, all 22539 valid query rows and 56 heads; two attention calls and zero DiT.',
        limitations=['Fixed-mu control is not an optimized-mu oracle or a lower bound on every fused design.',
            'P/softmax/PV implementation is unchanged, but actual P values/codes change with the correction; this is full operator output error.',
            'No conclusions about generation quality, speed, or the impossibility of learned/reoptimized compensation.'],
        cuda_initialized=False)


if __name__ == '__main__':
    path = RD / 'independent_summary.json'
    assert not path.exists(), 'Preserve existing summary'
    started = time.time()
    result = dict(experiment='E026', status='failed_stop', source_sha256=sha(__file__))
    try:
        result = run()
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['seconds'] = time.time() - started
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
        print(json.dumps(dict(status=result['status'], output=str(path), sha256=sha(path))))
