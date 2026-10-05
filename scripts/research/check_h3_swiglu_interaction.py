#!/usr/bin/env python3
"""E062 independent saved-artifact checks; NumPy arithmetic, CPU-only.

Run only after evaluate.json is complete. Does not import the experiment runner,
load model weights, rehash full checkpoints, execute forwards, or overwrite files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E062'
CASES = ('e010_p030_s05', 'e010_p036_s14')
BLOCKS = (0, 24, 49)
MODES = ('raw', 'centered', 'proportional_residual')
PARTS = ('a', 'b', 'c', 'total')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def raw(tensor):
    assert tensor.device.type == 'cpu'
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def array(tensor):
    if tensor.dtype == torch.bfloat16:
        bits = np.frombuffer(raw(tensor), dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float64).reshape(tuple(tensor.shape))
    return tensor.detach().numpy().astype(np.float64)


def signature(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw(value)).hexdigest())
    if isinstance(value, dict):
        return {k: signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [signature(v) for v in value]
    return value


def byte_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and raw(a) == raw(b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(byte_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(byte_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def load(record, files):
    path = Path(record['file'])
    assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], str(path)
    files[str(path)] = dict(record, sha256_recomputed=True)
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def rms(got, reference):
    assert got.shape == reference.shape and np.isfinite(got).all() and np.isfinite(reference).all()
    rr = float(np.sqrt(np.mean(reference*reference, dtype=np.float64)))
    er = float(np.sqrt(np.mean((got-reference)**2, dtype=np.float64)))
    return dict(reference_rms=rr, error_rms=er, relative_rms=er/max(rr, 1e-30),
                passed=(er == 0 if rr == 0 else er/rr < .01))


def silu(x):
    e = np.exp(-np.abs(x))
    return x*np.where(x >= 0, 1/(1+e), e/(1+e))


def bf16_round(x):
    # Finite FP32 -> BF16 round-to-nearest-even, returned as exactly decoded FP32.
    x = np.asarray(x, dtype=np.float32)
    assert np.isfinite(x).all()
    bits = x.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.view(np.float32)


def energy(parts, teacher):
    t = teacher-teacher.mean(axis=0, keepdims=True)
    denom = np.sum(t*t, axis=0, keepdims=True, dtype=np.float64)
    safe = np.where(denom > 0, denom, 1.)
    output = {}
    for mode in MODES:
        values = parts if mode == 'raw' else {k: v-v.mean(axis=0, keepdims=True) for k, v in parts.items()}
        if mode == 'proportional_residual':
            values = {k: v-t*(np.sum(t*v, axis=0, keepdims=True, dtype=np.float64)/safe) for k, v in values.items()}
        ab, c, total = values['a']+values['b'], values['c'], values['total']
        et, eab, ec = (float(np.sum(x*x, dtype=np.float64)) for x in (total, ab, c))
        cross = float(2*np.sum(ab*c, dtype=np.float64))
        net = et-eab
        output[mode] = dict(Etotal=et, Eab=eab, Ec=ec, cross=cross, net=net,
            net_over_total=net/et if et else None, signed_closure_error=net-ec-cross,
            identity=rms(ab+c, total), passes_20pct=bool(et > 0 and net/et >= .2))
    return output


def compare_stats(got, saved):
    maximum = 0.
    for mode in MODES:
        scale = max(got[mode]['Etotal'], got[mode]['Eab'], got[mode]['Ec'], 1e-30)
        for field in ('Etotal', 'Eab', 'Ec', 'cross', 'net', 'signed_closure_error'):
            difference = abs(got[mode][field]-saved[mode][field])/scale
            maximum = max(maximum, difference)
            assert difference < 1e-10, (mode, field, got[mode][field], saved[mode][field])
        assert got[mode]['passes_20pct'] == saved[mode]['passes_20pct']
        assert got[mode]['identity']['passed'] == saved[mode]['identity']['passed']
    return maximum


def sample_checks(payload):
    sample = payload['fp64_sample']
    g, u, gq, uq = (array(sample[k]) for k in ('g', 'u', 'gq', 'uq'))
    weight, gate = array(sample['weight']), array(sample['gate'])
    assert g.shape[0] == gate.shape[0] == 2 and weight.shape[0] == gate.shape[1] == 32
    assert g.shape[1] == weight.shape[1]
    for name, whole in (('g', 'teacher_gate'), ('u', 'teacher_up'), ('gq', 'native_gate'), ('uq', 'native_up')):
        assert byte_equal(sample[name], payload[whole][:2])
    assert byte_equal(sample['gate'], payload['adaln_gate'][:2, :32])
    phi, phiq = silu(g), silu(gq)
    exact = dict(a=(phiq-phi)*u, b=phi*(uq-u), c=(phiq-phi)*(uq-u), total=phiq*uq-phi*u)
    checks = {}
    for name, value in exact.items():
        checks['hidden_'+name] = rms(array(payload['hidden'][name][:2]), value)
        checks['saved_fp64_'+name] = rms(array(payload['fp64_hidden_reference'][name]), value)
        checks['projected_'+name] = rms(array(payload['projected'][name][:2, :32]), (value @ weight.T)*gate)
    corners = {k: array(v[:2]) for k, v in payload['hidden_bf16_four_corners'].items()}
    rounded = dict(a=corners['QB']-corners['BB'], b=corners['BQ']-corners['BB'],
        c=corners['QQ']-corners['QB']-corners['BQ']+corners['BB'], total=corners['QQ']-corners['BB'])
    for name, value in rounded.items():
        checks['actual_bf16_corner_projected_'+name] = rms(
            array(payload['projected_bf16_corners'][name][:2, :32]), (value @ weight.T)*gate)
    # roll(256) maps output rows 0/1 to input rows 256/257, preserving the full K dimension.
    rolled = (array(payload['native_up'][256:258]).astype(np.float32)-
              array(payload['teacher_up'][256:258]).astype(np.float32)).astype(np.float64)
    null = dict(a=exact['a'], b=phi*rolled, c=(phiq-phi)*rolled,
                total=phiq*(u+rolled)-phi*u)
    for name, value in null.items():
        checks['null_projected_'+name] = rms(array(payload['projected_null'][name][:2, :32]), (value @ weight.T)*gate)
    assert all(v['passed'] for v in checks.values()), checks
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=REPORTS/'evaluate_v2.json')
    args = parser.parse_args()
    output = REPORTS/'independent_check.json'
    assert not output.exists(), f'Refusing overwrite: {output}'
    started = time.monotonic()
    torch.set_num_threads(2)
    evaluation_path = args.evaluation
    evaluation = json.loads(evaluation_path.read_text())
    check_path = Path(evaluation['cpu_check_reference']['file'])
    checked = json.loads(check_path.read_text())
    assert evaluation['status'] == checked['status'] == 'complete'
    assert evaluation['attempted_dit_calls'] == evaluation['complete_dit_calls'] == len(evaluation['teacher_replays']) == 2
    assert evaluation['attempted_native_fc1_calls'] == evaluation['complete_native_fc1_calls'] == len(evaluation['cases']) == 6
    assert checked['cuda_initialized'] is False and not torch.cuda.is_initialized()
    assert sha(check_path) == evaluation['cpu_check_reference']['sha256']
    runners = [Path(p) for p in evaluation['sources'] if Path(p).name.startswith('probe_h3_swiglu_interaction') and Path(p).suffix == '.py']
    assert len(runners) == 1, runners
    runner = runners[0]
    assert sha(runner) == evaluation['sources'][str(runner)]['sha256']
    files, teachers, teacher_checks = {}, {}, []
    # E014 stores actual-input signatures only. E059 supplies concrete saved historical input tensors.
    old_path = ROOT/'results/research/E059/evaluate.json'
    old = json.loads(old_path.read_text())
    for row in evaluation['teacher_replays']:
        cid = row['case_id']
        teacher, history = load(row['artifact'], files), load(row['historical_artifact'], files)
        old_row = next(r for r in old['cases'] if r['case_id'] == cid and r['position'] == 'source_teacher' and r['mode'] == 'zero')
        old_input = load(old_row['artifact'], files)
        fields = {k: byte_equal(teacher[k], history[k]) for k in ('raw_outputs', 'velocities')}
        fields['actual_input_to_E059_concrete'] = byte_equal(teacher['actual_dit_inputs'], old_input['actual_dit_inputs'])
        assert all(fields.values())
        assert signature(teacher['actual_dit_inputs']) == history['actual_dit_inputs'] == row['actual_dit_input_signature'] == checked['inputs'][cid]['actual_dit_input_signature']
        for field in ('raw_outputs', 'velocities'):
            assert signature(teacher[field]) == row[field] == evaluation['references'][cid][field]
        assert row['runtime_audit']['sdpa_calls'] == 102 and row['runtime_audit']['scaled_mm_calls'] == row['runtime_audit']['disk_loads'] == 0
        teachers[cid] = teacher
        teacher_checks.append(dict(case_id=cid, byte_equal=fields, actual_input_signature_to_E014=True))
    assert set(teachers) == set(CASES)
    cells, maximum = [], 0.
    assert {(r['case_id'], r['block']) for r in evaluation['cases']} == {(cid, b) for cid in CASES for b in BLOCKS}
    for row in evaluation['cases']:
        p = load(row['artifact'], files)
        cid, block = row['case_id'], row['block']
        assert p['case_id'] == cid and p['block'] == block and row['status'] == 'complete'
        kw = teachers[cid]['actual_dit_inputs']['kwargs']
        video = array(kw['img_pos_info']['position_ids']).astype(np.int64).ravel()
        offsets = np.arange(512, dtype=np.int64)*(len(video)-1)//511
        indices = video[offsets]
        assert np.array_equal(array(p['indices']), indices) and np.array_equal(array(p['offsets']), offsets)
        assert indices.tolist() == checked['inputs'][cid]['selected_indices']
        combined = array(kw['inverse_indices']).astype(np.int64).ravel()*3+np.maximum(array(kw['token_tags']).astype(np.int64).ravel(), 0)
        assert np.array_equal(array(p['selected_combined_indices']), combined[indices])
        assert p['packet']['input_shape'][0] == kw['x'].shape[1] > 512
        assert signature(p['packet']) == row['packet_signature']
        assert row['runtime_audit']['scaled_mm_calls'] == 1 and row['runtime_audit']['sdpa_calls'] == row['runtime_audit']['disk_loads'] == 0
        assert row['fastpack_checks']['checked_calls'] == 1 and row['fastpack_checks']['invalid_calls'] == 0
        assert byte_equal(p['hidden_bf16_four_corners']['BB'], p['teacher_hidden_bf16'])
        assert byte_equal(p['hidden_bf16_four_corners']['QQ'], p['native_hidden_bf16'])
        numeric = sample_checks(p)
        teacher = array(p['teacher_output'])
        gate = array(p['adaln_gate']).astype(np.float32)
        td, nd = (array(p[k]).astype(np.float32) for k in ('teacher_down_bf16', 'native_down_bf16'))
        assert np.array_equal(teacher, (td*gate).astype(np.float64))
        actual32 = bf16_round(nd*gate)-bf16_round(td*gate)
        assert np.array_equal(array(p['actual_bf16_gated_difference']), actual32.astype(np.float64))
        groups = {}
        for name, saved, field in (('analytic', 'projected', 'metrics'), ('actual_bf16_corners', 'projected_bf16_corners', 'bf16_four_corner_metrics'), ('null', 'projected_null', 'null_metrics')):
            stats = energy({k: array(v) for k, v in p[saved].items()}, teacher)
            maximum = max(maximum, compare_stats(stats, row[field]))
            assert all(v['identity']['passed'] for v in stats.values())
            groups[name] = stats
        c = array(p['projected']['c'])
        for mode in MODES:
            denom = groups['analytic'][mode]['Etotal']
            ratio = groups['null'][mode]['net']/denom if denom else None
            groups['null'][mode]['net_over_original_total'] = ratio
            saved_ratio = row['null_metrics'][mode]['net_over_original_total']
            assert ratio is saved_ratio is None or abs(ratio-saved_ratio) < 1e-10*max(1., abs(ratio))
        # First reproduce the runner's FP32 subtraction, then independently evaluate FP64 total-c.
        actual = actual32.astype(np.float64)
        arithmetic = energy(dict(a=(actual32-c.astype(np.float32)).astype(np.float64), b=np.zeros_like(c), c=c, total=actual), teacher)
        maximum = max(maximum, compare_stats(arithmetic, row['actual_bf16_arithmetic_sensitivity']))
        arithmetic64 = energy(dict(a=actual-c, b=np.zeros_like(c), c=c, total=actual), teacher)
        uncertain = any((groups['analytic'][mode]['net'] > 0) != (other[mode]['net'] > 0) or
                        groups['analytic'][mode]['passes_20pct'] != other[mode]['passes_20pct']
                        for mode in ('raw', 'proportional_residual') for other in (groups['actual_bf16_corners'], arithmetic))
        assert uncertain == row['bf16_decision_uncertain']
        arithmetic_subtraction_decision_stable = all((arithmetic[m]['net'] > 0) == (arithmetic64[m]['net'] > 0) and
            arithmetic[m]['passes_20pct'] == arithmetic64[m]['passes_20pct'] for m in ('raw', 'proportional_residual'))
        corners = {k: array(v) for k, v in p['hidden_bf16_four_corners'].items()}
        corner_c = corners['QQ']-corners['QB']-corners['BQ']+corners['BB']
        hidden_c_difference = rms(corner_c, array(p['hidden']['c']))
        projected_c_difference = rms(array(p['projected_bf16_corners']['c']), c)
        cells.append(dict(case_id=cid, block=block, metrics=groups,
            arithmetic_sensitivity_runner_fp32_subtraction=arithmetic,
            arithmetic_sensitivity_independent_fp64_subtraction=arithmetic64,
            arithmetic_subtraction_decision_stable=arithmetic_subtraction_decision_stable,
            bf16_decision_uncertain=uncertain, numerical_checks=numeric,
            actual_bf16_hidden_c_vs_analytic_c=hidden_c_difference,
            actual_bf16_projected_c_vs_analytic_c=projected_c_difference,
            actual_bf16_gated_total_vs_analytic_total=rms(actual, array(p['projected']['total'])),
            primary_cell_passed=not uncertain and all(groups['analytic'][m]['passes_20pct'] for m in ('raw', 'proportional_residual'))))
        assert arithmetic_subtraction_decision_stable, 'FP32 vs FP64 subtraction changes arithmetic-control decision'
        print(f'E062 independent {cid} block {block}: {groups["analytic"]["raw"]["net_over_total"]:.9g}', flush=True)
        del p, corners, corner_c, c, gate, td, nd, teacher, actual, actual32
    gates = [dict(block=b, passed=all(r['primary_cell_passed'] for r in cells if r['block'] == b)) for b in BLOCKS]
    assert gates == evaluation['primary_gate']['by_block']
    gate = any(r['passed'] for r in gates)
    assert gate == evaluation['primary_gate']['passed'] and not torch.cuda.is_initialized()
    result = dict(experiment='E062', status='complete', seconds=time.monotonic()-started,
        cuda_initialized=False, new_model_forwards=0, script=dict(file=str(Path(__file__)), sha256=sha(__file__)),
        evaluation=dict(file=str(evaluation_path), sha256=sha(evaluation_path)),
        e059_input_history=dict(file=str(old_path), sha256=sha(old_path)),
        teacher_checks=teacher_checks, cells=cells, primary_gate=dict(by_block=gates, passed=gate),
        max_metric_difference_normalized_by_energy=maximum, files=list(files.values()),
        method='Independent NumPy float64 energies/SiLU/full-K sample projections; PyTorch only deserializes CPU tensors and supplies raw bytes.',
        scope=['Two teacher raw/velocity byte replays against E014; concrete actual input bytes against E059 plus E014 signatures.',
            'Six full-input native-fc1 cell artifacts freshly SHA256 checked; saved projection energies recomputed for raw/center/common-teacher proportional residual.',
            'Per cell numerical sample: first two of 512 selected rows, all hidden channels K, first 32 BF16 down output rows and actual AdaLN gates; every tested component relative RMS <1%.',
            'Saved actual BF16 hidden corners and gated down differences inspected separately from analytic SiLU interaction.',
            'Null uses the same three projections and original unpermuted total denominators; auxiliary, noncausal.'],
        limitations=['No full-checkpoint or full-down-weight reread/hash and no GPU replay.',
            'Stored projected tensors validate the full sampled energy readout; the independent down projection numerical check is only 2 rows x 32 outputs with full K.',
            '20% is an investment gate on six local diagnostic cells, not a significance, quality, generalization, or linear-representability conclusion.',
            'BF16 actual gated difference excludes the final residual-add rounding and later network propagation.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', seconds=result['seconds'], gate=gate, output=str(output), sha256=sha(output)), indent=2))


if __name__ == '__main__':
    main()
