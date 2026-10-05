#!/usr/bin/env python3
"""E063 independent NumPy audit of completed saved Q/K radial-screen artifacts.

No experiment imports, GPU calls, checkpoint reads, or quantizer/kernel replay.
The E062 checker supplies CPU deserialization and byte/hash utilities only.
"""
from __future__ import annotations
import argparse
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
import check_h3_swiglu_interaction as io

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E063'
CASES, BLOCKS = ('e010_p030_s05', 'e010_p036_s14'), (0, 24, 49)
ARM0, ARM32 = 'rank0_same_smooth', 'legacy_rank32'


def geometry_metrics(samples, norms):
    """Teacher-defined Euclidean projectors; epsilon is deliberately absent."""
    by_part, by_head, geometry, numerical = {}, {}, {}, {}
    for part in ('q', 'k', 'v'):
        teacher = io.array(samples['teacher']['raw'][part])
        norm2 = np.sum(teacher*teacher, axis=-1, keepdims=True, dtype=np.float64)
        energies, coefficients = {}, {}
        for arm in (ARM0, ARM32):
            value = io.array(samples[arm]['raw'][part])
            error = value-teacher
            assert np.isfinite(error).all()
            dot = np.sum(error*teacher, axis=-1, keepdims=True, dtype=np.float64)
            alpha = np.divide(dot, norm2, out=np.zeros_like(dot), where=norm2 > 0)
            radial = teacher*alpha
            tangent = error-radial
            e = {name: np.sum(x*x, axis=(0, 2), dtype=np.float64)
                 for name, x in (('raw', error), ('radial', radial), ('tangent', tangent))}
            residual = e['raw']-e['radial']-e['tangent']
            relative = np.abs(residual)/np.maximum(e['raw'], 1e-300)
            assert float(relative.max()) < 1e-10, (part, arm, relative)
            numerical[part+'/'+arm] = dict(max_relative_energy_closure=float(relative.max()),
                max_abs_radial_tangent_dot=float(np.max(np.abs(np.sum(radial*tangent, axis=-1)))))
            vdot = np.sum(value*teacher, axis=-1, keepdims=True, dtype=np.float64)
            coefficients[arm] = np.divide(vdot, norm2, out=np.zeros_like(vdot), where=norm2 > 0).squeeze(-1)
            if part != 'v':
                for stage in ('norm', 'rope'):
                    diff = io.array(samples[arm][stage][part])-io.array(samples['teacher'][stage][part])
                    assert np.isfinite(diff).all()
                    e[stage] = np.sum(diff*diff, axis=(0, 2), dtype=np.float64)
            energies[arm] = e
        heads = []
        for head in range(teacher.shape[1]):
            row = {f'E{rank}_{stage}': float(energies[arm][stage][head])
                   for rank, arm in ((0, ARM0), (32, ARM32)) for stage in energies[arm]}
            row.update({'G_'+stage: row['E0_'+stage]-row['E32_'+stage] for stage in energies[ARM0]})
            row['signed_gain_closure_error'] = row['G_raw']-row['G_radial']-row['G_tangent']
            heads.append(row)
        by_head[part] = heads
        by_part[part] = {key: float(np.sum([h[key] for h in heads], dtype=np.float64)) for key in heads[0]}
        geom = dict(teacher_zero_norm=int(np.count_nonzero(norm2 == 0)),
            teacher_min_mean_square=float(norm2.min()/teacher.shape[-1]),
            radial_nonpositive_counts={arm: int(np.count_nonzero((c <= 0) & (norm2.squeeze(-1) > 0))) for arm, c in coefficients.items()},
            radial_nonpositive_by_head={arm: np.sum((c <= 0) & (norm2.squeeze(-1) > 0), axis=0).tolist() for arm, c in coefficients.items()})
        if part != 'v':
            epsilon = norms[part]['eps']
            assert np.isfinite(epsilon) and epsilon > 0
            geom.update(epsilon=epsilon, teacher_at_or_below_epsilon=int(np.count_nonzero(norm2/teacher.shape[-1] <= epsilon)))
        geometry[part] = geom
    by_part['qk'] = {k: by_part['q'][k]+by_part['k'][k] for k in by_part['q']}
    e = by_part['qk']
    raw = e['G_raw']/e['E0_raw'] if e['E0_raw'] > 0 else None
    radial = e['G_radial']/e['G_raw'] if e['G_raw'] != 0 else None
    norm = e['G_norm']/e['E0_norm'] if e['E0_norm'] > 0 else None
    checks = dict(raw_gain_at_least_10pct=raw is not None and raw >= .1,
                  radial_gain_share_at_least_75pct=radial is not None and radial >= .75,
                  norm_gain_at_most_10pct=norm is not None and norm <= .1)
    gate = dict(raw_relative_gain=raw, radial_gain_over_raw_gain=radial, norm_relative_gain=norm,
                passed=all(checks.values()))
    return dict(by_part=by_part, by_head=by_head, geometry=geometry, numerical_checks=numerical,
                gate=gate, individual_gate_conditions=checks)


def compare_metric_row(got, expected):
    assert got.keys() == expected.keys()
    maximum = 0.
    for key, value in got.items():
        stage = key.split('_', 1)[1] if key != 'signed_gain_closure_error' else 'raw'
        scale = max(abs(got['E0_'+stage]), abs(got['E32_'+stage]), 1e-300)
        relative = abs(value-expected[key])/scale
        assert relative < 1e-10, (key, value, expected[key], scale)
        maximum = max(maximum, relative)
    return maximum


def compare_metrics(got, expected):
    maximum = max(compare_metric_row(row, expected['by_part'][part]) for part, row in got['by_part'].items())
    for part, heads in got['by_head'].items():
        assert len(heads) == len(expected['by_head'][part])
        for i, row in enumerate(heads):
            maximum = max(maximum, compare_metric_row(row, expected['by_head'][part][i]))
    for part, g in got['geometry'].items():
        for key, value in g.items():
            saved = expected['geometry'][part][key]
            if key == 'teacher_min_mean_square':
                assert abs(value-saved) <= 1e-12*max(abs(value), 1e-300)
            else:
                assert value == saved, (part, key, value, saved)
    assert got['gate']['passed'] == expected['gate']['passed']
    for key in ('raw_relative_gain', 'radial_gain_over_raw_gain', 'norm_relative_gain'):
        value, saved = got['gate'][key], expected['gate'][key]
        assert value is saved is None or abs(value-saved) <= 1e-10*max(1., abs(value))
    assert all(v['max_relative_energy_closure'] < 1e-10 for v in expected['numerical_checks'].values())
    return maximum


def check_samples(payload, teacher):
    kw = teacher['actual_dit_inputs']['kwargs']
    video = io.array(kw['img_pos_info']['position_ids']).astype(np.int64).ravel()
    offsets = np.arange(512, dtype=np.int64)*(len(video)-1)//511
    assert len(video) >= 512 and len(np.unique(video)) == len(video)
    assert np.array_equal(io.array(payload['offsets']), offsets)
    assert np.array_equal(io.array(payload['indices']), video[offsets])
    samples = payload['samples']
    assert set(samples) == {'teacher', ARM0, ARM32}
    shape = tuple(samples['teacher']['raw']['q'].shape)
    assert len(shape) == 3 and shape[0] == 512
    for arm in samples:
        assert set(samples[arm]['raw']) == {'q', 'k', 'v'}
        for stage in ('raw', 'norm', 'rope'):
            assert set(samples[arm][stage]) == ({'q', 'k', 'v'} if stage == 'raw' else {'q', 'k'})
            for value in samples[arm][stage].values():
                assert value.dtype == torch.bfloat16 and tuple(value.shape) == shape
                assert np.isfinite(io.array(value)).all()
    assert payload['source_weight']['shape'][0] == 3*shape[1]*shape[2]
    actual = {}
    for part in ('q', 'k'):
        assert tuple(payload['norms'][part]['gamma'].shape) == (shape[2],)
        actual[part] = {}
        for stage in ('norm', 'rope'):
            same = io.byte_equal(samples['teacher'][stage][part], payload['teacher_actual_'+stage+'_samples'][part])
            assert same and payload['full_teacher_replays'][part][stage+'_exact']
            actual[part][stage+'_samples_byte_exact'] = same
        assert payload['actual_layout_checks'][part]['exact']
        expected_shape = [int(kw['x'].shape[1]), shape[1], shape[2]]
        assert payload['actual_layout_checks'][part]['actual_input_signature']['shape'] == expected_shape
        assert payload['actual_layout_checks'][part]['actual_input_signature']['dtype'] == 'torch.bfloat16'
    assert payload['rope_freqs'].shape[0] == 512 and payload['rope_freqs'].shape[1] <= shape[2]
    packet = payload['activation_packet_signature']
    assert packet['original_shape'] == [int(kw['x'].shape[1]), payload['source_weight']['shape'][1]]
    assert packet['original_shape'][0] > 512
    assert packet['packed']['shape'] == [packet['original_shape'][0], packet['original_shape'][1]//2]
    assert io.signature(payload['activation_global']) == packet['global_scale']
    assert np.isfinite(io.array(payload['activation_global'])).all() and (io.array(payload['activation_global']) > 0).all()
    for value in payload['weight_globals'].values():
        assert np.isfinite(io.array(value)).all() and (io.array(value) > 0).all()
    assert payload['rank0_roundtrip_exact']
    return dict(sample_shape=list(shape), actual_teacher_norm_rope=actual,
        full_input_activation_packet_shape=packet['original_shape'], indices_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=REPORTS/'evaluate.json')
    args = parser.parse_args()
    output = REPORTS/'independent_check.json'
    assert not output.exists(), f'Refusing overwrite: {output}'
    started = time.monotonic()
    torch.set_num_threads(2)
    evaluation = json.loads(args.evaluation.read_text())
    checked_path = Path(evaluation['cpu_check_reference']['file'])
    checked = json.loads(checked_path.read_text())
    assert evaluation['status'] == checked['status'] == 'complete' and checked['cuda_initialized'] is False
    assert evaluation['attempted_dit_calls'] == evaluation['complete_dit_calls'] == len(evaluation['teacher_replays']) == 2
    assert evaluation['attempted_native_qkv_calls'] == evaluation['complete_native_qkv_calls'] == 12
    assert evaluation['activation_packs'] == len(evaluation['cases']) == 6
    assert io.sha(checked_path) == evaluation['cpu_check_reference']['sha256']
    runner_paths = [Path(p) for p in evaluation['sources'] if Path(p).name.startswith('probe_h3_qknorm_radial') and Path(p).suffix == '.py']
    assert len(runner_paths) == 1
    assert io.sha(runner_paths[0]) == evaluation['sources'][str(runner_paths[0])]['sha256']
    assert evaluation['attention_function']['name'] == '_comfy_attention_forward'
    assert evaluation['attention_function']['module'] == 'diffsynth.models.minimax_h3_dit_comfy'
    attention_source = evaluation['attention_function']['source']
    assert io.sha(attention_source['file']) == attention_source['sha256']
    files, teachers, teacher_checks = {}, {}, []
    history_path = ROOT/'results/research/E059/evaluate.json'
    history = json.loads(history_path.read_text())
    for row in evaluation['teacher_replays']:
        cid = row['case_id']
        p, old = io.load(row['artifact'], files), io.load(row['historical_artifact'], files)
        old_input_row = next(r for r in history['cases'] if r['case_id'] == cid and r['position'] == 'source_teacher' and r['mode'] == 'zero')
        old_input = io.load(old_input_row['artifact'], files)
        exact = {k: io.byte_equal(p[k], old[k]) for k in ('raw_outputs', 'velocities')}
        exact['actual_input_to_E059_concrete'] = io.byte_equal(p['actual_dit_inputs'], old_input['actual_dit_inputs'])
        assert all(exact.values())
        assert io.signature(p['actual_dit_inputs']) == old['actual_dit_inputs'] == row['actual_dit_input_signature'] == checked['inputs'][cid]['actual_dit_input_signature']
        for field in ('raw_outputs', 'velocities'):
            assert io.signature(p[field]) == row[field] == evaluation['references'][cid][field]
        audit = row['runtime_audit']
        assert audit['sdpa_calls'] == 102 and audit['scaled_mm_calls'] == audit['disk_loads'] == 0
        teachers[cid] = p
        teacher_checks.append(dict(case_id=cid, byte_equal=exact, actual_input_signature_to_E014=True))
    assert set(teachers) == set(CASES)
    assert {(r['case_id'], r['block']) for r in evaluation['cases']} == {(cid, b) for cid in CASES for b in BLOCKS}
    cells, maximum = [], 0.
    for row in evaluation['cases']:
        p = io.load(row['artifact'], files)
        cid, block = row['case_id'], row['block']
        assert (p['case_id'], p['block']) == (cid, block) and row['status'] == 'complete'
        sample_checks = check_samples(p, teachers[cid])
        assert io.array(p['indices']).astype(np.int64).tolist() == checked['inputs'][cid]['selected_indices']
        assert p['activation_packet_signature'] == row['activation_packet_signature']
        assert p['actual_layout_checks'] == row['layout_checks']
        assert p['full_teacher_replays'] == row['teacher_norm_rope_replay']
        assert io.signature(p['weight_globals']) == row['weight_globals']
        assert p['legacy_artifact'] == evaluation['layers'][str(block)]['native_artifact']
        assert p['source_weight']['sha256'] == evaluation['layers'][str(block)]['layer']['source_weight_sha256']
        assert row['shared_activation_packet_unchanged'] and row['rank0_roundtrip_exact']
        assert row['runtime_audit']['scaled_mm_calls'] == 2 and row['runtime_audit']['sdpa_calls'] == row['runtime_audit']['disk_loads'] == 0
        assert row['fastpack_checks']['checked_calls'] == 1 and row['fastpack_checks']['invalid_calls'] == 0
        recomputed = geometry_metrics(p['samples'], p['norms'])
        maximum = max(maximum, compare_metrics(recomputed, row['metrics']))
        cells.append(dict(case_id=cid, block=block, metrics=recomputed, sample_checks=sample_checks,
            shared_packet_source_contract_verified=True, runtime_counter_contract_verified=True))
        print(f'E063 independent {cid} block {block}: {recomputed["gate"]}', flush=True)
        del p
    gates = [dict(block=b, passed=all(c['metrics']['gate']['passed'] for c in cells if c['block'] == b)) for b in BLOCKS]
    gate = any(g['passed'] for g in gates)
    assert gates == evaluation['primary_gate']['by_block'] and gate == evaluation['primary_gate']['passed']
    assert not torch.cuda.is_initialized()
    result = dict(experiment='E063', status='complete', seconds=time.monotonic()-started,
        cuda_initialized=False, new_model_forwards=0,
        script=dict(file=str(Path(__file__)), sha256=io.sha(__file__)),
        byte_utility_source=dict(file=io.__file__, sha256=io.sha(io.__file__)),
        evaluation=dict(file=str(args.evaluation), sha256=io.sha(args.evaluation)),
        e059_input_history=dict(file=str(history_path), sha256=io.sha(history_path)),
        teacher_checks=teacher_checks, cells=cells, primary_gate=dict(by_block=gates, passed=gate),
        max_energy_normalized_metric_difference=maximum, files=list(files.values()),
        numerical_method='NumPy FP64 teacher-defined radial/tangent projection; radial=0 when teacher norm is zero; no epsilon in projector. Negative gains and radial fractions >1 retained.',
        scope=['Two teacher raw/velocity byte replays against E014, concrete actual-input byte replays against E059, and E014 input signature checks.',
            'Six cell files freshly SHA256 checked; fixed uniform indices and all saved Q/K/V, norm and RoPE sample dimensions verified.',
            'Actual teacher norm/RoPE hook samples byte-compared with saved full-shape local-replay samples.',
            'Per-head and aggregate signed raw/radial/tangent/norm/RoPE energies independently recomputed; three necessary-screen gates checked.',
            'Comfy source binding, recorded full-shape layout checks, shared packet signature/immutability declarations, two native GEMMs and one activation pack per cell audited.'],
        limitations=['The full activation packet, full QKV and full norm-input tensors are not saved; their complete identities are source/record contracts, not independently reconstructed CPU arrays.',
            'No CPU reproduction of the GPU quantizer/kernel and no fresh full checkpoint/native-weight hash.',
            '512 video rows per cell and two old teacher states; necessary-condition screen only, not attention functionality, generalization or quality.',
            'The rank0/rank32 comparison changes the complete weight decomposition and may change weight globals; it does not isolate the LR branch as an independent treatment.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', seconds=result['seconds'], gate=gate, output=str(output), sha256=io.sha(output)), indent=2))


if __name__ == '__main__':
    main()
