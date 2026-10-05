#!/usr/bin/env python3
"""Independent CPU/NumPy E064 saved-block four-corner audit.

Full arrays are read in row chunks. No experiment/quantizer imports or GPU calls.
Only the final independent_check.json is written; no existing artifact is changed.
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
import check_h3_swiglu_interaction as io

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E064'
CASES = ('e010_p030_s05', 'e010_p036_s14')
BLOCKS = (0, 12, 24, 36, 49)
COMPONENTS = ('qB', 'p', 'echo', 'e_out', 'e_in', 'rB')
CHUNK_ROWS = 128


def chunk_array(tensor, indices):
    assert tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16
    selected = tensor[np.asarray(indices, dtype=np.int64)].contiguous()
    bits = selected.view(torch.uint16).numpy().astype(np.uint32) << 16
    return bits.view(np.float32).astype(np.float64)


def differences(arrays, indices):
    hb, hq, bb, qb, bq, qq = [chunk_array(arrays[k], indices) for k in ('hB', 'hQ', 'BB', 'QB', 'BQ', 'QQ')]
    q = qb-bb
    p = bq-bb
    echo = (qq-bq)-q
    total = qq-bb
    before = hq-hb
    carry_remainder = p-before
    return np.stack((q, p, echo, total, before, carry_remainder), axis=0)


def gram_readout(gram):
    i = {name: n for n, name in enumerate(COMPONENTS)}
    energies = {name: float(gram[n, n]) for name, n in i.items()}
    baseline = float(gram[0, 0]+gram[1, 1]+2*gram[0, 1])
    total, ec = energies['e_out'], energies['echo']
    net = total-baseline
    inner_net = float(2*gram[3, 2]-gram[2, 2])
    return dict(energies=energies, gram=gram.tolist(), baseline_p_plus_qB_energy=baseline,
        net_echo=net, net_echo_inner_product=inner_net,
        net_echo_over_total=net/total if total > 0 else None,
        signed_energy_closure=net-inner_net,
        echo_cross_p_plus_qB=float(2*(gram[2, 0]+gram[2, 1])),
        BF16_transport_energy_change=energies['p']-energies['e_in'],
        carry_cross_rB=float(2*gram[4, 5]),
        passes_20pct=bool(total > 0 and net/total >= .2))


def modality_metrics(arrays, indices, saved_moments=None):
    """Two passes: means/raw Gram, then direct centered Gram (no cancellation)."""
    indices = np.asarray(indices, dtype=np.int64).ravel()
    assert indices.size and np.unique(indices).size == indices.size
    count, channels = indices.size, arrays['BB'].shape[1]
    sums = np.zeros((6, channels), dtype=np.float64)
    raw_gram = np.zeros((6, 6), dtype=np.float64)
    max_identity = max_carry_identity = max_output = 0.
    identity_sse = carry_identity_sse = 0.
    zero = dict(e_in=True, p=True, echo=True, rB=True)
    for start in range(0, count, CHUNK_ROWS):
        values = differences(arrays, indices[start:start+CHUNK_ROWS])
        assert np.isfinite(values).all()
        sums += np.sum(values, axis=1, dtype=np.float64)
        flat = values.reshape(6, -1)
        raw_gram += flat @ flat.T
        identity_error = values[0]+values[1]+values[2]-values[3]
        carry_error = values[4]+values[5]-values[1]
        max_identity = max(max_identity, float(np.max(np.abs(identity_error))))
        max_carry_identity = max(max_carry_identity, float(np.max(np.abs(carry_error))))
        max_output = max(max_output, float(np.max(np.abs(values[3]))))
        identity_sse += float(np.sum(identity_error*identity_error, dtype=np.float64))
        carry_identity_sse += float(np.sum(carry_error*carry_error, dtype=np.float64))
        for name, idx in (('e_in', 4), ('p', 1), ('echo', 2), ('rB', 5)):
            zero[name] = zero[name] and not bool(np.any(values[idx] != 0))
    means = sums/count
    centered_gram = np.zeros_like(raw_gram)
    max_centered_identity = 0.
    for start in range(0, count, CHUNK_ROWS):
        values = differences(arrays, indices[start:start+CHUNK_ROWS])-means[:, None, :]
        flat = values.reshape(6, -1)
        centered_gram += flat @ flat.T
        max_centered_identity = max(max_centered_identity,
            float(np.max(np.abs(values[0]+values[1]+values[2]-values[3]))))
    mean_gram = (sums @ sums.T)/count
    scale = max(float(np.max(np.diag(raw_gram))), 1e-300)
    bias_closure = float(np.max(np.abs(raw_gram-centered_gram-mean_gram)))/scale
    assert bias_closure < 1e-10, bias_closure
    identity_relative_rms = float(np.sqrt(identity_sse/max(raw_gram[3, 3], 1e-300)))
    carry_identity_relative_rms = float(np.sqrt(carry_identity_sse/max(raw_gram[1, 1], 1e-300)))
    assert max_identity <= 1e-12*max(1., max_output) and max_centered_identity <= 1e-12*max(1., max_output)
    assert max_carry_identity <= 1e-12*max(1., float(np.sqrt(raw_gram[1, 1])))
    moment_error = None
    if saved_moments is not None:
        moment_error = {}
        for name, value in (('channel_sums', sums), ('channel_means', means)):
            recorded = saved_moments[name]
            reference = io.array(recorded) if isinstance(recorded, torch.Tensor) else np.asarray(recorded, dtype=np.float64)
            assert reference.shape == value.shape
            relative = float(np.max(np.abs(value-reference)))/max(float(np.max(np.abs(value))), 1e-300)
            assert relative < 1e-10, (name, relative)
            moment_error[name+'_max_relative_difference'] = relative
    raw_stats, centered_stats = gram_readout(raw_gram), gram_readout(centered_gram)
    for stats in (raw_stats, centered_stats):
        escale = max(stats['energies']['e_out'], stats['baseline_p_plus_qB_energy'], stats['energies']['echo'], 1e-300)
        assert abs(stats['signed_energy_closure'])/escale < 1e-10
    return dict(row_count=int(count), channels=int(channels), component_order=list(COMPONENTS),
        raw=raw_stats, centered=centered_stats,
        channel_means_shape=list(means.shape),
        numpy_channel_means_sha256=hashlib.sha256(means.tobytes()).hexdigest(),
        mean_bias_gram=mean_gram.tolist(),
        mean_bias_energies={name: float(mean_gram[i, i]) for i, name in enumerate(COMPONENTS)},
        saved_moment_comparison=moment_error,
        e_out_bias_fraction=float(mean_gram[3, 3]/raw_gram[3, 3]) if raw_gram[3, 3] > 0 else None,
        numerical=dict(max_abs_four_corner_identity=max_identity, max_abs_carry_identity=max_carry_identity,
            four_corner_identity_relative_rms=identity_relative_rms, carry_identity_relative_rms=carry_identity_relative_rms,
            max_abs_centered_four_corner_identity=max_centered_identity,
            raw_equals_centered_plus_bias_max_relative_error=bias_closure), exact_zero=zero)


def same_tensor(a, b):
    """Byte equality in chunks, avoiding a second full-size serialization buffer."""
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype):
        return False
    if a.ndim == 0:
        return io.raw(a) == io.raw(b)
    return all(io.raw(a[start:start+CHUNK_ROWS]) == io.raw(b[start:start+CHUNK_ROWS])
               for start in range(0, a.shape[0], CHUNK_ROWS))


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return same_tensor(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def compare_statistics(got, recorded):
    assert got['row_count'] == recorded['rows'] and got['channels'] == recorded['channels']
    assert got['component_order'] == recorded['component_order']
    maximum = 0.
    for mode in ('raw', 'centered'):
        a, b = got[mode], recorded[mode]
        for name in COMPONENTS:
            value, ref = a['energies'][name], b['energies'][name]
            difference = abs(value-ref)/max(abs(value), abs(ref), 1e-300)
            assert difference < 1e-10, (mode, name, difference)
            maximum = max(maximum, difference)
        matrix, reference = np.asarray(a['gram']), np.asarray(b['gram'])
        scale = max(float(np.max(np.diag(matrix))), 1e-300)
        difference = float(np.max(np.abs(matrix-reference)))/scale
        assert difference < 1e-10
        maximum = max(maximum, difference)
        for left, right in (('baseline_p_plus_qB_energy', 'E_without_echo'), ('net_echo', 'net_echo')):
            difference = abs(a[left]-b[right])/max(a['energies']['e_out'], a['baseline_p_plus_qB_energy'], 1e-300)
            assert difference < 1e-10
            maximum = max(maximum, difference)
        assert a['passes_20pct'] == b['passes_20pct']
        x, y = a['net_echo_over_total'], b['net_echo_over_e_out']
        assert x is y is None or abs(x-y) < 1e-10*max(1., abs(x))
        for i, first in enumerate(COMPONENTS):
            for j in range(i+1, len(COMPONENTS)):
                assert abs(2*matrix[i, j]-b['twice_cross_terms'][first+','+COMPONENTS[j]])/scale < 1e-10
    for name, value in got['mean_bias_energies'].items():
        reference = recorded['mean_bias_energy'][name]
        scale = max(got['raw']['energies'][name], abs(value), 1e-300)
        assert abs(value-reference)/scale < 1e-10
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=REPORTS/'evaluate.json')
    args = parser.parse_args()
    output = REPORTS/'independent_check.json'
    assert not output.exists(), f'Refusing overwrite: {output}'
    started = time.monotonic()
    torch.set_num_threads(2)
    evaluation = json.loads(args.evaluation.read_text())
    assert evaluation['status'] == 'complete'
    checked_path = Path(evaluation['cpu_check_reference']['file'])
    assert io.sha(checked_path) == evaluation['cpu_check_reference']['sha256']
    checked = json.loads(checked_path.read_text())
    assert checked['status'] == 'complete' and checked['cuda_initialized'] is False
    assert evaluation['attempted_dit_calls'] == evaluation['complete_dit_calls'] == len(evaluation['full_paths']) == 6
    assert evaluation['attempted_local_calls'] == evaluation['complete_local_calls'] == len(evaluation['local_replays']) == 70
    assert len(evaluation['cases']) == 20 and evaluation['component_order'] == list(COMPONENTS)
    runner_paths = [Path(p) for p in evaluation['sources'] if Path(p).name.startswith('probe_h3_depth_four_corner') and Path(p).suffix == '.py']
    assert len(runner_paths) == 1
    assert io.sha(runner_paths[0]) == evaluation['sources'][str(runner_paths[0])]['sha256']
    assert io.sha(evaluation['plan']['file']) == evaluation['plan']['sha256']
    for field in ('sources', 'references', 'reference_reports', 'export_manifests', 'asset_binding'):
        assert checked[field] == evaluation[field]
    for record in evaluation['reference_reports'].values():
        assert Path(record['file']).stat().st_size == record['bytes'] and io.sha(record['file']) == record['sha256']
    files, loaded = {}, {}
    def load(record):
        path = str(record['file'])
        if path not in loaded:
            loaded[path] = io.load(record, files)
        else:
            assert all(files[path][k] == record[k] for k in ('file', 'bytes', 'sha256'))
        return loaded[path]
    history_path = ROOT/'results/research/E059/evaluate.json'
    old = json.loads(history_path.read_text())
    actual_history = {}
    for cid in CASES:
        rec = next(r for r in old['cases'] if r['case_id'] == cid and r['position'] == 'source_teacher' and r['mode'] == 'zero')
        actual_history[cid] = load(rec['artifact'])['actual_dit_inputs']
    paths = {(r['case_id'], r['arm']): r for r in evaluation['full_paths']}
    assert set(paths) == {(cid, arm) for cid in CASES for arm in ('bf16', 'plain', 'svd')}
    captures, domains, full_checks = {}, {}, []
    totals = dict(scaled_mm_calls=0, pack_calls=0, sdpa_calls=0, disk_loads=0)
    def account(row, full):
        expected_native = 0 if row['arm'] == 'bf16' else (200 if full else 4)
        audit, pack = row['runtime_audit'], row['fastpack_checks']
        assert audit['scaled_mm_calls'] == expected_native and audit['sdpa_calls'] == (102 if full else 2)
        assert audit['disk_loads'] == 0 and pack['checked_calls'] == expected_native and pack['invalid_calls'] == 0
        for key in ('scaled_mm_calls', 'sdpa_calls', 'disk_loads'):
            totals[key] += audit[key]
        totals['pack_calls'] += pack['checked_calls']
    for (cid, arm), row in paths.items():
        p, reference = load(row['artifact']), load(row['historical_artifact'])
        assert p['case_id'] == cid and p['arm'] == arm and row['replay_exact']
        assert row['historical_artifact'] == evaluation['references'][arm][cid]['artifact']
        exact = {key: same_tree(p[key], reference[key]) for key in ('raw_outputs', 'velocities')}
        exact['actual_inputs_to_concrete_history'] = same_tree(p['actual_dit_inputs'], actual_history[cid])
        assert all(exact.values()), (cid, arm, exact)
        actual_signature = io.signature(p['actual_dit_inputs'])
        assert actual_signature == reference['actual_dit_inputs'] == row['actual_dit_input_signature'] == checked['inputs'][cid]['actual_dit_input_signature']
        for key in ('raw_outputs', 'velocities'):
            assert io.signature(p[key]) == row[key] == evaluation['references'][arm][cid][key]
        kw = p['actual_dit_inputs']['kwargs']
        domain = {m: io.array(kw[field]['position_ids']).astype(np.int64).ravel() for m, field in
            (('video', 'img_pos_info'), ('audio', 'audio_pos_info'), ('text', 'text_pos_info'))}
        joined = np.concatenate(list(domain.values()))
        assert len(np.unique(joined)) == len(joined) and joined.min() >= 0 and joined.max() < kw['x'].shape[1]
        assert p['padding_rows'] == kw['x'].shape[1]-len(joined) == checked['inputs'][cid]['padding_rows']
        for m, idx in domain.items():
            assert np.array_equal(idx, io.array(p['modality_indices'][m]))
            assert len(idx) == checked['inputs'][cid]['modality_counts'][m]
        assert io.signature(p['modality_indices']) == checked['inputs'][cid]['modality_indices_signature']
        if cid in domains:
            assert all(np.array_equal(domain[m], domains[cid][m]) for m in domain)
        domains[cid] = domain
        assert set(row['blocks']) == {str(b) for b in BLOCKS}
        for block in BLOCKS:
            record = row['blocks'][str(block)]
            capture = load(record['artifact'])
            assert capture['input'].dtype == capture['output'].dtype == torch.bfloat16
            assert capture['input'].shape == capture['output'].shape == (kw['x'].shape[1], checked['hidden_size'])
            for value, field in (('input', 'input_signature'), ('kwargs', 'metadata_signature'), ('output', 'output_signature')):
                assert io.signature(capture[value]) == record[field]
            captures[(cid, arm, block)] = capture
        account(row, full=True)
        full_checks.append(dict(case_id=cid, arm=arm, exact=exact, actual_inputs_match_E014_signature=True,
            complete_block_capture_signatures_verified=5, modality_counts={m: len(v) for m, v in domain.items()}, padding_rows=p['padding_rows']))
        print(f'E064 independent path verified: {cid} {arm}', flush=True)
    local = {(r['case_id'], r['arm'], r['block'], r['corner']): r for r in evaluation['local_replays']}
    assert len(local) == 70
    expected_local = {(cid, 'bf16', b, 'BB') for cid in CASES for b in BLOCKS}
    expected_local |= {(cid, arm, b, corner) for cid in CASES for arm in ('plain', 'svd') for b in BLOCKS for corner in ('QB', 'QQ')}
    expected_local |= {(cid, 'bf16', b, 'BQ_from_'+arm) for cid in CASES for arm in ('plain', 'svd') for b in BLOCKS}
    assert set(local) == expected_local
    for (cid, arm, block, corner), row in local.items():
        source_arm = 'bf16' if corner in ('BB', 'QB') else (corner.removeprefix('BQ_from_') if corner.startswith('BQ_from_') else arm)
        captured = captures[(cid, source_arm, block)]
        assert row['actual_input_signature'] == dict(args=[io.signature(captured['input'])], kwargs=io.signature(captured['kwargs']))
        diagonal = corner in ('BB', 'QQ')
        assert row['diagonal_exact'] == diagonal
        if diagonal:
            assert row['output_signature'] == io.signature(captured['output'])
        if corner == 'BB':
            assert row['reused_by_arms'] == ['plain', 'svd']
        account(row, full=False)
    assert totals == evaluation['runtime_totals'] == dict(scaled_mm_calls=960, pack_calls=960, sdpa_calls=752, disk_loads=0)
    cells, maximum = [], 0.
    assert {(r['case_id'], r['arm'], r['block']) for r in evaluation['cases']} == {(cid, arm, b) for cid in CASES for arm in ('plain', 'svd') for b in BLOCKS}
    for row in evaluation['cases']:
        cid, arm, block = row['case_id'], row['arm'], row['block']
        assert row['status'] == 'complete'
        trow, qrow = paths[(cid, 'bf16')]['blocks'][str(block)], paths[(cid, arm)]['blocks'][str(block)]
        assert row['artifacts']['teacher_capture'] == trow['artifact'] and row['artifacts']['quantized_capture'] == qrow['artifact']
        teacher, quant = captures[(cid, 'bf16', block)], captures[(cid, arm, block)]
        assert same_tree(teacher['kwargs'], quant['kwargs'])
        assert trow['metadata_signature'] == qrow['metadata_signature'] == row['metadata_signature']
        assert row['input_signatures'] == dict(B=trow['input_signature'], Q=qrow['input_signature'])
        assert row['diagonal_sources']['QQ'] == local[(cid, arm, block, 'QQ')]
        cross = {c: load(row['artifacts'][c]) for c in ('QB', 'BQ')}
        for corner, p in cross.items():
            assert (p['case_id'], p['arm'], p['block'], p['corner']) == (cid, arm, block, corner)
            source = trow if corner == 'QB' else qrow
            assert p['input_signature'] == source['input_signature'] and p['metadata_signature'] == source['metadata_signature']
            call_key = (cid, arm, block, 'QB') if corner == 'QB' else (cid, 'bf16', block, 'BQ_from_'+arm)
            assert io.signature(p['output']) == local[call_key]['output_signature']
        arrays = dict(hB=teacher['input'], hQ=quant['input'], BB=teacher['output'], QQ=quant['output'],
                      QB=cross['QB']['output'], BQ=cross['BQ']['output'])
        assert all(x.device.type == 'cpu' and x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape == arrays['hB'].shape for x in arrays.values())
        moments = load(row['statistics_artifact'])
        metrics = {}
        for modality, indices in domains[cid].items():
            stats = modality_metrics(arrays, indices, moments[modality])
            maximum = max(maximum, compare_statistics(stats, row['metrics'][modality]))
            metrics[modality] = stats
        zero = None
        if block == 0:
            zero = dict(input_exact=same_tensor(arrays['hB'], arrays['hQ']),
                BB_BQ_exact=same_tensor(arrays['BB'], arrays['BQ']), QB_QQ_exact=same_tensor(arrays['QB'], arrays['QQ']))
            assert all(zero.values()) and zero == row['block0_exact']
            assert all(all(s['exact_zero'].values()) and all(s[m]['energies'][n] == 0 for m in ('raw', 'centered')
                for n in ('e_in', 'p', 'echo', 'rB')) for s in metrics.values())
        else:
            assert row['block0_exact'] is None
        cells.append(dict(case_id=cid, arm=arm, block=block, metrics=metrics, block0_exact=zero,
            identical_complete_metadata=True, cross_input_and_output_signatures_verified=True))
        print(f'E064 independent {cid} {arm} block {block}: raw={metrics["video"]["raw"]["net_echo_over_total"]}, centered={metrics["video"]["centered"]["net_echo_over_total"]}', flush=True)
    arms = {}
    for arm in ('plain', 'svd'):
        depths = []
        for block in BLOCKS[1:]:
            rows = [c for c in cells if c['arm'] == arm and c['block'] == block]
            assert len(rows) == 2
            depths.append(dict(block=block, passed=all(r['metrics']['video'][m]['passes_20pct'] for r in rows for m in ('raw', 'centered'))))
        passing = sum(d['passed'] for d in depths)
        arms[arm] = dict(depths=depths, passing_depths=passing, passed=passing >= 2)
    gate = dict(arms=arms, threshold=.2, required_depths=2, passed=any(a['passed'] for a in arms.values()))
    assert gate == evaluation['primary_gate']
    new_tensor_bytes = sum(r['bytes'] for r in files.values() if '/20261004/E064/' in r['file'])
    assert new_tensor_bytes == evaluation['tensor_bytes'] <= 32*1024**3
    assert not torch.cuda.is_initialized()
    result = dict(experiment='E064', status='complete', seconds=time.monotonic()-started,
        cuda_initialized=False, new_model_forwards=0, script=dict(file=str(Path(__file__)), sha256=io.sha(__file__)),
        byte_utility_source=dict(file=io.__file__, sha256=io.sha(io.__file__)),
        evaluation=dict(file=str(args.evaluation), sha256=io.sha(args.evaluation)),
        e059_concrete_input_reference=dict(file=str(history_path), sha256=io.sha(history_path)),
        full_path_checks=full_checks, local_replay_contracts_verified=70, runtime_totals=totals,
        component_order=list(COMPONENTS), cells=cells, primary_gate=gate,
        max_metric_relative_difference=maximum, new_tensor_bytes_verified=new_tensor_bytes, files=list(files.values()),
        numerical_method='Independent NumPy FP64 from all six actual BF16 arrays; 128-row chunks; raw means/Gram then a second direct-centering pass. No epsilon, clipping, synthetic tests or experiment statistics imports.',
        scope=['All three full paths reproduce both E014 states in actual input signatures and raw/velocity bytes; actual input tensors also byte-checked against E059.',
            'Every saved capture, cross output and statistics artifact freshly hashed; full hB/hQ/BB/QB/BQ/QQ rows used for all three disjoint modality domains.',
            'Full captured kwargs are byte-identical between B/Q paths; all 70 local input/output signatures and diagonal declarations checked.',
            'Raw/centered signed net, complete component Gram, mean-bias energies, identity carry and all four block0 exact-zero cells checked.',
            'No full model weights, quantizer or GPU kernel re-execution; recorded runtime counters corroborate the frozen source contract.'],
        limitations=['The source-screen concerns five fixed depths and two diagnostic states, not every layer or unseen prompts.',
            'Echo includes the full implemented quantized-block input dependence, including W/A effects, nonlinearities and BF16 arithmetic; it is not A4 attribution.',
            'Removing echo in the energy formula is a mathematical counterfactual, not a deployable correction or an endpoint/suffix intervention.',
            '20% and two repeated depths are investment gates, not significance or impossibility statements.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', seconds=result['seconds'], gate=gate,
        output=str(output), sha256=io.sha(output)), indent=2))


if __name__ == '__main__':
    main()
