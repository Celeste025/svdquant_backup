#!/usr/bin/env python3
"""Independent E066 saved-sample arithmetic and full-row aggregation; zero CUDA.

Unsaved full-M output channels and native kernels are not recomputed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'
import numpy as np
import torch
import check_h3_swiglu_interaction as io

ROOT = Path(__file__).resolve().parents[2]
NAMES = tuple(f'blocks.{block}.{kind}' for block in range(50)
              for kind in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2'))
ARMS = ('restart', 'carry')
DOMAINS = ('video', 'audio', 'text', 'joint')
EXPECTED_KEYS = ('e010_p030_s05/source_teacher', 'e010_p030_s05/next_teacher',
                 'e010_p036_s14/source_teacher', 'e010_p036_s14/next_teacher')


def unswizzle_bytes(scales, rows, columns):
    rp, cp = (rows+127)//128*128, (columns+3)//4*4
    raw = np.frombuffer(io.raw(scales), dtype=np.uint8)
    return (raw.reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
        .reshape(rp//128, cp//4, 128, 4).transpose(0, 2, 1, 3)
        .reshape(rp, cp)[:rows, :columns])


def ratios_summary(rows, domain):
    values = [r[domain]['carry_over_restart'] for r in rows]
    defined = [v for v in values if v is not None]
    return dict(count=len(rows), defined_ratios=len(defined),
        carry_better=sum(r[domain]['carry_sse'] < r[domain]['restart_sse'] for r in rows),
        carry_worse=sum(r[domain]['carry_sse'] > r[domain]['restart_sse'] for r in rows),
        equal=sum(r[domain]['carry_sse'] == r[domain]['restart_sse'] for r in rows),
        minimum=min(defined) if defined else None,
        median=statistics.median(defined) if defined else None,
        maximum=max(defined) if defined else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=ROOT/'results/research/E066')
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    ep = args.reports/'evaluate.json'
    destination = args.reports/'independent.json'
    assert not destination.exists()
    result = json.loads(ep.read_text())
    assert result['experiment'] == 'E066' and result['status'] == 'complete'
    assert result['complete_dit_calls'] == result['attempted_dit_calls'] == 4
    assert result['local_native_calls'] == result['attempted_local_native_calls'] == 3200
    assert result['full_native_calls'] == result['subset_native_calls'] == 1600
    assert result['activation_packs'] == 800
    assert len(result['cases']) == 4
    assert tuple(row['key'] for row in result['cases']) == EXPECTED_KEYS
    files, json_files, export_files, cached_references = {}, {}, {}, {}
    maximum_sample, maximum_aggregate = 0., 0.

    def verify(record):
        path = Path(record['file'])
        assert path.stat().st_size == record['bytes'] and io.sha(path) == record['sha256'], path
        return path

    def read_json(record):
        path = verify(record)
        json_files[str(path)] = record
        return json.loads(path.read_text())

    def load_reference(record):
        key = record['file']
        if key not in cached_references:
            cached_references[key] = io.load(record, files)
        return cached_references[key]

    def compare(actual, expected, *, sample=False):
        nonlocal maximum_sample, maximum_aggregate
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a-b)/np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300))) if a.size else 0.
        assert error < 1e-10, error
        if sample:
            maximum_sample = max(maximum_sample, error)
        else:
            maximum_aggregate = max(maximum_aggregate, error)

    def summarize(vectors, indices):
        teacher, restart, carry = [float(np.sum(vectors[name][indices], dtype=np.float64))
                                   for name in ('teacher', 'restart', 'carry')]
        return dict(rows=len(indices), teacher_energy=teacher, restart_sse=restart,
                    carry_sse=carry, carry_over_restart=carry/restart if restart else None)

    def check_metrics(actual, expected):
        assert set(actual) == set(expected) == set(DOMAINS)
        for domain in DOMAINS:
            assert actual[domain]['rows'] == expected[domain]['rows']
            for field in ('teacher_energy', 'restart_sse', 'carry_sse', 'carry_over_restart'):
                a, b = actual[domain][field], expected[domain][field]
                if a is None:
                    assert b is None
                else:
                    compare(a, b)

    for path, record in result['sources'].items():
        assert Path(path).resolve() == Path(record['file']).resolve()
        verify(record)
    parent = read_json(result['source_evaluation'])
    assert parent['experiment'] == 'E065b' and parent['status'] == 'complete'
    parent_checks = {name: read_json(record) for name, record in result['source_checks'].items()}
    assert set(parent_checks) == {'outputs', 'selection'}
    assert all(d['status'] == 'complete' and d['cuda_initialized'] is False for d in parent_checks.values())
    assert all(d['evaluation_sha256'] == result['source_evaluation']['sha256'] for d in parent_checks.values())
    assert parent_checks['selection']['final_manifest_selection_bindings'] == 400
    assert read_json(result['source_process_exit'])['exit_code'] == 0
    checked = read_json(result['cpu_check_reference'])
    assert checked['status'] == 'complete' and checked['cuda_initialized'] is False
    assert result['export_manifests'] == parent['exports']
    manifest_rows = {}
    for arm in ARMS:
        manifest = read_json(result['export_manifests'][arm])
        assert manifest['status'] == 'complete' and manifest['arm'] == arm
        assert tuple(row['name'] for row in manifest['layers']) == NAMES
        manifest_rows[arm] = {row['name']: row for row in manifest['layers']}
        assert set(result['selected_exports'][arm]) == set(NAMES)
        for name in NAMES:
            record = result['selected_exports'][arm][name]
            row = manifest_rows[arm][name]
            expected_path = (Path(result['export_manifests'][arm]['file']).parent/row['file']).resolve()
            assert Path(record['file']).resolve() == expected_path
            assert record['sha256'] == row['file_sha256'] and record['bytes'] == row['file_bytes']
            verify(record)
            export_files[str(expected_path)] = record

    summaries, sample_rows_checked = [], 0
    for case in result['cases']:
        key = case['key']
        payload = io.load(case['artifact'], files)
        actual = payload['actual_dit_inputs']
        assert io.signature(actual) == case['actual_dit_input_signature'] == checked['inputs'][key]['actual_dit_input_signature']
        references = parent['references'][key]
        assert payload['reference_artifacts'] == references
        concrete = load_reference(references['legacy_selected']['artifact'])
        assert io.byte_equal(actual, concrete['actual_dit_inputs'])
        teacher = load_reference(references['bf16']['artifact'])
        for field in ('raw_outputs', 'velocities'):
            assert io.byte_equal(payload[field], teacher[field])
            assert io.signature(payload[field]) == case[field] == references['bf16'][field]
        assert case['historical_replay_exact']
        audit = case['runtime_audit']
        assert audit['scaled_mm_calls'] == 800 and audit['sdpa_calls'] == 102 and audit['disk_loads'] == 0
        assert case['fastpack_checks']['checked_calls'] == 200 and case['fastpack_checks']['invalid_calls'] == 0
        kw = actual['kwargs']
        full_rows = kw['x'].shape[1]
        domains = {name: kw[field]['position_ids'].reshape(-1).long().numpy()
                   for name, field in (('video', 'img_pos_info'), ('audio', 'audio_pos_info'), ('text', 'text_pos_info'))}
        domains['joint'] = np.sort(np.concatenate(list(domains.values())))
        assert len(np.unique(domains['joint'])) == len(domains['joint'])
        assert domains['joint'].min() >= 0 and domains['joint'].max() < full_rows
        selected = domains['joint'][np.linspace(0, len(domains['joint'])-1, 512, dtype=np.float64).astype(np.int64)]
        assert len(np.unique(selected)) == 512
        assert np.array_equal(payload['sample_indices'].numpy(), selected)
        for domain in DOMAINS:
            assert np.array_equal(payload['modality_indices'][domain].numpy(), domains[domain])
            assert checked['row_checks'][key]['modality_indices'][domain] == domains[domain].tolist()
        assert checked['row_checks'][key]['sample_indices'] == selected.tolist()
        assert checked['row_checks'][key]['full_rows'] == full_rows
        assert set(case['layers']) == set(NAMES)
        layer_summaries = []
        for name in NAMES:
            row = case['layers'][name]
            data = io.load(row['artifact'], files)
            assert data['layer'] == name and data['case_id'] == case['case_id'] and data['position'] == case['position']
            assert data['full_rows'] == full_rows and data['packet_subset_bytes_verified']
            assert np.array_equal(data['sample_indices'].numpy(), selected)
            for domain in DOMAINS:
                assert np.array_equal(data['modality_indices'][domain].numpy(), domains[domain])
            xs, target = data['xs'], data['target']
            assert xs.dtype == target.dtype == torch.bfloat16 and xs.shape[0] == target.shape[0] == 512
            assert target.shape[1] == data['output_features']
            assert io.signature(xs) == data['source_signature'] and io.signature(target) == data['target_signature']
            packet = data['packet']
            assert packet['shape'] == list(xs.shape) and data['full_packet_shape'] == [full_rows, xs.shape[1]]
            assert io.signature(packet['global_scale']) == data['full_packet_global_signature']
            logical = unswizzle_bytes(packet['scales'], 512, xs.shape[1]//16)
            assert np.array_equal(logical, np.frombuffer(io.raw(data['logical_scale_rows']), dtype=np.uint8).reshape(logical.shape))
            vectors = {}
            for field in ('teacher', 'restart', 'carry'):
                value = data['row_energy'][field]
                assert value.dtype == torch.float64 and value.shape == (full_rows,)
                array = io.array(value)
                assert np.isfinite(array).all() and (array >= 0).all()
                vectors[field] = array
            sample_vectors = {'teacher': np.sum(io.array(target)**2, axis=1, dtype=np.float64)}
            subset_vectors = {'teacher': sample_vectors['teacher']}
            compare(sample_vectors['teacher'], vectors['teacher'][selected], sample=True)
            for arm in ARMS:
                assert data['export_files'][arm] == result['selected_exports'][arm][name]
                row_m = manifest_rows[arm][name]
                assert row_m['shape'] == [target.shape[1], xs.shape[1]]
                assert data['smooth_signature'] == row_m['smooth']
                output = data['outputs'][arm]
                assert output.dtype == torch.bfloat16 and output.shape == target.shape
                assert io.signature(output) == data['output_signatures'][arm]
                delta = io.array(output)-io.array(target)
                assert np.isfinite(delta).all()
                sample_vectors[arm] = np.sum(delta*delta, axis=1, dtype=np.float64)
                compare(sample_vectors[arm], vectors[arm][selected], sample=True)
                subset_output = data['subset_outputs'][arm]
                assert subset_output.dtype == torch.bfloat16 and subset_output.shape == target.shape
                assert io.signature(subset_output) == data['subset_output_signatures'][arm]
                subset_delta = io.array(subset_output)-io.array(target)
                assert np.isfinite(subset_delta).all()
                subset_vectors[arm] = np.sum(subset_delta*subset_delta, axis=1, dtype=np.float64)
            full_metrics = {domain: summarize(vectors, domains[domain]) for domain in DOMAINS}
            sample_metrics = {domain: summarize(sample_vectors, np.flatnonzero(np.isin(selected, domains[domain])))
                              for domain in DOMAINS}
            subset_metrics = {domain: summarize(subset_vectors, np.flatnonzero(np.isin(selected, domains[domain])))
                              for domain in DOMAINS}
            for expected in (row['metrics'], data['metrics']):
                check_metrics(full_metrics, expected)
            for expected in (row['sample_metrics'], data['sample_metrics']):
                check_metrics(sample_metrics, expected)
            for expected in (row['subset_metrics'], data['subset_metrics']):
                check_metrics(subset_metrics, expected)
            layer_summaries.append(dict(layer=name, full=full_metrics, sample=sample_metrics, subset=subset_metrics))
            sample_rows_checked += 512
        summaries.append(dict(key=key, layers=layer_summaries,
            full_by_domain={domain: ratios_summary([row['full'] for row in layer_summaries], domain) for domain in DOMAINS},
            sample_by_domain={domain: ratios_summary([row['sample'] for row in layer_summaries], domain) for domain in DOMAINS},
            subset_by_domain={domain: ratios_summary([row['subset'] for row in layer_summaries], domain) for domain in DOMAINS},
            full_by_type={kind: {domain: ratios_summary([row['full'] for row in layer_summaries if row['layer'].endswith(kind)], domain)
                                for domain in DOMAINS} for kind in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')}))
        print(f'E066 independent complete {key}', flush=True)
    assert sample_rows_checked == 4*200*512 and not torch.cuda.is_initialized()
    final = dict(experiment='E066', status='complete', seconds=time.monotonic()-started,
        cuda_initialized=False, new_model_forwards=0, script_sha256=io.sha(__file__),
        evaluation_sha256=io.sha(ep), cases=summaries, pt_files=files, json_files=json_files, export_files=export_files,
        verified_pt_count=len(files), verified_export_count=len(export_files),
        sample_rows_checked=sample_rows_checked, sample_output_tensors_checked=4*200*4,
        maximum_sample_relative_difference=maximum_sample,
        maximum_aggregate_relative_difference=maximum_aggregate,
        scope='Historical BF16 input/raw/velocity identity; full saved per-row energy aggregation; independent channelwise arithmetic on saved full-M samples and actual M512 outputs for both arms, 512 rows per layer/state; selected-export binding',
        limitations=['Unsaved full-M output channels and native kernels were not independently recomputed.',
                     'No video quality, generalization, or causal-composition claim.'])
    with destination.open('x') as stream:
        json.dump(final, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in final.items() if k not in ('cases', 'pt_files', 'json_files', 'export_files')}, indent=2))
    print(json.dumps([{k: v for k, v in row.items() if k != 'layers'} for row in summaries], indent=2))


if __name__ == '__main__':
    main()
