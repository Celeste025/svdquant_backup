#!/usr/bin/env python3
"""E065 saved-candidate selection, recurrence witnesses, and local SSE audit.

Independent CPU/NumPy arithmetic. Does not import the experiment runner or
reexecute SVD/native kernels; full candidate matrices are bound by hashes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
import torch
import check_h3_swiglu_interaction as io

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E065'
ARMS = ('restart', 'carry')
EXPECTED_NAMES = tuple(f'blocks.{block}.{suffix}' for block in range(50)
    for suffix in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2'))


def bind_manifest_row(manifest_path, row, name, history, source_weight_sha256):
    """Bind the installed manifest row to the independently checked selection.

    Continuation prefixes use absolute paths; newly exported layers use paths
    relative to the arm's manifest directory, as the frozen installer does.
    File contents are hashed once by the existing selected-export load below.
    """
    assert row['name'] == name
    selected = history['export']
    installed_path = (Path(manifest_path).parent/row['file']).resolve()
    assert installed_path == Path(selected['file']).resolve(), (name, installed_path, selected['file'])
    assert row['file_sha256'] == selected['sha256'] and row['file_bytes'] == selected['bytes']
    assert row['selected_k'] == history['selected_k']
    assert row['selected_score'] == history['selected_score']
    best = history['candidates'][history['selected_k']]
    assert row['source_weight_sha256'] == source_weight_sha256
    assert row['shape'] == best['packet']['original_shape']
    assert row['lr_a'] == best['lr_a'] and row['lr_b'] == best['lr_b']
    assert row['selected_decoded_q'] == best['decoded_q'] and row['selected_packet'] == best['packet']
    assert row['selected_local_outputs'] == history['selected_local_outputs']


def unswizzle_bytes(scales, rows, columns):
    rp, cp = (rows+127)//128*128, (columns+3)//4*4
    raw = np.frombuffer(io.raw(scales), dtype=np.uint8)
    return (raw.reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
        .reshape(rp//128, cp//4, 128, 4).transpose(0, 2, 1, 3)
        .reshape(rp, cp)[:rows, :columns])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=RD)
    args = parser.parse_args()
    reports = args.reports
    started = time.monotonic()
    torch.set_num_threads(2)
    dest = reports/'independent_selection.json'
    assert not dest.exists()
    ep = reports/'evaluate.json'
    result = json.loads(ep.read_text())
    assert result['status'] == 'complete'
    if result['experiment'] == 'E065b':
        n = result['reused_prefix_layers']
        assert 0 <= n < 200 and result['candidate_native_calls'] == (200-n)*128
        assert result['completed_candidate_native_calls'] == 25600
        assert result['calibration_activation_packs'] == result['shape_native_calls'] == result['calibration_bf16_calls'] == 0
        assert result['reused_calibration_activation_packs'] == 1600 and result['reused_shape_native_calls'] == 8
        assert result['reused_calibration_bf16_calls'] == 8
        source = result['source_report']
        assert io.sha(source['file']) == source['sha256']
        previous = json.loads(Path(source['file']).read_text())
        assert len(previous['search_layers']) == n and previous['complete_dit_calls'] == 9
        assert result['search_layers'][:n] == previous['search_layers']
    else:
        assert result['experiment'] == 'E065'
        assert result['candidate_native_calls'] == 25600
        assert result['calibration_activation_packs'] == 1600 and result['shape_native_calls'] == 8
        assert result['calibration_bf16_calls'] == 8
    assert len(result['calibration_cases']) == 8
    names = [entry['layer'] for entry in result['search_layers']]
    assert len(names) == len(set(names)) == 200 and tuple(names) == EXPECTED_NAMES
    expected_cases = [f'p{p}_s{s:02d}' for p in (1, 20, 46, 105) for s in (3, 14)]
    assert [c['case_id'] for c in result['calibration_cases']] == expected_cases
    files, json_files, summaries = {}, {}, []
    maximum = 0.

    def read_json(record):
        path = Path(record['file'])
        assert path.stat().st_size == record['bytes'] and io.sha(path) == record['sha256']
        json_files[str(path)] = record
        return json.loads(path.read_text())

    def compare(a, b):
        nonlocal maximum
        error = abs(a-b)/max(abs(a), abs(b), 1e-300)
        assert error < 1e-10, (a, b, error)
        maximum = max(maximum, error)

    assert set(result['exports']) == set(ARMS)
    manifests = {}
    for arm in ARMS:
        manifest = read_json(result['exports'][arm])
        assert manifest['status'] == 'complete' and manifest['arm'] == arm
        assert manifest['target_count'] == manifest['exact_roundtrip_count'] == 200
        manifest_names = [row['name'] for row in manifest['layers']]
        assert len(manifest_names) == len(set(manifest_names)) == 200
        assert tuple(manifest_names) == EXPECTED_NAMES
        # search_contract is the scope field emitted by both frozen assemblers.
        assert manifest['search_contract'] == result['search_contract']
        assert manifest['legacy_manifest'] == result['legacy_manifest'] and manifest['state'] == result['state']
        assert manifest['sources'] == result['sources']
        if 'scope' in manifest:
            assert manifest['scope'] == result['scope']
        manifests[arm] = manifest

    checked = read_json(result['cpu_check_reference'])
    assert checked['status'] == 'complete' and checked['cuda_initialized'] is False
    for case in result['calibration_cases']:
        contract = checked['calibration_checks'][case['case_id']]
        full = io.load(case['artifact'], files)
        assert full['actual_dit_input_signature'] == contract['actual_dit_input_signature']
        assert io.signature(full['raw_outputs']) == case['raw_outputs']
        assert full['selection']['indices'].tolist() == contract['selected_indices']
        assert full['selection']['modality_counts'] == contract['modality_counts']
        assert case['runtime_audit']['sdpa_calls'] == contract['expected_sdpa_calls']
        assert case['runtime_audit']['disk_loads'] == 0
        assert case['fastpack_checks']['checked_calls'] == 200

    for index, entry in enumerate(result['search_layers']):
        info = read_json(entry['metadata'])
        name = entry['layer']
        assert info['layer'] == name and info['candidate_zero_exact']
        samples = []
        for case in result['calibration_cases']:
            sample = io.load(case['layers'][name]['artifact'], files)
            assert sample['layer'] == name and sample['case_id'] == case['case_id']
            assert io.signature(sample['xs']) == sample['source_signature']
            assert io.signature(sample['target']) == sample['target_signature']
            assert sample['xs'].dtype == sample['target'].dtype == torch.bfloat16
            assert sample['xs'].shape[0] == sample['target'].shape[0] == 512
            assert np.unique(sample['indices'].numpy()).size == 512
            assert sum(sample['modality_counts'].values()) == 512
            contract = checked['calibration_checks'][case['case_id']]
            assert sample['indices'].tolist() == contract['selected_indices']
            assert sample['modality_counts'] == contract['modality_counts']
            packet = sample['packet']
            assert packet['shape'] == list(sample['xs'].shape)
            assert io.signature(packet['global_scale']) == sample['full_packet_global_signature']
            logical = unswizzle_bytes(packet['scales'], 512, sample['xs'].shape[-1]//16)
            expected = np.frombuffer(io.raw(sample['logical_scale_rows']), dtype=np.uint8).reshape(logical.shape)
            assert np.array_equal(logical, expected)
            samples.append(sample)
        first = None
        selections = {}
        for arm in ARMS:
            history = info['arms'][arm]
            manifest = manifests[arm]
            installed_row = manifest['layers'][index]
            bind_manifest_row(result['exports'][arm]['file'], installed_row, name,
                history, info['source_weight_sha256'])
            candidates = history['candidates']
            assert len(candidates) == 8
            previous = info['q0']
            for k, candidate in enumerate(candidates):
                assert candidate['k'] == k
                assert candidate['seed'] == 310000+1000*(index//4)+50*(index%4)+k
                assert candidate['roundtrip_exact'] and candidate['case_ids'] == expected_cases
                assert len(candidate['per_case_sse']) == 8 and all(np.isfinite(candidate['per_case_sse']))
                compare(sum(candidate['per_case_sse']), candidate['score'])
                assert candidate['parent_q'] == (info['q0'] if arm == 'restart' else previous)
                previous = candidate['actual_q']
                if k == 0:
                    fields = ('seed', 'parent_q', 'target', 'lr_a', 'lr_b', 'residual',
                        'decoded_q', 'actual_q', 'packet', 'per_case_sse', 'score', 'output_signatures')
                    common = {f: candidate[f] for f in fields}
                    if first is None:
                        first = common
                    else:
                        assert common == first
            best_k = min(range(8), key=lambda k: candidates[k]['score'])
            assert history['selected_k'] == best_k
            best = candidates[best_k]
            compare(history['selected_score'], best['score'])
            selected = io.load(history['selected_local_outputs'], files)
            assert selected['layer'] == name and selected['arm'] == arm and selected['selected_k'] == best_k
            assert selected['case_ids'] == expected_cases
            assert [io.signature(v) for v in selected['outputs']] == best['output_signatures']
            scores = []
            for output, sample in zip(selected['outputs'], samples, strict=True):
                delta = io.array(output)-io.array(sample['target'])
                assert np.isfinite(delta).all()
                scores.append(float(np.sum(delta*delta, dtype=np.float64)))
            for score, recorded in zip(scores, best['per_case_sse'], strict=True):
                compare(score, recorded)
            compare(sum(scores), best['score'])
            exported = io.load(history['export'], files)
            assert exported['name'] == name
            assert exported['format'] == manifest['format'] and exported['recipe'] == manifest['recipe']
            assert exported['shape'] == installed_row['shape']
            tensors = exported['tensors']
            assert io.signature(tensors['lr_a']) == best['lr_a'] and io.signature(tensors['lr_b']) == best['lr_b']
            assert io.signature(tensors['smooth']) == samples[0]['smooth_signature']
            assert io.signature(tensors['smooth']) == installed_row['smooth']
            assert io.signature(dict(codes=tensors['weight_packed'], scales=tensors['weight_scales_swizzled'],
                global_scale=tensors['weight_global'], original_shape=exported['shape'])) == best['packet']
            selections[arm] = dict(k=best_k, score=best['score'], first_score=candidates[0]['score'],
                selected_over_first=best['score']/candidates[0]['score'])
        witness = io.load(info['recurrence_witness'], files)
        assert witness['layer'] == name and len(witness['candidates']) == 16
        previous = {}
        for value in witness['candidates']:
            arm, k = value['arm'], value['k']
            assert arm in ARMS and 0 <= k < 8
            ws, parent, target, lowrank, residual, decoded = [io.array(value[f]).astype(np.float32)
                for f in ('ws', 'parent_q', 'target', 'lowrank', 'residual', 'decoded_q')]
            assert all(a.shape == (8, 8) for a in (ws, parent, target, lowrank, residual, decoded))
            assert np.array_equal(io.bf16_round(ws-parent), target)
            assert np.array_equal(io.bf16_round(ws-lowrank), residual)
            if k > 0 and arm == 'carry':
                assert np.array_equal(parent, previous[arm])
            previous[arm] = io.array(value['actual_q']).astype(np.float32)
        summaries.append(dict(layer=name, selections=selections,
            carry_over_restart=selections['carry']['score']/selections['restart']['score']))
        if (index+1) % 25 == 0:
            print(f'E065 independent selection {index+1}/200', flush=True)
    controls = []
    shape_controls = result['reused_selection_shape_controls'] if result['experiment'] == 'E065b' else result['selection_shape_controls']
    assert len(shape_controls) == 4
    for row in shape_controls:
        control = io.load(row['artifact'], files)
        full, subset, teacher = [io.array(control[k]) for k in ('full_output_rows', 'subset_output', 'teacher_output')]
        diff, quant = subset-full, full-teacher
        sse, quant_sse = float(np.sum(diff*diff)), float(np.sum(quant*quant))
        compare(sse, row['tile_difference_sse'])
        compare(quant_sse, row['full_native_quantization_sse'])
        assert io.byte_equal(control['full_output_rows'], control['subset_output']) == row['bitwise']
        controls.append(dict(layer=row['layer'], tile_sse=sse, quant_sse=quant_sse,
            tile_over_quant_sse=sse/quant_sse if quant_sse else None))
    assert not torch.cuda.is_initialized()
    final = dict(experiment=result['experiment'], status='complete', seconds=time.monotonic()-started,
        cuda_initialized=False, new_model_forwards=0, script_sha256=io.sha(__file__), evaluation_sha256=io.sha(ep),
        maximum_score_relative_difference=maximum, layers=summaries, shape_controls=controls,
        pt_files=files, json_files=json_files, verified_pt_count=len(files),
        carry_better_local_count=sum(r['carry_over_restart'] < 1 for r in summaries),
        candidate_zero_exact_layers=200, verified_candidate_records=3200,
        final_manifest_selection_bindings=400,
        scope='All selected local SSEs, argmins, first-candidate pairs, final installation manifests bound to selected exports, export factors/packet bytes and stored 8x8 recurrence witnesses',
        limitation='No SVD/native reexecution; complete parent/target matrices bound by recorded hashes, not independently recomputed; no full-M packet recapture. Argmin uses all recorded candidate scores, but independent SSE recomputation covers only selected candidates; unselected output tensors were not retained.')
    with dest.open('x') as stream:
        json.dump(final, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in final.items() if k not in ('layers', 'pt_files', 'json_files')}, indent=2))


if __name__ == '__main__':
    main()
