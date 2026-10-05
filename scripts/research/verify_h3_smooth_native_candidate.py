#!/usr/bin/env python3
"""E071 saved-artifact CPU verification; no runner/parser/model/SVD replay."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E071'
CASES, ROWS = ('p1_s03', 'p20_s03'), (22400, 22464)
NAME, M, K, RANK = 'blocks.0.attn.qkv_proj', 21504, 5376, 32
MODEL = Path('/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors')
WEIGHT_SHA = '1727c595c15e1c57f16832e961946975d1a6073f445e7b2e6ccd489554eac264'


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(file=str(path), bytes=path.stat().st_size, sha256=sha(path))


def raw(tensor):
    require(tensor.device.type == 'cpu', 'Non-CPU artifact tensor')
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy()


def tensor_record(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(memoryview(raw(tensor))).hexdigest())


def signature(value):
    if isinstance(value, torch.Tensor):
        return tensor_record(value)
    if isinstance(value, dict):
        return {k: signature(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [signature(v) for v in value]
    return value


def exact(left, right, label):
    require(left.dtype == right.dtype and left.shape == right.shape, f'Layout mismatch: {label}')
    a, b = raw(left), raw(right)
    for offset in range(0, a.size, 8 << 20):
        require(np.array_equal(a[offset:offset+(8 << 20)], b[offset:offset+(8 << 20)]), f'Byte mismatch: {label}')


def bf32(tensor):
    require(tensor.dtype == torch.bfloat16, 'Expected BF16 artifact')
    return (tensor.contiguous().view(torch.uint16).numpy().astype(np.uint32) << 16).view(np.float32)


def bf_bits(value):
    value = np.asarray(value, dtype=np.float32)
    require(np.isfinite(value).all(), 'Nonfinite BF16 conversion input')
    bits = value.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def bf_cast_exact(value, actual, label):
    require(actual.dtype == torch.bfloat16 and tuple(actual.shape) == value.shape, f'BF16 cast shape: {label}')
    require(np.array_equal(bf_bits(value), actual.contiguous().view(torch.uint16).numpy()), f'BF16 cast mismatch: {label}')


def tensor(value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and tuple(value.shape) == tuple(shape) and
            value.dtype == dtype, f'Tensor contract mismatch: {label}')


def close(actual, expected, tolerance, label):
    require(np.isfinite(actual) and np.isfinite(expected), f'Nonfinite scalar: {label}')
    error = abs(float(actual)-float(expected))/max(abs(float(expected)), 1e-300)
    require(error <= tolerance, f'{label} relative error {error} exceeds {tolerance}')
    return error


def scores(candidate, baseline):
    tensor(candidate, baseline.shape, torch.bfloat16, 'candidate attention')
    library, fp64 = 0.0, 0.0
    for offset in range(0, candidate.shape[0], 256):
        c, b = bf32(candidate[offset:offset+256]), bf32(baseline[offset:offset+256])
        diff = c-b
        rounded = (bf_bits(diff).astype(np.uint32) << 16).view(np.float32)
        library += float(np.sum(rounded*rounded, dtype=np.float64))
        d64 = c.astype(np.float64)-b.astype(np.float64)
        fp64 += float(np.sum(d64*d64, dtype=np.float64))
    return dict(library_pointwise_fp32_squares_fp64_sum=library, fp64_difference_sse=fp64)


def packet_check(packet, source, decoded=None):
    """NumPy encoding/decoding witness, never calls the GPU packer or GEMM."""
    rows, width = source.shape
    require(tuple(packet['original_shape']) == (rows, width), 'Packet original shape mismatch')
    tensor(packet['packed'], (rows, width//2), torch.uint8, 'packed codes')
    rp, cp = (rows+127)//128*128, (width//16+3)//4*4
    storage = packet['swizzled_scales']
    tensor(storage, (rp*cp,), torch.float8_e4m3fn, 'swizzled SF')
    sf_bits = storage.view(torch.uint8).numpy().reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
    sf_bits = sf_bits.reshape(rp//128, cp//4, 128, 4).transpose(0, 2, 1, 3).reshape(rp, cp)
    require(np.all(sf_bits[rows:] == 0) and np.all(sf_bits[:, width//16:] == 0), 'Nonzero SF padding')
    sf_bits = sf_bits[:rows, :width//16]
    global_value = packet['global_scale'].numpy()
    require(packet['global_scale'].dtype == torch.float32 and global_value.size == 1 and
            np.isfinite(global_value).all() and global_value.item() > 0, 'Invalid packet global scale')
    maximum = np.float32(0)
    for offset in range(0, rows, 256):
        x = bf32(source[offset:offset+256])
        require(np.isfinite(x).all(), 'Nonfinite packet source')
        maximum = np.maximum(maximum, np.abs(x).max())
    # Frozen CUDA packers implement Python-scalar division as FP32 reciprocal multiply.
    expected_global = np.maximum(np.maximum(maximum, np.float32(1e-12))*np.float32(1.0/2688.0), np.float32(1e-12))
    require(np.array_equal(global_value.reshape(-1).view(np.uint32), np.array([expected_global], dtype=np.float32).view(np.uint32)),
            'Full-tensor global scale differs')
    codes = np.arange(127, dtype=np.uint32)
    exponent, mantissa = codes >> 3, codes & 7
    fp8_lut = np.where(exponent == 0, mantissa.astype(np.float64)*2**-9,
                      (1+mantissa.astype(np.float64)/8)*np.exp2(exponent.astype(np.float64)-7)).astype(np.float32)
    sf_midpoints = (fp8_lut[:-1]+fp8_lut[1:])/np.float32(2)
    levels = np.array([-6,-4,-3,-2,-1.5,-1,-.5,0,.5,1,1.5,2,3,4,6], dtype=np.float32)
    midpoints = (levels[:-1]+levels[1:])/np.float32(2)
    nibble_map = np.array([15,14,13,12,11,10,9,0,1,2,3,4,5,6,7], dtype=np.uint8)
    signed_lut = np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], dtype=np.float32)
    packed = packet['packed'].numpy()
    for offset in range(0, rows, 128):
        end = min(rows, offset+128)
        x = bf32(source[offset:end]).reshape(end-offset, width//16, 16)
        ideal = np.maximum(np.abs(x).max(axis=2)*np.float32(1.0/6.0), np.float32(1e-12))/expected_global
        ideal = np.maximum(ideal, np.float32(np.finfo(np.float32).tiny))
        sf = np.searchsorted(sf_midpoints, ideal, side='left')
        tie = (sf < 126) & (ideal == sf_midpoints[np.minimum(sf, 125)])
        sf = (sf + (tie & ((sf & 1) == 1))).astype(np.uint8)
        require(np.array_equal(sf, sf_bits[offset:end]), 'E4M3 scale encoding differs')
        effective = fp8_lut[sf]*expected_global
        z = x/np.where(effective == 0, np.float32(1), effective)[..., None]
        z = np.where(effective[..., None] == 0, np.float32(0), z)
        nibbles = nibble_map[np.searchsorted(midpoints, z, side='left')]
        nibbles = np.where((nibbles == 0) & (z < 0), np.uint8(8), nibbles).reshape(end-offset, width)
        expected = nibbles[:, ::2] | (nibbles[:, 1::2] << 4)
        require(np.array_equal(expected, packed[offset:end]), 'Signed-lower E2M1 code encoding differs')
        if decoded is not None:
            reconstructed = signed_lut[nibbles]*np.repeat(effective, 16, axis=1)
            bf_cast_exact(reconstructed, decoded[offset:end], 'weight packet decoded BF16')
    return dict(shape=[rows, width], encoded_codes_and_scales_exact=True,
                global_full_tensor_exact=True, zero_padding_exact=True, decoded_bf16_exact=decoded is not None)


def candidate_math(candidate, export):
    """Elementwise source/scale/residual checks and saved-top32 SVD equations."""
    constants = {}
    for node in ast.parse((ROOT/'scripts/research/h3_native_nvfp4.py').read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('FORMAT', 'RECIPE'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    require(export['format'] == constants['FORMAT'] and export['recipe'] == constants['RECIPE'] and
            export['name'] == NAME and export['shape'] == [M, K], 'Frozen native export contract differs')
    e = export['tensors']
    require(set(e) == {'weight_packed', 'weight_scales_swizzled', 'weight_global', 'smooth', 'lr_a', 'lr_b', 'bias'} and
            e['bias'] is None, 'Export tensor/bias scope differs')
    for name in ('ws_bf16', 'lr_product_bf16', 'residual_bf16', 'decoded_q_bf16'):
        tensor(candidate[name], (M, K), torch.bfloat16, name)
    tensor(candidate['raw_scale_fp32'], (K,), torch.float32, 'raw ask scale')
    tensor(candidate['deploy_scale_bf16'], (K,), torch.bfloat16, 'deployment scale')
    raw_scale = candidate['raw_scale_fp32'].numpy()
    require(np.isfinite(raw_scale).all() and np.all(raw_scale > 0), 'Invalid raw scale')
    bf_cast_exact(raw_scale, candidate['deploy_scale_bf16'], 'raw FP32 scale to BF16')
    exact(candidate['deploy_scale_bf16'], e['smooth'], 'export scale')
    scale = bf32(e['smooth'])
    require(np.isfinite(scale).all() and np.all(scale > 0) and not np.all(scale == 1), 'Invalid/trivial deployment scale')
    for name in ('x', 'w'):
        tensor(candidate['span'][name], (K,), torch.float32, name+' span')
    with MODEL.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        require(0 < header_size <= 16*1024**2, 'Invalid source header extent')
        header = json.loads(stream.read(header_size))
    entry = header[NAME+'.weight']
    begin, end = entry['data_offsets']
    require(entry['dtype'] == 'BF16' and entry['shape'] == [M, K] and end-begin == 2*M*K and
            0 <= begin < end <= MODEL.stat().st_size-8-header_size, 'Source weight header differs')
    weights = np.memmap(MODEL, dtype='<u2', mode='r', offset=8+header_size+begin, shape=(M, K))
    source_hash, wspan = hashlib.sha256(), np.zeros(K, dtype=np.float32)
    for offset in range(0, M, 128):
        source = np.asarray(weights[offset:offset+128])
        source_hash.update(memoryview(source).cast('B'))
        w = (source.astype(np.uint32) << 16).view(np.float32)
        require(np.isfinite(w).all(), 'Source weight is nonfinite')
        wspan = np.maximum(wspan, np.abs(w).max(axis=0))
        bf_cast_exact(w*scale, candidate['ws_bf16'][offset:offset+128], 'single BF16 weight smoothing')
        ws = bf32(candidate['ws_bf16'][offset:offset+128])
        product = bf32(candidate['lr_product_bf16'][offset:offset+128])
        bf_cast_exact(ws-product, candidate['residual_bf16'][offset:offset+128], 'BF16 residual subtraction')
    del weights
    require(source_hash.hexdigest() == WEIGHT_SHA and
            np.array_equal(wspan.view(np.uint32), candidate['span']['w'].numpy().view(np.uint32)),
            'Selected source weight SHA or full-channel weight absmax differs')
    expected_raw = np.sqrt(candidate['span']['x'].numpy())/np.sqrt(wspan)
    formula_error = float(np.max(np.abs(raw_scale.astype(np.float64)-expected_raw.astype(np.float64))/
                                 np.maximum(np.abs(expected_raw.astype(np.float64)), 1e-300)))
    require(np.isfinite(formula_error) and formula_error <= 1e-6, 'Manual scale formula differs beyond FP32 tolerance')
    svd = candidate['svd']
    tensor(svd['u_top32'], (M, RANK), torch.float64, 'SVD U32')
    tensor(svd['vh_top32'], (RANK, K), torch.float64, 'SVD Vh32')
    tensor(svd['singular_values'], (K,), torch.float64, 'full spectrum')
    u, s, vh = svd['u_top32'].numpy(), svd['singular_values'].numpy(), svd['vh_top32'].numpy()
    require(np.isfinite(u).all() and np.isfinite(s).all() and np.isfinite(vh).all() and
            np.all(s >= 0) and np.all(s[:-1] >= s[1:]), 'Invalid top32/spectrum')
    ortho = dict(u=float(np.linalg.norm(u.T@u-np.eye(RANK))), vh=float(np.linalg.norm(vh@vh.T-np.eye(RANK))))
    require(max(ortho.values()) <= 1e-8, 'Independent top32 Gram gate failed')
    bf_cast_exact(vh, e['lr_a'], 'SVD A cast')
    bf_cast_exact(u*s[:RANK], e['lr_b'], 'SVD B cast')
    rows = np.array([i*(M-1)//15 for i in range(16)], dtype=np.int64)
    cols = np.array([i*(K-1)//15 for i in range(16)], dtype=np.int64)
    row_rhs, column_rhs = u[rows]*s[:RANK], vh[:, cols].T*s[:RANK]
    row_delta = bf32(candidate['ws_bf16'][torch.from_numpy(rows)]).astype(np.float64)@vh.T-row_rhs
    column_delta = bf32(candidate['ws_bf16'][:, torch.from_numpy(cols)]).astype(np.float64).T@u-column_rhs
    equations = dict(rows=float(np.linalg.norm(row_delta)/max(np.linalg.norm(row_rhs), 1e-300)),
                     columns=float(np.linalg.norm(column_delta)/max(np.linalg.norm(column_rhs), 1e-300)))
    require(max(equations.values()) <= 1e-8, 'Independent sampled top32 singular equations failed')
    weight_packet = dict(packed=e['weight_packed'], swizzled_scales=e['weight_scales_swizzled'],
                         global_scale=e['weight_global'], original_shape=(M, K))
    packet = packet_check(weight_packet, candidate['residual_bf16'], candidate['decoded_q_bf16'])
    return dict(source_weight_sha256=source_hash.hexdigest(), whole_checkpoint_rehashed=False,
                single_smoothing_exact=True, residual_subtraction_exact=True,
                bf16_scale_and_svd_factor_casts_exact=True, raw_scale_formula_relative_error=formula_error,
                raw_scale_formula_tolerance=1e-6, top32_gram_frobenius=ortho, sampled_singular_equations=equations,
                weight_packet=packet, lr_product_gemm_independently_repeated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPORTS/'independent.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite independent result')
    run, receipt = read(REPORTS/'run.json'), read(REPORTS/'process_exit.json')
    require(run.get('status') == 'complete' and receipt.get('exit_code') == receipt.get('worker_exit_code') == 0 and
            receipt.get('timeout_exit') is False, 'Wait for complete run and normal supervisor/worker exit0')
    started = time.monotonic()
    result = dict(experiment='E071', status='running', checker=record(__file__), reports={}, files={}, cases=[],
                  new_model_forwards=0, new_svd_calls=0, new_parser_calls=0, cuda_initialized=False)

    def verify(expected):
        path = Path(expected['file']).resolve()
        if str(path) not in result['files']:
            result['files'][str(path)] = record(path)
        actual = result['files'][str(path)]
        require(actual['sha256'] == expected['sha256'] and
                ('bytes' not in expected or actual['bytes'] == expected['bytes']), f'File identity differs: {path}')
        return path
    def report(path):
        row = record(path)
        result['files'][row['file']] = row
        result['reports'][Path(path).name] = row
        return read(path)

    try:
        launch = report(REPORTS/'launch.json')
        receipt = report(REPORTS/'process_exit.json')
        run, check = report(REPORTS/'run.json'), report(REPORTS/'check.json')
        require(all(v['experiment'] == 'E071' for v in (launch, receipt, run, check)) and
                run['status'] == check['status'] == 'complete' and run['phase'] == 'run' and check['phase'] == 'check',
                'Wrong experiment/phase/status')
        for name in ('runner', 'plan', 'cpu_check', 'supervisor'):
            verify(launch[name])
        require(Path(launch['runner']['file']).resolve() == (ROOT/'scripts/research/probe_h3_smooth_native_candidate.py').resolve() and
                Path(launch['output']).resolve() == (REPORTS/'run.json').resolve() and
                launch['cpu_check'] == run['cpu_check'], 'Frozen launch/report differs')
        require(receipt['exit_code'] == receipt['worker_exit_code'] == 0 and receipt['timeout_exit'] is False and
                'memory_stop' not in receipt and 'supervisor_error' not in receipt and
                receipt['launch_sha256'] == result['reports']['launch.json']['sha256'] and
                receipt['completed_output_sha256'] == result['reports']['run.json']['sha256'], 'Exit provenance differs')
        start, deadline = launch['budget_start_unix'], launch['deadline_unix']
        require(0 < deadline-start <= 900 and all(v['budget_start_unix'] == start and v['deadline_unix'] == deadline
                for v in (run, receipt)) and start <= receipt['start_epoch'] <= receipt['end_epoch'] <= deadline,
                'Original deadline differs')
        require(receipt['gpu'] == launch['gpu'] and receipt['gpu_before'] == '0, 0' and
                launch['environment']['CUDA_VISIBLE_DEVICES'] == str(launch['gpu']), 'GPU launch receipt differs')
        require(check['cuda_initialized'] is False and check['new_model_forwards'] == 0 and
                run['cuda_initialized'] is True and run['peak_allocated_gib'] < 60 and
                receipt['sampled_peak_gpu_used_mib'] <= 60*1024 and run['data_bytes'] <= 8*1024**3 and
                run['data_byte_limit'] == 8*1024**3 and 0 < run['seconds_total'] <= 900, 'Budget/preflight differs')
        fields = ('sources', 'plan', 'source_bridge', 'source_independent', 'source_exit', 'case_sources',
                  'source_weight_sha256', 'setup_check', 'setup_manifest', 'library_contract', 'torch')
        require(all(check[k] == run[k] for k in fields), 'CPU/run source or calibration contract differs')
        for path, row in run['sources'].items():
            require(Path(path).resolve() == Path(row['file']).resolve(), 'Source record path differs')
            verify(row)
        require(run['sources'][str(Path(launch['runner']['file']).resolve())] == launch['runner'] and
                run['source_weight_sha256'] == WEIGHT_SHA, 'Frozen runner/weight binding differs')
        parent = read(verify(run['source_bridge']))
        parent_check = read(verify(run['source_independent']))
        parent_exit = read(verify(run['source_exit']))
        require(parent['status'] == parent_check['status'] == 'complete' and
                parent_check['reports']['run_bridge_v2.json'] == run['source_bridge'] and
                parent_check['reports']['process_exit_bridge.json'] == run['source_exit'] and
                parent_exit['exit_code'] == parent_exit['worker_exit_code'] == 0 and
                parent_exit['completed_output_sha256'] == run['source_bridge']['sha256'] and
                run['case_sources'] == parent['prefix_cases'], 'E070 case/independent provenance differs')
        methods = ('calibrate', 'reset', '_reset', 'ask', '_ask', '_calibrate_wgts', 'tell', '_tell')
        require(run['library_calls'] == {name: 1 for name in methods}, 'Actual single-candidate library call counts differ')
        contract = run['library_contract']
        expected = dict(objective='OutputsError', granularity='Layer', strategy='Manual', alpha=.5, beta=.5,
            spans=[['AbsMax', 'AbsMax']], sample_batch_size=1, sample_size=-1, degree=2,
            develop_dtype='torch.float32', population_size=1, num_iters=1,
            w_quantizer=None, x_quantizer=None, y_quantizer=None, needs_quant=False)
        require(all(contract[k] == v for k, v in expected.items()) and
                set(contract['inherited_functions']) == set(methods), 'Real inherited Manual contract differs')
        for name, row in contract['inherited_functions'].items():
            module = 'smooth.py' if name in ('_reset', '_ask', '_tell') else 'search.py'
            require(Path(row['file']).resolve() == (ROOT/'third_party/deepcompressor/deepcompressor/calib'/module).resolve() and
                    str(Path(row['file']).resolve()) in run['sources'], 'Inherited library method source differs')
        counts = dict(attention_calls=4, reload_qkv_calls=2, native_scaled_mm_calls=4, activation_packs=4,
                      sdpa_calls=8, svd_calls=1, prefix_calls=0, complete_dit_calls=0,
                      text_encoder_calls=0, video_vae_calls=0, audio_vae_calls=0)
        require(all(run[k] == v and check[k] == 0 for k, v in counts.items()) and
                run['baseline_case_order'] == run['candidate_case_order'] == [0, 1] and
                run['candidate_installs'] == run['candidate_recovers'] == 1 and run['reloaded_export_tensors_exact'] is True,
                'Runtime counts, case order or export restoration differs')
        recovery = run['recovery']
        require(all(recovery[k] is True for k in ('original_module_same', 'original_storage_identity', 'all_original_hooks_restored')) and
                recovery['original_weight_sha256'] == WEIGHT_SHA and
                recovery['outstanding_state_dict'] == recovery['outstanding_hooks'] == 0, 'Original module restoration failed')

        global np, torch
        import numpy as np
        import torch
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), 'CPU checker initialized CUDA')
        result['environment'] = dict(numpy=np.__version__, torch=torch.__version__, cuda_visible_devices='')
        def load(row):
            return torch.load(verify(row), map_location='cpu', weights_only=True, mmap=True)
        candidate, export = load(run['candidate_artifact']), load(run['export'])
        require(signature(candidate) == run['candidate_tensor_signatures'] and
                tensor_record(candidate['raw_scale_fp32']) == run['best_raw_scale_signature'] and
                tensor_record(candidate['deploy_scale_bf16']) == run['deployment_scale_signature'], 'Saved candidate/get_best binding differs')
        result['candidate_math'] = candidate_math(candidate, export)
        require(run['weight_roundtrip']['numeric_exact'] is True and
                run['weight_roundtrip']['compatibility'] == 'h3_e005_sf0_canonical_zero_numeric_equivalence_v1' and
                run['weight_roundtrip']['signed_zero_byte_equality_required'] is False and
                run['weight_roundtrip']['decoded'] == tensor_record(candidate['decoded_q_bf16']) and
                max(run['svd_gram'].values()) <= 1e-8, 'Runner numerical receipts differ')
        for key in ('case_sources', 'baseline_cases', 'candidate_cases', 'reload_cases'):
            require([r['case_id'] for r in run[key]] == list(CASES), f'Case count/order differs: {key}')
        xspan = np.zeros(K, dtype=np.float32)
        numeric_scores = []
        scale = bf32(export['tensors']['smooth'])
        new_bytes = run['candidate_artifact']['bytes'] + run['export']['bytes'] + run['model_setup']['bytes']
        verify(run['model_setup'])
        for index, case_id in enumerate(CASES):
            source_row, baseline_row = run['case_sources'][index], run['baseline_cases'][index]
            candidate_row, reload_row = run['candidate_cases'][index], run['reload_cases'][index]
            source, current, reloaded = load(source_row['artifact']), load(candidate_row['artifact']), load(reload_row['artifact'])
            new_bytes += candidate_row['artifact']['bytes'] + reload_row['artifact']['bytes']
            source_sig = signature(source['attention_inputs'])
            require(source['case_id'] == current['case_id'] == reloaded['case_id'] == case_id and
                    source_sig == source_row['input_signature'] == baseline_row['attention_input_signature'] ==
                    candidate_row['attention_input_signature'] == current['attention_inputs_signature'] == reloaded['attention_inputs_signature'],
                    'Full source input/kwargs identity differs')
            require(baseline_row['source_artifact'] == candidate_row['source_artifact'] == source_row['artifact'] and
                    baseline_row['byte_exact_e070'] is True and
                    baseline_row['output_signature'] == source_row['output_signature'] == tensor_record(source['output']),
                    'Actual BF16 baseline does not bind complete E070 output')
            x = source['attention_inputs']['args'][0]
            tensor(x, (ROWS[index], K), torch.bfloat16, 'original complete input')
            tensor(current['attention_output'], (ROWS[index], K), torch.bfloat16, 'candidate attention output')
            require(tensor_record(current['attention_output']) == candidate_row['output_signature'], 'Attention output signature differs')
            for row, native, sdpa in ((baseline_row, 0, 2), (candidate_row, 1, 2), (reload_row, 1, 0)):
                audit = row['runtime_audit']
                require(audit['scaled_mm_calls'] == native and audit['sdpa_calls'] == sdpa and audit['disk_loads'] == 0 and
                        row['fastpack_checks']['checked_calls'] == native and
                        row['fastpack_checks']['invalid_calls'] == 0 and
                        row['fastpack_checks']['compatibility'] == 'h3_e005_sf0_canonical_zero_numeric_equivalence_v1',
                        'Per-stage native/attention/pack count differs')
            require(reload_row['byte_exact_candidate'] is True, 'Reload byte-exact receipt absent')
            for name, shape in (('qkv_output', (ROWS[index], M)), ('x_s', (ROWS[index], K)),
                                ('lr_a_output', (ROWS[index], RANK)), ('lr_b_output_sample', (512, M))):
                tensor(current[name], shape, torch.bfloat16, name)
                exact(current[name], reloaded[name], case_id+'.'+name)
                values = current[name].contiguous().view(torch.uint16).numpy()
                for offset in range(0, shape[0], 256):
                    require(np.all((values[offset:offset+256] & np.uint16(0x7f80)) != np.uint16(0x7f80)), 'Nonfinite branch/QKV evidence')
            for name in ('packed', 'swizzled_scales', 'global_scale'):
                exact(current['activation_packet'][name], reloaded['activation_packet'][name], case_id+'.packet.'+name)
            require(current['activation_packet']['original_shape'] == reloaded['activation_packet']['original_shape'], 'Reload packet full shape differs')
            tensor(current['sample_indices'], (512,), torch.int64, 'LR sample indices')
            exact(current['sample_indices'], reloaded['sample_indices'], 'LR sample indices')
            expected_indices = np.array([i*(ROWS[index]-1)//511 for i in range(512)], dtype=np.int64)
            require(np.array_equal(current['sample_indices'].numpy(), expected_indices), 'Preselected LR sample rows differ')
            for value, row in ((current, candidate_row), (reloaded, reload_row)):
                xs_sig = tensor_record(value['x_s'])
                evidence = value['lr_input_evidence']
                require(evidence['a_input_is_pack_source'] is True and evidence['b_input_is_a_output'] is True and
                        evidence['branch_a_calls'] == evidence['branch_b_calls'] == 1 and
                        evidence['input_dtype'] == 'torch.bfloat16' and
                        evidence['pack_source_signature'] == evidence['a_input_signature'] == xs_sig == row['x_s_signature'] and
                        tensor_record(value['qkv_output']) == row['qkv_signature'] and
                        signature(value['activation_packet']) == row['packet_signature'], 'Actual LR/pack source or output evidence differs')
            for offset in range(0, ROWS[index], 128):
                values = bf32(x[offset:offset+128])
                require(np.isfinite(values).all(), 'Nonfinite source activation')
                xspan = np.maximum(xspan, np.abs(values).max(axis=0))
                bf_cast_exact(values/scale, current['x_s'][offset:offset+128], 'actual LR xs from original x/scale')
            pack = packet_check(current['activation_packet'], current['x_s'])
            case_scores = scores(current['attention_output'], source['output'])
            library_error = close(case_scores['library_pointwise_fp32_squares_fp64_sum'], run['scores']['library_per_case'][case_id],
                                  2e-6, 'Library score '+case_id)
            fp64_error = close(case_scores['fp64_difference_sse'], run['scores']['fp64_per_case'][case_id], 1e-10, 'FP64 score '+case_id)
            numeric_scores.append(case_scores)
            result['cases'].append(dict(case_id=case_id, complete_input_signature=source_sig, rows=ROWS[index],
                candidate_reload_qkv_and_evidence_byte_exact=True, actual_xs_elementwise_exact=True,
                activation_packet=pack, independent_scores=case_scores, library_relative_difference=library_error,
                fp64_relative_difference=fp64_error))
            del source, current, reloaded, x
        require(np.array_equal(xspan.view(np.uint32), candidate['span']['x'].numpy().view(np.uint32)), 'Full-case x absmax differs')
        reported = run['scores']
        require(reported['library_best'] == reported['library_reported'] and
                float(np.float32(reported['library_per_case'][CASES[0]])+np.float32(reported['library_per_case'][CASES[1]])) ==
                reported['library_reported'], 'Actual tell/get_best/FP32 sequential score binding differs')
        low_total = sum(v['library_pointwise_fp32_squares_fp64_sum'] for v in numeric_scores)
        fp64_total = sum(v['fp64_difference_sse'] for v in numeric_scores)
        result['score_checks'] = dict(library_cpu_sum=low_total, library_gpu_tell=reported['library_reported'],
            library_relative_difference=close(low_total, reported['library_reported'], 2e-6, 'Total library score'),
            fp64_cpu_sse=fp64_total, fp64_reported=reported['fp64_total'],
            fp64_relative_difference=close(fp64_total, reported['fp64_total'], 1e-10, 'Total FP64 score'),
            library_definition='BF16 pointwise subtraction, FP32 square; CPU FP64 accumulation of these rounded squares',
            library_tolerance=2e-6, fp64_tolerance=1e-10, candidate_count=1, ranking_claim=False)
        require(new_bytes == run['data_bytes'], 'Saved artifact byte accounting differs')
        result['receipts'] = dict(counts=counts, library_calls=run['library_calls'], recovery=recovery,
            seconds_total=run['seconds_total'], svd_seconds=run['svd_seconds'], data_bytes=new_bytes,
            peak_allocated_gib=run['peak_allocated_gib'], sampled_peak_gpu_used_mib=receipt['sampled_peak_gpu_used_mib'], exit_code=0)
        result['scope_limits'] = [
            'No model, search/parser, GPU kernel or full SVD was independently rerun; live API/forward/input-object identities remain source-bound receipts.',
            'BF16 baseline forwards were not saved twice; their live output signatures bind the independently verified complete E070 outputs.',
            'CPU verifies full saved xs against source input/scale, full packet encoding/decoding, and candidate/reloaded outputs; it does not reproduce native GEMM.',
            'Saved GPU BF16 B@A product is checked only as the input to exact residual subtraction. Neither that full GEMM nor LR A/B GEMMs were rerun.',
            'Top32 Gram and sampled singular equations are checked; full unsaved U/Vh finiteness and exact full-SVD execution remain runner receipts.',
            'Whole-checkpoint hash is not repeated; the entire selected source weight is rehashed.',
            'One Manual candidate supports wiring/export consistency only: no ranking, full-recipe calibration, new-method or perceptual-quality conclusion.'
        ]
        require(not torch.cuda.is_initialized(), 'CPU checker initialized CUDA')
        result.update(status='complete', verdict='pass_with_scope_limits')
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    result['seconds_total'] = time.monotonic()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output), seconds_total=result['seconds_total'])))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
