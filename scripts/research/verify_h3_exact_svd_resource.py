#!/usr/bin/env python3
"""Independent E070-B CPU artifact verification; no runner import or SVD.

Requires completed evaluate.json and supervisor exit0 before loading tensors.
NumPy recomputes top32 singular equations and orthogonality. Only the selected
BF16 source weight is rehashed; the entire checkpoint is not rehashed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E070'
DIRECTORY = REPORTS/'exact_svd'
RUNNER = ROOT/'scripts/research/probe_h3_exact_svd_resource.py'
RUNNER_SHA = 'a400967abd67e342d9e03d2d9e00f503538460f34491875406ce1c888c6b62a6'
COMMIT = '69f3473f5e1c1504bae35cc50c7858ef900a9b17'
WEIGHT_SHA = '398fbc666059bf253c09db92db9c56a077815603082469329567deb40a2ba78d'
KEY, M, N, RANK = 'blocks.0.mlp.fc1.weight', 28672, 5376, 32
TOL = 1e-8


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(file=str(path), bytes=path.stat().st_size, sha256=digest(path))


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DIRECTORY/'independent.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an independent result')
    # These preconditions intentionally precede imports and result creation.
    evaluation = read(DIRECTORY/'evaluate.json')
    receipt = read(REPORTS/'process_exit_svd.json')
    require(evaluation.get('status') == 'complete', 'Wait for complete E070-B evaluation')
    require(receipt.get('exit_code') == receipt.get('worker_exit_code') == 0 and
            receipt.get('timeout_exit') is False, 'Wait for normal supervisor and worker exit0')
    started = time.monotonic()
    result = dict(experiment='E070', arm='exact_svd', status='running', files={}, reports={},
                  checker=record(__file__), new_model_forwards=0, new_svd_calls=0,
                  new_quantization_calls=0, cuda_initialized=False,
                  purpose='Baseline engineering resource/numerical pilot; no new method or quality claim')

    def verify(expected):
        path = Path(expected['file']).resolve()
        actual = result['files'].get(str(path))
        if actual is None:
            actual = record(path)
            result['files'][str(path)] = actual
        require(actual['sha256'] == expected['sha256'] and
                ('bytes' not in expected or actual['bytes'] == expected['bytes']), f'File identity changed: {path}')
        return path

    def report(path):
        value = record(path)
        result['reports'][Path(path).name] = value
        result['files'][value['file']] = value
        return read(path)

    try:
        launch = report(REPORTS/'launch_svd.json')
        receipt = report(REPORTS/'process_exit_svd.json')
        evaluation = report(DIRECTORY/'evaluate.json')
        check = report(DIRECTORY/'check.json')
        svd_started = report(DIRECTORY/'svd_started.json')
        svd_finished = report(DIRECTORY/'svd_finished.json')
        require(launch['experiment'] == receipt['experiment'] == evaluation['experiment'] == check['experiment'] == 'E070',
                'Wrong experiment')
        require(launch['arm'] == receipt['arm'] == 'svd' and evaluation['phase'] == 'evaluate' and
                check['phase'] == 'check', 'Wrong phase/arm')
        require(check['status'] == evaluation['status'] == svd_finished['status'] == 'complete' and
                check['torch_imported'] is False and check['cuda_imported'] is False and
                check['cuda_initialized'] is False and check['selected_weight_loaded'] is False,
                'Preflight or evaluation status differs')
        require(receipt['exit_code'] == receipt['worker_exit_code'] == 0 and receipt['timeout_exit'] is False and
                'memory_stop' not in receipt and 'supervisor_error' not in receipt, 'Supervisor did not complete normally')
        require(receipt['launch_sha256'] == result['reports']['launch_svd.json']['sha256'] and
                receipt['completed_output_sha256'] == result['reports']['evaluate.json']['sha256'],
                'Supervisor launch/output binding differs')
        start, deadline = launch['budget_start_unix'], launch['deadline_unix']
        require(0 < deadline-start <= 1800 and all(v['budget_start_unix'] == start and
                v['deadline_unix'] == deadline for v in (receipt, evaluation)), 'Shared deadline differs')
        require(start <= receipt['start_epoch'] <= receipt['end_epoch'] <= deadline and
                start <= svd_started['unix'] <= svd_finished['unix'] <= deadline and
                svd_started['deadline_unix'] == deadline, 'Reported timeline outside original budget')
        for name in ('runner', 'plan', 'cpu_check', 'supervisor'):
            verify(launch[name])
        require(Path(launch['runner']['file']).resolve() == RUNNER.resolve() and
                launch['runner']['sha256'] == RUNNER_SHA and
                Path(launch['cpu_check']['file']).resolve() == (DIRECTORY/'check.json').resolve() and
                Path(launch['output']).resolve() == (DIRECTORY/'evaluate.json').resolve(), 'Launch source/report differs')
        require(launch['environment']['CUDA_VISIBLE_DEVICES'] == str(launch['gpu']) and
                receipt['gpu'] == launch['gpu'] and receipt['gpu_before'] == '0, 0', 'GPU launch receipt differs')
        verify(evaluation['cpu_check'])
        require(evaluation['binding'] == check['binding'], 'Evaluation/preflight binding differs')
        binding = evaluation['binding']
        for name, expected in binding['sources'].items():
            require(Path(name).resolve() == Path(expected['file']).resolve(), 'Source key/path mismatch')
            verify(expected)
        require(binding['sources'][str(RUNNER)]['sha256'] == RUNNER_SHA and
                binding['upstream']['commit'] == COMMIT and binding['upstream']['call'] == 'torch.linalg.svd(weight.double())' and
                binding['rank'] == RANK and binding['source_weight_sha256'] == WEIGHT_SHA and
                evaluation['source_weight_verified'] is True, 'Weight/upstream contract differs')
        upstream = verify(binding['upstream']['source'])
        for path in (upstream, RUNNER):
            calls = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call) and
                     ast.unparse(n) == 'torch.linalg.svd(weight.double())']
            require(len(calls) == 1 and not calls[0].keywords, 'Expected one exact default full-SVD expression')
        require(binding['upstream']['full_matrices'] == 'default True; no argument supplied' and
                binding['upstream']['driver'] == 'default; no argument supplied', 'Upstream call options differ')
        full_shapes = dict(u=[M, M], s=[N], vh=[N, N])
        require(binding['expected_full_outputs'] == svd_finished['shapes'] == full_shapes,
                'Default full-matrices output shapes differ')
        require(evaluation['attempted_svd_calls'] == evaluation['returned_svd_calls'] == evaluation['complete_svd_calls'] == 1 and
                evaluation['new_model_forwards'] == evaluation['new_quantization_calls'] == 0 and
                all(check[k] == 0 for k in ('attempted_svd_calls', 'returned_svd_calls', 'complete_svd_calls',
                                           'new_model_forwards', 'new_quantization_calls')),
                'SVD/model/quantization call counts differ')
        require(evaluation['full_outputs_finite'] is True and evaluation['cuda_initialized'] is True,
                'GPU full-output finite check absent')
        require(0 < evaluation['svd_expression_synchronized_seconds'] <= evaluation['seconds_total'] <= 1800 and
                svd_finished['seconds'] == evaluation['svd_expression_synchronized_seconds'], 'Resource time invalid')
        require(evaluation['overall_peak_allocated_bytes'] < 60*1024**3 and
                receipt['sampled_peak_gpu_used_mib'] <= 60*1024 and
                evaluation['data_bytes'] <= 256*1024**2, 'Resource budget exceeded')
        numerical = evaluation['numerical_checks']
        for name in ('full_rank_selected_rows', 'full_rank_selected_columns'):
            require(math.isfinite(numerical[name]['relative_frobenius']) and
                    numerical[name]['relative_frobenius'] <= TOL, 'Runner full-rank reconstruction gate failed')
        require(math.isfinite(numerical['spectrum_energy_relative_error']) and
                numerical['spectrum_energy_relative_error'] <= TOL and
                all(math.isfinite(v['frobenius']) and v['frobenius'] <= TOL
                    for v in numerical['orthogonality'].values()), 'Runner numerical gate failed')

        import numpy as np
        import torch
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), 'CPU checker unexpectedly initialized CUDA')
        result['environment'] = dict(numpy=np.__version__, torch=torch.__version__, python=sys.version,
                                     cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        artifact_path = verify(evaluation['artifact'])
        require(artifact_path.stat().st_size <= 256*1024**2, 'Artifact exceeds output budget')
        tensors = torch.load(artifact_path, map_location='cpu', weights_only=True, mmap=True)
        schema = dict(u_top32=((M, RANK), torch.float64), s_top32=((RANK,), torch.float64),
                      vh_top32=((RANK, N), torch.float64), singular_values=((N,), torch.float64),
                      a_bf16=((RANK, N), torch.bfloat16), b_bf16=((M, RANK), torch.bfloat16),
                      row_indices=((16,), torch.int64), column_indices=((16,), torch.int64),
                      source_rows_fp64=((16, N), torch.float64), source_columns_fp64=((M, 16), torch.float64))
        require(set(tensors) == set(schema) == set(evaluation['artifact_tensors']), 'Artifact tensor schema differs')
        arrays, tensor_records = {}, {}
        for name, (shape, dtype) in schema.items():
            value = tensors[name]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and tuple(value.shape) == shape and
                    value.dtype == dtype and value.is_contiguous(), f'Tensor format differs: {name}')
            raw = value.view(torch.uint8).numpy()
            observed = dict(shape=list(shape), dtype=str(dtype), sha256=hashlib.sha256(memoryview(raw)).hexdigest())
            require(observed == evaluation['artifact_tensors'][name], f'Tensor hash differs: {name}')
            array = value.view(torch.uint16).numpy() if dtype == torch.bfloat16 else value.numpy()
            if dtype == torch.bfloat16:
                require(np.isfinite((array.astype(np.uint32) << 16).view(np.float32)).all(), f'Nonfinite {name}')
            else:
                require(np.isfinite(array).all(), f'Nonfinite {name}')
            arrays[name], tensor_records[name] = array, observed
        result['artifact_tensors'] = tensor_records
        rows = np.array([i*(M-1)//15 for i in range(16)], dtype=np.int64)
        cols = np.array([i*(N-1)//15 for i in range(16)], dtype=np.int64)
        require(np.array_equal(arrays['row_indices'], rows) and np.array_equal(arrays['column_indices'], cols) and
                numerical['selected_rows'] == rows.tolist() and numerical['selected_columns'] == cols.tolist(),
                'Preselected sample indices differ')

        model = binding['model']
        model_path = Path(model['file'])
        stat = model_path.stat()
        require(stat.st_size == model['bytes'] and stat.st_mtime_ns == model['mtime_ns'] and
                model['whole_file_rehashed_this_probe'] is False, 'Historical checkpoint metadata differs')
        with model_path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            require(0 < header_size <= 16*1024**2, 'Invalid safetensors header size')
            header_bytes = stream.read(header_size)
        header = json.loads(header_bytes)
        entry = header[KEY]
        require(header_size == binding['safetensors']['header_bytes'] and
                hashlib.sha256(header_bytes).hexdigest() == binding['safetensors']['header_sha256'] and
                entry == binding['safetensors']['entry'] and binding['safetensors']['key'] == KEY and
                entry['dtype'] == 'BF16' and entry['shape'] == [M, N], 'Selected safetensors entry changed')
        begin, end = entry['data_offsets']
        require(end-begin == M*N*2 and 0 <= begin < end <= stat.st_size-8-header_size, 'Selected tensor extent invalid')
        weights = np.memmap(model_path, dtype='<u2', mode='r', offset=8+header_size+begin, shape=(M, N))
        def from_bf16(value):
            return (np.asarray(value).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        source_hash, energy = hashlib.sha256(), 0.0
        for offset in range(0, M, 256):
            chunk = weights[offset:offset+256]
            source_hash.update(memoryview(np.asarray(chunk)).cast('B'))
            values = from_bf16(chunk)
            require(np.isfinite(values).all(), 'Nonfinite selected source weight')
            energy += float(np.sum(values*values, dtype=np.float64))
        require(source_hash.hexdigest() == WEIGHT_SHA, 'Full selected-weight tensor hash differs')
        require(np.array_equal(from_bf16(weights[rows]), arrays['source_rows_fp64']) and
                np.array_equal(from_bf16(weights[:, cols]), arrays['source_columns_fp64']),
                'Saved FP64 source samples differ from selected checkpoint weight')
        del weights
        result['source_identity'] = dict(model=model, header_sha256=hashlib.sha256(header_bytes).hexdigest(),
            selected_tensor_sha256=source_hash.hexdigest(), selected_tensor_bytes=M*N*2,
            whole_checkpoint_rehashed=False, selected_tensor_fully_rehashed=True, source_samples_exact=True)

        u, s, vh = arrays['u_top32'], arrays['singular_values'], arrays['vh_top32']
        require(np.array_equal(arrays['s_top32'], s[:RANK]) and np.all(s >= 0) and np.all(s[:-1] >= s[1:]),
                'Singular spectrum ordering/top32 differs')
        def residual_stats(actual, expected):
            error = actual-expected
            sse = float(np.sum(error*error, dtype=np.float64))
            ref = float(np.sum(expected*expected, dtype=np.float64))
            return dict(sse=sse, reference_energy=ref, relative_frobenius=math.sqrt(sse/max(ref, 1e-300)),
                        max_abs=float(np.max(np.abs(error))), elements=int(error.size))
        row_equation = residual_stats(arrays['source_rows_fp64'] @ vh.T, u[rows]*s[:RANK])
        column_equation = residual_stats(arrays['source_columns_fp64'].T @ u, vh[:, cols].T*s[:RANK])
        require(row_equation['relative_frobenius'] <= TOL and column_equation['relative_frobenius'] <= TOL,
                'Independent sampled singular-vector equation failed')
        gram_checks = {}
        for name, gram in (('u_top32', u.T @ u), ('vh_top32', vh @ vh.T)):
            residual = gram-np.eye(RANK, dtype=np.float64)
            frobenius = float(np.sqrt(np.sum(residual*residual, dtype=np.float64)))
            reported = np.array(numerical['orthogonality'][name]['residual'], dtype=np.float64)
            require(frobenius <= TOL and reported.shape == (RANK, RANK) and np.isfinite(reported).all() and
                    np.max(np.abs(reported-residual)) <= 1e-11, 'Independent top32 Gram differs or fails')
            gram_checks[name] = dict(frobenius=frobenius, max_abs=float(np.max(np.abs(residual))),
                max_abs_difference_from_gpu_report=float(np.max(np.abs(reported-residual))))
        def bf16_rne_bits(value):
            # PyTorch BFloat16 scalar conversion rounds the FP32 intermediate to nearest-even.
            f32 = np.asarray(value, dtype=np.float32)
            bits = f32.view(np.uint32)
            return ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
        require(np.array_equal(bf16_rne_bits(vh), arrays['a_bf16']) and
                np.array_equal(bf16_rne_bits(u*s[:RANK]), arrays['b_bf16']), 'Independent BF16 A/B casting differs')
        spectrum_energy = float(np.sum(s*s, dtype=np.float64))
        spectrum_error = abs(spectrum_energy-energy)/max(energy, 1e-300)
        require(spectrum_error <= TOL and
                abs(energy-numerical['source_frobenius_energy'])/max(energy, 1e-300) <= 1e-11 and
                abs(spectrum_energy-numerical['spectrum_energy'])/max(spectrum_energy, 1e-300) <= 1e-11,
                'Independent source/spectrum Frobenius energy differs')
        result['independent_numerics'] = dict(tolerance=TOL, row_singular_equation=row_equation,
            column_singular_equation=column_equation, top32_orthogonality=gram_checks,
            bf16_casting_exact=True, source_frobenius_energy=energy, spectrum_energy=spectrum_energy,
            spectrum_energy_relative_error=spectrum_error,
            equation_reference='W_rows @ V32 versus U_rows*S32; W_columns.T @ U32 versus V_columns*S32')
        result['resource_receipts'] = dict(svd_expression_synchronized_seconds=evaluation['svd_expression_synchronized_seconds'],
            checks_and_extraction_seconds=evaluation['checks_and_extraction_seconds'],
            seconds_total=evaluation['seconds_total'], overall_peak_allocated_bytes=evaluation['overall_peak_allocated_bytes'],
            sampled_peak_gpu_used_mib=receipt['sampled_peak_gpu_used_mib'], data_bytes=evaluation['data_bytes'],
            worker_exit_code=receipt['worker_exit_code'], supervisor_exit_code=receipt['exit_code'])
        result['scope_limits'] = [
            'No SVD or model forward was repeated; saved top32 equations, Gram matrices and source bytes were independently checked.',
            'Full U/Vh were not saved. Their finite checks, selected full-spectrum reconstructions and non-top32 orthogonality remain GPU-runner receipts.',
            'Whole-checkpoint SHA is historical metadata; this check independently rehashed only the safetensors header and entire selected BF16 tensor.',
            'Device-memory sampling is not a continuous proof of opaque-library peak usage; allocator and supervisor observations are reported separately.',
            'One unsmoothed original-weight decomposition does not establish costs for all residuals, smoothing candidates or complete PTQ.',
            'This baseline engineering pilot provides no novel method, local/full-model error improvement or perceptual-quality evidence.'
        ]
        require(not torch.cuda.is_initialized(), 'CPU checker initialized CUDA')
        result.update(status='complete', verdict='pass_with_scope_limits', cuda_initialized=False)
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
