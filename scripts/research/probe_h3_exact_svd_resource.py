#!/usr/bin/env python3
"""E070: one real-weight, upstream exact FP64 full-SVD resource pilot.

The check path uses only the Python standard library and a safetensors header.
Evaluation imports Torch lazily, loads one BF16 weight, and calls SVD once.
No model forward, smoothing, quantization, fitting, or alternative decomposition.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'results/research/E070/exact_svd'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E070/exact_svd')
PLAN = ROOT / 'research_state/06_experiments/E070_h3_baseline_entry_plan.md'
E069 = ROOT / 'results/research/E069'
INVENTORY = Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export/manifest.json')
MODEL = Path('/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors')
KEY = 'blocks.0.mlp.fc1.weight'
SHAPE = (28672, 5376)
RANK = 32
COMMIT = '69f3473f5e1c1504bae35cc50c7858ef900a9b17'
UPSTREAM_SHA = '3eb734678fa8f44fbe5eb57cfdcd0c55546b948d837780df16813cecaaabb619'
MODEL_SHA = 'a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4'
WEIGHT_SHA = '398fbc666059bf253c09db92db9c56a077815603082469329567deb40a2ba78d'
MAX_ALLOCATED = 60 * 1024**3
MAX_DATA = 256 * 1024**2
MAX_SECONDS = 1800


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return dict(file=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def verify_record(value):
    path = Path(value['file'])
    require(path.is_file() and path.stat().st_size == value['bytes'] and sha256(path) == value['sha256'],
            f'Source binding changed: {path}')
    return path


def load_complete(path):
    value = json.loads(Path(path).read_text())
    require(value.get('status') == 'complete', f'Complete source required: {path}')
    return value


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prerequisites():
    require(PLAN.is_file(), 'E070 plan must be frozen before either phase')
    review = load_complete(E069 / 'independent_review.json')
    require(review['upstream_commit'] == COMMIT and review['verdict'] == 'pass_with_scope_limits', 'E069 review differs')
    audit_path = verify_record(review['audit'])
    audit = load_complete(audit_path)
    manifest_path = verify_record(review['upstream_manifest'])
    upstream_manifest = load_complete(manifest_path)
    require(audit['upstream_commit'] == upstream_manifest['commit'] == COMMIT, 'Upstream commit changed')
    upstream = E069 / 'upstream/deepcompressor/nn/patch/lowrank.py'
    source = file_record(upstream)
    require(source['sha256'] == UPSTREAM_SHA and source == review['checked_files'][str(upstream)], 'Pinned upstream source differs')
    calls = [node for node in ast.walk(ast.parse(upstream.read_text())) if isinstance(node, ast.Call)
             and ast.unparse(node) == 'torch.linalg.svd(weight.double())']
    require(len(calls) == 1, 'Pinned exact SVD expression not unique')
    inventory = load_complete(INVENTORY)
    rows = [row for row in inventory['layers'] if row['name'] + '.weight' == KEY]
    require(len(rows) == 1 and tuple(rows[0]['shape']) == SHAPE and rows[0]['source_weight_sha256'] == WEIGHT_SHA,
            'Historical source weight identity differs')
    old_model = inventory['sources'][str(MODEL)]
    stat = MODEL.stat()
    require(old_model['sha256'] == MODEL_SHA and stat.st_size == old_model['bytes'] and stat.st_mtime_ns == old_model['mtime_ns'],
            'Historical model file metadata differs')
    with MODEL.open('rb') as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, 'Invalid safetensors prefix')
        header_bytes = struct.unpack('<Q', prefix)[0]
        require(0 < header_bytes <= 16 * 1024**2, 'Unexpected safetensors header size')
        raw_header = stream.read(header_bytes)
    header = json.loads(raw_header)
    selected = header[KEY]
    start, stop = selected['data_offsets']
    require(selected['dtype'] == 'BF16' and tuple(selected['shape']) == SHAPE and
            stop-start == math.prod(SHAPE)*2 and 0 <= start < stop <= stat.st_size-8-header_bytes,
            'Selected safetensors header differs')
    paths = (Path(__file__), PLAN, INVENTORY, E069/'independent_review.json', audit_path, manifest_path, upstream)
    return dict(sources={str(path.resolve()): file_record(path) for path in paths},
        upstream=dict(commit=COMMIT, source=source, call='torch.linalg.svd(weight.double())', line=calls[0].lineno,
                      full_matrices='default True; no argument supplied', driver='default; no argument supplied'),
        model=dict(file=str(MODEL), sha256=MODEL_SHA, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                   whole_file_rehashed_this_probe=False, identity_basis='E009 recorded hash plus matching size/mtime; runtime selected tensor hash'),
        safetensors=dict(header_bytes=header_bytes, header_sha256=hashlib.sha256(raw_header).hexdigest(),
                         metadata=header.get('__metadata__'), key=KEY, entry=selected),
        source_weight_sha256=WEIGHT_SHA, rank=RANK,
        expected_full_outputs=dict(u=[SHAPE[0], SHAPE[0]], s=[SHAPE[1]], vh=[SHAPE[1], SHAPE[1]]))


def selected_indices(size):
    return [index*(size-1)//15 for index in range(16)]


def data_bytes():
    return sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()) if DATA.exists() else 0


def guard(args, torch=None):
    require(time.time() < args.deadline_unix, 'E070 shared absolute deadline reached')
    require(data_bytes() <= MAX_DATA, 'E070 output exceeded 256 MiB')
    if torch is not None:
        require(torch.cuda.max_memory_allocated() < MAX_ALLOCATED, 'E070 allocated memory reached 60 GiB')


def memory(torch):
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())


def tensor_sha(tensor):
    array = tensor.detach().cpu().contiguous().view(sys.modules['torch'].uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def error_stats(error, reference):
    require(bool(error.isfinite().all()), 'Nonfinite reconstruction residual')
    sse, energy = float(error.square().sum()), float(reference.square().sum())
    return dict(sse=sse, reference_energy=energy, relative_frobenius=math.sqrt(sse/max(energy, 1e-300)),
                max_abs=float(error.abs().max()), elements=error.numel())


def evaluate(args, report):
    guard(args)
    checked = load_complete(REPORTS / 'check.json')
    require(checked['binding'] == report['binding'] and checked['torch_imported'] is False and
            checked['cuda_imported'] is False, 'CPU source/header binding differs')
    report['cpu_check'] = file_record(REPORTS / 'check.json')
    DATA.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors import safe_open
    require(torch.cuda.device_count() == 1, 'Expose exactly one GPU to this probe')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    require(properties.total_memory > MAX_ALLOCATED, 'GPU cannot support the 60 GiB allocation cap')
    # The allocator cap is not a measurement of opaque CUDA-library allocations.
    torch.cuda.set_per_process_memory_fraction(MAX_ALLOCATED / properties.total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    report.update(torch=torch.__version__, cuda_version=torch.version.cuda,
                  device=dict(name=properties.name, total_memory=properties.total_memory,
                              capability=list(torch.cuda.get_device_capability())),
                  environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF')},
                  allocator_cap_bytes=MAX_ALLOCATED, external_supervisor_required_for_opaque_cuda_memory=True)
    guard(args, torch)
    with safe_open(str(MODEL), framework='pt', device='cpu') as source:
        cpu_weight = source.get_tensor(KEY)
    require(tuple(cpu_weight.shape) == SHAPE and cpu_weight.dtype == torch.bfloat16 and
            tensor_sha(cpu_weight) == WEIGHT_SHA, 'Loaded real BF16 weight differs from E009')
    report['source_weight_verified'] = True
    weight = cpu_weight.to(device='cuda')
    del cpu_weight
    require(bool(torch.isfinite(weight).all()), 'Nonfinite source weight')
    torch.cuda.synchronize()
    report['before_svd_memory'] = memory(torch)
    guard(args, torch)
    torch.cuda.reset_peak_memory_stats()
    report['attempted_svd_calls'] = 1
    save_new(REPORTS / 'svd_started.json', dict(status='running', unix=time.time(), pid=os.getpid(),
        deadline_unix=args.deadline_unix, source_weight_sha256=WEIGHT_SHA, shape=list(SHAPE),
        expression=report['binding']['upstream']['call'], before_svd_memory=report['before_svd_memory']))
    print('E070 phase=svd_started calls=1 dtype=float64 full_matrices=default', flush=True)
    start = time.perf_counter()
    u, s, vh = torch.linalg.svd(weight.double())
    report['returned_svd_calls'] = 1
    torch.cuda.synchronize()
    report['svd_expression_synchronized_seconds'] = time.perf_counter()-start
    report['complete_svd_calls'] = 1
    report['svd_memory'] = memory(torch)
    save_new(REPORTS / 'svd_finished.json', dict(status='complete', unix=time.time(),
        seconds=report['svd_expression_synchronized_seconds'], memory=report['svd_memory'],
        shapes=dict(u=list(u.shape), s=list(s.shape), vh=list(vh.shape))))
    print('E070 phase=svd_finished numerical_checks_next', flush=True)
    guard(args, torch)
    require(tuple(u.shape) == (SHAPE[0], SHAPE[0]) and tuple(s.shape) == (SHAPE[1],) and
            tuple(vh.shape) == (SHAPE[1], SHAPE[1]) and u.dtype == s.dtype == vh.dtype == torch.float64,
            'Full FP64 SVD output contract differs')
    require(bool(torch.isfinite(s).all()) and bool((s >= 0).all()) and bool((s[:-1] >= s[1:]).all()),
            'Singular spectrum is nonfinite, negative, or unordered')
    for vectors in (u, vh):
        for offset in range(0, vectors.shape[0], 1024):
            guard(args, torch)
            require(bool(torch.isfinite(vectors[offset:offset+1024]).all()), 'Nonfinite full SVD vectors')
    report['full_outputs_finite'] = True
    check_start = time.perf_counter()
    rows, cols = selected_indices(SHAPE[0]), selected_indices(SHAPE[1])
    row_reconstruction = (u[rows, :SHAPE[1]] * s) @ vh
    row_target = weight[rows].double()
    row_stats = error_stats(row_reconstruction-row_target, row_target)
    del row_reconstruction, row_target
    column_sse = column_energy = column_max = weight_energy = 0.0
    for offset in range(0, SHAPE[0], 1024):
        guard(args, torch)
        stop = min(SHAPE[0], offset+1024)
        reconstructed = (u[offset:stop, :SHAPE[1]] * s) @ vh[:, cols]
        target = weight[offset:stop, cols].double()
        local = error_stats(reconstructed-target, target)
        column_sse += local['sse']
        column_energy += local['reference_energy']
        column_max = max(column_max, local['max_abs'])
        weight_energy += float(weight[offset:stop].double().square().sum())
        del reconstructed, target
    column_stats = dict(sse=column_sse, reference_energy=column_energy,
        relative_frobenius=math.sqrt(column_sse/max(column_energy, 1e-300)), max_abs=column_max,
        elements=SHAPE[0]*len(cols))
    u_columns = sorted(set(selected_indices(SHAPE[0]) + [SHAPE[1]-1, SHAPE[1]]))
    v_rows = selected_indices(SHAPE[1])
    def orthogonality(vectors):
        gram = vectors.T @ vectors
        residual = gram-torch.eye(gram.shape[0], device='cuda', dtype=torch.float64)
        require(bool(torch.isfinite(residual).all()), 'Nonfinite orthogonality residual')
        return dict(frobenius=float(residual.norm()), max_abs=float(residual.abs().max()),
                    residual=residual.cpu().tolist())
    ortho = dict(u_selected_columns=orthogonality(u[:, u_columns]),
                 vh_selected_rows=orthogonality(vh[v_rows].T),
                 u_top32=orthogonality(u[:, :RANK]), vh_top32=orthogonality(vh[:RANK].T))
    spectrum_energy = float(s.square().sum())
    report['numerical_checks'] = dict(selected_rows=rows, selected_columns=cols,
        sample_selection='16 floor-spaced indices index*(size-1)//15, including both endpoints',
        full_rank_selected_rows=row_stats, full_rank_selected_columns=column_stats,
        u_orthogonality_columns=u_columns, vh_orthogonality_rows=v_rows, orthogonality=ortho,
        source_frobenius_energy=weight_energy, spectrum_energy=spectrum_energy,
        spectrum_energy_relative_error=abs(spectrum_energy-weight_energy)/max(weight_energy, 1e-300),
        scope='GPU FP64 residual checks; no claim of independent SVD recomputation or rank32 quality improvement')
    require(row_stats['relative_frobenius'] <= 1e-8 and column_stats['relative_frobenius'] <= 1e-8 and
            all(value['frobenius'] <= 1e-8 for value in ortho.values()) and
            report['numerical_checks']['spectrum_energy_relative_error'] <= 1e-8, 'FP64 numerical residual gate failed')
    # Match the pinned branch: A=Vh[:32], B=U[:,:32]*S[:32], each cast to the original BF16 dtype.
    a = vh[:RANK].to(weight.dtype)
    b = (u[:, :RANK] * s[:RANK]).to(weight.dtype)
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Nonfinite BF16 branch factors')
    artifact = dict(u_top32=u[:, :RANK].contiguous().cpu(), s_top32=s[:RANK].clone().cpu(),
        vh_top32=vh[:RANK].contiguous().cpu(), singular_values=s.cpu(),
        a_bf16=a.contiguous().cpu(), b_bf16=b.contiguous().cpu(),
        row_indices=torch.tensor(rows, dtype=torch.int64), column_indices=torch.tensor(cols, dtype=torch.int64),
        source_rows_fp64=weight[rows].double().cpu(), source_columns_fp64=weight[:, cols].double().cpu())
    torch.cuda.synchronize()
    report['checks_and_extraction_seconds'] = time.perf_counter()-check_start
    report['after_checks_memory'] = memory(torch)
    report['overall_peak_allocated_bytes'] = max(report[name]['peak_allocated_bytes'] for name in
                                               ('before_svd_memory', 'svd_memory', 'after_checks_memory'))
    guard(args, torch)
    require(sum(value.numel()*value.element_size() for value in artifact.values()) < MAX_DATA,
            'Selected factors exceed artifact budget')
    artifact_path = DATA / 'blocks.0.mlp.fc1.exact_svd.pt'
    with artifact_path.open('xb') as stream:
        torch.save(artifact, stream)
    report['artifact'] = file_record(artifact_path)
    report['artifact_tensors'] = {name: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value))
                                  for name, value in artifact.items()}
    guard(args, torch)
    report.update(status='complete', data_bytes=data_bytes(), cuda_initialized=torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'evaluate'))
    parser.add_argument('--budget-start-unix', type=float)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if args.phase == 'evaluate' and (args.budget_start_unix is None or args.deadline_unix is None or
            not 0 < args.deadline_unix-args.budget_start_unix <= MAX_SECONDS or
            not args.budget_start_unix <= time.time() < args.deadline_unix):
        parser.error('Evaluation requires the original shared start/deadline, at most 1800 seconds apart')
    output = REPORTS / (args.phase + '.json')
    REPORTS.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f'Refusing output overwrite: {output}')
    with (REPORTS / (args.phase + '.lock')).open('x') as stream:
        stream.write(str(os.getpid()))
    start = time.perf_counter()
    report = dict(experiment='E070', phase=args.phase, status='running', pid=os.getpid(),
        attempted_svd_calls=0, returned_svd_calls=0, complete_svd_calls=0, new_model_forwards=0,
        new_quantization_calls=0, budget_start_unix=args.budget_start_unix, deadline_unix=args.deadline_unix,
        meaning='Single exact-SVD resource and numerical-contract pilot; no method or quality evidence')
    try:
        if args.phase == 'evaluate':
            def timeout(signum, frame):
                raise TimeoutError('E070 shared absolute deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, args.deadline_unix-time.time()))
            report['deadline_contract'] = 'SIGALRM plus phase guards; blocking native calls require the root external supervisor'
            save_new(REPORTS / 'evaluate_started.json', dict(status='running', unix=time.time(), pid=os.getpid(),
                     deadline_unix=args.deadline_unix))
        report['binding'] = prerequisites()
        if args.phase == 'check':
            report['torch_imported'] = 'torch' in sys.modules
            report['cuda_imported'] = any(name == 'torch.cuda' or name.startswith('torch.cuda.') for name in sys.modules)
            require(not report['torch_imported'] and not report['cuda_imported'], 'CPU check imported Torch/CUDA')
            report.update(status='complete', cuda_initialized=False, selected_weight_loaded=False)
        else:
            evaluate(args, report)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      data_bytes=data_bytes())
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            report['failure_memory'] = memory(torch)
    finally:
        if args.phase == 'evaluate':
            signal.setitimer(signal.ITIMER_REAL, 0)
        report['seconds_total'] = time.perf_counter()-start
        save_new(output, report)
    print(json.dumps(dict(status=report['status'], report=str(output), complete_svd_calls=report['complete_svd_calls'],
                         seconds_total=report['seconds_total']), indent=2), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
