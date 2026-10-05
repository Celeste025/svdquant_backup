#!/usr/bin/env python3
"""Independent CPU checks of saved E070-A v2 bridge artifacts.

Never imports the runner or DeepCompressor, invokes its parser, or runs a model.
The original v1 import/check failure remains separate from the frozen v2 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'results/research/E070'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E070/bridge_v2')
RUNNER = ROOT/'scripts/research/probe_h3_calibration_bridge_v2.py'
RUNNER_SHA = '8374c00f4421efd79b7b9da0dc7d352e4aeb976a8b88de359e8aaaafdb5477c4'
CASES = ('p1_s03', 'p20_s03')
ROWS = (22400, 22464)


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
    parser.add_argument('--output', type=Path, default=REPORTS/'independent_bridge_v2.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite independent result')
    # Do not create an independent result prematurely, or import tensor libraries.
    run = read(REPORTS/'run_bridge_v2.json')
    receipt = read(REPORTS/'process_exit_bridge.json')
    require(run.get('status') == 'complete', 'Wait for complete E070-A v2 run')
    require(receipt.get('exit_code') == receipt.get('worker_exit_code') == 0 and
            receipt.get('timeout_exit') is False, 'Wait for normal bridge supervisor/worker exit0')
    started = time.monotonic()
    result = dict(experiment='E070', arm='bridge_v2', status='running', checker=record(__file__),
                  reports={}, files={}, cases=[], new_model_forwards=0, new_parser_calls=0,
                  new_quantization_calls=0, cuda_initialized=False)

    def verify(expected):
        path = Path(expected['file']).resolve()
        key = str(path)
        if key not in result['files']:
            result['files'][key] = record(path)
        actual = result['files'][key]
        require(actual['sha256'] == expected['sha256'] and
                ('bytes' not in expected or actual['bytes'] == expected['bytes']), f'File changed: {path}')
        return path

    def report(path):
        row = record(path)
        result['reports'][Path(path).name] = row
        result['files'][row['file']] = row
        return read(path)

    try:
        launch = report(REPORTS/'launch_bridge.json')
        receipt = report(REPORTS/'process_exit_bridge.json')
        check = report(REPORTS/'check_bridge_v2.json')
        run = report(REPORTS/'run_bridge_v2.json')
        require(launch['experiment'] == receipt['experiment'] == 'E070' and
                launch['arm'] == receipt['arm'] == 'bridge' and
                check['experiment'] == run['experiment'] == 'E070_bridge', 'Experiment/arm changed')
        require(check['status'] == run['status'] == 'complete' and check['phase'] == 'check' and
                run['phase'] == 'run' and check['cuda_initialized'] is False and
                check['new_model_forwards'] == 0 and run['cuda_initialized'] is True, 'Phase/check status invalid')
        require(receipt['exit_code'] == receipt['worker_exit_code'] == 0 and receipt['timeout_exit'] is False and
                'memory_stop' not in receipt and 'supervisor_error' not in receipt, 'Supervisor failed')
        require(receipt['launch_sha256'] == result['reports']['launch_bridge.json']['sha256'] and
                receipt['completed_output_sha256'] == result['reports']['run_bridge_v2.json']['sha256'],
                'Launch/result exit receipt binding differs')
        for name in ('runner', 'plan', 'cpu_check', 'supervisor'):
            verify(launch[name])
        require(Path(launch['runner']['file']).resolve() == RUNNER.resolve() and
                launch['runner']['sha256'] == RUNNER_SHA and
                Path(launch['cpu_check']['file']).resolve() == (REPORTS/'check_bridge_v2.json').resolve() and
                Path(launch['output']).resolve() == (REPORTS/'run_bridge_v2.json').resolve(), 'Frozen v2 paths/source differ')
        start, deadline = launch['budget_start_unix'], launch['deadline_unix']
        require(0 < deadline-start <= 900 and all(v['budget_start_unix'] == start and v['deadline_unix'] == deadline
                for v in (run, receipt)) and start <= receipt['start_epoch'] <= receipt['end_epoch'] <= deadline,
                'Shared budget/receipt timeline differs')
        require(launch['environment']['CUDA_VISIBLE_DEVICES'] == str(launch['gpu']) and
                receipt['gpu'] == launch['gpu'] and receipt['gpu_before'] == '0, 0', 'GPU launch receipt differs')
        require(run['wall_budget_seconds'] == 900 and run['data_byte_limit'] == 4*1024**3 and
                run['data_bytes'] < 4*1024**3 and run['peak_allocated_gib'] < 60 and
                receipt['sampled_peak_gpu_used_mib'] <= 60*1024 and 0 < run['seconds_total'] <= 900,
                'Resource budget failed')
        same_fields = ('sources', 'plan', 'source_evaluation', 'source_capture_artifacts', 'source_input_checks',
                       'calibration_inputs', 'setup_check', 'setup_manifest', 'environment_probe', 'actual_cache_import', 'torch')
        require(all(check[name] == run[name] for name in same_fields) and check['inputs'] == run['inputs'],
                'CPU-check/run identity changed')
        verify(run['cpu_check_reference'])
        require(run['cpu_check_reference'] == launch['cpu_check'] and run['plan'] == launch['plan'],
                'CPU/plan launch binding differs')
        for path, value in run['sources'].items():
            require(Path(path).resolve() == Path(value['file']).resolve(), 'Source key/path mismatch')
            verify(value)
        require(run['sources'][str(RUNNER)]['sha256'] == RUNNER_SHA, 'Runtime source differs from frozen v2')
        parent = read(verify(run['source_evaluation']))
        require(parent['status'] == 'complete' and parent['experiment'] == 'E065b', 'Wrong calibration provenance')
        environment = read(verify(run['environment_probe']))
        require(environment['status'] == 'complete' and environment['returncode'] == 0, 'Real import environment failed')
        api = run['actual_cache_import']
        expected_config = dict(objective='OutputsError', granularity='Layer', sample_batch_size=1, sample_size=-1,
                               tensor_type='Weights', quantizers_enabled=False, fit_or_ask_called=False)
        require(api['status'] == 'complete' and api['config'] == expected_config and
                api['parse_ipts_owner'] == 'deepcompressor.calib.search.SearchBasedCalibrator' and
                api['actual_parser_class'] == 'deepcompressor.calib.smooth.SmoothCalibrator', 'Actual parser contract differs')
        expected_api_files = dict(TensorCache='data/cache.py', TensorsCache='data/cache.py',
            SearchBasedCalibrator='calib/search.py', SmoothCalibrator='calib/smooth.py', get_smooth_span='calib/smooth.py')
        require(set(api['classes_and_functions']) == set(expected_api_files), 'Actual API member set differs')
        for name, relative in expected_api_files.items():
            expected_path = (ROOT/'third_party/deepcompressor/deepcompressor'/relative).resolve()
            require(Path(api['classes_and_functions'][name]).resolve() == expected_path and
                    str(expected_path) in run['sources'], 'Actual API source identity differs')
        zeros = ('complete_dit_calls', 'activation_packs', 'new_candidate_weights', 'calibration_fit_calls',
                 'text_encoder_calls', 'video_vae_calls', 'audio_vae_calls')
        require(all(run[name] == check[name] == 0 for name in zeros) and
                run['attempted_prefix_calls'] == run['complete_prefix_calls'] ==
                run['attempted_wrapper_calls'] == run['complete_wrapper_calls'] == 2 and
                all(check[name] == 0 for name in ('attempted_prefix_calls', 'complete_prefix_calls',
                                                'attempted_wrapper_calls', 'complete_wrapper_calls')),
                'Forward/calibration call budget differs')
        require(run['runtime_totals'] == dict(sdpa_calls=12, scaled_mm_calls=0, disk_loads=0) and
                check['runtime_totals'] == dict(sdpa_calls=0, scaled_mm_calls=0, disk_loads=0), 'Runtime totals differ')
        contract = dict(actual_parse_ipts=True, actual_repartition=True, actual_extract=True,
            objective='OutputsError', granularity='Layer', sample_batch_size=1, sample_size=-1,
            case_indices=[0, 1], full_rows=list(ROWS), byte_exact_all=True, immutable_inputs_all=True,
            x_acts_used_for_actual_absmax=True, no_token_slices=True, no_fit=True)
        require(run['bridge_contract'] == contract, 'Bridge acceptance contract differs')

        import numpy as np
        import torch
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), 'CPU checker initialized CUDA')
        result['environment'] = dict(numpy=np.__version__, torch=torch.__version__, python=sys.version,
                                     cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        def raw_bytes(tensor):
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu', 'Expected CPU tensor')
            return tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        def tensor_record(tensor):
            return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                        sha256=hashlib.sha256(memoryview(raw_bytes(tensor))).hexdigest())
        def signature(value):
            if isinstance(value, torch.Tensor):
                return tensor_record(value)
            if isinstance(value, dict):
                return {k: signature(v) for k, v in sorted(value.items())}
            if isinstance(value, (tuple, list)):
                return [signature(v) for v in value]
            return value
        def byte_equal(left, right, label):
            if isinstance(left, torch.Tensor):
                require(isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype,
                        f'Tensor layout differs: {label}')
                lraw, rraw = raw_bytes(left), raw_bytes(right)
                for offset in range(0, lraw.size, 8 << 20):
                    require(np.array_equal(lraw[offset:offset+(8 << 20)], rraw[offset:offset+(8 << 20)]),
                            f'Tensor bytes differ: {label}')
            elif isinstance(left, dict):
                require(isinstance(right, dict) and set(left) == set(right), f'Dictionary differs: {label}')
                for key in left:
                    byte_equal(left[key], right[key], label+'.'+str(key))
            elif isinstance(left, (tuple, list)):
                require(type(left) is type(right) and len(left) == len(right), f'Sequence differs: {label}')
                for index, (a, b) in enumerate(zip(left, right)):
                    byte_equal(a, b, label+f'[{index}]')
            else:
                require(type(left) is type(right) and left == right, f'Scalar differs: {label}')
        def load(value, trusted_raw=False):
            return torch.load(verify(value), map_location='cpu', weights_only=not trusted_raw, mmap=True)
        def finite(value, label):
            if isinstance(value, torch.Tensor):
                flat = value.contiguous().reshape(-1)
                for offset in range(0, flat.numel(), 1 << 20):
                    part = flat[offset:offset+(1 << 20)]
                    if part.dtype == torch.bfloat16:
                        bits = part.view(torch.uint16).numpy().astype(np.uint32) << 16
                        array = bits.view(np.float32)
                    else:
                        array = part.numpy()
                    require(np.isfinite(array).all(), f'Nonfinite tensor: {label}')
            elif isinstance(value, dict):
                for key, part in value.items():
                    finite(part, label+'.'+str(key))
            elif isinstance(value, (tuple, list)):
                for index, part in enumerate(value):
                    finite(part, label+f'[{index}]')
        require([v['id'] for v in run['calibration_inputs']] == list(CASES) and
                [v['case_id'] for v in run['prefix_cases']] == list(CASES) and
                [v['case_id'] for v in run['wrapper_cases']] == list(CASES), 'Case order/count changed')
        span = np.zeros(5376, dtype=np.float32)
        x_signatures = []
        artifact_bytes = 0
        for index, case_id in enumerate(CASES):
            case = run['calibration_inputs'][index]
            require(case == next(v for v in parent['calibration_inputs'] if v['id'] == case_id) and
                    case['prompt_id'] == (1, 20)[index] and case['step'] == 3 and case['kind'] == 'raw_dit',
                    'Frozen calibration state differs')
            raw = load(case['artifact'], trusted_raw=True)
            require(set(raw) == {'input_args', 'input_kwargs', 'meta'}, 'Original raw schema changed')
            raw_signature = signature(dict(args=raw['input_args'], kwargs=raw['input_kwargs']))
            source = parent['calibration_checks'][case_id]
            captured_record = next(v['artifact'] for v in parent['calibration_cases'] if v['case_id'] == case_id)
            require(run['source_capture_artifacts'][case_id] == captured_record and
                    run['source_input_checks'][case_id] == source, 'Historical capture binding differs')
            old_capture = load(captured_record)
            require(raw_signature == source['actual_dit_input_signature'] == old_capture['actual_dit_input_signature'] ==
                    run['inputs'][case_id]['actual_dit_input_signature'] and
                    raw['meta'] == source['metadata'] == old_capture['metadata'] == run['inputs'][case_id]['metadata'],
                    'Original raw metadata/input signature differs')
            before, after = run['prefix_cases'][index], run['wrapper_cases'][index]
            for row, kind in ((before, 'prefix'), (after, 'wrapper')):
                require(Path(row['artifact']['file']).resolve() == (DATA/kind/(case_id+'.pt')).resolve(),
                        'Unexpected v2 artifact path')
                counts = row['runtime_audit']
                require(counts['sdpa_calls'] == (4 if kind == 'prefix' else 2) and
                        counts['scaled_mm_calls'] == counts['disk_loads'] == 0 and
                        row['full_rows'] == ROWS[index] and row['input_kwargs_unchanged'] is True,
                        'Per-call runtime/layout contract differs')
                artifact_bytes += row['artifact']['bytes']
            prefix, replay = load(before['artifact']), load(after['artifact'])
            require(prefix['case_id'] == replay['case_id'] == case_id and replay['case_index'] == index and
                    prefix['actual_dit_input_signature'] == raw_signature and prefix['metadata'] == raw['meta'] and
                    after['case_index'] == index and after['byte_exact_reference'] is True, 'Per-case payload identity differs')
            p_inputs, r_inputs = prefix['attention_inputs'], replay['attention_inputs']
            require(set(p_inputs) == {'args', 'kwargs'} and len(p_inputs['args']) == 1 and
                    set(p_inputs['kwargs']) == {'rope_freqs', 'cu_seqlens', 'max_seqlen'}, 'Attention interface changed')
            x = p_inputs['args'][0]
            require(x.shape == (ROWS[index], 5376) and x.dtype == torch.bfloat16 and
                    prefix['output'].shape == replay['output'].shape == x.shape and
                    prefix['output'].dtype == replay['output'].dtype == torch.bfloat16,
                    'Full attention tensor M/channel/dtype differs')
            byte_equal(p_inputs, r_inputs, case_id+'.attention_inputs')
            byte_equal(prefix['output'], replay['output'], case_id+'.attention_output')
            p_sig = signature(p_inputs)
            require(p_sig == before['input_signature'] == after['input_signature'] and
                    tensor_record(prefix['output']) == before['output_signature'] == after['output_signature'],
                    'Actual full input/output signatures differ')
            finite(p_inputs, case_id+'.inputs')
            finite(prefix['output'], case_id+'.output')
            cu = p_inputs['kwargs']['cu_seqlens']
            cu_array = cu.numpy()
            require(cu.dtype == torch.int32 and cu.ndim == 1 and int(cu_array[0]) == 0 and
                    int(cu_array[-1]) == ROWS[index] and np.all(np.diff(cu_array) >= 0) and
                    np.count_nonzero(np.diff(cu_array) > 0) == 2 and
                    cu_array.tolist() == run['inputs'][case_id]['cu_seqlens'] and
                    p_inputs['kwargs']['max_seqlen'] == raw['input_kwargs']['packed_seq_params']['max_seqlen_q'],
                    'Packed sequence boundaries/max length differ')
            byte_equal(cu, raw['input_kwargs']['packed_seq_params']['cu_seqlens_q'], case_id+'.original_cu')
            x_signatures.append(tensor_record(x))
            for offset in range(0, ROWS[index], 256):
                bits = x[offset:offset+256].contiguous().view(torch.uint16).numpy().astype(np.uint32) << 16
                span = np.maximum(span, np.abs(bits.view(np.float32)).max(axis=0))
            result['cases'].append(dict(case_id=case_id, case_index=index, full_rows=ROWS[index],
                actual_dit_input_signature=raw_signature, attention_input_signature=p_sig,
                attention_output_signature=before['output_signature'], full_input_kwargs_byte_exact=True,
                full_output_byte_exact=True, source_state_bound=True, prefix_sdpa=4, wrapper_sdpa=2))
            del prefix, replay, raw, old_capture, p_inputs, r_inputs, x

        cache = load(run['cache_bridge'])
        require(Path(run['cache_bridge']['file']).resolve() == (DATA/'cache_bridge.pt').resolve(), 'Wrong v2 cache artifact')
        require(cache['sample_count'] == 2 and cache['visited_case_indices'] == [0, 1] and
                cache['actual_parse_ipts'] is True and cache['actual_repartition'] is True and cache['actual_extract'] is True and
                cache['repartition'] == dict(max_batch_size=1, max_size=-1, standardize=False, reshape=True),
                'Saved parser/extract evidence differs')
        require(len(cache['raw_indices']) == len(cache['parsed_indices']) == 2, 'Case-index cache size changed')
        for index, (raw, parsed) in enumerate(zip(cache['raw_indices'], cache['parsed_indices'])):
            require(raw.shape == parsed.shape == (1, 1) and raw.dtype == parsed.dtype == torch.int64 and
                    int(raw.item()) == int(parsed.item()) == index, 'Case-index tensor changed')
            byte_equal(raw, parsed, f'case_index[{index}]')
            expected_extract = signature(dict(args=[parsed], kwargs={}))
            require(cache['extraction_signatures'][index] == expected_extract, 'Extracted index signature differs')
        require(signature(cache['raw_indices']) == cache['raw_signature'] == cache['parsed_signature'] ==
                signature(cache['parsed_indices']), 'Raw/parsed full index signatures differ')
        for name, value in check['index_cache_check'].items():
            require(cache[name] == value, f'Runtime index cache differs from real CPU parser check: {name}')
        require(cache['x_acts_shapes'] == [[ROWS[0], 5376], [ROWS[1], 5376]] and
                cache['x_acts_signatures'] == x_signatures and cache['span_reference_exact'] is True and
                cache['x_acts_repartitioned'] is False and cache['span'].shape == (5376,) and
                cache['span'].dtype == torch.float32 and tensor_record(cache['span']) == cache['span_signature'],
                'Actual x_acts/span domain/layout/signature differs')
        saved_span = cache['span'].numpy()
        require(np.isfinite(saved_span).all() and np.array_equal(span.view(np.uint32), saved_span.view(np.uint32)),
                'Independent all-row channel absmax does not match saved actual span bytes')
        setup = read(verify(run['model_setup']))
        require(setup['status'] == 'complete' and setup['complete_dit_calls'] == 0,
                'BF16 setup completed unexpected DiT calls')
        artifact_bytes += run['cache_bridge']['bytes'] + run['model_setup']['bytes']
        require(artifact_bytes == run['data_bytes'], 'Artifact byte accounting differs')
        result['cache_bridge'] = dict(raw_and_parsed_indices_exact=True, visited_case_indices=[0, 1],
            x_acts_shapes=cache['x_acts_shapes'], x_acts_signatures=x_signatures,
            span_signature=cache['span_signature'], independently_recomputed_full_rows=sum(ROWS),
            span_all_channels_byte_exact=True, span_min=float(span.min()), span_max=float(span.max()))
        result['resource_receipts'] = dict(seconds_total=run['seconds_total'], data_bytes=artifact_bytes,
            peak_allocated_gib=run['peak_allocated_gib'], sampled_peak_gpu_used_mib=receipt['sampled_peak_gpu_used_mib'],
            prefix_calls=2, wrapper_calls=2, complete_dit_calls=0, sdpa_calls=12, worker_exit_code=0, supervisor_exit_code=0)
        result['scope_limits'] = [
            'Independent CPU comparison of saved complete inputs/kwargs/outputs and full-row per-channel absmax; no model or parser replay.',
            'Real parser execution, immutable live model identity and forward counters are source-bound runner receipts, not independently instrumented executions.',
            'Two original variable-M cases validate only this no-intervention bridge, not complete smoothing/LR calibration, selected candidates or deployment.',
            'No model weights were refitted or changed; no method novelty, quantization error improvement or perceptual quality is established.',
            'Supervisor device-memory peak is sampled, not a continuous bound on every opaque-library allocation.'
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
