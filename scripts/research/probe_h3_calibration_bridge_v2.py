#!/usr/bin/env python3
"""E070-A v2 (unwrap decorated function for source identity; no protocol change): real variable-length H3 attention through a case-index eval cache.

Two BF16 DiT prefixes stop at blocks.0.attn, followed by exactly two attention
replays. Actual DeepCompressor cache/parser APIs are used without fitting.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DC = ROOT/'third_party/deepcompressor'
sys.path[:0] = [str(HERE), str(DC)]
import probe_h3_plain_baseline as base
import torch

DATA = Path('/data1/models/svdquant-wjq/research/20261004/E070/bridge_v2')
REPORTS = ROOT/'results/research/E070'
PLAN = ROOT/'research_state/06_experiments/E070_h3_baseline_entry_plan.md'
ENVIRONMENT_PROBE = REPORTS/'environment_probe.json'
SOURCE_REPORT = ROOT/'results/research/E065b/evaluate.json'
CASE_IDS = ('p1_s03', 'p20_s03')
TARGET = 'blocks.0.attn'
WALL_SECONDS, MAX_DATA_BYTES = 900, 4*1024**3
save, file_record, signature = base.save, base.file_record, base.tree_signature
load_complete = base.inherited.load_complete


def require(value, message):
    if not value:
        raise RuntimeError(message)


def phase_report(phase):
    return REPORTS/(phase+'_bridge_v2.json')


def load_api(report):
    """Import real library classes; failures stop before opening any model."""
    try:
        from deepcompressor.data.cache import TensorCache, TensorsCache
        from deepcompressor.data.common import TensorType
        from deepcompressor.calib.config import (
            SmoothCalibConfig, SmoothSpanMode, SearchBasedCalibObjective,
            SearchBasedCalibGranularity, SearchBasedCalibStrategy)
        from deepcompressor.calib.smooth import SmoothCalibrator, get_smooth_span
        from deepcompressor.calib.search import SearchBasedCalibrator
    except BaseException as exc:
        report['actual_cache_import'] = dict(status='failed', error=repr(exc))
        raise RuntimeError('Actual DeepCompressor cache/parser import failed; no stand-in or model execution') from exc
    require(SmoothCalibrator._parse_ipts is SearchBasedCalibrator._parse_ipts,
            'Expected inherited actual SearchBasedCalibrator._parse_ipts')
    config = SmoothCalibConfig(objective=SearchBasedCalibObjective.OutputsError,
        granularity=SearchBasedCalibGranularity.Layer, strategy=SearchBasedCalibStrategy.GridSearch,
        sample_batch_size=1, sample_size=-1, alpha=.5, beta=-2, num_grids=20,
        spans=[(SmoothSpanMode.AbsMax, SmoothSpanMode.AbsMax)], allow_low_rank=False)
    parser = SmoothCalibrator(tensor_type=TensorType.Weights, config=config,
        w_quantizer=None, x_quantizer=None, y_quantizer=None, develop_dtype=torch.float32)
    require(parser.objective == SearchBasedCalibObjective.OutputsError and
            parser.granularity == SearchBasedCalibGranularity.Layer and
            parser.config.sample_batch_size == 1 and parser.config.sample_size == -1 and not parser.needs_quant,
            'Actual parser configuration differs from the frozen no-fit contract')
    members = dict(TensorCache=TensorCache, TensorsCache=TensorsCache,
        SearchBasedCalibrator=SearchBasedCalibrator, SmoothCalibrator=SmoothCalibrator,
        get_smooth_span=get_smooth_span)
    paths = {name: str(Path(inspect.getfile(inspect.unwrap(value))).resolve()) for name, value in members.items()}
    require(all(Path(p).is_relative_to(DC.resolve()) for p in paths.values()), 'Unexpected DeepCompressor installation')
    report['actual_cache_import'] = dict(status='complete', classes_and_functions=paths,
        parse_ipts_owner='deepcompressor.calib.search.SearchBasedCalibrator',
        actual_parser_class='deepcompressor.calib.smooth.SmoothCalibrator',
        config=dict(objective='OutputsError', granularity='Layer', sample_batch_size=1, sample_size=-1,
                    tensor_type='Weights', quantizers_enabled=False, fit_or_ask_called=False))
    return types.SimpleNamespace(TensorCache=TensorCache, TensorsCache=TensorsCache, parser=parser,
        get_smooth_span=get_smooth_span, AbsMax=SmoothSpanMode.AbsMax)


def prerequisites(report):
    require(PLAN.is_file(), 'E070 frozen plan required')
    environment = load_complete(ENVIRONMENT_PROBE)
    require(environment['returncode'] == 0, 'Existing environment import probe must pass')
    for name in ('TORCH_EXTENSIONS_DIR', 'TORCH_CUDA_ARCH_LIST', 'CUDA_HOME'):
        require(os.environ.get(name) == environment['environment'][name], f'Frozen environment differs: {name}')
    parent = load_complete(SOURCE_REPORT)
    require(parent['experiment'] == 'E065b' and len(parent['calibration_inputs']) == 8 and
            len(parent['calibration_cases']) == 8, 'Complete inherited E065 calibration identities required')
    base.check_sources(parent['sources'])
    old_check_path = base.verify_file(parent['e014_check'])
    old_check = load_complete(old_check_path)
    require(old_check['cuda_initialized'] is False, 'Inherited E014 model setup check is not CPU-only')
    base.check_sources(old_check['sources'])
    e014_manifest = json.loads(base.verify_file(old_check['manifest']).read_text())
    require(old_check['manifest']['file'] == str(base.MANIFEST.resolve()), 'Unexpected E014 setup manifest')
    asset = load_complete(base.verify_file(e014_manifest['asset_reference']))
    for path, row in asset['assets'].items():
        stat = Path(path).stat()
        require(stat.st_size == row['bytes'] and stat.st_mtime_ns == row['mtime_ns'], 'Original model asset changed')
    cases, captures, checks = [], {}, {}
    for case_id in CASE_IDS:
        row = next(r for r in parent['calibration_inputs'] if r['id'] == case_id)
        require(row['kind'] == 'raw_dit' and row['step'] == 3, 'Frozen calibration case changed')
        base.verify_file(row['artifact'])
        captured = next(r for r in parent['calibration_cases'] if r['case_id'] == case_id)
        base.verify_file(captured['artifact'])
        cases.append(row)
        captures[case_id] = captured['artifact']
        checks[case_id] = parent['calibration_checks'][case_id]
    require([r['prompt_id'] for r in cases] == [1, 20], 'Do not substitute calibration prompts')
    api = load_api(report)
    sources = {p: file_record(p) for p in set(parent['sources']) | set(old_check['sources'])}
    for path in (Path(__file__), PLAN, ENVIRONMENT_PROBE, SOURCE_REPORT, old_check_path,
                 Path(base.__file__), Path(old_check['manifest']['file'])):
        sources[str(path.resolve())] = file_record(path)
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('deepcompressor') and path and Path(path).suffix == '.py':
            require(Path(path).resolve().is_relative_to(DC.resolve()), 'Mixed DeepCompressor source roots')
            sources[str(Path(path).resolve())] = file_record(path)
    report.update(sources=sources, plan=file_record(PLAN), source_evaluation=file_record(SOURCE_REPORT),
        source_capture_artifacts=captures, source_input_checks=checks, calibration_inputs=cases,
        setup_check=parent['e014_check'], setup_manifest=old_check['manifest'],
        environment_probe=file_record(ENVIRONMENT_PROBE), torch=torch.__version__,
        environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'TORCH_EXTENSIONS_DIR',
            'TORCH_CUDA_ARCH_LIST', 'CUDA_HOME', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION')})
    return cases, e014_manifest, api


def load_raw(case):
    payload = torch.load(base.verify_file(case['artifact']), map_location='cpu', weights_only=False, mmap=True)
    require(set(payload) == {'input_args', 'input_kwargs', 'meta'}, 'Original raw calibration schema changed')
    require(payload['meta']['prompt_id'] == case['prompt_id'] and payload['meta']['step'] == case['step'],
            'Original calibration metadata changed')
    value = dict(args=payload['input_args'], kwargs=payload['input_kwargs'])
    require(not value['args'] and value['kwargs'].get('control_hints') is None, 'Unexpected positional/control input')
    return value, payload['meta']


def positive_segments(cu):
    require(cu.ndim == 1 and cu.dtype in (torch.int32, torch.int64) and int(cu[0]) == 0 and
            bool((cu[1:] >= cu[:-1]).all()), 'Invalid cumulative packed sequence lengths')
    return int((cu[1:] > cu[:-1]).sum())


def parsed_index_cache(api):
    ids = [torch.tensor([[i]], dtype=torch.int64) for i in range(len(CASE_IDS))]
    cache = api.TensorCache(data=ids, channels_dim=1, num_cached=2, num_total=2, num_samples=2,
                            orig_device=torch.device('cpu'))
    # This real inherited method directly invokes TensorCache.repartition with
    # max_batch_size=1, max_size=-1, standardize=False, reshape=True.
    parsed = api.parser._parse_ipts(api.TensorsCache(cache), set_device=True)
    require(parsed.num_tensors == 1 and len(parsed.front().data) == 2,
            'Actual parser dropped or duplicated a case-index sample')
    extracted = [parsed.extract(i, {}) for i in range(2)]
    require(all(len(r.args) == 1 and not r.kwargs and r.args[0].shape == (1, 1) and
                r.args[0].dtype == torch.int64 and int(r.args[0].item()) == i for i, r in enumerate(extracted)),
            'Actual repartition/extract changed case-index identity')
    require(signature(ids) == signature(parsed.front().data), 'Actual index cache data changed')
    return parsed, dict(raw_indices=ids, parsed_indices=parsed.front().data,
        raw_signature=signature(ids), parsed_signature=signature(parsed.front().data),
        extraction_signatures=[signature(dict(args=r.args, kwargs=r.kwargs)) for r in extracted],
        repartition=dict(max_batch_size=1, max_size=-1, standardize=False, reshape=True),
        sample_count=2, actual_parse_ipts=True, actual_repartition=True, actual_extract=True)


@torch.inference_mode()
def cpu_check(report, cases, api):
    require(not torch.cuda.is_initialized(), 'Bridge check must be CPU-only')
    report['inputs'] = {}
    for case in cases:
        value, metadata = load_raw(case)
        source = report['source_input_checks'][case['id']]
        actual = signature(value)
        require(actual == source['actual_dit_input_signature'] and metadata == source['metadata'],
                'Raw case identity differs from E065 capture')
        captured = torch.load(base.verify_file(report['source_capture_artifacts'][case['id']]),
                              map_location='cpu', weights_only=True, mmap=True)
        require(captured['actual_dit_input_signature'] == actual and captured['metadata'] == metadata,
                'E065 calibration artifact does not bind this raw input')
        kw = value['kwargs']
        x = kw['x']
        require(x.ndim == 3 and x.shape[0] == 1 and x.dtype == torch.bfloat16, 'Raw H3 x layout changed')
        cu = kw['packed_seq_params']['cu_seqlens_q']
        refcu = kw['refiner_packed_seq_params']['cu_seqlens_q']
        segments, ref_segments = positive_segments(cu), positive_segments(refcu)
        require(int(cu[-1]) == x.shape[1] == source['full_rows'], 'Original packed M changed')
        require(segments == 2 and ref_segments == 1, 'Frozen two-case SDPA budget changed')
        report['inputs'][case['id']] = dict(artifact=case['artifact'], metadata=metadata,
            actual_dit_input_signature=actual, full_rows=x.shape[1], cu_seqlens=cu.tolist(),
            refiner_cu_seqlens=refcu.tolist(), expected_prefix_sdpa=2*ref_segments+segments,
            expected_wrapper_sdpa=segments)
    require([report['inputs'][case]['full_rows'] for case in CASE_IDS] == [22400, 22464],
            'Frozen cases must preserve their distinct actual M values')
    _, index = parsed_index_cache(api)
    report['index_cache_check'] = {k: v for k, v in index.items() if k not in ('raw_indices', 'parsed_indices')}
    require(not torch.cuda.is_initialized(), 'Actual cache/parser check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, new_model_forwards=0)


def guard(args, report):
    require(time.time() < args.deadline_unix, 'E070 bridge original deadline reached')
    require(report['attempted_prefix_calls'] <= 2 and report['attempted_wrapper_calls'] <= 2 and
            report['complete_dit_calls'] == 0, 'E070 bridge model-call budget exceeded')
    require(report['data_bytes'] < MAX_DATA_BYTES, 'E070 bridge data budget reached 4 GiB')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E070 bridge peak allocation reached 60 GiB')


def write_tensor(payload, path, report):
    require(not path.exists(), f'Refusing artifact overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    record = file_record(path)
    report['data_bytes'] += record['bytes']
    require(report['data_bytes'] < MAX_DATA_BYTES, 'E070 bridge data budget reached 4 GiB')
    return record


def model_identity(model):
    return {n: (id(t), None if torch.is_inference(t) else t._version, t.untyped_storage().data_ptr(),
                tuple(t.shape), str(t.dtype), str(t.device))
            for n, t in list(model.named_parameters())+list(model.named_buffers())}


class PrefixCaptured(RuntimeError):
    pass


@torch.inference_mode()
def capture_prefix(pipe, case, checked, args, report):
    guard(args, report)
    require(report['attempted_prefix_calls'] < 2, 'No additional prefix allowed')
    value, metadata = load_raw(case)
    expected = checked['inputs'][case['id']]
    require(signature(value) == expected['actual_dit_input_signature'], 'Raw input changed after CPU check')
    target = pipe.dit.get_submodule(TARGET)
    require(not target._forward_pre_hooks and not target._forward_hooks, 'Unexpected attention hooks')
    captured, raw_inputs = {}, []

    def dit_pre(_module, positional, kwargs):
        actual = signature(dict(args=positional, kwargs=kwargs))
        require(actual == expected['actual_dit_input_signature'], 'Actual prefix DiT input changed')
        raw_inputs.append(actual)

    def attn_pre(_module, positional, kwargs):
        require(not captured, 'Target attention was reached more than once')
        require(len(positional) == 1 and set(kwargs) == {'rope_freqs', 'cu_seqlens', 'max_seqlen'},
                'Original attention signature changed')
        require(positional[0].ndim == 2 and positional[0].shape[0] == expected['full_rows'] and
                positional[0].dtype == torch.bfloat16, 'Attention M/dtype differs from raw case')
        # The frozen tree_cpu helper uses blocking CUDA->CPU copies.
        actual = base.inherited.tree_cpu(dict(args=positional, kwargs=kwargs))
        captured['attention_inputs'] = actual
        captured['input_signature'] = signature(actual)
        captured['live_inputs'] = dict(args=positional, kwargs=kwargs)

    def attn_post(_module, _pos, output):
        require('attention_inputs' in captured and 'output' not in captured, 'Unexpected attention capture sequence')
        require(signature(captured['live_inputs']) == captured['input_signature'], 'Attention mutated x/kwargs')
        require(torch.is_tensor(output) and output.shape == captured['attention_inputs']['args'][0].shape and
                output.dtype == torch.bfloat16 and bool(torch.isfinite(output).all()), 'Invalid real attention output')
        captured['output'] = output.detach().cpu()
        captured.pop('live_inputs')
        raise PrefixCaptured('Stopped immediately after real blocks.0.attn')

    def forbid_complete(_module, _pos, _output):
        report['complete_dit_calls'] += 1
        raise RuntimeError('Full DiT completion forbidden by E070-A')

    handles = [pipe.dit.register_forward_pre_hook(dit_pre, with_kwargs=True),
               pipe.dit.register_forward_hook(forbid_complete),
               target.register_forward_pre_hook(attn_pre, with_kwargs=True),
               target.register_forward_hook(attn_post)]
    audit = base.RuntimeAudit()
    audit.phase = 'resident_bf16'
    report['attempted_prefix_calls'] += 1
    save(report, args.output)
    try:
        with audit.installed():
            try:
                base.gpu_call(pipe, dict(kind='raw_dit'), value)()
            except PrefixCaptured:
                pass
            else:
                raise RuntimeError('Prefix did not stop at the target attention')
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    row = dict(audit.row())
    require(row['sdpa_calls'] == expected['expected_prefix_sdpa'] and row['scaled_mm_calls'] == row['disk_loads'] == 0,
            'Prefix BF16 runtime path changed')
    require(len(raw_inputs) == 1 and 'output' in captured, 'Incomplete real prefix capture')
    report['complete_prefix_calls'] += 1
    report['runtime_totals']['sdpa_calls'] += row['sdpa_calls']
    payload = dict(case_id=case['id'], metadata=metadata, actual_dit_input_signature=raw_inputs[0],
        attention_inputs=captured['attention_inputs'], output=captured['output'])
    artifact = write_tensor(payload, DATA/'prefix'/(case['id']+'.pt'), report)
    report['prefix_cases'].append(dict(case_id=case['id'], artifact=artifact,
        input_signature=captured['input_signature'], output_signature=base.tensor_record(captured['output']),
        full_rows=expected['full_rows'], input_kwargs_unchanged=True, runtime_audit=row))
    guard(args, report)
    save(report, args.output)
    del value, captured
    gc.collect()
    torch.cuda.empty_cache()
    return payload


class CaseIndexAttention(torch.nn.Module):
    """Only the index is cached by the generic evaluation-input parser."""
    def __init__(self, attention, cases, args, report):
        super().__init__()
        self.attention = attention
        self.cases = tuple(cases)
        self.args = args
        self.report = report
        self.visited = []
        self.records = []
        self.input_signatures = tuple(signature(c['attention_inputs']) for c in self.cases)

    @torch.inference_mode()
    def forward(self, case_id):
        report, args = self.report, self.args
        guard(args, report)
        require(torch.is_tensor(case_id) and case_id.shape == (1, 1) and case_id.dtype == torch.int64,
                'Evaluation bridge requires one real case-index scalar [1,1]')
        index = int(case_id.item())
        require(0 <= index < len(self.cases) and index not in self.visited and
                report['attempted_wrapper_calls'] < 2, 'Repeated, invalid or excessive wrapper case')
        case = self.cases[index]
        frozen = case['attention_inputs']
        expected = self.input_signatures[index]
        require(signature(frozen) == expected, 'Immutable full-case inputs changed before wrapper')
        gpu = base.tree_device(frozen, 'cuda')
        require(signature(gpu) == expected, 'Full case transfer changed input bytes or kwargs')
        audit = base.RuntimeAudit()
        audit.phase = 'resident_bf16'
        report['attempted_wrapper_calls'] += 1
        save(report, args.output)
        with audit.installed():
            output = self.attention(*gpu['args'], **gpu['kwargs'])
            torch.cuda.synchronize()
        cpu_output = output.detach().cpu()
        require(signature(gpu) == expected == signature(frozen), 'Wrapper attention mutated real x/kwargs')
        require(cpu_output.shape == frozen['args'][0].shape and cpu_output.dtype == torch.bfloat16 and
                bool(torch.isfinite(cpu_output).all()), 'Wrapper changed actual per-case M or output dtype')
        require(torch.equal(cpu_output, case['output']) and
                base.tensor_record(cpu_output) == base.tensor_record(case['output']), 'Wrapper replay is not byte-exact')
        counts = dict(audit.row())
        expected_sdpa = positive_segments(frozen['kwargs']['cu_seqlens'])
        require(counts['sdpa_calls'] == expected_sdpa and counts['scaled_mm_calls'] == counts['disk_loads'] == 0,
                'Wrapper BF16 attention runtime path changed')
        self.visited.append(index)
        report['complete_wrapper_calls'] += 1
        report['runtime_totals']['sdpa_calls'] += counts['sdpa_calls']
        artifact = write_tensor(dict(case_id=case['case_id'], case_index=index,
            attention_inputs=frozen, output=cpu_output), DATA/'wrapper'/(case['case_id']+'.pt'), report)
        row = dict(case_id=case['case_id'], case_index=index, artifact=artifact,
            input_signature=expected, output_signature=base.tensor_record(cpu_output),
            full_rows=cpu_output.shape[0], byte_exact_reference=True, input_kwargs_unchanged=True,
            runtime_audit=counts)
        self.records.append(row)
        report['wrapper_cases'].append(row)
        save(report, args.output)
        guard(args, report)
        return output


@torch.inference_mode()
def run(args, report, cases, e014_manifest, api):
    checked = load_complete(phase_report('check'))
    require(checked['cuda_initialized'] is False and checked['new_model_forwards'] == 0,
            'Complete CPU-only bridge check required')
    for field in ('sources', 'plan', 'source_evaluation', 'source_capture_artifacts', 'source_input_checks',
                  'calibration_inputs', 'setup_check', 'setup_manifest', 'environment_probe', 'actual_cache_import', 'torch'):
        require(checked[field] == report[field], f'Frozen bridge check binding changed: {field}')
    report['cpu_check_reference'] = file_record(phase_report('check'))
    report['inputs'] = checked['inputs']
    require(not DATA.exists(), 'Refusing existing bridge data directory')
    DATA.mkdir(parents=True)
    guard(args, report)
    old_check = load_complete(base.verify_file(report['setup_check']))
    setup = dict(experiment='E070_inherited_E014_BF16_setup', status='running', complete_dit_calls=0,
                 sources=old_check['sources'])
    setup_path = DATA/'setup_bf16.json'
    setup_args = types.SimpleNamespace(arm='bf16', deadline_unix=args.deadline_unix,
        check_report=Path(report['setup_check']['file']), output=setup_path)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    report['data_bytes'] += setup_path.stat().st_size
    require(pipe.text_encoder is None and pipe.video_vae is None and pipe.audio_vae is None,
            'Bridge process must contain only the resident BF16 DiT')
    identity = model_identity(pipe.dit)
    payloads = [capture_prefix(pipe, case, checked, args, report) for case in cases]
    require(model_identity(pipe.dit) == identity, 'Prefix changed model storage/version identity')
    parsed, cache_evidence = parsed_index_cache(api)
    require({k: v for k, v in cache_evidence.items() if k not in ('raw_indices', 'parsed_indices')} ==
            checked['index_cache_check'], 'Runtime index parser differs from CPU check')
    # This separate actual activation cache preserves whole variable-M tensors.
    # It is used for span statistics, never passed to eval-input repartition.
    x_cache = api.TensorCache(data=[c['attention_inputs']['args'][0] for c in payloads], channels_dim=1,
        num_cached=2, num_total=2, num_samples=2, orig_device=torch.device('cpu'))
    x_acts = api.TensorsCache(x_cache)
    standardized = x_acts.front().get_standardized_data(reshape=False)
    require([tuple(x.shape) for x in standardized] == [(22400, 5376), (22464, 5376)],
            'Span activation cache did not preserve actual variable-M tensors')
    span = api.get_smooth_span(standardized, group_shape=[-1, -1], span_mode=api.AbsMax,
                               device='cpu', dtype=torch.float32)
    reference_span = torch.stack([x.abs().amax(dim=0).float() for x in standardized]).amax(dim=0)
    require(torch.equal(span, reference_span) and span.shape == (5376,) and bool(torch.isfinite(span).all()),
            'Actual activation span differs from full-case channel absmax')
    cache_evidence.update(x_acts_shapes=[list(x.shape) for x in standardized],
        x_acts_signatures=[base.tensor_record(x) for x in standardized],
        span=span, span_signature=base.tensor_record(span), span_reference_exact=True,
        x_acts_repartitioned=False, scope='Actual full attention x only; case indices and kwargs excluded from span')
    wrapper = CaseIndexAttention(pipe.dit.get_submodule(TARGET), payloads, args, report)
    for index in range(len(parsed.front().data)):
        guard(args, report)
        extracted = parsed.extract(index, {})
        output = wrapper(*extracted.args, **extracted.kwargs)
        require(output.shape[0] == checked['inputs'][CASE_IDS[index]]['full_rows'], 'Wrapper returned a sliced sequence')
        del output
        gc.collect()
        torch.cuda.empty_cache()
    require(wrapper.visited == [0, 1] and model_identity(pipe.dit) == identity,
            'Wrapper visitation or model identity changed')
    require(all(signature(c['attention_inputs']) == sig for c, sig in zip(payloads, wrapper.input_signatures, strict=True)),
            'Immutable full case cache changed during evaluation')
    cache_evidence['visited_case_indices'] = wrapper.visited
    report['cache_bridge'] = write_tensor(cache_evidence, DATA/'cache_bridge.pt', report)
    report['bridge_contract'] = dict(actual_parse_ipts=True, actual_repartition=True, actual_extract=True,
        objective='OutputsError', granularity='Layer', sample_batch_size=1, sample_size=-1,
        case_indices=wrapper.visited, full_rows=[22400, 22464], byte_exact_all=True, immutable_inputs_all=True,
        x_acts_used_for_actual_absmax=True, no_token_slices=True, no_fit=True)
    require(report['attempted_prefix_calls'] == report['complete_prefix_calls'] == 2 and
            report['attempted_wrapper_calls'] == report['complete_wrapper_calls'] == 2 and
            report['complete_dit_calls'] == 0 and report['runtime_totals'] ==
            dict(sdpa_calls=12, scaled_mm_calls=0, disk_loads=0), 'Final E070 bridge count contract failed')
    guard(args, report)
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'run'))
    parser.add_argument('--budget-start-unix', type=float)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.output = phase_report(args.phase)
    require(not args.output.exists(), f'Refusing report overwrite: {args.output}')
    if args.phase == 'run' and (args.deadline_unix is None or args.budget_start_unix is None or
            not 0 < args.deadline_unix-args.budget_start_unix <= WALL_SECONDS or
            not args.budget_start_unix <= time.time() < args.deadline_unix):
        parser.error('Run requires one original start/deadline pair, at most 900 seconds apart')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    report = dict(experiment='E070_bridge', phase=args.phase, status='running',
        budget_start_unix=args.budget_start_unix, deadline_unix=args.deadline_unix,
        wall_budget_seconds=WALL_SECONDS, data_byte_limit=MAX_DATA_BYTES, data_bytes=0,
        attempted_prefix_calls=0, complete_prefix_calls=0, attempted_wrapper_calls=0, complete_wrapper_calls=0,
        complete_dit_calls=0, activation_packs=0, new_candidate_weights=0, calibration_fit_calls=0,
        text_encoder_calls=0, video_vae_calls=0, audio_vae_calls=0,
        runtime_totals=dict(sdpa_calls=0, scaled_mm_calls=0, disk_loads=0), prefix_cases=[], wrapper_cases=[],
        scope='Actual packed-sequence bridge only; not a complete calibration, quantization or quality result')
    started = time.monotonic()
    try:
        if args.phase == 'run':
            def expired(_signum, _frame):
                raise TimeoutError('E070 bridge original absolute deadline reached')
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        cases, manifest, api = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, cases, api)
        else:
            run(args, report, cases, manifest, api)
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if args.phase == 'run':
            signal.setitimer(signal.ITIMER_REAL, 0)
        report['seconds_total'] = time.monotonic()-started
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if torch.cuda.is_initialized():
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        print(json.dumps(dict(status=report['status'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
