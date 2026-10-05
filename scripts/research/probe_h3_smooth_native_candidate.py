#!/usr/bin/env python3
"""E071: one real SmoothCalibrator candidate, scored and exported natively.

Only candidate installation/recovery is adapted. The library's calibration,
span reset, ask, output-error scoring and tell implementations remain inherited.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
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
import probe_h3_calibration_bridge_v2 as bridge
import export_h3_native_nvfp4 as exporter
from h3_native_nvfp4 import NativeH3Linear, FORMAT, RECIPE
from h3_nvfp4_zero_sf_compat import (pack_activation_legacy_zero_sf as pack_activation_legacy,
    collect_fastpack_checks, COMPATIBILITY)
from minimax_h3_svdquant_common import nvfp4_qdq, H3_DIT_PATH
from safetensors import safe_open
import torch
from torch.overrides import TorchFunctionMode

DATA = Path('/data1/models/svdquant-wjq/research/20261004/E071')
REPORTS = ROOT/'results/research/E071'
PLAN = ROOT/'research_state/06_experiments/E071_h3_native_candidate_plan.md'
E070 = ROOT/'results/research/E070'
SOURCE_RUN = E070/'run_bridge_v2.json'
SOURCE_INDEPENDENT = E070/'independent_bridge_v2.json'
SOURCE_EXIT = E070/'process_exit_bridge.json'
INVENTORY = Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export/manifest.json')
UPSTREAM = ROOT/'results/research/E069/upstream/deepcompressor/nn/patch/lowrank.py'
TARGET = 'blocks.0.attn.qkv_proj'
CASES = ('p1_s03', 'p20_s03')
ROWS = (22400, 22464)
SHAPE = (21504, 5376)
WEIGHT_SHA = '1727c595c15e1c57f16832e961946975d1a6073f445e7b2e6ccd489554eac264'
WALL_SECONDS, MAX_BYTES, MAX_GPU = 900, 8*1024**3, 60*1024**3
save, record, signature, tensor_record = base.save, base.file_record, base.tree_signature, base.tensor_record
require, load_complete = bridge.require, base.inherited.load_complete


def report_path(phase):
    return REPORTS/(phase+'.json')


def load_api(report):
    from deepcompressor.data.cache import TensorCache, TensorsCache
    from deepcompressor.data.common import TensorType
    from deepcompressor.calib.config import (SmoothCalibConfig, SmoothSpanMode,
        SearchBasedCalibObjective, SearchBasedCalibGranularity, SearchBasedCalibStrategy)
    from deepcompressor.calib.smooth import SmoothCalibrator
    from deepcompressor.calib.search import SearchBasedCalibrator

    class NativeCandidateCalibrator(SmoothCalibrator):
        def _process_wgts_centric_mod(self, wgts, mods, update_state_dict=True, **kwargs):
            require(update_state_dict and not kwargs and len(wgts) == len(mods) == 1,
                    'Unexpected candidate installation arguments')
            require(wgts[0] is self.executor.original.weight and mods[0] is self.executor.original,
                    'Calibrator selected a different parameter/module')
            self.executor.install(self.candidate, self.span_pairs[0])

        def _recover_mod(self):
            self.executor.recover()
            super()._recover_mod()

    for method in ('calibrate', 'reset', '_reset', 'ask', '_ask', '_calibrate_wgts', 'tell', '_tell'):
        require(getattr(NativeCandidateCalibrator, method) is getattr(SmoothCalibrator, method),
                f'Library method must remain inherited: {method}')
    config = SmoothCalibConfig(objective=SearchBasedCalibObjective.OutputsError,
        granularity=SearchBasedCalibGranularity.Layer, strategy=SearchBasedCalibStrategy.Manual,
        alpha=.5, beta=.5, spans=[(SmoothSpanMode.AbsMax, SmoothSpanMode.AbsMax)],
        sample_batch_size=1, sample_size=-1, degree=2, allow_low_rank=True)
    calibrator = NativeCandidateCalibrator(tensor_type=TensorType.Weights, config=config,
        w_quantizer=None, x_quantizer=None, y_quantizer=None, develop_dtype=torch.float32)
    require(calibrator.population_size == calibrator.num_iters == 1 and
            calibrator.alpha_beta_pairs == [(.5, .5)], 'Exactly one real Manual candidate required')
    require(not any((calibrator.needs_quant, calibrator.needs_w_quant, calibrator.needs_x_quant,
                     calibrator.needs_y_quant)), 'Default quantizer path must be disabled')
    functions = {name: getattr(SmoothCalibrator, name) for name in
        ('calibrate', 'reset', '_reset', 'ask', '_ask', '_calibrate_wgts', 'tell', '_tell')}
    report['library_contract'] = dict(objective='OutputsError', granularity='Layer', strategy='Manual',
        alpha=.5, beta=.5, spans=[['AbsMax', 'AbsMax']], sample_batch_size=1, sample_size=-1,
        degree=2, develop_dtype='torch.float32', population_size=1, num_iters=1,
        w_quantizer=None, x_quantizer=None, y_quantizer=None, needs_quant=False,
        executor='Explicit native module install/recover adapter; not the default library Quantizer or LR hook path',
        score='BF16 subtraction, then FP32 square/sum per case and sequential case accumulation',
        inherited_functions={name: dict(file=str(Path(inspect.getfile(inspect.unwrap(fn))).resolve()),
            function=inspect.unwrap(fn).__qualname__) for name, fn in functions.items()},
        fallback_group_shapes=dict(w=calibrator.w_group_shape, x=calibrator.x_group_shape,
                                   y=calibrator.y_group_shape))
    require(all(Path(row['file']).is_relative_to(DC.resolve()) for row in
                report['library_contract']['inherited_functions'].values()), 'Unexpected library installation')
    return types.SimpleNamespace(TensorCache=TensorCache, TensorsCache=TensorsCache,
        parser=calibrator, calibrator=calibrator, functions=functions,
        SearchBasedCalibrator=SearchBasedCalibrator)


def prerequisites(report):
    require(PLAN.is_file(), 'Frozen E071 plan required')
    parent = load_complete(SOURCE_RUN)
    independent = load_complete(SOURCE_INDEPENDENT)
    receipt = json.loads(SOURCE_EXIT.read_text())
    require(record(SOURCE_RUN) == independent['reports']['run_bridge_v2.json'] and
            record(SOURCE_EXIT) == independent['reports']['process_exit_bridge.json'],
            'E070 independent source/exit binding changed')
    require(receipt['exit_code'] == receipt['worker_exit_code'] == 0 and receipt['timeout_exit'] is False and
            receipt['completed_output_sha256'] == record(SOURCE_RUN)['sha256'], 'E070 must have normal exit0')
    require(parent['bridge_contract']['byte_exact_all'] and parent['complete_prefix_calls'] == 2 and
            parent['complete_wrapper_calls'] == 2 and parent['runtime_totals']['sdpa_calls'] == 12,
            'Completed real bridge required')
    base.check_sources(parent['sources'])
    for row in parent['prefix_cases']:
        base.verify_file(row['artifact'])
    require(tuple(row['case_id'] for row in parent['prefix_cases']) == CASES and
            tuple(row['full_rows'] for row in parent['prefix_cases']) == ROWS, 'Frozen bridge cases changed')
    environment = load_complete(E070/'environment_probe.json')
    for key in ('CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'TORCH_EXTENSIONS_DIR'):
        require(os.environ.get(key) == environment['environment'][key], f'Frozen environment differs: {key}')
    svd_run = load_complete(E070/'exact_svd/evaluate.json')
    svd_check = load_complete(E070/'exact_svd/independent.json')
    require(svd_check['reports']['evaluate.json'] == record(E070/'exact_svd/evaluate.json') and
            svd_run['complete_svd_calls'] == 1, 'Completed exact-SVD resource pilot required')
    require(record(UPSTREAM) == svd_run['binding']['upstream']['source'], 'Pinned upstream SVD source changed')
    inventory = load_complete(INVENTORY)
    row = next(r for r in inventory['layers'] if r['name'] == TARGET)
    require(tuple(row['shape']) == SHAPE and row['source_weight_sha256'] == WEIGHT_SHA,
            'Frozen QKV source identity changed')
    old_check = load_complete(base.verify_file(parent['setup_check']))
    manifest = json.loads(base.verify_file(parent['setup_manifest']).read_text())
    asset = load_complete(base.verify_file(manifest['asset_reference']))
    for path, binding in asset['assets'].items():
        stat = Path(path).stat()
        require(stat.st_size == binding['bytes'] and stat.st_mtime_ns == binding['mtime_ns'],
                f'Original model asset changed: {path}')
    api = load_api(report)
    paths = set(parent['sources']) | set(old_check['sources'])
    paths.update(str(p.resolve()) for p in (Path(__file__), PLAN, SOURCE_RUN, SOURCE_INDEPENDENT, SOURCE_EXIT,
        INVENTORY, UPSTREAM, E070/'exact_svd/evaluate.json', E070/'exact_svd/independent.json',
        Path(bridge.__file__), Path(exporter.__file__), HERE/'h3_native_nvfp4.py',
        HERE/'h3_nvfp4_zero_sf_compat.py', HERE/'wan_native_nvfp4.py'))
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('deepcompressor') and path and Path(path).suffix == '.py':
            require(Path(path).resolve().is_relative_to(DC.resolve()), 'Mixed library installations')
            paths.add(str(Path(path).resolve()))
    report.update(sources={p: record(p) for p in sorted(paths)}, plan=record(PLAN),
        source_bridge=record(SOURCE_RUN), source_independent=record(SOURCE_INDEPENDENT), source_exit=record(SOURCE_EXIT),
        case_sources=parent['prefix_cases'], source_weight_sha256=WEIGHT_SHA,
        setup_check=parent['setup_check'], setup_manifest=parent['setup_manifest'], torch=torch.__version__,
        source_svd=record(E070/'exact_svd/evaluate.json'), upstream_svd=record(UPSTREAM),
        asset_identity=dict(reference=manifest['asset_reference'], validation='Historical full SHA plus fresh size/mtime'),
        environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME',
            'TORCH_CUDA_ARCH_LIST', 'TORCH_EXTENSIONS_DIR', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION')})
    return parent, manifest, api


def load_cases(report):
    cases = []
    for row in report['case_sources']:
        payload = torch.load(base.verify_file(row['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(payload['case_id'] == row['case_id'] and signature(payload['attention_inputs']) == row['input_signature'] and
                tensor_record(payload['output']) == row['output_signature'], 'Saved full attention identity changed')
        require(payload['attention_inputs']['args'][0].shape == (row['full_rows'], SHAPE[1]) and
                set(payload['attention_inputs']['kwargs']) == {'rope_freqs', 'cu_seqlens', 'max_seqlen'},
                'Frozen full attention schema changed')
        cases.append(payload)
    return cases


def actual_caches(api, cases):
    x = api.TensorCache(data=[case['attention_inputs']['args'][0] for case in cases], channels_dim=1,
        num_cached=2, num_total=2, num_samples=2, orig_device=torch.device('cpu'))
    indices = api.TensorCache(data=[torch.tensor([[i]], dtype=torch.int64) for i in range(2)], channels_dim=1,
        num_cached=2, num_total=2, num_samples=2, orig_device=torch.device('cpu'))
    return api.TensorsCache(x), api.TensorsCache(indices)


@contextmanager
def library_call_audit(api, report):
    require(sys.getprofile() is None, 'Do not replace an existing Python profiler')
    names = {inspect.unwrap(fn).__code__: name for name, fn in api.functions.items()}
    calls = {name: 0 for name in api.functions}
    def profile(frame, event, _arg):
        if event == 'call' and frame.f_code in names:
            name = names[frame.f_code]
            calls[name] += 1
            if name == 'tell':
                values = frame.f_locals['error']
                require(len(values) == 1 and values[0].dtype == torch.float32 and values[0].numel() == 1,
                        'Real tell received an unexpected error tensor')
                report.setdefault('scores', {})['library_reported'] = float(values[0].item())
    try:
        sys.setprofile(profile)
        yield calls
    finally:
        sys.setprofile(None)
        report['library_calls'] = calls


@torch.inference_mode()
def cpu_check(report, api):
    require(not torch.cuda.is_initialized(), 'CPUcheck must not initialize CUDA')
    cases = load_cases(report)
    with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as source:
        weight = source.get_tensor(TARGET+'.weight')
        require(TARGET+'.bias' not in source.keys(), 'Frozen target is bias-free')
    require(tuple(weight.shape) == SHAPE and weight.dtype == torch.bfloat16 and
            tensor_record(weight)['sha256'] == WEIGHT_SHA, 'Actual CPU target weight identity differs')
    parameter = torch.nn.Parameter(weight, requires_grad=False)
    x_acts, eval_inputs = actual_caches(api, cases)
    parsed, index_check = bridge.parsed_index_cache(api)
    require(len(parsed.front().data) == len(eval_inputs.front().data) == 2, 'Actual index parser changed cases')
    with library_call_audit(api, report):
        api.calibrator.reset(x_wgts=[parameter], x_acts=x_acts)
        scale = api.calibrator.ask()
    require(report['library_calls'] == dict(calibrate=0, reset=1, _reset=1, ask=1, _ask=1,
                                            _calibrate_wgts=0, tell=0, _tell=0), 'CPU reset/ask call audit changed')
    x_span, w_span = api.calibrator.span_pairs[0]
    raw_x = torch.stack([c['attention_inputs']['args'][0].abs().amax(0).float() for c in cases]).amax(0)
    raw_w = weight.abs().amax(0).float()
    require(torch.equal(x_span, raw_x) and torch.equal(w_span, raw_w), 'Actual None-quantizer span reset differs')
    require(scale.dtype == torch.float32 and scale.shape == (SHAPE[1],) and
            bool(torch.isfinite(scale).all()) and bool((scale > 0).all()) and bool((scale != 1).any()),
            'Invalid/nontrivial CPU candidate scale')
    report['inputs'] = {row['case_id']: dict(source_artifact=row['artifact'],
        attention_input_signature=row['input_signature'], reference_output_signature=row['output_signature'],
        full_rows=row['full_rows']) for row in report['case_sources']}
    report['cpu_span_check'] = dict(x_span=tensor_record(x_span), w_span=tensor_record(w_span),
        raw_scale_fp32=tensor_record(scale), actual_reset=True, actual_ask=True,
        actual_none_quantizer_group_shapes=report['library_contract']['fallback_group_shapes'],
        independent_absmax_exact=True, scale_values=scale.tolist(),
        scope='CPU reset/ask only; no candidate installation, SVD, scoring or model forward')
    report['index_cache_check'] = {k: v for k, v in index_check.items() if k not in ('raw_indices', 'parsed_indices')}
    require(not torch.cuda.is_initialized(), 'CPU span check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, new_model_forwards=0)


def guard(args, report):
    require(time.time() < args.deadline_unix, 'E071 original absolute deadline reached')
    require(report['attention_calls'] <= 4 and report['reload_qkv_calls'] <= 2 and report['activation_packs'] <= 4 and
            report['svd_calls'] <= 1 and report['complete_dit_calls'] == report['prefix_calls'] == 0,
            'E071 call allocation exceeded')
    require(report['data_bytes'] <= MAX_BYTES, 'E071 new-data allocation exceeded')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < MAX_GPU, 'E071 peak allocation reached 60 GiB')


def write_tensor(payload, path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    result = record(path)
    report['data_bytes'] += result['bytes']
    require(report['data_bytes'] <= MAX_BYTES, 'E071 new data exceeded 8 GiB')
    return result


def packet_cpu(packet):
    return dict(packed=packet.packed.detach().cpu(), swizzled_scales=packet.swizzled_scales.detach().cpu(),
        global_scale=packet.global_scale.detach().cpu(), original_shape=list(packet.original_shape))


def hook_identity(model):
    return {name: {field: tuple((key, id(value)) for key, value in getattr(module, field).items()) for field in
        ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')}
        for name, module in model.named_modules()}


class BranchInputAudit(TorchFunctionMode):
    """Observe actual F.linear operands without replacing any Torch function."""
    def __init__(self, executor, module):
        self.executor, self.module = executor, module

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        current = self.executor.current
        is_linear = func is torch.nn.functional.linear
        is_a = is_linear and len(args) >= 2 and args[1] is self.module.lr_a
        is_b = is_linear and len(args) >= 2 and args[1] is self.module.lr_b
        if is_a:
            require(current['branch_a_calls'] == 0 and args[0] is current['live_xs'],
                    'LR A must read the exact unquantized activation-pack source object')
            actual = args[0].detach().cpu()
            require(tensor_record(actual) == current['pack_source_signature'], 'Packer mutated the LR input')
            current['x_s'] = actual
            current['branch_a_calls'] += 1
        if is_b:
            require(current['branch_b_calls'] == 0 and args[0] is current['live_a_output'],
                    'LR B must read the actual BF16 first-linear output')
            current['branch_b_calls'] += 1
        output = func(*args, **kwargs)
        if is_a:
            require(output.dtype == torch.bfloat16, 'First LR output is not BF16')
            current['live_a_output'] = output
            current['lr_a_output'] = output.detach().cpu()
        if is_b:
            require(output.dtype == torch.bfloat16, 'Second LR output is not BF16')
            current['lr_b_output_sample'] = output.index_select(0, current['sample_indices'].to('cuda')).detach().cpu()
        return output


class NativeExecutor:
    def __init__(self, attention, original, cases, args, report):
        self.attention, self.original, self.cases = attention, original, cases
        self.args, self.report = args, report
        self.phase, self.native, self.current = 'baseline', None, None
        self.installs, self.recovers = 0, 0

    def pack(self, xs, *, chunk_rows=1024):
        current, report = self.current, self.report
        require(current is not None and 'activation_packet' not in current, 'Unexpected activation pack')
        require(report['activation_packs'] < 4, 'Activation pack budget exhausted')
        current['live_xs'] = xs
        current['pack_source_signature'] = tensor_record(xs)
        report['activation_packs'] += 1
        packet = base.pack_activation_fast(xs, chunk_rows=chunk_rows)
        current['activation_packet'] = packet_cpu(packet)
        return packet

    def begin(self, index):
        require(self.current is None, 'Overlapping native forwards')
        rows = ROWS[index]
        self.current = dict(case_id=CASES[index], branch_a_calls=0, branch_b_calls=0,
            sample_indices=torch.linspace(0, rows-1, 512, dtype=torch.float64).to(torch.int64))

    def finish(self, qkv_output):
        current = self.current
        require(current['branch_a_calls'] == current['branch_b_calls'] == 1 and
                qkv_output.shape == (current['x_s'].shape[0], SHAPE[0]) and qkv_output.dtype == torch.bfloat16,
                'Incomplete actual native branch evidence')
        current['qkv_output'] = qkv_output.detach().cpu()
        current['lr_input_evidence'] = dict(a_input_is_pack_source=True, b_input_is_a_output=True,
            pack_source_signature=current['pack_source_signature'], a_input_signature=tensor_record(current['x_s']),
            branch_a_calls=1, branch_b_calls=1, input_dtype='torch.bfloat16',
            observer='TorchFunctionMode observes original F.linear operands; no function replacement')
        for key in ('live_xs', 'live_a_output', 'branch_a_calls', 'branch_b_calls', 'pack_source_signature'):
            current.pop(key, None)
        self.current = None
        return current

    @torch.inference_mode()
    def install(self, candidate, spans):
        report, args = self.report, self.args
        guard(args, report)
        require(self.installs == 0 and self.phase == 'baseline' and self.attention.qkv_proj is self.original,
                'Only one original-module candidate install is allowed')
        require(report['baseline_case_order'] == [0, 1], 'Candidate installed before both real baselines')
        require(candidate.dtype == torch.float32 and candidate.shape == (SHAPE[1],), 'Unexpected raw scale')
        smooth = candidate.to(torch.bfloat16)
        require(bool(torch.isfinite(smooth).all()) and bool((smooth > 0).all()) and bool((smooth != 1).any()),
                'BF16 deployment scale must be finite, positive and non-identity')
        ws = self.original.weight*smooth
        require(ws.dtype == torch.bfloat16 and tuple(ws.shape) == SHAPE, 'Smoothed weight contract changed')
        report['svd_calls'] += 1
        report['active_stage'] = 'exact_svd'
        save(report, args.output)
        torch.cuda.synchronize()
        start = time.monotonic()
        u, s, vh = torch.linalg.svd(ws.double())
        torch.cuda.synchronize()
        report['svd_seconds'] = time.monotonic()-start
        require(tuple(u.shape) == (SHAPE[0], SHAPE[0]) and tuple(vh.shape) == (SHAPE[1], SHAPE[1]) and
                tuple(s.shape) == (SHAPE[1],) and u.dtype == s.dtype == vh.dtype == torch.float64,
                'Default full FP64 SVD shape/dtype changed')
        finite = bool(torch.isfinite(s).all())
        for tensor in (u, vh):
            for offset in range(0, tensor.shape[0], 1024):
                finite = finite and bool(torch.isfinite(tensor[offset:offset+1024]).all())
        require(finite and bool((s >= 0).all()) and bool((s[:-1] >= s[1:]).all()), 'Invalid full SVD outputs')
        u32, vh32 = u[:, :32].contiguous(), vh[:32].contiguous()
        gram_u = u32.T@u32-torch.eye(32, dtype=torch.float64, device='cuda')
        gram_v = vh32@vh32.T-torch.eye(32, dtype=torch.float64, device='cuda')
        report['svd_gram'] = dict(u_frobenius=float(gram_u.norm()), vh_frobenius=float(gram_v.norm()))
        require(max(report['svd_gram'].values()) <= 1e-8, 'SVD top32 Gram gate failed')
        a, b = vh32.to(torch.bfloat16), (u32*s[:32]).to(torch.bfloat16)
        svd_saved = dict(u_top32=u32.cpu(), singular_values=s.cpu(), vh_top32=vh32.cpu())
        del u, vh, u32, vh32, gram_u, gram_v, s
        gc.collect()
        torch.cuda.empty_cache()
        guard(args, report)
        product = b@a
        residual = ws-product
        packet = pack_activation_legacy(residual, chunk_rows=1024)
        decoded = packet.decode(dtype=torch.bfloat16, chunk_rows=1024)
        independent = nvfp4_qdq(residual, element_size=1024)
        require(bool(torch.isfinite(independent).all()) and torch.equal(decoded, independent),
                'Independent legacy QDQ versus packed decode numeric roundtrip failed')
        self.native = NativeH3Linear(packet, smooth, a, b, self.original.bias,
                                    activation_packer=self.pack, chunk_rows=1024)
        evidence = dict(raw_scale_fp32=candidate.detach().cpu(), deploy_scale_bf16=smooth.cpu(),
            ws_bf16=ws.cpu(), lr_product_bf16=product.cpu(), residual_bf16=residual.cpu(),
            decoded_q_bf16=decoded.cpu(), svd=svd_saved,
            span=dict(x=spans[0].cpu(), w=spans[1].cpu()))
        report['candidate_artifact'] = write_tensor(evidence, DATA/'candidate.pt', report)
        report['candidate_tensor_signatures'] = signature(evidence)
        report['weight_roundtrip'] = dict(numeric_exact=True, compatibility=COMPATIBILITY, decoded=tensor_record(decoded),
            independent_qdq=tensor_record(independent), signed_zero_byte_equality_required=False)
        report['export'] = write_tensor(exporter.export_payload(self.native, TARGET), DATA/'export.pt', report)
        self.attention.qkv_proj = self.native
        self.phase = 'candidate'
        self.installs += 1
        report['candidate_installs'] = self.installs
        report['active_stage'] = 'candidate_attention'
        del evidence, svd_saved, ws, product, residual, decoded, independent
        guard(args, report)
        save(report, args.output)

    def recover(self):
        if self.attention.qkv_proj is self.native and self.native is not None:
            self.attention.qkv_proj = self.original
            self.recovers += 1
        require(self.attention.qkv_proj is self.original, 'Original QKV object recovery failed')
        self.phase = 'recovered'
        self.report['candidate_recovers'] = self.recovers


def fp64_sse(output, reference):
    total = 0.0
    for start in range(0, output.shape[0], 2048):
        delta = output[start:start+2048].double()-reference[start:start+2048].double()
        total += float(delta.square().sum())
    return total


class CaseAttention(torch.nn.Module):
    def __init__(self, executor):
        super().__init__()
        self.executor = executor
        self.attention = executor.attention

    @torch.inference_mode()
    def forward(self, case_id):
        ex, report = self.executor, self.executor.report
        guard(ex.args, report)
        require(case_id.shape == (1, 1) and case_id.dtype == torch.int64, 'Only [1,1] case indices are eval inputs')
        index = int(case_id.item())
        require(ex.phase in ('baseline', 'candidate') and index in (0, 1), 'Unexpected case/phase')
        order = report[ex.phase+'_case_order']
        require(index == len(order) and len(order) < 2, 'Each phase must visit each full case exactly once in order')
        case = ex.cases[index]
        expected = report['case_sources'][index]['input_signature']
        require(signature(case['attention_inputs']) == expected, 'Immutable input changed before attention')
        gpu = base.tree_device(case['attention_inputs'], 'cuda')
        require(signature(gpu) == expected, 'Actual full attention input transfer changed bytes')
        audit = base.RuntimeAudit()
        audit.phase = 'resident_bf16' if ex.phase == 'baseline' else 'native'
        report['attention_calls'] += 1
        save(report, ex.args.output)
        captured = []
        handle = None
        if ex.phase == 'candidate':
            ex.begin(index)
            handle = ex.native.register_forward_hook(lambda _m, _p, out: captured.append(out))
        try:
            with audit.installed(), collect_fastpack_checks() as packs:
                if ex.phase == 'candidate':
                    with BranchInputAudit(ex, ex.native):
                        output = self.attention(*gpu['args'], **gpu['kwargs'])
                else:
                    output = self.attention(*gpu['args'], **gpu['kwargs'])
                torch.cuda.synchronize()
        finally:
            if handle is not None:
                handle.remove()
        require(signature(gpu) == expected == signature(case['attention_inputs']), 'Attention mutated complete inputs/kwargs')
        require(output.shape == (ROWS[index], SHAPE[1]) and output.dtype == torch.bfloat16 and
                bool(torch.isfinite(output).all()), 'Invalid real attention output')
        counts = dict(audit.row())
        native_count = int(ex.phase == 'candidate')
        require(counts['sdpa_calls'] == 2 and counts['scaled_mm_calls'] == native_count and
                counts['disk_loads'] == 0 and packs.summary['checked_calls'] == native_count,
                'Actual attention/pack count differs')
        report['sdpa_calls'] += counts['sdpa_calls']
        report['native_scaled_mm_calls'] += counts['scaled_mm_calls']
        cpu_output = output.detach().cpu()
        row = dict(case_id=CASES[index], source_artifact=report['case_sources'][index]['artifact'],
            attention_input_signature=expected, output_signature=tensor_record(cpu_output), runtime_audit=counts,
            fastpack_checks=packs.summary)
        if ex.phase == 'baseline':
            require(tensor_record(cpu_output) == tensor_record(case['output']), 'Actual BF16 baseline must byte-replay E070')
            row['byte_exact_e070'] = True
            report['baseline_cases'].append(row)
        else:
            require(len(captured) == 1, 'Expected exactly one actual candidate QKV output')
            payload = ex.finish(captured.pop())
            payload.update(attention_inputs_signature=expected, attention_output=cpu_output)
            row['artifact'] = write_tensor(payload, DATA/'candidate_cases'/(CASES[index]+'.pt'), report)
            row['qkv_signature'] = tensor_record(payload['qkv_output'])
            row['x_s_signature'] = tensor_record(payload['x_s'])
            row['packet_signature'] = signature(payload['activation_packet'])
            report['candidate_cases'].append(row)
            reference = case['output'].to(output.device)
            # Duplicate readout only: actual library scoring below remains unmodified.
            low = (output-reference).to(torch.float32).square().sum()
            report['scores']['library_per_case'][CASES[index]] = float(low)
            report['scores']['fp64_per_case'][CASES[index]] = fp64_sse(cpu_output, case['output'])
            del payload, reference, low
        order.append(index)
        save(report, ex.args.output)
        guard(ex.args, report)
        return output


@torch.inference_mode()
def replay_export(ex, report, args):
    payload = torch.load(base.verify_file(report['export']), map_location='cpu', weights_only=True, mmap=True)
    require(payload['name'] == TARGET and payload['shape'] == list(SHAPE), 'Export target changed')
    module = NativeH3Linear.from_export(payload, device='cuda', activation_packer=ex.pack, chunk_rows=1024)
    require(module is not ex.native and signature(exporter.export_payload(module, TARGET)) == signature(payload),
            'Reload must create a new byte-identical native module')
    for index, case in enumerate(ex.cases):
        guard(args, report)
        expected = report['case_sources'][index]['input_signature']
        require(signature(case['attention_inputs']) == expected, 'Immutable source changed before export replay')
        x = case['attention_inputs']['args'][0].to('cuda')
        ex.begin(index)
        report['reload_qkv_calls'] += 1
        audit = base.RuntimeAudit()
        audit.phase = 'native'
        with audit.installed(), collect_fastpack_checks() as packs, BranchInputAudit(ex, module):
            output = module(x)
            torch.cuda.synchronize()
        counts = dict(audit.row())
        require(counts['sdpa_calls'] == counts['disk_loads'] == 0 and counts['scaled_mm_calls'] == 1 and
                packs.summary['checked_calls'] == 1, 'Export replay native count differs')
        report['native_scaled_mm_calls'] += counts['scaled_mm_calls']
        current = ex.finish(output)
        current['attention_inputs_signature'] = expected
        candidate = torch.load(base.verify_file(report['candidate_cases'][index]['artifact']),
                               map_location='cpu', weights_only=True, mmap=True)
        for key in ('qkv_output', 'x_s', 'activation_packet', 'lr_a_output', 'sample_indices', 'lr_b_output_sample'):
            require(signature(current[key]) == signature(candidate[key]), f'Export replay byte mismatch: {key}')
        artifact = write_tensor(current, DATA/'reload_cases'/(CASES[index]+'.pt'), report)
        report['reload_cases'].append(dict(case_id=CASES[index], artifact=artifact, byte_exact_candidate=True,
            qkv_signature=tensor_record(current['qkv_output']), x_s_signature=tensor_record(current['x_s']),
            packet_signature=signature(current['activation_packet']), runtime_audit=counts, fastpack_checks=packs.summary))
        del x, output, current, candidate
        gc.collect()
        torch.cuda.empty_cache()
        save(report, args.output)
    report['reloaded_export_tensors_exact'] = True


@torch.inference_mode()
def run(args, report, manifest, api):
    checked = load_complete(report_path('check'))
    for key in ('sources', 'plan', 'source_bridge', 'source_independent', 'source_exit', 'case_sources',
                'source_weight_sha256', 'setup_check', 'setup_manifest', 'library_contract', 'torch'):
        require(checked[key] == report[key], f'CPU check source contract changed: {key}')
    require(checked['cuda_initialized'] is False and checked['new_model_forwards'] == 0, 'CPU-only completed check required')
    report['cpu_check'] = record(report_path('check'))
    cases = load_cases(report)
    DATA.mkdir(parents=True, exist_ok=False)
    require(torch.cuda.device_count() == 1, 'Expose exactly one GPU')
    properties = torch.cuda.get_device_properties(0)
    require(properties.total_memory > MAX_GPU, 'Insufficient GPU capacity for allocation cap')
    torch.cuda.set_per_process_memory_fraction(MAX_GPU/properties.total_memory, 0)
    guard(args, report)
    old_check = load_complete(base.verify_file(report['setup_check']))
    setup = dict(experiment='E071_inherited_E014_BF16_setup', status='running', complete_dit_calls=0,
                 sources=old_check['sources'])
    setup_path = DATA/'setup_bf16.json'
    pipe = base.setup_model(types.SimpleNamespace(arm='bf16', deadline_unix=args.deadline_unix,
        check_report=Path(report['setup_check']['file']), output=setup_path), setup, manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = record(setup_path)
    report['data_bytes'] += setup_path.stat().st_size
    require(pipe.text_encoder is None and pipe.video_vae is None and pipe.audio_vae is None, 'Only resident BF16 DiT allowed')
    attention = pipe.dit.get_submodule('blocks.0.attn')
    original = attention.qkv_proj
    require(type(original) is torch.nn.Linear and tuple(original.weight.shape) == SHAPE and original.bias is None and
            tensor_record(original.weight)['sha256'] == WEIGHT_SHA, 'Actual resident source QKV changed')
    original_identity, original_hooks = bridge.model_identity(pipe.dit), hook_identity(pipe.dit)
    ex = NativeExecutor(attention, original, cases, args, report)
    api.calibrator.executor = ex
    wrapper = CaseAttention(ex)
    x_acts, eval_inputs = actual_caches(api, cases)
    try:
        report['active_stage'] = 'real_calibrate'
        save(report, args.output)
        with library_call_audit(api, report):
            best = api.calibrator.calibrate(x_wgts=[original.weight], x_acts=x_acts, x_mods=[original],
                eval_inputs=eval_inputs, eval_module=wrapper, eval_kwargs={})
    finally:
        ex.recover()
        api.SearchBasedCalibrator._recover_mod(api.calibrator)
        report['recovery'] = dict(original_module_same=attention.qkv_proj is original,
            original_weight_sha256=tensor_record(original.weight)['sha256'],
            original_storage_identity=bridge.model_identity(pipe.dit) == original_identity,
            all_original_hooks_restored=hook_identity(pipe.dit) == original_hooks,
            outstanding_state_dict=len(api.calibrator._state_dict), outstanding_hooks=len(api.calibrator._hooks))
        save(report, args.output)
    require(report['library_calls'] == {name: 1 for name in api.functions}, 'Real one-candidate library call count differs')
    require(api.calibrator.iter == 1 and api.calibrator.is_done() and ex.installs == ex.recovers == 1,
            'One complete candidate/install/recovery required')
    recovery = report['recovery']
    require(recovery['original_module_same'] and recovery['original_storage_identity'] and
            recovery['all_original_hooks_restored'] and recovery['original_weight_sha256'] == WEIGHT_SHA and
            recovery['outstanding_state_dict'] == recovery['outstanding_hooks'] == 0, 'Original model restoration failed')
    candidate = torch.load(base.verify_file(report['candidate_artifact']), map_location='cpu', weights_only=True, mmap=True)
    require(tensor_record(best) == tensor_record(candidate['raw_scale_fp32']), 'Real get_best must return raw FP32 scale')
    report['best_raw_scale_signature'] = tensor_record(best)
    report['deployment_scale_signature'] = tensor_record(candidate['deploy_scale_bf16'])
    report['scores']['library_best'] = float(api.calibrator.best_error[0].item())
    require(report['scores']['library_best'] == report['scores']['library_reported'], 'Real best/tell mismatch')
    low_scores = [report['scores']['library_per_case'][key] for key in CASES]
    low_total = float(torch.tensor(low_scores[0], dtype=torch.float32)+torch.tensor(low_scores[1], dtype=torch.float32))
    require(low_total == report['scores']['library_reported'], 'Readout differs from unchanged real library score')
    report['scores']['fp64_total'] = sum(report['scores']['fp64_per_case'].values())
    del candidate
    replay_export(ex, report, args)
    require(report['baseline_case_order'] == report['candidate_case_order'] == [0, 1] and
            report['attention_calls'] == 4 and report['reload_qkv_calls'] == 2 and
            report['native_scaled_mm_calls'] == report['activation_packs'] == 4 and
            report['sdpa_calls'] == 8 and report['svd_calls'] == 1, 'Final E071 count contract failed')
    require(bridge.model_identity(pipe.dit) == original_identity and hook_identity(pipe.dit) == original_hooks,
            'Export replay changed original model')
    guard(args, report)
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'run'))
    parser.add_argument('--budget-start-unix', type=float)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.output = report_path(args.phase)
    require(not args.output.exists(), f'Refusing report overwrite: {args.output}')
    if args.phase == 'run' and (args.budget_start_unix is None or args.deadline_unix is None or
            not 0 < args.deadline_unix-args.budget_start_unix <= WALL_SECONDS or
            not args.budget_start_unix <= time.time() < args.deadline_unix):
        parser.error('Run requires the original start/deadline pair, at most 900 seconds apart')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    report = dict(experiment='E071', phase=args.phase, status='running',
        budget_start_unix=args.budget_start_unix, deadline_unix=args.deadline_unix,
        wall_budget_seconds=WALL_SECONDS, data_byte_limit=MAX_BYTES, data_bytes=0,
        attention_calls=0, reload_qkv_calls=0, native_scaled_mm_calls=0, activation_packs=0, sdpa_calls=0,
        svd_calls=0, prefix_calls=0, complete_dit_calls=0, text_encoder_calls=0, video_vae_calls=0, audio_vae_calls=0,
        baseline_case_order=[], candidate_case_order=[], baseline_cases=[], candidate_cases=[], reload_cases=[],
        scores=dict(library_per_case={}, fp64_per_case={}),
        scope='Single native-compatible candidate wiring/export smoke; not stock YAML byte reproduction, search or quality evidence')
    started = time.monotonic()
    try:
        if args.phase == 'run':
            def expired(_signum, _frame):
                raise TimeoutError('E071 original absolute deadline reached')
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        _parent, manifest, api = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, api)
        else:
            run(args, report, manifest, api)
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
