#!/usr/bin/env python3
"""E065: fixed-smooth carry-Q versus fresh-Q restart, eight candidates each.

Eight independent full BF16 calibration calls, then one legacy replay and
eight full native diagnostic calls. No initialization sweep or shifted inputs.
"""
from __future__ import annotations

import argparse
import gc
import inspect
from pathlib import Path
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe_h3_lowrank_initialization as e061
from h3_native_nvfp4 import install_native_h3
from wan_native_nvfp4 import PackedNVFP4, swizzle_scales, unswizzle_scales
import torch
import torch.nn.functional as F

base, old, inherited, builder, exporter = e061.base, e061.old, e061.inherited, e061.builder, e061.exporter
NativeH3Linear, TARGET_NAMES, FORMAT, RECIPE = e061.NativeH3Linear, e061.TARGET_NAMES, e061.FORMAT, e061.RECIPE
LowRankBranch, nvfp4_qdq, pack_activation_legacy = e061.LowRankBranch, e061.nvfp4_qdq, e061.pack_activation_legacy
safe_open, H3_DIT_PATH = e061.safe_open, e061.H3_DIT_PATH
save, file_record, signature, require, key = e061.save, e061.file_record, e061.signature, e061.require, e061.key
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E065')
REPORTS = ROOT/'results/research/E065'
PLAN = ROOT/'research_state/06_experiments/E065_h3_carry_q_plan.md'
CACHE = ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s'
ARMS = ('restart', 'carry')
PROMPTS, STEPS = (1, 20, 46, 105), (3, 14)
N_CANDIDATES, ROWS, CHUNK_ROWS = 8, 512, 1024
MAX_CALLS, MAX_DATA_BYTES, WALL_SECONDS = 17, 128*1024**3, 1800
MAX_LOCAL_CANDIDATE_CALLS, MAX_SHAPE_CALLS = 200*2*8*8, 8
WITNESS_ROWS, WITNESS_COLS = 8, 8


def guard(args, report, *, before_forward=False):
    require(time.time() < args.deadline_unix, 'E065 absolute deadline reached')
    require(report['attempted_dit_calls'] <= MAX_CALLS, 'E065 full-call budget exceeded')
    if before_forward:
        require(report['attempted_dit_calls'] < MAX_CALLS, 'E065 seventeen-forward budget exhausted')
    require(report['candidate_native_calls'] <= MAX_LOCAL_CANDIDATE_CALLS and
            report['shape_native_calls'] <= MAX_SHAPE_CALLS, 'E065 local native budget exceeded')
    require(report['data_bytes'] < MAX_DATA_BYTES, 'E065 new-data budget reached 128 GiB')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E065 peak reached 60 GiB')


# Clone only the unwrapped implementations into a private namespace. Neither
# E061 nor its dependency modules are monkey-patched. In particular __file__,
# PLAN, DATA, REPORTS and guard must not retain E061's bindings.
_E061_GLOBALS = dict(e061.__dict__)
_E061_GLOBALS.update(__file__=__file__, DATA=DATA, REPORTS=REPORTS, PLAN=PLAN,
                     ARMS=ARMS, MAX_CALLS=MAX_CALLS, guard=guard)


def clone_e061(name, *, inference=False):
    original = inspect.unwrap(getattr(e061, name))
    cloned = types.FunctionType(original.__code__, _E061_GLOBALS, original.__name__,
                                original.__defaults__, original.__closure__)
    cloned.__kwdefaults__ = original.__kwdefaults__
    return torch.inference_mode()(cloned) if inference else cloned


_prerequisites = clone_e061('prerequisites')
_cpu_check = clone_e061('cpu_check', inference=True)
_run_case = clone_e061('run_case', inference=True)
install_export = clone_e061('install_export')


def prerequisites(report):
    manifest, e014_manifest, items, legacy = _prerequisites(report)
    paths = (Path(e061.__file__), Path(__file__), PLAN,
             ROOT/'scripts/collect_minimax_h3_calib_standard.py',
             HERE/'h3_nvfp4_zero_sf_compat.py', HERE/'h3_nvfp4_fastpack.py',
             HERE/'wan_native_nvfp4.py',
             ROOT/'third_party/deepcompressor/deepcompressor/calib/lowrank.py')
    report['sources'].update({str(p.resolve()): file_record(p) for p in paths})
    calibration = []
    for pid in PROMPTS:
        for step in STEPS:
            path = CACHE/f'p{pid}'/f'sample_p{pid}_s{step:02d}.pt'
            calibration.append(dict(id=f'p{pid}_s{step:02d}', prompt_id=pid, step=step,
                                    kind='raw_dit', artifact=file_record(path)))
    require(len(calibration) == 8 and not set(PROMPTS).intersection({30, 36}), 'Calibration/diagnostic split changed')
    report.update(calibration_inputs=calibration,
                  reuse=dict(source=file_record(e061.__file__),
                             helpers=['prerequisites', 'cpu_check', 'run_case', 'install_export'],
                             isolated_module_globals=True),
                  search_contract=dict(arms=list(ARMS), candidates=8, rank=32, q=40, niter=2,
                    seed='310000+1000*block+50*local_index+k',
                    objective='sum of eight native local-output SSEs; FP64 difference and accumulation',
                    selection='first candidate at minimum score; no early stop',
                    carry='latest candidate Q, never selected-best Q',
                    activation_domain='one complete-input pack per layer/case; slice codes/SF; unchanged global'))
    return manifest, e014_manifest, items, legacy


def load_calibration(case):
    # These are trusted existing project cache files, using their frozen schema.
    payload = torch.load(base.verify_file(case['artifact']), map_location='cpu', weights_only=False, mmap=True)
    require(set(payload) == {'input_args', 'input_kwargs', 'meta'}, 'Calibration payload schema changed')
    require(payload['meta']['prompt_id'] == case['prompt_id'] and payload['meta']['step'] == case['step'],
            'Calibration prompt/step metadata differs')
    value = dict(args=payload['input_args'], kwargs=payload['input_kwargs'])
    require(not value['args'] and value['kwargs'].get('control_hints') is None, 'Unexpected raw DiT args/control hints')
    return value, payload['meta']


def select_rows(value):
    kw = value['kwargs']
    domains = {name: kw[field]['position_ids'].reshape(-1).long().cpu()
               for name, field in (('video', 'img_pos_info'), ('audio', 'audio_pos_info'), ('text', 'text_pos_info'))}
    joined = torch.cat(list(domains.values())).sort().values
    count = kw['x'].shape[1]
    require(joined.numel() >= ROWS and joined.unique().numel() == joined.numel() and
            int(joined[0]) >= 0 and int(joined[-1]) < count, 'Invalid nonpadding joint domain')
    offsets = torch.linspace(0, joined.numel()-1, ROWS, dtype=torch.float64).long()
    indices = joined.index_select(0, offsets)
    require(indices.unique().numel() == ROWS, 'Joint sample rows repeat')
    counts = {name: int(torch.isin(indices, rows).sum()) for name, rows in domains.items()}
    require(sum(counts.values()) == ROWS, 'Selected rows outside modality domains')
    return dict(indices=indices, full_rows=count, nonpadding_rows=joined.numel(),
                padding_rows=count-joined.numel(), modality_counts=counts,
                domain_signature=signature(domains))


@torch.inference_mode()
def cpu_check(report, manifest, items, legacy):
    _cpu_check(report, manifest, items, legacy)
    report['calibration_checks'] = {}
    for case in report['calibration_inputs']:
        value, metadata = load_calibration(case)
        selected = select_rows(value)
        require(value['kwargs']['x'].dtype == torch.bfloat16, 'Calibration x must be BF16')
        cu = value['kwargs']['packed_seq_params']['cu_seqlens_q']
        require(50*int((cu[1:] > cu[:-1]).sum())+2 == 102, 'Calibration packed segments differ from full-call audit')
        report['calibration_checks'][case['id']] = dict(actual_dit_input_signature=signature(value),
            metadata=metadata, selection_signature=signature(selected),
            selected_indices=selected['indices'].tolist(), modality_counts=selected['modality_counts'],
            full_rows=selected['full_rows'], padding_rows=selected['padding_rows'])
    require(not torch.cuda.is_initialized(), 'E065 CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, calibration_input_count=8,
                  diagnostic_teacher_count=4, complete_dit_calls=0,
                  checked_weight_headers=200, new_model_forwards=0)


def save_tensor(value, path, report):
    require(not path.exists(), f'Refusing existing artifact {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(value, stream)
    record = file_record(path)
    report['data_bytes'] += record['bytes']
    require(report['data_bytes'] < MAX_DATA_BYTES, 'E065 new-data budget reached 128 GiB')
    return record


def save_json_artifact(value, path, report):
    require(not path.exists(), f'Refusing existing JSON artifact {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    save(value, path)
    record = file_record(path)
    report['data_bytes'] += record['bytes']
    require(report['data_bytes'] < MAX_DATA_BYTES, 'E065 new-data budget reached 128 GiB')
    return record


def packet_signature(packet):
    return signature(dict(codes=packet.packed, scales=packet.swizzled_scales,
                          global_scale=packet.global_scale, original_shape=list(packet.original_shape)))


def subset_packet(packet, indices):
    n, k2 = packet.packed.shape
    logical = unswizzle_scales(packet.swizzled_scales, n, k2*2//16)
    codes = packet.packed.index_select(0, indices).contiguous()
    # Index uint8 storage: CUDA float8 index_select is not a supported ABI.
    scales = logical.contiguous().view(torch.uint8).index_select(0, indices).contiguous().view(torch.float8_e4m3fn)
    subset = PackedNVFP4(codes, None, packet.global_scale,
                         swizzle_scales(scales), (ROWS, k2*2), packet.recipe)
    decoded_sf = unswizzle_scales(subset.swizzled_scales, ROWS, k2*2//16)
    require(torch.equal(codes, packet.packed.index_select(0, indices)) and
            torch.equal(decoded_sf.contiguous().view(torch.uint8), scales.view(torch.uint8)) and
            torch.equal(subset.global_scale.view(torch.uint8), packet.global_scale.view(torch.uint8)),
            'Subset packet changed codes, logical SF bytes, or full-input global')
    return subset, scales


def packet_payload(packet):
    return dict(codes=packet.packed.cpu(), scales=packet.swizzled_scales.cpu(),
                global_scale=packet.global_scale.cpu(), shape=list(packet.original_shape), recipe=packet.recipe)


def packet_from_payload(payload, device):
    return PackedNVFP4(payload['codes'].to(device), None, payload['global_scale'].to(device),
                       payload['scales'].to(device), tuple(payload['shape']), payload['recipe'])


@torch.inference_mode()
def shape_control(xs, teacher, full, subset, indices, name, row, legacy_dir, args, report):
    guard(args, report)
    path = legacy_dir/row['file']
    require(file_record(path)['sha256'] == row['file_sha256'], 'Shape-control legacy export changed')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    module = NativeH3Linear.from_export(payload, device='cuda', activation_packer=base.pack_activation_fast,
                                       chunk_rows=CHUNK_ROWS)
    outputs = []
    for activation, values in ((full, xs), (subset, xs.index_select(0, indices))):
        report['shape_native_calls'] += 1
        guard(args, report)
        main = module.main_from_packet(activation, include_bias=True)
        result = main + 1.0*F.linear(F.linear(values, module.lr_a), module.lr_b)
        outputs.append((result.index_select(0, indices) if activation is full else result).cpu())
        del main, result
    full_rows, sub_rows = outputs
    require(bool(torch.isfinite(full_rows).all()) and bool(torch.isfinite(sub_rows).all()), 'Nonfinite shape control')
    delta = sub_rows.double()-full_rows.double()
    target = teacher.index_select(0, indices).cpu()
    tile_sse = float(delta.square().sum())
    quant_sse = float((full_rows.double()-target.double()).square().sum())
    artifact = save_tensor(dict(layer=name, full_output_rows=full_rows, subset_output=sub_rows,
        teacher_output=target, indices=indices.cpu()), DATA/'shape_controls'/(name+'.pt'), report)
    report['selection_shape_controls'].append(dict(layer=name, artifact=artifact,
        bitwise=base.tensor_record(full_rows)==base.tensor_record(sub_rows),
        max_abs=float(delta.abs().max()), tile_difference_sse=tile_sse,
        full_native_quantization_sse=quant_sse,
        tile_over_quant_sse=tile_sse/quant_sse if quant_sse > 0 else None,
        scope='M512 selection is not claimed bitwise equal to full-M native execution; diagnostic only'))
    del module, payload, outputs


@torch.inference_mode()
def capture_calibration(pipe, case, checked, state, legacy, args, report):
    guard(args, report, before_forward=True)
    value, metadata = load_calibration(case)
    expected = checked['calibration_checks'][case['id']]
    require(signature(value) == expected['actual_dit_input_signature'] and metadata == expected['metadata'],
            'Calibration input changed since CPU check')
    selected = select_rows(value)
    require(signature(selected) == expected['selection_signature'], 'Calibration row selection changed')
    indices = selected['indices'].cuda()
    layers, actual, handles, pending = {}, [], [], {}
    old_rows = {r['name']: r for r in legacy['layers']}
    legacy_dir = Path(report['legacy_manifest']['file']).parent
    def observe(_module, pos, kw):
        actual.append(signature(dict(args=pos, kwargs=kw)))
        require(actual[-1] == expected['actual_dit_input_signature'], 'Actual calibration DiT input differs')
    handles.append(pipe.dit.register_forward_pre_hook(observe, with_kwargs=True))
    for name in TARGET_NAMES:
        def pre_hook(module, positional, name=name):
            guard(args, report)
            require(name not in layers and name not in pending, 'Repeated calibration linear')
            x = positional[0]
            require(x.ndim == 2 and x.shape[0] == selected['full_rows'] and
                    x.dtype == torch.bfloat16, 'Calibration linear input dimensions/dtype changed')
            smooth = state['layers'][name]['smooth'].to(device='cuda', dtype=torch.bfloat16)
            xs = x/smooth
            full = base.pack_activation_fast(xs, chunk_rows=CHUNK_ROWS)
            report['calibration_activation_packs'] += 1
            subset, logical_rows = subset_packet(full, indices)
            xs_rows = xs.index_select(0, indices).cpu()
            detail = dict(layer=name, case_id=case['id'], xs=xs_rows,
                indices=selected['indices'], modality_counts=selected['modality_counts'],
                packet=packet_payload(subset), logical_scale_rows=logical_rows.cpu(),
                full_xs_signature=base.tensor_record(xs),
                full_packet_global_signature=base.tensor_record(full.global_scale),
                full_packet_shape=list(full.original_shape), full_packet_signature=packet_signature(full),
                smooth_signature=base.tensor_record(smooth), source_signature=base.tensor_record(xs_rows),
                source_weight_sha256=old_rows[name]['source_weight_sha256'], packet_subset_bytes_verified=True)
            control = (xs, full, subset) if case['id'] == 'p1_s03' and name.startswith('blocks.0.') else None
            pending[name] = (detail, control)
            del smooth, xs, full, subset, logical_rows, detail, xs_rows
        def post_hook(module, positional, output, name=name):
            detail, control = pending.pop(name)
            require(output.ndim == 2 and output.shape[0] == selected['full_rows'] and
                    output.dtype == torch.bfloat16, 'Calibration linear output dimensions/dtype changed')
            detail['target'] = output.index_select(0, indices).cpu()
            detail['target_signature'] = base.tensor_record(detail['target'])
            artifact = save_tensor(detail, DATA/'calibration'/name/(case['id']+'.pt'), report)
            layers[name] = dict(artifact=artifact, source_signature=detail['source_signature'],
                                target_signature=detail['target_signature'])
            if control is not None:
                xs, full, subset = control
                shape_control(xs, output, full, subset, indices, name, old_rows[name], legacy_dir, args, report)
            del detail, control
        module = pipe.dit.get_submodule(name)
        handles.extend((module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)))
    audit = base.RuntimeAudit()
    audit.phase = 'resident_bf16'
    report['attempted_dit_calls'] += 1
    report['active_call'] = dict(arm='calibration_bf16', case_id=case['id'])
    save(report, args.output)
    try:
        with audit.installed(), base.collect_fastpack_checks() as fastpack:
            result = base.gpu_call(pipe, dict(kind='raw_dit'), value)()
            torch.cuda.synchronize()
        report['complete_dit_calls'] += 1
        report['calibration_bf16_calls'] += 1
    finally:
        for handle in handles:
            handle.remove()
    audit_row = audit.row()
    shape_calls = 8 if case['id'] == 'p1_s03' else 0
    require(audit_row['sdpa_calls'] == 102 and audit_row['disk_loads'] == 0 and
            audit_row['scaled_mm_calls'] == shape_calls, 'BF16 calibration/shape-control runtime audit differs')
    require(len(actual) == 1 and not pending and set(layers) == set(TARGET_NAMES) and fastpack.summary['checked_calls'] == 200,
            'Calibration capture/pack count differs')
    raw, raw_records = base.cpu_outputs(result)
    output = save_tensor(dict(case_id=case['id'], actual_dit_input_signature=actual[0],
        raw_outputs=raw, selection=selected, metadata=metadata), DATA/'calibration_full'/(case['id']+'.pt'), report)
    report['calibration_cases'].append(dict(case_id=case['id'], artifact=output, layers=layers,
        raw_outputs=raw_records, runtime_audit=audit_row, fastpack_checks=fastpack.summary))
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save(report, args.output)
    print(f'E065 captured {case["id"]}: 200 full-input packets', flush=True)
    del result, raw, value
    gc.collect()
    guard(args, report)


def score_candidate(module, samples, args, report):
    scores, outputs = [], []
    for sample in samples:
        report['candidate_native_calls'] += 1
        guard(args, report)
        main = module.main_from_packet(sample['packet'], include_bias=True)
        branch = 1.0*F.linear(F.linear(sample['xs'], module.lr_a), module.lr_b)
        output = main+branch
        difference = output.double()-sample['target'].double()
        require(bool(torch.isfinite(difference).all()), 'Nonfinite candidate local output')
        scores.append(float(difference.square().sum(dtype=torch.float64)))
        outputs.append(output.cpu())
    # Python binary64 sums in fixed calibration-case order.
    return sum(scores), scores, outputs


@torch.inference_mode()
def export_candidates(args, report, legacy, state):
    exports = {}
    for arm in ARMS:
        directory = DATA/arm
        require(not directory.exists(), f'Refusing existing export directory {directory}')
        directory.mkdir(parents=True)
        exports[arm] = dict(format=FORMAT, recipe=RECIPE, status='exporting', arm=arm,
            layers=[], target_count=0, exact_roundtrip_count=0, sources=report['sources'],
            legacy_manifest=report['legacy_manifest'], state=report['state'], search_contract=report['search_contract'])
    report['export_progress'] = {arm: 0 for arm in ARMS}
    old_rows = {r['name']: r for r in legacy['layers']}
    with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as source:
        for index, name in enumerate(TARGET_NAMES):
            guard(args, report)
            started = time.monotonic()
            w_cpu = source.get_tensor(name+'.weight').to(torch.bfloat16).contiguous()
            source_sha = base.tensor_record(w_cpu)['sha256']
            require(source_sha == old_rows[name]['source_weight_sha256'], 'Original BF16 weight hash mismatch')
            smooth = state['layers'][name]['smooth'].to(device='cuda', dtype=torch.bfloat16)
            require(bool(torch.isfinite(smooth).all()) and bool((smooth > 0).all()), 'Invalid frozen smooth')
            ws = w_cpu.cuda()*smooth
            del w_cpu
            q0 = nvfp4_qdq(ws, element_size=CHUNK_ROWS)
            q0_sig = base.tensor_record(q0)
            wr = torch.linspace(0, ws.shape[0]-1, WITNESS_ROWS, dtype=torch.float64).long().cuda()
            wc = torch.linspace(0, ws.shape[1]-1, WITNESS_COLS, dtype=torch.float64).long().cuda()
            def witness(tensor):
                return tensor.index_select(0, wr).index_select(1, wc).cpu()
            samples = []
            for case in report['calibration_cases']:
                record = case['layers'][name]['artifact']
                sample = torch.load(base.verify_file(record), map_location='cpu', weights_only=True, mmap=True)
                require(sample['layer'] == name and sample['case_id'] == case['case_id'] and
                        base.tensor_record(sample['xs']) == sample['source_signature'] and
                        base.tensor_record(sample['target']) == sample['target_signature'], 'Saved calibration row hashes changed')
                require(sample['smooth_signature'] == base.tensor_record(smooth), 'Calibration smoothing changed')
                samples.append(dict(case_id=case['case_id'], artifact=record,
                    packet=packet_from_payload(sample['packet'], 'cuda'),
                    xs=sample['xs'].cuda(), target=sample['target'].cuda()))
            histories, first, witness_values = {}, None, []
            for arm in ARMS:
                q_previous, previous_sig = q0, q0_sig
                best_score, best_payload, best_outputs, best_k = None, None, None, None
                history = []
                for k in range(N_CANDIDATES):
                    guard(args, report)
                    seed = 310000+1000*(index//4)+50*(index%4)+k
                    parent_q = q0 if arm == 'restart' else q_previous
                    parent_sig = q0_sig if arm == 'restart' else previous_sig
                    target = ws-parent_q
                    torch.manual_seed(seed)
                    branch = LowRankBranch(ws.shape[1], ws.shape[0], rank=32, weight=target)
                    a, b = branch.a.weight.detach(), branch.b.weight.detach()
                    require(a.dtype == b.dtype == torch.bfloat16 and
                            tuple(a.shape) == (32, ws.shape[1]) and tuple(b.shape) == (ws.shape[0], 32), 'LR ABI changed')
                    lowrank = b@a
                    residual = ws-lowrank
                    packet = pack_activation_legacy(residual, chunk_rows=CHUNK_ROWS)
                    independent = nvfp4_qdq(residual, element_size=CHUNK_ROWS)
                    decoded = packet.decode(dtype=torch.bfloat16, chunk_rows=CHUNK_ROWS)
                    require(bool(torch.isfinite(independent).all()) and torch.equal(decoded, independent),
                            f'{name}/{arm}/{k}: independent packed weight roundtrip failed')
                    module = NativeH3Linear(packet, smooth, a, b, chunk_rows=CHUNK_ROWS)
                    total, scores, outputs = score_candidate(module, samples, args, report)
                    candidate = dict(k=k, seed=seed, score=total, case_ids=[s['case_id'] for s in samples],
                        per_case_sse=scores, parent_q=parent_sig, target=base.tensor_record(target),
                        lr_a=base.tensor_record(a), lr_b=base.tensor_record(b),
                        residual=base.tensor_record(residual), decoded_q=base.tensor_record(decoded),
                        actual_q=base.tensor_record(independent),
                        packet=packet_signature(packet), roundtrip_exact=True,
                        output_signatures=[base.tensor_record(o) for o in outputs])
                    if k == 0:
                        common = {field: candidate[field] for field in
                            ('seed', 'parent_q', 'target', 'lr_a', 'lr_b', 'residual', 'decoded_q', 'actual_q',
                             'packet', 'per_case_sse', 'score', 'output_signatures')}
                        if arm == 'restart':
                            first = common
                        else:
                            require(common == first, f'{name}: candidate zero arms differ')
                    witness_values.append(dict(arm=arm, k=k, ws=witness(ws), parent_q=witness(parent_q),
                        target=witness(target), lowrank=witness(lowrank), residual=witness(residual),
                        decoded_q=witness(decoded), actual_q=witness(independent)))
                    history.append(candidate)
                    if best_score is None or total < best_score:
                        best_score, best_k = total, k
                        best_payload, best_outputs = exporter.export_payload(module, name), outputs
                    # Latest candidate, not argmin: this is the carry-Q recurrence.
                    q_previous, previous_sig = independent, candidate['actual_q']
                    del branch, a, b, lowrank, residual, target, packet, decoded, module, outputs, independent
                    guard(args, report)
                require(best_k == min(range(8), key=lambda k: history[k]['score']), 'First-tie best selection differs')
                path = DATA/arm/'layers'/(name+'.pt')
                record = save_tensor(best_payload, path, report)
                report['export_bytes'] += record['bytes']
                best = history[best_k]
                require(base.tensor_record(best_payload['tensors']['lr_a']) == best['lr_a'] and
                        base.tensor_record(best_payload['tensors']['lr_b']) == best['lr_b'], 'Export factors not from best candidate')
                exported_packet = signature(dict(codes=best_payload['tensors']['weight_packed'],
                    scales=best_payload['tensors']['weight_scales_swizzled'],
                    global_scale=best_payload['tensors']['weight_global'], original_shape=best_payload['shape']))
                require(exported_packet == best['packet'], 'Export packet not from the same selected candidate')
                selected_outputs = save_tensor(dict(layer=name, arm=arm, selected_k=best_k,
                    case_ids=[s['case_id'] for s in samples], outputs=best_outputs,
                    output_signatures=best['output_signatures']), DATA/'selected_local_outputs'/arm/(name+'.pt'), report)
                item = dict(name=name, shape=list(ws.shape), file=str(path.relative_to(DATA/arm)),
                    file_sha256=record['sha256'], file_bytes=record['bytes'], source_weight_sha256=source_sha,
                    smooth=base.tensor_record(best_payload['tensors']['smooth']), lr_a=best['lr_a'], lr_b=best['lr_b'],
                    roundtrip=dict(exact=True, changed_elements=0, elements=ws.numel()), bias=None,
                    residual_dtype='torch.bfloat16', lr_multiplier=1.0, selected_k=best_k,
                    selected_score=best_score, selected_decoded_q=best['decoded_q'], selected_packet=best['packet'],
                    selected_local_outputs=selected_outputs)
                exports[arm]['layers'].append(item)
                report['export_progress'][arm] += 1
                histories[arm] = dict(selected_k=best_k, selected_score=best_score, candidates=history,
                                     export=record, selected_local_outputs=selected_outputs)
                del q_previous, best_payload, best_outputs
            witness_record = save_tensor(dict(layer=name, row_indices=wr.cpu(), column_indices=wc.cpu(),
                candidates=witness_values, scope='Fixed 8x8 coordinate arithmetic witness, not full SVD reproduction'),
                DATA/'candidate_witnesses'/(name+'.pt'), report)
            search_record = save_json_artifact(dict(layer=name, source_weight_sha256=source_sha,
                q0=q0_sig, candidate_zero_exact=True, arms=histories, recurrence_witness=witness_record),
                DATA/'candidate_metadata'/(name+'.json'), report)
            report['search_layers'].append(dict(layer=name, metadata=search_record, candidate_zero_exact=True,
                selected={arm: dict(k=histories[arm]['selected_k'], score=histories[arm]['selected_score'])
                          for arm in ARMS}, recurrence_witness=witness_record))
            report['active_export'] = dict(layer=name, seconds=time.monotonic()-started)
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            save(report, args.output)
            print(f'E065 searched {index+1}/200 {name}', flush=True)
            del samples, ws, smooth, q0, witness_values, sample
            gc.collect()
    report['exports'] = {}
    for arm, result in exports.items():
        require(len(result['layers']) == 200, 'Incomplete selected export')
        result.update(status='complete', target_count=200, exact_roundtrip_count=200)
        path = DATA/arm/'manifest.json'
        require(not path.exists(), f'Refusing manifest {path}')
        save(result, path)
        record = file_record(path)
        report['exports'][arm] = record
        report['data_bytes'] += record['bytes']
        report['export_bytes'] += record['bytes']
    require(report['candidate_native_calls'] == MAX_LOCAL_CANDIDATE_CALLS, 'Incomplete candidate evaluation count')
    torch.cuda.empty_cache()
    guard(args, report)
    save(report, args.output)


def run_case(pipe, item, arm, checked, cpu_pipe, args, report):
    _run_case(pipe, item, arm, checked, cpu_pipe, args, report)
    report['diagnostic_complete_dit_calls'] += 1
    report['data_bytes'] += report['cases'][-1]['artifact']['bytes']
    guard(args, report)
    save(report, args.output)


def diagnostic_readout(report, items):
    rows = []
    for item in items:
        if item['input_role'] == 'next_shifted':
            continue
        k = key(item)
        scores = {arm: report['reference_metrics'][k][arm]['video']['sse']
                  for arm in ('legacy_selected', 'plain')}
        for arm in ARMS:
            row = next(r for r in report['cases'] if
                       (r['case_id'], r['position'], r['arm']) == (item['id'], item['input_role'], arm))
            scores[arm] = row['metrics']['video']['sse']
        rows.append(dict(case_id=item['id'], position=item['input_role'], video_sse=scores,
            carry_over_restart=scores['carry']/scores['restart'] if scores['restart'] else None,
            carry_over_legacy=scores['carry']/scores['legacy_selected'] if scores['legacy_selected'] else None))
    require(len(rows) == 4, 'Expected four diagnostic teacher rows')
    return dict(rows=rows, metric='video velocity FP64 SSE; individual states, no pooled acceptance gate',
                shifted_evaluation='not authorized', method_claim=False)


@torch.inference_mode()
def evaluate(args, report, manifest, e014_manifest, items, legacy):
    checked = old.load_complete(REPORTS/'check.json')
    for field in ('sources', 'e060_audit', 'legacy_manifest', 'state', 'references',
                  'reference_reports', 'e059_evaluation', 'calibration_inputs', 'search_contract', 'reuse'):
        require(checked[field] == report[field], f'E065 CPU binding changed: {field}')
    require(checked['cuda_initialized'] is False, 'E065 check was not CPU-only')
    require(not DATA.exists(), f'Refusing existing E065 data directory {DATA}')
    DATA.mkdir(parents=True)
    report['cpu_check_reference'] = file_record(REPORTS/'check.json')
    report['reference_metrics'] = checked['reference_metrics']
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    require(torch.cuda.get_device_capability() == (12, 0), 'E065 requires SM120')
    state = torch.load(report['state']['file'], map_location='cpu', weights_only=True, mmap=True)
    require(set(state['layers']) == set(TARGET_NAMES) and state['config'] == legacy['state_config'], 'Legacy state layout differs')
    setup_path = DATA/'setup_bf16.json'
    old_check = old.load_complete(manifest['e014_check']['file'])
    setup = dict(experiment='E065_inherited_E014_setup', status='running', complete_dit_calls=0, sources=old_check['sources'])
    setup_args = types.SimpleNamespace(arm='bf16', deadline_unix=args.deadline_unix,
        check_report=Path(manifest['e014_check']['file']), output=setup_path)
    pipe = base.setup_model(setup_args, setup, e014_manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    report['data_bytes'] += report['model_setup']['bytes']
    for case in report['calibration_inputs']:
        capture_calibration(pipe, case, checked, state, legacy, args, report)
    require(report['calibration_bf16_calls'] == 8 and report['calibration_activation_packs'] == 1600 and
            report['shape_native_calls'] == 8 and len(report['selection_shape_controls']) == 4,
            'Calibration/shape-control stage incomplete')
    # Release the 200 BF16 target weights before allocating candidate workspaces.
    guard(args, report)
    before = base.non_target_identity(pipe.dit)
    report['legacy_installation'] = install_native_h3(pipe.dit, e014_manifest['svd_export_dir'],
        activation_packer=base.pack_activation_fast, chunk_rows=CHUNK_ROWS)
    require(base.non_target_identity(pipe.dit) == before, 'Legacy installation changed non-target tensors')
    builder.configure_schedule(pipe, manifest['settings'])
    cpu_pipe = builder.make_cpu_pipeline(manifest['settings'])
    first = next(i for i in items if (i['id'], i['input_role']) == ('e010_p030_s05', 'source_teacher'))
    run_case(pipe, first, 'legacy_selected', checked, cpu_pipe, args, report)
    export_candidates(args, report, legacy, state)
    del state
    teachers = [i for i in items if i['input_role'] != 'next_shifted']
    require(len(teachers) == 4, 'Diagnostic teacher count differs')
    for arm in ARMS:
        install_export(pipe, arm, args, report)
        for item in teachers:
            run_case(pipe, item, arm, checked, cpu_pipe, args, report)
    require(report['complete_dit_calls'] == report['attempted_dit_calls'] == MAX_CALLS and
            len(report['cases']) == 9, 'E065 final full-call count differs')
    report.update(status='complete', active_call=None, diagnostic_readout=diagnostic_readout(report, items),
                  decision='Report fixed-smooth carry versus restart; no automatic follow-up or new-method claim')
    guard(args, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'evaluate'), required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or REPORTS/f'{args.phase}.json'
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    started = time.time()
    if args.phase == 'evaluate':
        require(args.deadline_unix is not None and 0 < args.deadline_unix-started <= WALL_SECONDS,
                'One root-assigned absolute deadline at most 1800 seconds must cover all GPU work')
    report = dict(experiment='E065', phase=args.phase, status='running', start_epoch=started,
        deadline_unix=args.deadline_unix, wall_budget_seconds=WALL_SECONDS, max_dit_calls=MAX_CALLS,
        data_byte_limit=MAX_DATA_BYTES, data_bytes=0, export_bytes=0, attempted_dit_calls=0, complete_dit_calls=0,
        calibration_bf16_calls=0, diagnostic_complete_dit_calls=0,
        calibration_activation_packs=0, candidate_native_calls=0,
        shape_native_calls=0, peak_allocated_bytes=0, cases=[], calibration_cases=[],
        selection_shape_controls=[], search_layers=[],
        scope='Fixed legacy smooth; eight candidates each; carry latest Q versus fresh Q0; strong-baseline comparison')
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E065 absolute wall deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, e014_manifest, items, legacy = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, manifest, items, legacy)
        else:
            evaluate(args, report, manifest, e014_manifest, items, legacy)
        report['elapsed_seconds'] = time.time()-started
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', elapsed_seconds=time.time()-started, error=traceback.format_exc())
        save(report, args.output)
        raise
    finally:
        if args.phase == 'evaluate':
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
