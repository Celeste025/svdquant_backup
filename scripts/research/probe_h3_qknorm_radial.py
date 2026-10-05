#!/usr/bin/env python3
"""E063: same-smooth rank0/rank32 QK radial-gain necessary-condition screen."""
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
import probe_h3_swiglu_interaction_v2 as prior
from h3_native_nvfp4 import NativeH3Linear
from h3_plain_native_nvfp4 import PlainNativeH3Linear, pack_activation_legacy
from minimax_h3_svdquant_common import H3_DIT_PATH, nvfp4_qdq
from diffsynth.models import minimax_h3_dit_comfy as comfy
from safetensors import safe_open
import torch
import torch.nn.functional as F

base, old = prior.base, prior.old
save, file_record, signature, require, cpu = prior.save, prior.file_record, prior.signature, prior.require, prior.cpu
CASE_IDS, BLOCKS, ROWS = prior.CASE_IDS, prior.BLOCKS, prior.ROWS
PLAN = ROOT/'research_state/06_experiments/E063_h3_qknorm_radial_plan.md'
REPORTS = ROOT/'results/research/E063'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E063')
LIMIT = 2*1024**3
ARM0, ARM32 = 'rank0_same_smooth', 'legacy_rank32'


def prerequisites(report):
    require(PLAN.is_file(), 'E063 plan required')
    binding = {}
    manifest = base.prerequisites(types.SimpleNamespace(manifest=base.MANIFEST, plan=base.PLAN), binding)
    checked = old.load_complete(base.REPORTS/'E014_check_bf16.json')
    history = old.load_complete(base.REPORTS/'E014_evaluate_bf16.json')
    require(checked['sources'] == binding['sources'] == history['sources'] and
        checked['cuda_initialized'] is False and history['arm'] == 'bf16', 'E014 provenance changed')
    export = Path(manifest['svd_export_dir'])
    legacy = old.load_complete(export/'manifest.json')
    layers = {}
    for block in BLOCKS:
        name = f'blocks.{block}.attn.qkv_proj'
        row = next(r for r in legacy['layers'] if r['name'] == name)
        artifact = file_record(export/row['file'])
        require(row['bias'] is None and artifact['sha256'] == row['file_sha256'], 'Legacy QKV export changed')
        layers[str(block)] = dict(layer=row, native_artifact=artifact)
    sources = dict(binding['sources'])
    extra = (Path(__file__), PLAN, Path(prior.__file__), Path(inspect.getfile(comfy)),
        Path(inspect.getfile(comfy._apply_rope)), HERE/'h3_plain_native_nvfp4.py', HERE/'h3_native_nvfp4.py')
    sources.update({str(p.resolve()): file_record(p) for p in extra})
    references = {cid: next(r for r in history['cases'] if r['id'] == cid) for cid in CASE_IDS}
    for row in references.values():
        base.verify_file(row['artifact'])
    report.update(sources=sources, plan=file_record(PLAN), layers=layers, references=references,
        inherited_check=file_record(base.REPORTS/'E014_check_bf16.json'),
        inherited_evaluation=file_record(base.REPORTS/'E014_evaluate_bf16.json'),
        legacy_manifest=file_record(export/'manifest.json'), asset_binding=binding['asset_binding'],
        attention_function=dict(module=comfy._comfy_attention_forward.__module__,
            name=comfy._comfy_attention_forward.__name__, source=file_record(inspect.getfile(comfy._comfy_attention_forward))),
        layout='[rows,3,heads,head_dim] from actual ComfyPruned bound method; verified against q_norm/k_norm inputs')
    return manifest, [next(c for c in manifest['cases'] if c['id'] == cid) for cid in CASE_IDS]


@torch.inference_mode()
def cpu_check(report, cases):
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    torch.set_num_threads(6)
    pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['inputs'] = {}
    for case in cases:
        value = base.load_case(case)
        validated = base.validate_case(case, value, pipe)
        actual = prior.packed_actual(pipe, value)
        ref = report['references'][case['id']]
        payload = torch.load(base.verify_file(ref['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(validated == ref['input_signature'] == payload['input_signature'], 'Historical input changed')
        require(signature(actual) == ref['actual_dit_inputs'] == payload['actual_dit_inputs'], 'Actual input changed')
        for field in ('raw_outputs', 'velocities'):
            require(signature(payload[field]) == ref[field], 'Historical output hash changed')
        selected = prior.selection(actual)
        report['inputs'][case['id']] = dict(input_signature=validated, actual_dit_input_signature=signature(actual),
            selection_signature=signature(selected), selected_indices=selected['indices'].tolist(),
            video_count=selected['video_count'], packed_segment=selected['packed_segment'])
    with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as source:
        for spec in report['layers'].values():
            row = spec['layer']
            require(source.get_slice(row['name']+'.weight').get_shape() == row['shape'], 'Original QKV shape changed')
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, input_count=2, rows_per_cell=ROWS, fixed_blocks=list(BLOCKS))


def guard(args, report):
    require(time.time() < args.deadline_unix, 'E063 absolute deadline reached')
    require(report['attempted_dit_calls'] <= 2 and report['attempted_native_qkv_calls'] <= 12 and
        report['activation_packs'] <= 6, 'E063 call/pack budget exceeded')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E063 peak reached 60 GiB')
    require(report['tensor_bytes'] < LIMIT, 'E063 tensor files reached 2 GiB')


def save_tensor(value, path, report):
    require(not path.exists(), f'Refusing existing artifact {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(value, stream)
    rec = file_record(path)
    report['tensor_bytes'] += rec['bytes']
    require(report['tensor_bytes'] < LIMIT, 'E063 tensor files reached 2 GiB')
    return rec


def split_qkv(value, attn):
    require(value.ndim == 2 and value.shape[-1] == 3*attn.num_heads*attn.head_dim, 'QKV dimensions changed')
    qkv = value.view(value.shape[0], 3, attn.num_heads, attn.head_dim)
    return dict(q=qkv[:, 0], k=qkv[:, 1], v=qkv[:, 2])


def packet_signature(packet):
    return signature(dict(packed=packet.packed.cpu(), scales=packet.swizzled_scales.cpu(),
        global_scale=packet.global_scale.cpu(), original_shape=list(packet.original_shape)))


def gain_metrics(samples, norms):
    """FP64 signed SSE changes; every radial projector is teacher-defined."""
    def sse_by_head(x):
        return x.square().sum((0, 2))
    groups, head_groups, checks, geometry = {}, {}, {}, {}
    for part in ('q', 'k', 'v'):
        teacher = samples['teacher']['raw'][part].double()
        norm2 = teacher.square().sum(-1, keepdim=True)
        safe = torch.where(norm2 > 0, norm2, torch.ones_like(norm2))
        energies, radial_coefficients = {}, {}
        for arm in (ARM0, ARM32):
            value = samples[arm]['raw'][part].double()
            error = value-teacher
            radial = teacher*((error*teacher).sum(-1, keepdim=True)/safe)
            tangent = error-radial
            energies[arm] = dict(raw=sse_by_head(error), radial=sse_by_head(radial), tangent=sse_by_head(tangent))
            relative = (energies[arm]['raw']-energies[arm]['radial']-energies[arm]['tangent']).abs()/energies[arm]['raw'].clamp_min(1e-300)
            require(bool(torch.isfinite(error).all()) and float(relative.max()) < 1e-10, 'FP64 radial energy closure failed')
            checks[f'{part}/{arm}'] = dict(max_relative_energy_closure=float(relative.max()),
                max_abs_radial_tangent_dot=float((radial*tangent).sum(-1).abs().max()))
            radial_coefficients[arm] = (value*teacher).sum(-1)/safe.squeeze(-1)
            if part != 'v':
                for stage in ('norm', 'rope'):
                    diff = samples[arm][stage][part].double()-samples['teacher'][stage][part].double()
                    require(bool(torch.isfinite(diff).all()), 'Nonfinite norm/RoPE difference')
                    energies[arm][stage] = sse_by_head(diff)
        per_head = []
        for h in range(teacher.shape[1]):
            row = {f'E{rank}_{stage}': float(energies[arm][stage][h])
                for rank, arm in ((0, ARM0), (32, ARM32)) for stage in energies[arm]}
            for stage in energies[ARM0]:
                row['G_'+stage] = row['E0_'+stage]-row['E32_'+stage]
            row['signed_gain_closure_error'] = row['G_raw']-row['G_radial']-row['G_tangent']
            per_head.append(row)
        head_groups[part] = per_head
        groups[part] = {k: sum(row[k] for row in per_head) for k in per_head[0]}
        geometry[part] = dict(teacher_zero_norm=int((norm2 == 0).sum()),
            teacher_zero_norm_by_head=(norm2.squeeze(-1) == 0).sum(0).tolist(),
            radial_negative_counts={arm: int(((coef < 0) & (norm2.squeeze(-1) > 0)).sum()) for arm, coef in radial_coefficients.items()},
            teacher_min_mean_square=float(norm2.min()/teacher.shape[-1]),
            radial_nonpositive_counts={arm: int(((coef <= 0) & (norm2.squeeze(-1) > 0)).sum()) for arm, coef in radial_coefficients.items()},
            radial_nonpositive_by_head={arm: ((coef <= 0) & (norm2.squeeze(-1) > 0)).sum(0).tolist() for arm, coef in radial_coefficients.items()})
        if part != 'v':
            eps = norms[part]['eps']
            geometry[part].update(epsilon=eps,
                teacher_at_or_below_epsilon=int((norm2/teacher.shape[-1] <= eps).sum()))
    groups['qk'] = {k: groups['q'][k]+groups['k'][k] for k in groups['q']}
    for name, group in groups.items():
        scale = max(group['E0_raw']+group['E32_raw'], 1e-300)
        require(abs(group['signed_gain_closure_error'])/scale < 1e-10, f'{name}: signed gain closure failed')
    e = groups['qk']
    raw_gain = e['G_raw']/e['E0_raw'] if e['E0_raw'] > 0 else None
    radial_share = e['G_radial']/e['G_raw'] if e['G_raw'] != 0 else None
    norm_gain = e['G_norm']/e['E0_norm'] if e['E0_norm'] > 0 else None
    gate = dict(raw_relative_gain=raw_gain, radial_gain_over_raw_gain=radial_share, norm_relative_gain=norm_gain,
        passed=raw_gain is not None and radial_share is not None and norm_gain is not None and
            raw_gain >= .1 and radial_share >= .75 and norm_gain <= .1)
    return dict(by_part=groups, by_head=head_groups, geometry=geometry, numerical_checks=checks, gate=gate)


@torch.inference_mode()
def capture_teacher(pipe, case, checked, args, report):
    guard(args, report)
    require(report['attempted_dit_calls'] < 2, 'Teacher call budget exhausted')
    value = base.load_case(case)
    cpu_pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    selected = prior.selection(prior.packed_actual(cpu_pipe, value))
    expected = checked['inputs'][case['id']]
    require(base.validate_case(case, value, cpu_pipe) == expected['input_signature'] and
        signature(selected) == expected['selection_signature'], 'CPU input/selection changed')
    indices = selected['indices'].cuda()
    captures = {b: {} for b in BLOCKS}
    handles, actual, raw = [], [], []
    context = {'active': None}
    require(type(pipe.dit) is comfy.MiniMaxH3DiTComfyPruned, 'Actual model is not ComfyPruned')
    def dit_pre(_module, pos, kw):
        actual.append(cpu(dict(args=pos, kwargs=kw)))
        require(signature(actual[0]) == expected['actual_dit_input_signature'], 'Actual teacher input mismatch')
    def dit_post(_module, pos, output):
        raw.append(output)
    handles.extend((pipe.dit.register_forward_pre_hook(dit_pre, with_kwargs=True), pipe.dit.register_forward_hook(dit_post)))
    for block in BLOCKS:
        attn, cap = pipe.dit.blocks[block].attn, captures[block]
        require(attn.forward.__func__ is comfy._comfy_attention_forward, 'Attention bound function changed')
        def attn_pre(_module, pos, kw, block=block, cap=cap):
            context['active'] = block
            cap['rope_freqs'] = kw['rope_freqs'].cpu()
        def attn_post(_module, pos, output):
            context['active'] = None
        def qkv_post(_module, pos, output, cap=cap):
            require(pos[0].dtype == output.dtype == torch.bfloat16, 'Actual QKV dtype changed')
            cap['full_input'] = pos[0].cpu()
            cap['teacher_qkv'] = output.cpu()
            cap['norm_actual'], cap['layout_checks'] = {}, {}
        handles.extend((attn.register_forward_pre_hook(attn_pre, with_kwargs=True),
            attn.register_forward_hook(attn_post), attn.qkv_proj.register_forward_hook(qkv_post)))
        for part in ('q', 'k'):
            def norm_hook(_module, pos, output, part=part, cap=cap, attn=attn):
                observed = pos[0].cpu()
                expected_part = split_qkv(cap['teacher_qkv'], attn)[part]
                require(signature(observed) == signature(expected_part), 'Comfy QKV layout does not match actual norm input bytes')
                cap['layout_checks'][part] = dict(exact=True, actual_input_signature=signature(observed))
                cap['norm_actual'][part] = dict(full_signature=signature(output.cpu()), sample=output.index_select(0, indices).cpu())
            handles.append(getattr(attn, part+'_norm').register_forward_hook(norm_hook))
    original_sdpa = comfy._sdpa_varlen_attention
    def capture_rope(q, k, v, **kwargs):
        block = context['active']
        if block is not None:
            cap = captures[block]
            cap['rope_actual'] = {name: dict(full_signature=signature(tensor.cpu()), sample=tensor.index_select(0, indices).cpu())
                for name, tensor in (('q', q), ('k', k))}
            require(signature(v.cpu()) == signature(split_qkv(cap['teacher_qkv'], pipe.dit.blocks[block].attn)['v']),
                'Comfy V layout changed at attention consumer')
        return original_sdpa(q, k, v, **kwargs)
    audit = base.RuntimeAudit()
    audit.phase = 'resident_bf16'
    report['attempted_dit_calls'] += 1
    report['active_case'] = case['id']
    save(report, args.output)
    comfy._sdpa_varlen_attention = capture_rope
    try:
        with audit.installed():
            result = base.gpu_call(pipe, case, value)()
            torch.cuda.synchronize()
        report['complete_dit_calls'] += 1
    finally:
        comfy._sdpa_varlen_attention = original_sdpa
        for handle in handles:
            handle.remove()
    base.audit_contract(audit.row(), 'bf16')
    require(len(raw) == len(actual) == 1, 'Teacher hook count changed')
    raw_outputs, raw_sig = base.cpu_outputs(raw[0])
    velocities, vel_sig = base.cpu_outputs(result)
    ref = report['references'][case['id']]
    require(raw_sig == ref['raw_outputs'] and vel_sig == ref['velocities'], 'Teacher historical replay not byte exact')
    artifact = save_tensor(dict(case_id=case['id'], actual_dit_inputs=actual[0], raw_outputs=raw_outputs,
        velocities=velocities), DATA/'teacher'/f'{case["id"]}.pt', report)
    report['teacher_replays'].append(dict(case_id=case['id'], artifact=artifact, historical_artifact=ref['artifact'],
        replay_exact=True, actual_dit_input_signature=signature(actual[0]), raw_outputs=raw_sig,
        velocities=vel_sig, runtime_audit=audit.row()))
    save(report, args.output)
    return captures, selected


@torch.inference_mode()
def local_cell(pipe, case, block, cap, selected, args, report):
    guard(args, report)
    attn = pipe.dit.blocks[block].attn
    spec = report['layers'][str(block)]
    require(attn.forward.__func__ is comfy._comfy_attention_forward, 'Comfy function binding changed')
    payload = torch.load(base.verify_file(spec['native_artifact']), map_location='cpu', weights_only=True, mmap=True)
    native = NativeH3Linear.from_export(payload, device='cuda', activation_packer=base.pack_activation_fast, chunk_rows=1024)
    weight = attn.qkv_proj.weight
    weight_sig = signature(weight.cpu())
    require(weight.dtype == torch.bfloat16 and weight_sig['sha256'] == spec['layer']['source_weight_sha256'], 'Original BF16 QKV weight mismatch')
    ws = weight*native.smooth
    weight_packet = pack_activation_legacy(ws, chunk_rows=1024)
    independently_quantized = nvfp4_qdq(ws, element_size=1024)
    decoded = weight_packet.decode(dtype=torch.bfloat16, chunk_rows=1024)
    require(torch.equal(decoded, independently_quantized) and bool(torch.isfinite(decoded).all()), 'Fresh rank0 independent QDQ roundtrip failed')
    rank0 = PlainNativeH3Linear(weight_packet, bias=None, chunk_rows=1024)
    del ws, independently_quantized, decoded
    xs = cap.pop('full_input').cuda()/native.smooth
    require(report['activation_packs'] < 6, 'Activation pack budget exhausted')
    with base.collect_fastpack_checks() as fastpack:
        packet = native.pack_input(xs)
    report['activation_packs'] += 1
    require(fastpack.summary['checked_calls'] == 1, 'Exactly one activation pack per cell required')
    before = packet_signature(packet)
    indices = selected['indices'].cuda()
    rope = cap['rope_freqs'].cuda()
    norms = {part: dict(gamma=getattr(attn, part+'_norm').weight.cpu(), eps=float(getattr(attn, part+'_norm').eps)) for part in ('q', 'k')}
    samples = {}
    replay_checks = {}
    def readings(qkv, label):
        parts = split_qkv(qkv, attn)
        out = dict(raw={part: value.index_select(0, indices).cpu() for part, value in parts.items()}, norm={}, rope={})
        for part in ('q', 'k'):
            normalized = getattr(attn, part+'_norm')(parts[part])
            rotated = comfy._apply_rope(normalized, rope)
            if label == 'teacher':
                require(signature(normalized.cpu()) == cap['norm_actual'][part]['full_signature'], 'Full teacher norm replay mismatch')
                require(signature(rotated.cpu()) == cap['rope_actual'][part]['full_signature'], 'Full teacher RoPE replay mismatch')
                replay_checks[part] = dict(norm_exact=True, rope_exact=True)
            out['norm'][part] = normalized.index_select(0, indices).cpu()
            out['rope'][part] = rotated.index_select(0, indices).cpu()
        return out
    teacher = cap.pop('teacher_qkv').cuda()
    samples['teacher'] = readings(teacher, 'teacher')
    del teacher
    audit = base.RuntimeAudit()
    audit.phase = 'native'
    require(report['attempted_native_qkv_calls'] <= 10, 'Native QKV call budget exhausted')
    report['active_local'] = dict(case_id=case['id'], block=block)
    with audit.installed():
        report['attempted_native_qkv_calls'] += 1
        save(report, args.output)
        output0 = rank0.main_from_packet(packet, include_bias=True)
        torch.cuda.synchronize()
        report['complete_native_qkv_calls'] += 1
        require(packet_signature(packet) == before, 'Rank0 mutated shared activation packet')
        samples[ARM0] = readings(output0, ARM0)
        del output0
        guard(args, report)
        report['attempted_native_qkv_calls'] += 1
        save(report, args.output)
        branch = 1.0*F.linear(F.linear(xs, native.lr_a), native.lr_b)
        main = native.main_from_packet(packet, include_bias=True)
        output32 = main+branch
        torch.cuda.synchronize()
        report['complete_native_qkv_calls'] += 1
        require(packet_signature(packet) == before, 'Rank32 mutated shared activation packet')
        samples[ARM32] = readings(output32, ARM32)
    require(audit.row()['scaled_mm_calls'] == 2 and audit.row()['sdpa_calls'] == audit.row()['disk_loads'] == 0, 'Local QKV audit changed')
    metrics = gain_metrics(samples, norms)
    data = dict(case_id=case['id'], block=block, indices=selected['indices'], offsets=selected['offsets'],
        samples=samples, norms=norms, rope_freqs=cap['rope_freqs'].index_select(0, selected['indices']),
        activation_packet_signature=before, activation_global=packet.global_scale.cpu(),
        weight_globals={ARM0: rank0.weight_global.cpu(), ARM32: native.weight_global.cpu()},
        source_weight=weight_sig, legacy_artifact=spec['native_artifact'],
        rank0_weight_packet_signature=packet_signature(weight_packet), rank0_roundtrip_exact=True,
        actual_layout_checks=cap['layout_checks'], full_teacher_replays=replay_checks,
        teacher_actual_norm_samples={p: v['sample'] for p, v in cap['norm_actual'].items()},
        teacher_actual_rope_samples={p: v['sample'] for p, v in cap['rope_actual'].items()})
    artifact = save_tensor(data, DATA/'cells'/f'{case["id"]}_b{block:02d}.pt', report)
    report['cases'].append(dict(case_id=case['id'], block=block, artifact=artifact, status='complete',
        metrics=metrics, shared_activation_packet_unchanged=True, rank0_roundtrip_exact=True,
        layout_checks=cap['layout_checks'], teacher_norm_rope_replay=replay_checks,
        activation_packet_signature=before, weight_globals=signature(data['weight_globals']),
        runtime_audit=audit.row(), fastpack_checks=fastpack.summary))
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save(report, args.output)
    print(f'E063 {case["id"]} block {block}: {metrics["gate"]}', flush=True)
    del native, rank0, packet, weight_packet, xs, branch, main, output32, payload
    gc.collect()
    guard(args, report)


@torch.inference_mode()
def evaluate(args, report, manifest, cases):
    checked = old.load_complete(REPORTS/'check.json')
    for field in ('sources', 'layers', 'references', 'legacy_manifest', 'asset_binding', 'attention_function'):
        require(report[field] == checked[field], f'E063 CPU binding changed: {field}')
    require(checked['cuda_initialized'] is False, 'E063 CPU check initialized CUDA')
    report['cpu_check_reference'] = file_record(REPORTS/'check.json')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    inherited_check = base.REPORTS/'E014_check_bf16.json'
    old_check = old.load_complete(inherited_check)
    setup_path = DATA/'setup_bf16.json'
    require(not setup_path.exists(), 'Refusing existing E063 setup')
    setup = dict(experiment='E063_inherited_E014_setup', status='running', complete_dit_calls=0, sources=old_check['sources'])
    setup_args = types.SimpleNamespace(arm='bf16', check_report=inherited_check,
        output=setup_path, deadline_unix=args.deadline_unix)
    pipe = base.setup_model(setup_args, setup, manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    for case in cases:
        captures, selected = capture_teacher(pipe, case, checked, args, report)
        for block in BLOCKS:
            local_cell(pipe, case, block, captures.pop(block), selected, args, report)
        del captures
        gc.collect()
    require(report['attempted_dit_calls'] == report['complete_dit_calls'] == 2 and
        report['attempted_native_qkv_calls'] == report['complete_native_qkv_calls'] == 12 and
        report['activation_packs'] == len(report['cases']) == 6, 'E063 final call/pack counts differ')
    gates = []
    for block in BLOCKS:
        rows = [r for r in report['cases'] if r['block'] == block]
        gates.append(dict(block=block, passed=len(rows) == 2 and all(r['metrics']['gate']['passed'] for r in rows)))
    passed = any(r['passed'] for r in gates)
    report.update(status='complete', active_case=None, active_local=None,
        primary_gate=dict(by_block=gates, passed=passed),
        decision='Eligible to propose a later joint-attention diagnostic; no functional claim' if passed else
            'Stop this radial-gain mismatch candidate; no broader QK-calibration conclusion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'evaluate'))
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or REPORTS/f'{args.phase}.json'
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    started = time.time()
    if args.phase == 'evaluate':
        require(args.deadline_unix is not None and 0 < args.deadline_unix-started <= 900,
            'Require one absolute deadline <=900 seconds')
    report = dict(experiment='E063', phase=args.phase, status='running', start_epoch=started,
        deadline_unix=args.deadline_unix, tensor_bytes=0, tensor_byte_limit=LIMIT,
        attempted_dit_calls=0, complete_dit_calls=0, attempted_native_qkv_calls=0,
        complete_native_qkv_calls=0, activation_packs=0, teacher_replays=[], cases=[],
        scope='Necessary-condition screen only; two teacher DiTs, twelve local native GEMMs, six shared activation packs',
        geometry='FP64 teacher radial projector dot(e,q)/dot(q,q); q=0 gives radial=0 and tangent=e; no norm epsilon',
        limits='No new attention counterfactual, no full native DiT, no training/video/quality claim')
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E063 absolute deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, cases = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, cases)
        else:
            evaluate(args, report, manifest, cases)
        report['elapsed_seconds'] = time.time()-started
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc(), elapsed_seconds=time.time()-started)
        save(report, args.output)
        raise
    finally:
        if args.phase == 'evaluate':
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
