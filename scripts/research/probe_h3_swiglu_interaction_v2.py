#!/usr/bin/env python3
"""E062: fixed-input native fc1 SwiGLU four corners; local diagnostic only."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe_h3_plain_baseline as base
import probe_h3_crossmodal_propagation as old
import prepare_h3_crossmodal_states as builder
from h3_native_nvfp4 import NativeH3Linear
from minimax_h3_svdquant_common import H3_DIT_PATH
from safetensors import safe_open
import torch
import torch.nn.functional as F

PLAN = ROOT/'research_state/06_experiments/E062_h3_swiglu_interaction_plan.md'
REPORTS = ROOT/'results/research/E062'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E062')
CASE_IDS = ('e010_p030_s05', 'e010_p036_s14')
BLOCKS = (0, 24, 49)
ROWS, SHIFT = 512, 256
LIMIT = 4*1024**3
save, file_record, signature, require = base.save, base.file_record, base.tree_signature, old.require
cpu = base.inherited.tree_cpu


def prerequisites(report):
    require(PLAN.is_file(), 'E062 plan required')
    binding = {}
    manifest = base.prerequisites(types.SimpleNamespace(manifest=base.MANIFEST, plan=base.PLAN), binding)
    checked = old.load_complete(base.REPORTS/'E014_check_bf16.json')
    require(checked['sources'] == binding['sources'] and checked['cuda_initialized'] is False,
            'Inherited E014 CPU/source contract changed')
    history = old.load_complete(base.REPORTS/'E014_evaluate_bf16.json')
    require(history['sources'] == checked['sources'] and history['arm'] == 'bf16', 'BF16 history source changed')
    export = Path(manifest['svd_export_dir'])
    legacy = old.load_complete(export/'manifest.json')
    layers = {}
    for block in BLOCKS:
        rows = {}
        for suffix in ('fc1', 'fc2'):
            name = f'blocks.{block}.mlp.{suffix}'
            row = next(r for r in legacy['layers'] if r['name'] == name)
            require(row['bias'] is None, 'Expected bias-free MLP')
            rows[suffix] = row
        record = file_record(export/rows['fc1']['file'])
        require(record['sha256'] == rows['fc1']['file_sha256'], 'Selected legacy native fc1 changed')
        layers[str(block)] = dict(fc1=rows['fc1'], fc2=rows['fc2'], native_artifact=record)
    cases = [next(c for c in manifest['cases'] if c['id'] == cid) for cid in CASE_IDS]
    references = {cid: next(r for r in history['cases'] if r['id'] == cid) for cid in CASE_IDS}
    for row in references.values():
        base.verify_file(row['artifact'])
    sources = dict(binding['sources'])
    sources.update({str(p.resolve()): file_record(p) for p in (Path(__file__), PLAN, HERE/'prepare_h3_crossmodal_states.py')})
    report.update(sources=sources, plan=file_record(PLAN), inherited_manifest=file_record(base.MANIFEST),
        inherited_check=file_record(base.REPORTS/'E014_check_bf16.json'), inherited_evaluation=file_record(base.REPORTS/'E014_evaluate_bf16.json'),
        legacy_manifest=file_record(export/'manifest.json'), asset_binding=binding['asset_binding'],
        layers=layers, references=references)
    return manifest, cases


def packed_actual(pipe, value):
    captured = []
    class Stop(Exception):
        pass
    def capture(*args, **kwargs):
        captured.append({'args': args, 'kwargs': kwargs})
        raise Stop
    state = value['state']
    packed = base.make_packed(pipe, value['embedding'], value['text_token_tags'], state)
    try:
        base.model_fn_minimax_h3(dit=capture, video_latents=state['video']['latents_before'],
            audio_latents=state['audio']['latents_before'], packed=packed, prompt_embeds=value['embedding'],
            timestep_video=state['video']['timestep'].reshape(1), timestep_audio=state['audio']['timestep'].reshape(1))
    except Stop:
        pass
    require(len(captured) == 1, 'Expected one zero-forward CPU sentinel')
    return captured[0]


def selection(actual):
    kw = actual['kwargs']
    video = kw['img_pos_info']['position_ids'].reshape(-1).long()
    require(video.numel() >= ROWS and video.unique().numel() == video.numel(), 'Insufficient/distinct video rows')
    offsets = torch.div(torch.arange(ROWS)*(video.numel()-1), ROWS-1, rounding_mode='floor')
    indices = video[offsets]
    tags = kw['token_tags'].reshape(-1).long()
    require(tags[video].unique().numel() == 1, 'Video mapping mixes modalities')
    combined = kw['inverse_indices'].reshape(-1).long()*3+tags.clamp(min=0)
    segments = kw['packed_seq_params']['cu_seqlens_q'].long()
    sample_segments = torch.bucketize(indices, segments[1:], right=True)
    require(sample_segments.unique().numel() == 1, 'Null must remain within one packed video sample')
    return dict(indices=indices, offsets=offsets, video_count=video.numel(), combined_indices=combined,
        selected_combined_indices=combined[indices], video_tag=int(tags[video[0]]),
        packed_segment=int(sample_segments[0]), selection='floor(i*(Nvideo-1)/511), i=0..511, actual img_pos_info')


@torch.inference_mode()
def cpu_check(report, cases):
    require(not torch.cuda.is_initialized(), 'CPU check must not initialize CUDA')
    torch.set_num_threads(6)
    pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['inputs'] = {}
    for case in cases:
        value = base.load_case(case)
        sig = base.validate_case(case, value, pipe)
        actual = packed_actual(pipe, value)
        history = report['references'][case['id']]
        payload = torch.load(base.verify_file(history['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(sig == history['input_signature'] == payload['input_signature'], 'Historical input changed')
        require(signature(actual) == history['actual_dit_inputs'] == payload['actual_dit_inputs'], 'Actual input mismatch')
        for name in ('raw_outputs', 'velocities'):
            require(signature(payload[name]) == history[name], 'Historical output mismatch')
        chosen = selection(actual)
        report['inputs'][case['id']] = dict(input_signature=sig, actual_dit_input_signature=signature(actual),
            selection_signature=signature(chosen), selected_indices=chosen['indices'].tolist(),
            video_count=chosen['video_count'], packed_segment=chosen['packed_segment'])
        print(f'E062 CPU input valid {case["id"]}', flush=True)
    with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as checkpoint:
        for block in BLOCKS:
            for suffix in ('fc1', 'fc2'):
                row = report['layers'][str(block)][suffix]
                require(checkpoint.get_slice(row['name']+'.weight').get_shape() == row['shape'], 'MLP shape changed')
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, complete_dit_calls=0, input_count=2,
        fixed_blocks=list(BLOCKS), rows_per_cell=ROWS, null_shift=SHIFT)


def guard(args, report):
    require(time.time() < args.deadline_unix, 'E062 absolute deadline reached')
    require(report['attempted_dit_calls'] <= 2 and report['attempted_native_fc1_calls'] <= 6, 'E062 call budget exceeded')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'E062 peak reached 60 GiB')
    require(report['tensor_bytes'] < LIMIT, 'E062 tensor storage reached 4 GiB')


def save_tensor(payload, path, report):
    require(not path.exists(), f'Refusing existing artifact {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    record = file_record(path)
    report['tensor_bytes'] += record['bytes']
    require(report['tensor_bytes'] < LIMIT, 'E062 tensor storage reached 4 GiB')
    return record


def rms_check(got, ref):
    error = got.double()-ref.double()
    rms = float(ref.double().square().mean().sqrt())
    diff = float(error.square().mean().sqrt())
    return dict(reference_rms=rms, error_rms=diff, relative_rms=diff/max(rms, 1e-30),
        passed=bool(torch.isfinite(error).all()) and (diff == 0 if rms == 0 else diff/rms < .01))


def energy(parts, teacher):
    values = {k: v.double() for k, v in parts.items()}
    t = teacher.double()
    tc = t-t.mean(0, keepdim=True)
    denom = tc.square().sum(0, keepdim=True)
    # Exact constant teacher channels have no proportional direction.
    safe = denom.clamp_min(1e-300)
    stats = {}
    for mode in ('raw', 'centered', 'proportional_residual'):
        v = values if mode == 'raw' else {k: x-x.mean(0, keepdim=True) for k, x in values.items()}
        if mode == 'proportional_residual':
            v = {k: x-tc*((tc*x).sum(0, keepdim=True)/safe) for k, x in v.items()}
        ab, c, total = v['a']+v['b'], v['c'], v['total']
        et, eab, ec = [float(x.square().sum()) for x in (total, ab, c)]
        cross = float(2*(ab*c).sum())
        net = et-eab
        stats[mode] = dict(Etotal=et, Eab=eab, Ec=ec, cross=cross, net=net,
            net_over_total=net/et if et > 0 else None,
            signed_closure_error=net-ec-cross,
            identity=rms_check(ab+c, total),
            passes_20pct=et > 0 and net/et >= .2)
    return stats


def four_corners(g, u, gq, uq, *, bf16=False):
    dtype = torch.bfloat16 if bf16 else torch.float32
    g, u, gq, uq = [x.to(dtype) for x in (g, u, gq, uq)]
    phi, phiq = F.silu(g), F.silu(gq)
    corners = dict(BB=phi*u, QB=phiq*u, BQ=phi*uq, QQ=phiq*uq)
    if bf16:
        f = {k: v.float() for k, v in corners.items()}
        return dict(a=f['QB']-f['BB'], b=f['BQ']-f['BB'],
            c=(f['QQ']-f['QB'])-(f['BQ']-f['BB']), total=f['QQ']-f['BB']), corners
    dphi, eu = phiq-phi, uq-u
    return dict(a=dphi*u, b=phi*eu, c=dphi*eu,
        total=corners['QQ']-corners['BB']), corners


@torch.inference_mode()
def capture_teacher(pipe, case, checked, args, report):
    guard(args, report)
    require(report['attempted_dit_calls'] < 2, 'Two teacher calls already attempted')
    value = base.load_case(case)
    cpu_pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    require(base.validate_case(case, value, cpu_pipe) == checked['inputs'][case['id']]['input_signature'], 'Input changed')
    actual_cpu = packed_actual(cpu_pipe, value)
    selected = selection(actual_cpu)
    require(signature(selected) == checked['inputs'][case['id']]['selection_signature'], 'Selected rows changed')
    indices = selected['indices'].cuda()
    captures = {b: {} for b in BLOCKS}
    actual, raw = [], []
    handles = []
    def input_hook(_module, pos, kw):
        actual.append(cpu({'args': pos, 'kwargs': kw}))
        require(signature(actual[0]) == checked['inputs'][case['id']]['actual_dit_input_signature'], 'Actual GPU input mismatch')
    def raw_hook(_module, pos, output):
        raw.append(output)
    handles.extend((pipe.dit.register_forward_pre_hook(input_hook, with_kwargs=True), pipe.dit.register_forward_hook(raw_hook)))
    for block in BLOCKS:
        module, cap = pipe.dit.blocks[block], captures[block]
        def block_pre(_module, pos, kw, cap=cap):
            require(torch.equal(kw['combined_indices'].cpu(), selected['combined_indices']), 'Actual AdaLN indices differ')
            cap['combined_indices_verified'] = True
        def adaln_hook(_module, pos, output, cap=cap):
            require(len(output) == 6 and output[5].dtype == torch.bfloat16, 'Expected actual BF16 MLP AdaLN gate')
            cap['adaln_gate'] = output[5].index_select(0, selected['selected_combined_indices'].cuda()).cpu()
        def fc1_hook(_module, pos, output, cap=cap):
            require(pos[0].dtype == output.dtype == torch.bfloat16, 'BF16 fc1 contract changed')
            cap['full_input'] = pos[0].detach().cpu()
            cap['teacher_fc1'] = output.index_select(0, indices).cpu()
        def fc2_hook(_module, pos, output, cap=cap):
            cap['teacher_hidden_bf16'] = pos[0].index_select(0, indices).cpu()
            cap['teacher_down_bf16'] = output.index_select(0, indices).cpu()
        handles.extend((module.register_forward_pre_hook(block_pre, with_kwargs=True),
            module.adaln_proj.register_forward_hook(adaln_hook), module.mlp.fc1.register_forward_hook(fc1_hook),
            module.mlp.fc2.register_forward_hook(fc2_hook)))
    audit = base.RuntimeAudit()
    audit.phase = 'resident_bf16'
    report['attempted_dit_calls'] += 1
    report['active_case'] = case['id']
    save(report, args.output)
    try:
        with audit.installed():
            result = base.gpu_call(pipe, case, value)()
            torch.cuda.synchronize()
        report['complete_dit_calls'] += 1
    finally:
        for handle in handles:
            handle.remove()
    base.audit_contract(audit.row(), 'bf16')
    require(len(actual) == len(raw) == 1, 'Teacher capture count changed')
    raw_outputs, raw_records = base.cpu_outputs(raw[0])
    velocities, velocity_records = base.cpu_outputs(result)
    history = report['references'][case['id']]
    require(raw_records == history['raw_outputs'] and velocity_records == history['velocities'], 'Historical teacher replay not byte exact')
    artifact = save_tensor(dict(case_id=case['id'], actual_dit_inputs=actual[0], raw_outputs=raw_outputs,
        velocities=velocities), DATA/'teacher'/f'{case["id"]}.pt', report)
    report['teacher_replays'].append(dict(case_id=case['id'], artifact=artifact, historical_artifact=history['artifact'],
        actual_dit_input_signature=signature(actual[0]), raw_outputs=raw_records, velocities=velocity_records,
        replay_exact=True, runtime_audit=audit.row()))
    save(report, args.output)
    del raw, result, actual, value
    gc.collect()
    guard(args, report)
    return captures, selected


@torch.inference_mode()
def local_cell(pipe, case, block, cap, selected, args, report):
    guard(args, report)
    require(report['attempted_native_fc1_calls'] < 6, 'Six local native calls already attempted')
    spec = report['layers'][str(block)]
    payload = torch.load(base.verify_file(spec['native_artifact']), map_location='cpu', weights_only=True, mmap=True)
    native = NativeH3Linear.from_export(payload, device='cuda', activation_packer=base.pack_activation_fast, chunk_rows=1024)
    packet_log = {}
    original_packer = native.activation_packer
    def log_packer(x, **kwargs):
        packet = original_packer(x, **kwargs)
        packet_log.update(global_scale=packet.global_scale.cpu(), input_shape=list(x.shape),
            smooth_input_signature=signature(x.cpu()), packed_shape=list(packet.packed.shape))
        return packet
    native.activation_packer = log_packer
    full_input = cap.pop('full_input').cuda()
    indices = selected['indices'].cuda()
    audit = base.RuntimeAudit()
    audit.phase = 'native'
    report['attempted_native_fc1_calls'] += 1
    report['active_local'] = dict(case_id=case['id'], block=block)
    save(report, args.output)
    with audit.installed(), base.collect_fastpack_checks() as fastpack:
        qfull = native(full_input)
        torch.cuda.synchronize()
    report['complete_native_fc1_calls'] += 1
    require(audit.row()['scaled_mm_calls'] == 1 and audit.row()['sdpa_calls'] == audit.row()['disk_loads'] == 0 and
            fastpack.summary['checked_calls'] == 1, 'Local native audit changed')
    native_sample = qfull.index_select(0, indices)
    gq_full, uq_full = qfull.chunk(2, -1)
    hidden_full = F.silu(gq_full)*uq_full
    down = pipe.dit.blocks[block].mlp.fc2
    require(down.bias is None and down.weight.dtype == torch.bfloat16, 'Teacher down changed')
    native_down = F.linear(hidden_full, down.weight).index_select(0, indices).cpu()
    native_hidden = hidden_full.index_select(0, indices).cpu()
    if str(block) not in report['down_weights']:
        weight = down.weight.cpu()
        weight_record = base.tensor_record(weight)
        require(weight_record['sha256'] == spec['fc2']['source_weight_sha256'], 'Original BF16 down hash mismatch')
        report['down_weights'][str(block)] = dict(tensor=weight_record, original_key=spec['fc2']['name']+'.weight',
            checkpoint=str(H3_DIT_PATH), stored_scope='First 32 output rows in each cell fp64_sample')
        del weight
    teacher = cap['teacher_fc1'].cuda()
    g, u = teacher.chunk(2, -1)
    gq, uq = native_sample.chunk(2, -1)
    require(torch.equal((F.silu(g)*u).cpu(), cap['teacher_hidden_bf16']), 'Actual BF16 teacher SwiGLU differs')
    require(torch.equal((F.silu(gq)*uq).cpu(), native_hidden), 'Actual BF16 native SwiGLU differs')
    del qfull, gq_full, uq_full, hidden_full, full_input, native, payload
    gate = cap['adaln_gate'].cuda().float()
    w32 = down.weight.float()
    def project(h):
        return F.linear(h.float(), w32)*gate
    hidden, corners = four_corners(g, u, gq, uq)
    hidden16, corners16 = four_corners(g, u, gq, uq, bf16=True)
    projected = {k: project(v) for k, v in hidden.items()}
    projected16 = {k: project(v) for k, v in hidden16.items()}
    teacher_output = cap['teacher_down_bf16'].float()*cap['adaln_gate'].float()
    # Fixed eu marginal-preserving null; neither condition-preserving nor causal.
    eu = uq.float()-u.float()
    rolled = eu.roll(SHIFT, 0)
    require(torch.equal(rolled.sort(0).values, eu.sort(0).values), 'Null changed marginal multiset')
    dphi, phi = F.silu(gq.float())-F.silu(g.float()), F.silu(g.float())
    null_hidden = dict(a=hidden['a'], b=phi*rolled, c=dphi*rolled,
        total=F.silu(gq.float())*(u.float()+rolled)-phi*u.float())
    null_projected = {k: project(v) for k, v in null_hidden.items()}
    sample = dict(g=g[:2].cpu(), u=u[:2].cpu(), gq=gq[:2].cpu(), uq=uq[:2].cpu(),
        weight=down.weight[:32].cpu(), gate=cap['adaln_gate'][:2, :32])
    sg, su, sqg, squ = [sample[k].double() for k in ('g', 'u', 'gq', 'uq')]
    ph, phq = F.silu(sg), F.silu(sqg)
    exact = dict(a=(phq-ph)*su, b=ph*(squ-su), c=(phq-ph)*(squ-su), total=phq*squ-ph*su)
    numerical = dict(hidden_identity=rms_check(hidden['a']+hidden['b']+hidden['c'], hidden['total']),
        projected_identity=rms_check(projected['a']+projected['b']+projected['c'], projected['total']))
    for name in exact:
        numerical['hidden_fp64_'+name] = rms_check(hidden[name][:2].cpu(), exact[name])
        ref = F.linear(exact[name], sample['weight'].double())*sample['gate'].double()
        numerical['projected_fp64_'+name] = rms_check(projected[name][:2, :32].cpu(), ref)
    projected, projected16, null_projected = cpu(projected), cpu(projected16), cpu(null_projected)
    stats, stats16, null_stats = [energy(v, teacher_output) for v in (projected, projected16, null_projected)]
    actual_bf16_total = (native_down*cap['adaln_gate']).float()-(cap['teacher_down_bf16']*cap['adaln_gate']).float()
    # Arithmetic sensitivity: retain the same c, replace total by actual BF16 delta.
    # total-c is a numerical counterfactual, not a deployable BF16 intervention.
    arithmetic_control = dict(a=actual_bf16_total-projected['c'], b=torch.zeros_like(projected['c']),
        c=projected['c'], total=actual_bf16_total)
    arithmetic_stats = energy(arithmetic_control, teacher_output)
    for mode in stats:
        null_stats[mode]['net_over_original_total'] = null_stats[mode]['net']/stats[mode]['Etotal'] if stats[mode]['Etotal'] else None
    uncertain = any((stats[mode]['net'] > 0) != (comparison[mode]['net'] > 0) or
        stats[mode]['passes_20pct'] != comparison[mode]['passes_20pct']
        for mode in ('raw', 'proportional_residual') for comparison in (stats16, arithmetic_stats))
    checks_ok = all(v['passed'] for v in numerical.values()) and all(
        v['identity']['passed'] for group in (stats, stats16, null_stats) for v in group.values())
    artifact_payload = dict(case_id=case['id'], block=block, indices=selected['indices'], offsets=selected['offsets'],
        selected_combined_indices=selected['selected_combined_indices'], teacher_gate=g.cpu(), teacher_up=u.cpu(),
        native_gate=gq.cpu(), native_up=uq.cpu(), adaln_gate=cap['adaln_gate'],
        hidden=cpu(hidden), hidden_bf16_four_corners=cpu(corners16), projected=projected,
        projected_bf16_corners=projected16, projected_null=null_projected, teacher_output=teacher_output,
        teacher_hidden_bf16=cap['teacher_hidden_bf16'], native_hidden_bf16=native_hidden,
        teacher_down_bf16=cap['teacher_down_bf16'], native_down_bf16=native_down,
        actual_bf16_gated_difference=actual_bf16_total,
        fp64_sample=sample, fp64_hidden_reference=exact, down_weight=report['down_weights'][str(block)],
        packet=packet_log, native_weight_global=torch.load(spec['native_artifact']['file'], map_location='cpu', weights_only=True, mmap=True)['tensors']['weight_global'])
    artifact = save_tensor(artifact_payload, DATA/'cells'/f'{case["id"]}_b{block:02d}.pt', report)
    row = dict(case_id=case['id'], block=block, artifact=artifact, status='complete' if checks_ok else 'failed_numeric',
        metrics=stats, bf16_four_corner_metrics=stats16, null_metrics=null_stats, numerical_checks=numerical,
        actual_bf16_arithmetic_sensitivity=arithmetic_stats,
        arithmetic_control_scope='net_actual=||actual BF16 gated delta||^2-||actual BF16 gated delta-FP32 Dc||^2; numerical sensitivity, not deployment',
        bf16_decision_uncertain=uncertain, bf16_projected_c_difference=rms_check(projected16['c'], projected['c']),
        actual_bf16_total_difference=rms_check(artifact_payload['actual_bf16_gated_difference'], projected['total']),
        runtime_audit=audit.row(), fastpack_checks=fastpack.summary, packet_signature=signature(packet_log))
    report['cases'].append(row)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save(report, args.output)
    require(checks_ok, 'Four-corner / FP64 numerical control failed; artifact preserved')
    print(f'E062 {case["id"]} block {block}: net/total={stats["raw"]["net_over_total"]:.6g}', flush=True)
    del w32, gate, artifact_payload
    gc.collect()
    guard(args, report)


@torch.inference_mode()
def evaluate(args, report, manifest, cases):
    checked = old.load_complete(REPORTS/'check_v2.json')
    for field in ('sources', 'layers', 'references', 'legacy_manifest', 'asset_binding'):
        require(report[field] == checked[field], f'CPU binding changed: {field}')
    require(checked['cuda_initialized'] is False, 'E062 CPU check initialized CUDA')
    report['cpu_check_reference'] = file_record(REPORTS/'check_v2.json')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    old_check = old.load_complete(base.REPORTS/'E014_check_bf16.json')
    setup_path = DATA/'setup_bf16.json'
    require(not setup_path.exists(), 'Refusing existing setup')
    setup = dict(experiment='E062_inherited_E014_setup', status='running', complete_dit_calls=0, sources=old_check['sources'])
    setup_args = types.SimpleNamespace(arm='bf16', check_report=base.REPORTS/'E014_check_bf16.json',
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
    require(report['complete_dit_calls'] == report['attempted_dit_calls'] == 2 and
        report['complete_native_fc1_calls'] == report['attempted_native_fc1_calls'] == len(report['cases']) == 6, 'Final call budget mismatch')
    gates = []
    for block in BLOCKS:
        rows = [r for r in report['cases'] if r['block'] == block]
        passes = len(rows) == 2 and all(not r['bf16_decision_uncertain'] and all(
            r['metrics'][mode]['passes_20pct'] for mode in ('raw', 'proportional_residual')) for r in rows)
        gates.append(dict(block=block, passed=passes))
    report.update(status='complete', primary_gate=dict(threshold=.2, by_block=gates, passed=any(g['passed'] for g in gates)),
        active_case=None, active_local=None, elapsed_seconds=time.time()-report['start_epoch'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'evaluate'))
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or REPORTS/f'{args.phase}_v2.json'
    require(not args.output.exists(), f'Refusing existing report {args.output}')
    start = time.time()
    if args.phase == 'evaluate':
        require(args.deadline_unix is not None and 0 < args.deadline_unix-start <= 900, 'Require one absolute deadline <=900 seconds')
    report = dict(experiment='E062', phase=args.phase, status='running', start_epoch=start, deadline_unix=args.deadline_unix,
        attempted_dit_calls=0, complete_dit_calls=0, attempted_native_fc1_calls=0, complete_native_fc1_calls=0,
        tensor_bytes=0, tensor_byte_limit=LIMIT, cases=[], teacher_replays=[], down_weights={},
        scope='Two BF16 historical replays and six full-input native fc1 replays; local sampled diagnostic; no quality claim',
        null_scope='Roll eu by 256 of 512 fixed video rows; preserves marginals, breaks content conditions, noncausal',
        projection='Per-output-channel center then remove centered teacher gated-down output direction; same basis for all terms')
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E062 absolute deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, cases = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, cases)
        else:
            evaluate(args, report, manifest, cases)
        report['elapsed_seconds'] = time.time()-start
        save(report, args.output)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc(), elapsed_seconds=time.time()-start)
        save(report, args.output)
        raise
    finally:
        if args.phase == 'evaluate':
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
