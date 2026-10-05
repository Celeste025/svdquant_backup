#!/usr/bin/env python3
"""E064: actual BF16/plain/SVD depth paths and complete-block four corners."""
from __future__ import annotations
import argparse
import copy
import gc
from pathlib import Path
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe_h3_swiglu_interaction_v2 as inherited
from h3_native_nvfp4 import NativeH3Linear, TARGET_NAMES
from h3_plain_native_nvfp4 import PlainNativeH3Linear, install_plain_h3
import torch

base, old = inherited.base, inherited.old
save, file_record, signature, require, cpu = inherited.save, inherited.file_record, inherited.signature, inherited.require, inherited.cpu
PLAN = ROOT/'research_state/06_experiments/E064_h3_depth_four_corner_plan.md'
REPORTS = ROOT/'results/research/E064'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E064')
BLOCKS, ARMS = (0, 12, 24, 36, 49), ('bf16', 'plain', 'svd')
CASE_IDS = inherited.CASE_IDS
COMPONENTS = ('qB', 'p', 'echo', 'e_out', 'e_in', 'rB')
LIMIT, CHUNK = 32*1024**3, 256


def prerequisites(report):
    require(PLAN.is_file(), 'E064 frozen plan required')
    binding = {}
    manifest = base.prerequisites(types.SimpleNamespace(manifest=base.MANIFEST, plan=base.PLAN), binding)
    check_path = base.REPORTS/'E014_check_bf16.json'
    checked = old.load_complete(check_path)
    require(checked['sources'] == binding['sources'] and checked['cuda_initialized'] is False, 'E014 CPU binding changed')
    refs, reference_reports = {}, {}
    for arm in ARMS:
        path = base.REPORTS/f'E014_evaluate_{arm}.json'
        history = old.load_complete(path)
        require(history['sources'] == checked['sources'] and history['arm'] == arm, 'E014 historical source/arm changed')
        refs[arm] = {cid: next(r for r in history['cases'] if r['id'] == cid) for cid in CASE_IDS}
        for ref in refs[arm].values():
            base.verify_file(ref['artifact'])
        reference_reports[arm] = file_record(path)
    exports = {arm: file_record(Path(manifest[f'{arm}_export_dir'])/'manifest.json') for arm in ('plain', 'svd')}
    sources = dict(binding['sources'])
    sources.update({str(p.resolve()): file_record(p) for p in
        (Path(__file__), PLAN, Path(inherited.__file__), HERE/'h3_native_nvfp4.py', HERE/'h3_plain_native_nvfp4.py')})
    report.update(sources=sources, plan=file_record(PLAN), references=refs, reference_reports=reference_reports,
        inherited_check=file_record(check_path), export_manifests=exports, asset_binding=binding['asset_binding'])
    cases = [next(c for c in manifest['cases'] if c['id'] == cid) for cid in CASE_IDS]
    return manifest, cases


def modality_indices(actual):
    kw = actual['kwargs']
    indices = {name: kw[key]['position_ids'].reshape(-1).long().cpu() for name, key in
        (('video', 'img_pos_info'), ('audio', 'audio_pos_info'), ('text', 'text_pos_info'))}
    joined = torch.cat(list(indices.values()))
    count = kw['x'].shape[1]
    require(joined.numel() == joined.unique().numel() and int(joined.min()) >= 0 and int(joined.max()) < count,
        'Modality row indices overlap or exceed actual sequence')
    return indices, count-joined.numel()


def components(arrays, indices):
    hB, hQ, BB, QB, BQ, QQ = [arrays[k].index_select(0, indices).double() for k in ('hB', 'hQ', 'BB', 'QB', 'BQ', 'QQ')]
    qB, p = QB-BB, BQ-BB
    echo, eout, ein = (QQ-BQ)-qB, QQ-BB, hQ-hB
    return torch.stack((qB, p, echo, eout, ein, p-ein))


def statistics(arrays, indices):
    """Two CPU FP64 row passes; explicit centering avoids Gram subtraction."""
    require(all(x.device.type == 'cpu' and x.dtype == torch.bfloat16 and x.ndim == 2
        and x.shape == arrays['hB'].shape for x in arrays.values()), 'Expected matching complete CPU BF16 arrays')
    result = {}
    for modality, rows in indices.items():
        require(rows.numel() > 0, 'Empty modality domain')
        sums = torch.zeros((len(COMPONENTS), arrays['hB'].shape[1]), dtype=torch.float64)
        gram = torch.zeros((len(COMPONENTS), len(COMPONENTS)), dtype=torch.float64)
        max_identity, max_output = 0., 0.
        for start in range(0, rows.numel(), CHUNK):
            v = components(arrays, rows[start:start+CHUNK])
            require(bool(torch.isfinite(v).all()), 'Nonfinite four-corner component')
            sums += v.sum(1)
            flat = v.flatten(1)
            gram += flat@flat.t()
            max_identity = max(max_identity, float((v[3]-v[0]-v[1]-v[2]).abs().max()))
            max_output = max(max_output, float(v[3].abs().max()))
        means = sums/rows.numel()
        centered = torch.zeros_like(gram)
        centered_identity = 0.
        for start in range(0, rows.numel(), CHUNK):
            v = components(arrays, rows[start:start+CHUNK])-means[:, None, :]
            centered_identity = max(centered_identity, float((v[3]-v[0]-v[1]-v[2]).abs().max()))
            flat = v.flatten(1)
            centered += flat@flat.t()
        require(max_identity <= 1e-12*max(1., max_output) and centered_identity <= 1e-12*max(1., max_output),
            'FP64 four-corner identity failed')
        modes = {}
        for mode, matrix in (('raw', gram), ('centered', centered)):
            energy = {name: float(matrix[i, i]) for i, name in enumerate(COMPONENTS)}
            cross = {f'{a},{b}': float(2*matrix[i, j]) for i, a in enumerate(COMPONENTS)
                for j, b in enumerate(COMPONENTS) if i < j}
            without_echo = float(matrix[0, 0]+matrix[1, 1]+2*matrix[0, 1])
            net = energy['e_out']-without_echo
            formula = energy['echo']+float(2*(matrix[0, 2]+matrix[1, 2]))
            require(abs(net-formula) <= 1e-10*max(energy['e_out'], without_echo, 1.), 'FP64 net echo energy closure failed')
            modes[mode] = dict(energies=energy, gram=matrix.tolist(), twice_cross_terms=cross,
                E_without_echo=without_echo, net_echo=net, net_echo_over_e_out=net/energy['e_out'] if energy['e_out'] > 0 else None,
                signed_energy_closure_error=net-formula,
                passes_20pct=energy['e_out'] > 0 and net/energy['e_out'] >= .2)
        result[modality] = dict(rows=rows.numel(), channels=arrays['hB'].shape[1], component_order=list(COMPONENTS),
            channel_sums=sums, channel_means=means,
            mean_bias_energy={name: float(rows.numel()*means[i].square().sum()) for i, name in enumerate(COMPONENTS)},
            identity_max_abs=max_identity, centered_identity_max_abs=centered_identity, **modes)
    return result


@torch.inference_mode()
def cpu_check(report, manifest, cases):
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    torch.set_num_threads(6)
    pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['inputs'] = {}
    hidden_bytes = 0
    export = old.load_complete(base.verify_file(report['export_manifests']['svd']))
    hidden_size = next(r for r in export['layers'] if r['name'] == 'blocks.0.attn.qkv_proj')['shape'][1]
    for case in cases:
        value = base.load_case(case)
        validated = base.validate_case(case, value, pipe)
        actual = inherited.packed_actual(pipe, value)
        indices, padding = modality_indices(actual)
        sig = signature(actual)
        for arm in ARMS:
            ref = report['references'][arm][case['id']]
            payload = torch.load(base.verify_file(ref['artifact']), map_location='cpu', weights_only=True, mmap=True)
            require(validated == ref['input_signature'] == payload['input_signature'] and
                sig == ref['actual_dit_inputs'] == payload['actual_dit_inputs'], 'Historical actual input differs')
            for field in ('raw_outputs', 'velocities'):
                require(signature(payload[field]) == ref[field], 'Historical raw/velocity tensor hashes changed')
        report['inputs'][case['id']] = dict(input_signature=validated, actual_dit_input_signature=sig,
            modality_indices_signature=signature(indices), modality_counts={k: v.numel() for k, v in indices.items()}, padding_rows=padding)
        hidden_bytes += actual['kwargs']['x'].shape[1]*hidden_size*2
    # Across both cases: three paths*five blocks*two hidden arrays + two arms*five blocks*two cross outputs.
    estimate = hidden_bytes*(3*len(BLOCKS)*2+2*len(BLOCKS)*2)
    require(estimate < LIMIT, 'Required complete hidden tensors cannot fit E064 storage budget')
    t = torch.arange(64).reshape(8, 8).to(torch.bfloat16)
    fixture = dict(hB=t, hQ=t+.5, BB=t*2, QB=t*2+1, BQ=t*2+2, QQ=t*2+4)
    check = statistics(fixture, {'video': torch.arange(8)})['video']
    require(check['raw']['energies']['echo'] == 64 and check['centered']['energies']['echo'] == 0, 'CPU four-corner/centering control failed')
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False, input_count=2, hidden_size=hidden_size,
        required_hidden_tensor_bytes=estimate, storage_estimate_scope='Full path input/output plus cross outputs; metadata/storage overhead excluded',
        cpu_math_control=True, planned_counts=dict(full_dit=6, local_blocks=70, native_gemm=960, activation_packs=960, sdpa=752))


def guard(args, report):
    require(time.time() < args.deadline_unix, 'E064 absolute deadline reached')
    require(report['attempted_dit_calls'] <= 6 and report['attempted_local_calls'] <= 80, 'E064 call budget exceeded')
    totals = report['runtime_totals']
    require(totals['scaled_mm_calls'] <= 960 and totals['pack_calls'] <= 960 and totals['sdpa_calls'] <= 772,
        'E064 operator budget exceeded')
    base.inherited.memory_guard()
    require(torch.cuda.max_memory_allocated() <= 60*1024**3, 'E064 allocation peak exceeded 60 GiB')
    require(report['tensor_bytes'] <= LIMIT, 'E064 tensor files exceeded 32 GiB')


def save_tensor(value, path, report):
    require(not path.exists(), f'Refusing existing artifact {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(value, stream)
    row = file_record(path)
    report['tensor_bytes'] += row['bytes']
    require(report['tensor_bytes'] <= LIMIT, 'E064 tensor files exceeded 32 GiB')
    return row


def account(report, audit, fastpack):
    row = audit.row()
    report['active_runtime_accounted'] = True
    for key in ('scaled_mm_calls', 'sdpa_calls', 'disk_loads'):
        report['runtime_totals'][key] += row[key]
    report['runtime_totals']['pack_calls'] += fastpack.summary['checked_calls']


@torch.inference_mode()
def capture_path(pipe, case, arm, checked, args, report):
    guard(args, report)
    require(report['attempted_dit_calls'] < 6, 'Six full calls exhausted')
    value = base.load_case(case)
    cpu_pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    require(base.validate_case(case, value, cpu_pipe) == checked['inputs'][case['id']]['input_signature'], 'Source input changed')
    captures, handles, actual, raw = {}, [], [], []
    report['active_full_captures'] = captures
    expected = checked['inputs'][case['id']]
    def dit_pre(_module, pos, kw):
        actual.append(cpu(dict(args=pos, kwargs=kw)))
        require(signature(actual[0]) == expected['actual_dit_input_signature'], 'Actual full DiT input changed')
    def dit_post(_module, pos, output):
        raw.append(output)
    handles.extend((pipe.dit.register_forward_pre_hook(dit_pre, with_kwargs=True), pipe.dit.register_forward_hook(dit_post)))
    for block in BLOCKS:
        pending = {}
        def pre(_module, pos, kw, pending=pending):
            require(len(pos) == 1 and pos[0].dtype == torch.bfloat16, 'Block input contract changed')
            pending.update(input=pos[0].cpu(), kwargs=cpu(kw))
        def post(_module, pos, output, block=block, pending=pending):
            require(output.dtype == torch.bfloat16, 'Block output contract changed')
            pending['output'] = output.cpu()
            path = DATA/'captures'/arm/f'{case["id"]}_b{block:02d}.pt'
            captures[str(block)] = dict(artifact=save_tensor(pending, path, report),
                input_signature=signature(pending['input']), metadata_signature=signature(pending['kwargs']),
                output_signature=signature(pending['output']))
            pending.clear()
            save(report, args.output)
        handles.extend((pipe.dit.blocks[block].register_forward_pre_hook(pre, with_kwargs=True),
            pipe.dit.blocks[block].register_forward_hook(post)))
    audit = base.RuntimeAudit()
    audit.phase = 'resident_bf16' if arm == 'bf16' else 'native'
    report['active_runtime_audit'] = audit.row()
    report['active_runtime_accounted'] = False
    report['attempted_dit_calls'] += 1
    report['active_call'] = dict(kind='full', case_id=case['id'], arm=arm)
    save(report, args.output)
    try:
        with audit.installed(), base.collect_fastpack_checks() as fastpack:
            result = base.gpu_call(pipe, case, value)()
            torch.cuda.synchronize()
        report['complete_dit_calls'] += 1
    finally:
        for handle in handles:
            handle.remove()
    account(report, audit, fastpack)
    base.audit_contract(audit.row(), arm)
    require(fastpack.summary['checked_calls'] == (0 if arm == 'bf16' else 200), 'Full activation pack count changed')
    require(len(actual) == len(raw) == 1 and len(captures) == 5, 'Capture count mismatch')
    raw_outputs, raw_sig = base.cpu_outputs(raw[0])
    velocities, vel_sig = base.cpu_outputs(result)
    ref = report['references'][arm][case['id']]
    require(raw_sig == ref['raw_outputs'] and vel_sig == ref['velocities'], 'Historical full-path replay not byte exact')
    indices, padding = modality_indices(actual[0])
    require(signature(indices) == expected['modality_indices_signature'], 'Actual modality mapping changed')
    artifact = save_tensor(dict(case_id=case['id'], arm=arm, actual_dit_inputs=actual[0], raw_outputs=raw_outputs,
        velocities=velocities, modality_indices=indices, padding_rows=padding), DATA/'full'/f'{case["id"]}_{arm}.pt', report)
    entry = dict(case_id=case['id'], arm=arm, artifact=artifact, blocks=captures, replay_exact=True,
        historical_artifact=ref['artifact'], actual_dit_input_signature=signature(actual[0]),
        raw_outputs=raw_sig, velocities=vel_sig, runtime_audit=audit.row(), fastpack_checks=fastpack.summary)
    report['full_paths'].append(entry)
    report['active_full_captures'] = None
    save(report, args.output)
    del value, actual, raw, result
    gc.collect()
    guard(args, report)
    return entry, indices


def load_capture(row):
    value = torch.load(base.verify_file(row['artifact']), map_location='cpu', weights_only=True, mmap=True)
    require(signature(value['input']) == row['input_signature'] and signature(value['kwargs']) == row['metadata_signature'] and
        signature(value['output']) == row['output_signature'], 'Captured block tensors changed')
    return value


@torch.inference_mode()
def local_call(module, value, arm, case_id, block, corner, expected_output, args, report):
    guard(args, report)
    require(report['attempted_local_calls'] < 80, 'Local block budget exhausted')
    expected = dict(args=[signature(value['input'])], kwargs=signature(value['kwargs']))
    observed = []
    def pre(_module, pos, kw):
        observed.append(signature(cpu(dict(args=pos, kwargs=kw))))
        require(observed[0] == expected, 'Local block actual input/metadata changed')
    handle = module.register_forward_pre_hook(pre, with_kwargs=True)
    x = value['input'].cuda()
    kw = base.tree_device(value['kwargs'], 'cuda')
    audit = base.RuntimeAudit()
    audit.phase = 'resident_bf16' if arm == 'bf16' else 'native'
    report['active_runtime_audit'] = audit.row()
    report['active_runtime_accounted'] = False
    report['attempted_local_calls'] += 1
    report['active_call'] = dict(kind='local', case_id=case_id, arm=arm, block=block, corner=corner)
    save(report, args.output)
    try:
        with audit.installed(), base.collect_fastpack_checks() as fastpack:
            output = module(x, **kw)
            torch.cuda.synchronize()
        report['complete_local_calls'] += 1
    finally:
        handle.remove()
    account(report, audit, fastpack)
    row = audit.row()
    n = 0 if arm == 'bf16' else 4
    require(row['scaled_mm_calls'] == n and row['sdpa_calls'] == 2 and row['disk_loads'] == 0 and
        fastpack.summary['checked_calls'] == n and len(observed) == 1, 'Local block runtime contract changed')
    output = output.cpu()
    output_sig = signature(output)
    if expected_output is not None:
        require(output_sig == expected_output, 'Diagonal block replay not byte exact')
    record = dict(case_id=case_id, arm=arm, block=block, corner=corner, actual_input_signature=observed[0],
        output_signature=output_sig, diagonal_exact=expected_output is not None,
        runtime_audit=row, fastpack_checks=fastpack.summary)
    report['local_replays'].append(record)
    save(report, args.output)
    guard(args, report)
    return output, record


@torch.inference_mode()
def swap_plain_to_svd(pipe, manifest, args, report):
    export_dir = Path(manifest['svd_export_dir'])
    exported = old.load_complete(base.verify_file(report['export_manifests']['svd']))
    require({r['name'] for r in exported['layers']} == set(TARGET_NAMES), 'SVD target set changed')
    untouched = base.non_target_identity(pipe.dit)
    for row in exported['layers']:
        guard(args, report)
        name = row['name']
        previous = pipe.dit.get_submodule(name)
        require(type(previous) is PlainNativeH3Linear, 'Expected resident plain module before SVD swap')
        path = export_dir/row['file']
        require(file_record(path)['sha256'] == row['file_sha256'], 'SVD export hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        require(payload['name'] == name and payload['shape'] == [previous.out_features, previous.in_features], 'SVD swap shape changed')
        replacement = NativeH3Linear.from_export(payload, device='cuda', activation_packer=base.pack_activation_fast, chunk_rows=1024).eval()
        parent, attr = name.rsplit('.', 1)
        setattr(pipe.dit.get_submodule(parent), attr, replacement)
        del previous, replacement, payload
    require(base.non_target_identity(pipe.dit) == untouched, 'SVD swap changed non-target identities')
    report['svd_swap'] = dict(target_count=200, manifest=report['export_manifests']['svd'], non_target_identity_preserved=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def four_corner(pipe, backups, teacher_row, quant_row, indices, arm, case_id, block, args, report):
    trow, qrow = teacher_row['blocks'][str(block)], quant_row['blocks'][str(block)]
    teacher, quant = load_capture(trow), load_capture(qrow)
    require(trow['metadata_signature'] == qrow['metadata_signature'], 'B/Q actual block metadata differs')
    QB, _ = local_call(pipe.dit.blocks[block], teacher, arm, case_id, block, 'QB', None, args, report)
    _, qq_replay = local_call(pipe.dit.blocks[block], quant, arm, case_id, block, 'QQ', qrow['output_signature'], args, report)
    backup = backups[block]
    try:
        backup.to('cuda')
        guard(args, report)
        BQ, _ = local_call(backup, quant, 'bf16', case_id, block, f'BQ_from_{arm}', None, args, report)
    finally:
        backup.to('cpu')
    artifacts = dict(teacher_capture=trow['artifact'], quantized_capture=qrow['artifact'])
    for corner, output, value in (('QB', QB, teacher), ('BQ', BQ, quant)):
        artifacts[corner] = save_tensor(dict(output=output, input_signature=signature(value['input']),
            metadata_signature=signature(value['kwargs']), case_id=case_id, arm=arm, block=block, corner=corner),
            DATA/'cross'/arm/f'{case_id}_b{block:02d}_{corner}.pt', report)
    arrays = dict(hB=teacher['input'], hQ=quant['input'], BB=teacher['output'], QQ=quant['output'], QB=QB, BQ=BQ)
    zero = None
    if block == 0:
        zero = dict(input_exact=signature(arrays['hB']) == signature(arrays['hQ']),
            BB_BQ_exact=signature(arrays['BB']) == signature(arrays['BQ']),
            QB_QQ_exact=signature(arrays['QB']) == signature(arrays['QQ']))
        require(all(zero.values()), 'Block0 exact four-corner zero control failed')
    metrics = statistics(arrays, indices)
    if block == 0:
        require(all(metrics[m][mode]['energies'][name] == 0 for m in indices for mode in ('raw', 'centered')
            for name in ('p', 'echo', 'e_in', 'rB')), 'Block0 components must be strictly zero')
    row = dict(case_id=case_id, arm=arm, block=block, status='complete', artifacts=artifacts,
        metadata_signature=trow['metadata_signature'], input_signatures=dict(B=trow['input_signature'], Q=qrow['input_signature']),
        diagonal_sources=dict(BB='shared BF16 diagonal replay, one per case/block', QQ=qq_replay),
        block0_exact=zero, metrics=metrics)
    means = {m: {name: metrics[m].pop(name) for name in ('channel_sums', 'channel_means')} for m in indices}
    row['statistics_artifact'] = save_tensor(means, DATA/'statistics'/arm/f'{case_id}_b{block:02d}.pt', report)
    report['cases'].append(row)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save(report, args.output)
    print(f'E064 {case_id} {arm} block {block}: net={metrics["video"]["raw"]["net_echo_over_e_out"]}', flush=True)
    del teacher, quant, QB, BQ, arrays
    gc.collect()
    guard(args, report)


@torch.inference_mode()
def independent_bf16_blocks(pipe, args, report):
    backups, records = {}, {}
    for block in BLOCKS:
        guard(args, report)
        original = pipe.dit.blocks[block]
        require(not original._forward_hooks and not original._forward_pre_hooks, 'Unexpected original block hooks')
        backup = copy.deepcopy(original).cpu().eval()
        require(backup is not original and backup.attn is not original.attn and
            backup.attn.forward.__self__ is backup.attn and
            backup.attn.forward.__func__ is original.attn.forward.__func__, 'Deepcopy did not rebind Comfy attention')
        before = signature(cpu(original.state_dict()))
        require(signature(backup.state_dict()) == before, 'Independent CPU block bytes differ from BF16 original')
        original_params = dict(original.named_parameters())
        require(all(p is not original_params[name] and p.device.type == 'cpu' for name, p in backup.named_parameters()),
            'CPU block backup aliases original GPU parameters')
        backups[block] = backup
        records[str(block)] = dict(state_signature=before, independent_parameters=True, comfy_method_rebound=True)
    report['bf16_block_backups'] = records
    save(report, args.output)
    return backups


def primary_gate(cases):
    arms = {}
    for arm in ('plain', 'svd'):
        depths = []
        for block in BLOCKS[1:]:
            rows = [r for r in cases if r['arm'] == arm and r['block'] == block]
            require(len(rows) == 2, 'Expected two states at every nonzero depth')
            passed = all(r['metrics']['video'][mode]['passes_20pct'] for r in rows for mode in ('raw', 'centered'))
            depths.append(dict(block=block, passed=passed))
        arms[arm] = dict(depths=depths, passing_depths=sum(r['passed'] for r in depths),
            passed=sum(r['passed'] for r in depths) >= 2)
    return dict(arms=arms, threshold=.2, required_depths=2, passed=any(v['passed'] for v in arms.values()))


@torch.inference_mode()
def evaluate(args, report, manifest, cases):
    checked = old.load_complete(REPORTS/'check.json')
    for field in ('sources', 'references', 'reference_reports', 'export_manifests', 'asset_binding'):
        require(report[field] == checked[field], f'E064 CPU source binding changed: {field}')
    require(checked['cuda_initialized'] is False and checked['cpu_math_control'], 'E064 CPU check incomplete')
    report['cpu_check_reference'] = file_record(REPORTS/'check.json')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    setup_path = DATA/'setup_bf16.json'
    require(not setup_path.exists(), 'Refusing existing E064 model setup')
    inherited_check = base.REPORTS/'E014_check_bf16.json'
    original_check = old.load_complete(inherited_check)
    setup = dict(experiment='E064_inherited_E014_setup', status='running', sources=original_check['sources'], complete_dit_calls=0)
    setup_args = types.SimpleNamespace(arm='bf16', check_report=inherited_check, output=setup_path, deadline_unix=args.deadline_unix)
    pipe = base.setup_model(setup_args, setup, manifest)
    setup['status'] = 'complete'
    save(setup, setup_path)
    report['model_setup'] = file_record(setup_path)
    backups = independent_bf16_blocks(pipe, args, report)
    teachers, domains = {}, {}
    for case in cases:
        teacher, indices = capture_path(pipe, case, 'bf16', checked, args, report)
        teachers[case['id']], domains[case['id']] = teacher, indices
        for block in BLOCKS:
            source = teacher['blocks'][str(block)]
            value = load_capture(source)
            _, record = local_call(pipe.dit.blocks[block], value, 'bf16', case['id'], block, 'BB',
                source['output_signature'], args, report)
            record['reused_by_arms'] = ['plain', 'svd']
            del value
    guard(args, report)
    untouched = base.non_target_identity(pipe.dit)
    report['plain_installation'] = install_plain_h3(pipe.dit, manifest['plain_export_dir'],
        activation_packer=base.pack_activation_fast, chunk_rows=1024)
    require(report['plain_installation']['target_count'] == 200 and base.non_target_identity(pipe.dit) == untouched,
        'Plain installation changed target count or non-target identity')
    # The original installer clears original weight pointers. Verify CPU copies survived.
    for block, backup in backups.items():
        require(signature(backup.state_dict()) == report['bf16_block_backups'][str(block)]['state_signature'],
            'Original installer changed independent BF16 backup')
    save(report, args.output)
    for arm in ('plain', 'svd'):
        if arm == 'svd':
            swap_plain_to_svd(pipe, manifest, args, report)
        for case in cases:
            quantized, indices = capture_path(pipe, case, arm, checked, args, report)
            require(signature(indices) == signature(domains[case['id']]), 'Quantized path modality domain changed')
            for block in BLOCKS:
                four_corner(pipe, backups, teachers[case['id']], quantized, indices, arm, case['id'], block, args, report)
    require(report['complete_dit_calls'] == report['attempted_dit_calls'] == 6 and
        report['complete_local_calls'] == report['attempted_local_calls'] == 70 and len(report['cases']) == 20,
        'E064 final completed/attempted counts changed')
    require(report['runtime_totals'] == dict(scaled_mm_calls=960, pack_calls=960, sdpa_calls=752, disk_loads=0),
        'E064 final actual operator counts changed')
    report['primary_gate'] = primary_gate(report['cases'])
    report.update(status='complete', active_call=None,
        diagonal_policy='10 BF16 BB replays shared between two quantized arms; 20 quantized QQ replays; no duplicate diagonal files',
        decision='Eligible to propose a later suffix intervention; no mechanism attribution' if report['primary_gate']['passed'] else
            'Stop this repeated harmful input-dependence candidate; no broader quantization claim')
    guard(args, report)


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
        require(args.deadline_unix is not None and 0 < args.deadline_unix-started <= 1800,
            'Require one absolute deadline <=1800 seconds including setup/copies/statistics')
    report = dict(experiment='E064', phase=args.phase, status='running', start_epoch=started,
        deadline_unix=args.deadline_unix, tensor_bytes=0, tensor_byte_limit=LIMIT,
        attempted_dit_calls=0, complete_dit_calls=0, attempted_local_calls=0, complete_local_calls=0,
        runtime_totals=dict(scaled_mm_calls=0, pack_calls=0, sdpa_calls=0, disk_loads=0),
        full_paths=[], local_replays=[], cases=[],
        scope='Actual depth paths; fixed historical H3 signed-tie-lower native recipes; source diagnostic only',
        component_order=list(COMPONENTS), centering='Same full modality rows, each component mean per hidden channel',
        max_counts=dict(full_dit=6, local_blocks=80, native_gemm=960, activation_packs=960, sdpa=772))
    try:
        if args.phase == 'evaluate':
            def timeout(_signal, _frame):
                raise TimeoutError('E064 absolute deadline reached')
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix-time.time())
        manifest, cases = prerequisites(report)
        if args.phase == 'check':
            cpu_check(report, manifest, cases)
        else:
            evaluate(args, report, manifest, cases)
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
