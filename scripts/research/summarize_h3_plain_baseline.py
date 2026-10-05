#!/usr/bin/env python3
"""Independent CPU audit of E014 artifacts; imports no experiment/model runner.

Numerical metrics are recomputed from raw tensors in FP64. Weight provenance is
streamed from the original safetensors; the 200 GPU quantizer/decoder equality
tests are inherited, explicitly, from the frozen exporter rather than rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('bf16', 'svd', 'plain')
MODS = ('video', 'audio')
CASES = ('e009_p001_s00', 'e010_p030_s05', 'e010_p036_s14')
SUFFIXES = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')
TARGETS = {f'blocks.{i}.{suffix}' for i in range(50) for suffix in SUFFIXES}
CHUNK = 1 << 20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    require(tensor.device.type == 'cpu', 'CPU tensors required')
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    h = hashlib.sha256()
    for start in range(0, data.numel(), 8 << 20):
        h.update(data[start:start+(8 << 20)].numpy().tobytes())
    return h.hexdigest()


def tensor_record(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=tensor_sha(tensor))


def signature(value):
    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, dict):
        return {key: signature(v) for key, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [signature(v) for v in value]
    return value


def load_tensor_file(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


class Files:
    def __init__(self):
        self.checked = {}

    def verify(self, path, digest=None, size=None):
        path = Path(path).resolve()
        stat = path.stat()
        old = self.checked.get(str(path))
        if old is None:
            old = dict(file=str(path), sha256=file_sha(path), bytes=stat.st_size,
                       mtime_ns=stat.st_mtime_ns)
            self.checked[str(path)] = old
        require(old['bytes'] == stat.st_size and old['mtime_ns'] == stat.st_mtime_ns,
                f'File mutated during independent audit: {path}')
        require(digest is None or old['sha256'] == digest, f'SHA mismatch: {path}')
        require(size is None or old['bytes'] == size, f'Size mismatch: {path}')
        return old

    def record(self, row):
        return self.verify(row['file'], row['sha256'], row.get('bytes'))

    def json(self, path, complete=True):
        record = self.verify(path)
        value = json.loads(Path(path).read_text())
        if complete:
            require(value.get('status') == 'complete', f'Incomplete report: {path}')
        return value, record

    def sources(self, rows):
        for path, record in rows.items():
            self.verify(path, record['sha256'], record.get('bytes'))


def ratio(a, b):
    return a / b if b > 0 else None


def metrics(got, ref):
    require(got.shape == ref.shape and got.dtype == ref.dtype, 'Metric shape/dtype mismatch')
    g, r = got.reshape(-1), ref.reshape(-1)
    err2 = ref2 = got2 = dot = maxabs = 0.
    changed = 0
    for start in range(0, g.numel(), CHUNK):
        gv, rv = g[start:start+CHUNK].double(), r[start:start+CHUNK].double()
        require(bool(torch.isfinite(gv).all()) and bool(torch.isfinite(rv).all()), 'Nonfinite output')
        delta = gv-rv
        err2 += float(delta.square().sum())
        ref2 += float(rv.square().sum())
        got2 += float(gv.square().sum())
        dot += float((gv*rv).sum())
        changed += int((gv != rv).sum())
        maxabs = max(maxabs, float(delta.abs().max()))
    return dict(elements=g.numel(), changed_elements=changed, exact=changed == 0,
                error_energy=err2, reference_energy=ref2, output_energy=got2,
                error_l2=math.sqrt(err2), reference_l2=math.sqrt(ref2),
                nmse=ratio(err2, ref2), cosine=ratio(dot, math.sqrt(ref2*got2)),
                output_reference_dot=dot, max_abs=maxabs,
                undefined_nmse_reason=None if ref2 > 0 else 'zero reference energy',
                accumulation_dtype='FP64 CPU')


def error_alignment(plain, svd, ref):
    dot = p2 = s2 = 0.
    p, s, r = (x.reshape(-1) for x in (plain, svd, ref))
    for start in range(0, p.numel(), CHUNK):
        ep = p[start:start+CHUNK].double()-r[start:start+CHUNK].double()
        es = s[start:start+CHUNK].double()-r[start:start+CHUNK].double()
        dot += float((ep*es).sum())
        p2 += float(ep.square().sum())
        s2 += float(es.square().sum())
    return dict(signed_dot=dot, cosine=ratio(dot, math.sqrt(p2*s2)),
                scope='Error-vector alignment only; not a causal mechanism')


def video_rows(latent):
    b, c, t, h, w = latent.shape
    require(b == 1 and h % 2 == 0 and w % 2 == 0, 'Unexpected video geometry')
    return latent.reshape(b, c, t, h//2, 2, w//2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, c*4).contiguous()


def audio_rows(latent):
    return latent.permute(0, 2, 1).reshape(-1, latent.shape[1]).contiguous()


def source_case(case, checked, files):
    files.record(case['reference'])
    historical, _ = files.json(case['reference']['file'])
    if case['kind'] == 'raw_dit':
        files.record(case['sample'])
        sample = torch.load(case['sample']['file'], map_location='cpu', weights_only=False, mmap=True)
        expected = signature(dict(args=sample['input_args'], kwargs=sample['input_kwargs']))
        require(expected == checked, 'Raw case differs from CPU check')
        require(historical['expected_endpoint_sha256'] == case['bf16_raw_output_sha256'], 'E009 reference changed')
        return dict(input_signature=expected, actual_dit_inputs=expected, state=None)
    files.record(case['prepared'])
    prepared = load_tensor_file(case['prepared']['file'])
    state = {}
    for modality in MODS:
        files.record(case['state'][modality])
        state[modality] = load_tensor_file(case['state'][modality]['file'])
    for key, value in dict(embedding=prepared['embedding'], text_token_tags=prepared['text_token_tags'], state=state).items():
        require(signature(value) == checked[key], f'CPU input proof differs from original {key}')
    trajectory = next(c for c in historical['cases'] if c['prompt_id'] == case['prompt_id'])
    historical_call = next(c for c in trajectory['dit_calls'] if c['step'] == case['step'])
    require({m: historical_call[m+'_sha256'] for m in MODS} == case['bf16_raw_output_sha256'], 'E010 reference changed')
    for modality in MODS:
        historical_step = next(c for c in trajectory['steps'] if c['step'] == case['step'] and c['modality'] == modality)
        require(all(case['state'][modality][k] == historical_step[k] for k in ('file', 'sha256')), 'Wrong E010 step')
    # Independent packing checks for these unconditional-anchor-free FL2VA cases.
    packed = checked['packed']
    text_len = prepared['embedding'].shape[0]
    v = video_rows(state['video']['latents_before'])
    a = audio_rows(state['audio']['latents_before'])
    used = text_len + len(a) + len(v)
    seq_len = ((used+63)//64)*64
    require(seq_len == packed['seq_len'], 'Unexpected padded sequence length')
    audio_pos = torch.arange(text_len, text_len+len(a))
    img_pos = torch.arange(text_len+len(a), used)
    text_pos = torch.arange(text_len)
    for name, val in dict(audio_pos=audio_pos, img_pos=img_pos, text_pos=text_pos,
                          cu_seqlens=torch.tensor([0, used, seq_len], dtype=torch.int32)).items():
        require(tensor_record(val) == packed[name], f'Packed {name} is not original layout')
    x = torch.zeros((1, seq_len, v.shape[1]), dtype=v.dtype)
    ax = torch.zeros((1, seq_len, a.shape[1]), dtype=a.dtype)
    x[0, img_pos] = v
    ax[0, audio_pos] = a
    timesteps = torch.full((seq_len,), 1.-float(state['video']['timestep'])/1000, dtype=torch.float32)
    timesteps[audio_pos] = 1.-float(state['audio']['timestep'])/1000
    unique, inverse = torch.unique(timesteps, sorted=True, return_inverse=True)
    kw = dict(x=tensor_record(x), audio_x=tensor_record(ax), img_position_ids=packed['img_position_ids'],
              unique_timesteps=tensor_record(unique), inverse_indices=tensor_record(inverse),
              token_tags=packed['token_tags'], prompt_embeds=checked['embedding'],
              img_pos_info={'position_ids': packed['img_pos']}, audio_pos_info={'position_ids': packed['audio_pos']},
              text_pos_info={'position_ids': packed['text_pos']},
              img_pos_for_infer_output_info={'position_ids': packed['img_pos']},
              packed_seq_params={'cu_seqlens_q': packed['cu_seqlens'], 'max_seqlen_q': used},
              refiner_packed_seq_params={'cu_seqlens_q': tensor_record(torch.tensor([0, text_len, text_len], dtype=torch.int32)),
                                         'max_seqlen_q': text_len},
              update_mask=None, skip_mask_out_condition=True, use_gradient_checkpointing=False,
              use_gradient_checkpointing_offload=False, control_hints=None)
    return dict(input_signature=checked, actual_dit_inputs={'args': [], 'kwargs': kw}, state=state)


def execution_contract(row, checks, arm):
    expected = dict(sdpa_calls=102, scaled_mm_calls=0 if arm == 'bf16' else 200, disk_loads=0)
    require({k: row[k] for k in expected} == expected, f'Runtime counts mismatch: {arm}')
    require(row['qkv_dtypes'] == [['torch.bfloat16']*3], 'Attention used another precision')
    check_flags(checks, arm)


def check_flags(checks, arm):
    if arm == 'bf16':
        require(checks is None, 'BF16 has unexpected quantizer checks')
    else:
        require(checks['checked_calls'] == 200 and checks['invalid_calls'] == 0, 'Invalid native contract')
        ids = checks['affected_call_indices_zero_based']
        require(len(ids) == checks['calls_with_nonzero_input_zero_sf'] and len(set(ids)) == len(ids)
                and all(0 <= i < 200 for i in ids), 'Invalid SF0 call counts')


def export_audit(manifest, files, checked):
    directory = Path(manifest['plain_export_dir'])
    plain, plain_record = files.json(directory/'manifest.json')
    files.sources(plain['sources'])
    require(plain['target_count'] == plain['exact_roundtrip_count'] == len(plain['layers']) == 200, 'Plain export incomplete')
    require({r['name'] for r in plain['layers']} == TARGETS, 'Plain target set changed')
    require(plain['recipe']['rank'] == 0 and plain['recipe']['smoothing'] is None, 'Not plain recipe')
    files.record(plain['cpu_check'])
    files.record(plain['asset_inventory_reference'])
    inventory, _ = files.json(plain['asset_inventory_reference']['file'])
    checkpoint = plain['source_checkpoint']
    require(checkpoint['file'] in inventory['assets'], 'Original checkpoint absent from asset inventory')
    require({k: v for k, v in checkpoint.items() if k != 'file'} == inventory['assets'][checkpoint['file']], 'Checkpoint provenance changed')
    stat = Path(checkpoint['file']).stat()
    require(stat.st_size == checkpoint['bytes'] and stat.st_mtime_ns == checkpoint['mtime_ns'], 'Checkpoint asset changed')
    old_record = checked['inherited_binding']['export_manifest']
    files.record(old_record)
    svd, _ = files.json(old_record['file'])
    require(svd['target_count'] == svd['exact_roundtrip_count'] == 200, 'SVD export incomplete')
    svd_rows = {r['name']: r for r in svd['layers']}
    require(set(svd_rows) == TARGETS, 'SVD target set changed')
    layers = []
    with safe_open(checkpoint['file'], framework='pt', device='cpu') as handle:
        for row in plain['layers']:
            name = row['name']
            require(row['key'] == name+'.weight' and row['dtype'] == 'BF16', 'Wrong original weight key/dtype')
            weight = handle.get_tensor(row['key'])
            require(list(weight.shape) == row['shape'] and weight.dtype == torch.bfloat16, 'Original W shape/dtype mismatch')
            source_sha = tensor_sha(weight)
            require(source_sha == row['source_weight_sha256'] == svd_rows[name]['source_weight_sha256'], 'Plain/SVD do not derive from same original W')
            del weight
            path = directory/row['file']
            require(path.resolve().parent == directory.resolve(), 'Escaped export payload')
            files.verify(path, row['file_sha256'], row['file_bytes'])
            payload = load_tensor_file(path)
            require(payload['format'] == plain['format'] and payload['recipe'] == plain['recipe']
                    and payload['name'] == name and payload['shape'] == row['shape']
                    and payload['source_weight_sha256'] == source_sha, 'Payload source/recipe mismatch')
            tensors = payload['tensors']
            require(set(tensors) == {'weight_packed', 'weight_scales_swizzled', 'weight_global', 'bias'}, 'Unexpected plain state')
            n, k = row['shape']
            require(tensors['weight_packed'].shape == (n, k//2) and tensors['weight_packed'].dtype == torch.uint8, 'Invalid code shape/dtype')
            require(tensors['weight_scales_swizzled'].dtype == torch.float8_e4m3fn
                    and tensors['weight_scales_swizzled'].numel() == ((n+127)//128)*128*((k//16+3)//4)*4, 'Invalid scale storage')
            global_scale = tensors['weight_global']
            require(global_scale.dtype == torch.float32 and global_scale.numel() == 1
                    and bool(torch.isfinite(global_scale).all()) and float(global_scale) > 0, 'Invalid tensor global')
            bias = None if row['bias_key'] is None else handle.get_tensor(row['bias_key'])
            require((bias is None) == (tensors['bias'] is None), 'Bias presence mismatch')
            if bias is not None:
                require(torch.equal(bias, tensors['bias']) and tensor_sha(bias) == row['bias_sha256'], 'Original bias changed')
            rt = row['roundtrip']
            require(rt['exact'] and rt['changed_elements'] == rt['err2'] == rt['max_abs'] == rt['nonzero_byte_differences'] == 0,
                    'Inherited original-W QDQ roundtrip failed')
            require(rt['elements'] == n*k, 'Roundtrip element count mismatch')
            if name.startswith('blocks.0.'):
                smoke = row['smoke']
                require(smoke['fast_vs_independent_e005_packet_bytes_exact'] and smoke['activation_roundtrip']['exact']
                        and smoke['samepacket_main']['nmse'] <= 1e-4, 'Representative arithmetic proof failed')
            layers.append(dict(name=name, original_weight_sha256=source_sha, payload_sha256=row['file_sha256'],
                               elements=n*k, zero_sign_differences=rt['zero_sign_differences'], roundtrip_exact=True))
            del payload, tensors, bias
            if len(layers) % 25 == 0:
                print(f'CPU source/payload audit {len(layers)}/200', flush=True)
    return dict(status='complete', manifest=plain_record, svd_manifest=old_record,
                target_count=200, original_weight_hash_count=200, inherited_exact_roundtrip_count=200,
                layers=layers, source_checkpoint=checkpoint,
                scope='Independent streaming original-W hashes, payload SHA/metadata/bias; numeric QDQ/decoder parity inherited from frozen GPU exporter, not re-quantized on CPU')


def common_report(report, files, checked, manifest_record, plan_record):
    files.sources(report['sources'])
    require(report['sources'] == checked['sources'], 'Runner sources changed between phases')
    require(report['manifest'] == manifest_record and report['plan'] == plan_record, 'Different protocol')
    files.record(report['cpu_check_reference'])
    require(report['checked_inputs'] == checked['inputs'], 'Different CPU input check')
    require(report['non_target_identity_preserved'], 'Non-target identity changed')
    expected = checked['inherited_binding']
    require(report['inherited_binding']['sdpa_enabled'] == expected['sdpa_enabled']
            and report['inherited_binding']['torch'] == expected['torch'], 'Backend differs')
    require(report['device']['visible_devices'] == '5' and report['device']['capability'] == [12, 0], 'Different GPU assignment')


def evaluate_audit(reports, manifest, checked, files):
    output = []
    for case in manifest['cases']:
        source = source_case(case, checked['inputs'][case['id']], files)
        tensors, rows = {}, {}
        for arm in ARMS:
            rows[arm] = row = next(r for r in reports[arm]['cases'] if r['id'] == case['id'])
            require(row['status'] == 'complete' and row['kind'] == case['kind'], 'Case incomplete/kind changed')
            files.record(row['artifact'])
            data = load_tensor_file(row['artifact']['file'])
            require(data['case_id'] == case['id'] and data['arm'] == arm, 'Wrong tensor artifact')
            for field in ('input_signature', 'actual_dit_inputs'):
                require(row[field] == data[field] == source[field], f'Actual/source input mismatch: {case["id"]} {arm} {field}')
            for modality in MODS:
                require(tensor_record(data['raw_outputs'][modality]) == row['raw_outputs'][modality], 'Raw output hash mismatch')
                require(bool(torch.isfinite(data['raw_outputs'][modality]).all()), 'Nonfinite raw output')
            if case['kind'] == 'model_fn':
                require(data['velocities'] is not None, 'Missing model_fn output')
                for modality, pack in [('video', video_rows), ('audio', audio_rows)]:
                    v = data['velocities'][modality]
                    require(tensor_record(v) == row['velocities'][modality], 'Velocity hash mismatch')
                    require(tensor_record(pack(-v)) == row['raw_outputs'][modality], 'Raw output / velocity sign-layout mismatch')
            else:
                require(data['velocities'] is None and row['velocities'] is None, 'Raw case unexpectedly has velocity')
            execution_contract(row['runtime_audit'], row['fastpack_checks'], arm)
            tensors[arm] = data
        bf16 = rows['bf16']
        require({m: bf16['raw_outputs'][m]['sha256'] for m in MODS} == case['bf16_raw_output_sha256'], 'Historical BF16 raw replay not exact')
        require(bf16['historical_bf16_replay']['raw_sha_exact'] is True, 'Runner raw replay flag false')
        if case['kind'] == 'model_fn':
            for modality in MODS:
                require(tensor_record(tensors['bf16']['velocities'][modality]) == tensor_record(source['state'][modality]['noise_pred']), 'Historical BF16 velocity not exact')
            require(bf16['historical_bf16_replay']['velocity_exact'] == {m: True for m in MODS}, 'Runner velocity replay flag false')
        result = dict(case_id=case['id'], split=case['split'], input_equal=True,
                      input_signature=source['input_signature'], actual_dit_inputs=source['actual_dit_inputs'],
                      historical_bf16_replay=bf16['historical_bf16_replay'], modalities={},
                      runtime={arm: {'audit': rows[arm]['runtime_audit'], 'checks': rows[arm]['fastpack_checks']} for arm in ARMS})
        for modality in MODS:
            ref = tensors['bf16']['raw_outputs'][modality]
            values = {arm: metrics(tensors[arm]['raw_outputs'][modality], ref) for arm in ('svd', 'plain')}
            values['plain_over_svd_error_energy'] = ratio(values['plain']['error_energy'], values['svd']['error_energy'])
            values['error_alignment'] = error_alignment(tensors['plain']['raw_outputs'][modality], tensors['svd']['raw_outputs'][modality], ref)
            values['raw_output_sha256'] = {arm: rows[arm]['raw_outputs'][modality]['sha256'] for arm in ARMS}
            if case['kind'] == 'model_fn':
                vm = {arm: metrics(tensors[arm]['velocities'][modality], tensors['bf16']['velocities'][modality]) for arm in ('svd', 'plain')}
                for arm in vm:
                    require(math.isclose(vm[arm]['error_energy'], values[arm]['error_energy'], rel_tol=1e-12)
                            and math.isclose(vm[arm]['reference_energy'], values[arm]['reference_energy'], rel_tol=1e-12), 'Packing altered error energy')
                values['velocity_metrics'] = vm
            result['modalities'][modality] = values
        output.append(result)
        print(f'CPU raw metrics complete: {case["id"]}', flush=True)
    return output


def benchmark_audit(benches, evaluations, files, launcher, report_paths):
    stages = launcher['stages']
    expected_names = ['export']+[f'{phase}_{arm}' for phase in ('evaluate', 'bench') for arm in ARMS]
    require([s['name'] for s in stages] == expected_names, 'Launcher order differs')
    require(launcher['planned_dit_calls'] == 24 and sum(s['planned_dit_calls'] for s in stages) == 24, 'Unexpected call allocation')
    require(launcher['seconds_total'] <= launcher['wall_budget_seconds'] == 2700, 'Wall budget exceeded')
    uuid = None
    pids, previous_end = [], None
    for stage in stages:
        require(stage['status'] == 'complete' and stage['returncode'] == 0, 'Incomplete stage')
        files.verify(stage['report'], stage['report_sha256'])
        require(stage['gpu_before']['memory_mib'] <= 64 and stage['gpu_before']['utilization_percent'] <= 1, 'GPU was not idle')
        if stage['name'] != 'export':
            device = stage['gpu_before']
            require(device['index'] == 5, 'Stage ran on a different GPU index')
            uuid = device['uuid'] if uuid is None else uuid
            require(device['uuid'] == uuid, 'Stage GPU UUID changed')
        command = stage['command']
        require(float(command[command.index('--deadline-unix')+1]) == launcher['deadline_epoch'], 'Deadline reset')
        if 'pid' in stage:
            require(stage['pid'] > 0 and stage['pid'] not in pids, 'Process PID was reused across recorded stages')
            require(stage['start_epoch'] < stage['end_epoch'] <= launcher['deadline_epoch'], 'Invalid process interval')
            require(previous_end is None or stage['start_epoch'] >= previous_end, 'Recorded processes overlapped')
            previous_end = stage['end_epoch']
            pids.append(stage['pid'])
    output = {}
    names = ('cuda_ms', 'host_forward_ms_excluding_checks', 'flag_validation_ms', 'host_ms_including_checks')
    for arm in ARMS:
        report = benches[arm]
        require(report['phase'] == 'bench' and report['arm'] == arm and report['complete_dit_calls'] == 5, 'Benchmark phase/call mismatch')
        files.record(report['evaluate_reference'])
        require(Path(report['evaluate_reference']['file']).resolve() == report_paths[('evaluate', arm)].resolve(), 'Bench binds a different evaluation')
        expected = evaluations[arm]['cases'][0]['raw_outputs']
        expected_sha = {m: expected[m]['sha256'] for m in MODS}
        require(len(report['repeats']) == 3 and [r['repeat'] for r in report['repeats']] == [0, 1, 2], 'Wrong repeats')
        for repeat in [report['warmup'], *report['repeats'], report['profile']]:
            require(repeat['raw_output_sha256'] == expected_sha, 'Bench output differs from evaluation')
            check_flags(repeat['fastpack_checks'], arm)
        for repeat in [report['warmup'], *report['repeats']]:
            require(all(math.isfinite(repeat[k]) and repeat[k] >= 0 for k in names), 'Invalid timing')
            require(math.isclose(repeat['host_forward_ms_excluding_checks']+repeat['flag_validation_ms'],
                                 repeat['host_ms_including_checks'], rel_tol=1e-10, abs_tol=1e-7), 'Timing components do not sum')
        execution_contract(report['profile']['runtime_audit'], report['profile']['fastpack_checks'], arm)
        files.record(report['profile']['trace'])
        trace = json.loads(Path(report['profile']['trace']['file']).read_text())
        kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
        native_kernels = [e for e in kernels if 'sm120' in e.get('name', '').lower() and 'e2m1' in e.get('name', '').lower()]
        require(bool(native_kernels) == (arm != 'bf16'), 'Actual native CUDA kernel evidence differs')
        latency = {}
        for key in names:
            values = [r[key] for r in report['repeats']]
            latency[key] = dict(values=values, median=statistics.median(values), min=min(values), max=max(values))
            require(latency[key] == report['latency_ms'][key], 'Latency reducer discrepancy')
        storage = report['resident_model_storage']
        require(sum(v['bytes'] for v in storage['categories'].values()) == storage['unique_cuda_storage_bytes'], 'Storage categories do not sum')
        memory = report['steady_memory']
        require(memory['peak_reserved_bytes'] >= memory['peak_allocated_bytes'] >= report['steady_before']['allocated_bytes'], 'Invalid allocator peaks')
        require(report['startup_peak_allocated_bytes'] <= 60*1024**3 and memory['peak_allocated_bytes'] <= 60*1024**3, 'Memory budget exceeded')
        output[arm] = dict(latency_ms=latency, storage=storage, startup_seconds=report['startup_seconds'],
                           startup_peak_allocated_bytes=report['startup_peak_allocated_bytes'], steady_memory=memory,
                           raw_output_sha256=expected_sha, warmup=report['warmup'],
                           profile_runtime=report['profile']['runtime_audit'], profile_checks=report['profile']['fastpack_checks'],
                           actual_native_kernel_events=len(native_kernels), actual_cuda_kernel_events=len(kernels))
    return dict(gpu_uuid=uuid, arms=output, recorded_distinct_pids=pids,
                process_evidence='Frozen launcher creates one new Popen per arm/phase and waits serially; stage commands/GPU UUID checked. Resume stages record distinct PIDs and nonoverlapping intervals; inherited stages lack PID-level evidence.',
                comparison='Only E014 matched three-arm timing; do not compare excluded-check time against E009 including-check headline.')


def run(args, result):
    files = Files()
    freeze, freeze_rec = files.json(args.report_dir/'frozen_contract.json')
    for path, digest in freeze['files'].items():
        files.verify(path, digest)
    manifest_path = ROOT/'research_state/06_experiments/E014_h3_plain_baseline_manifest.json'
    plan_path = manifest_path.with_name('E014_h3_plain_baseline_plan.md')
    manifest, _ = files.json(manifest_path, complete=False)
    require(tuple(c['id'] for c in manifest['cases']) == CASES and tuple(manifest['arms']) == ARMS, 'Fixed experiment changed')
    checked, checked_rec = files.json(args.report_dir/'E014_check_bf16.json')
    require(checked['cuda_initialized'] is False and checked['complete_dit_calls'] == 0, 'CPU check was not CPU-only')
    files.sources(checked['sources'])
    files.record(checked['manifest']); files.record(checked['plan'])
    launcher_path = args.launcher or args.report_dir/'launcher_resume1.json'
    if args.launcher is None and not launcher_path.exists():
        launcher_path = args.report_dir/'launcher.json'
    launcher, launcher_rec = files.json(launcher_path)
    require(launcher['freeze_sha256'] == freeze_rec['sha256'], 'Launcher freeze changed')
    if 'previous_launcher' in launcher:
        files.record(launcher['previous_launcher'])
        files.record(launcher['continuation_source'])
        previous, previous_rec = files.json(launcher['previous_launcher']['file'], complete=False)
        require(previous['status'] == 'failed_stop', 'Expected preserved launcher failure')
        require(launcher['deadline_epoch'] == previous['deadline_epoch']
                and launcher['start_epoch'] == previous['start_epoch']
                and launcher['freeze_sha256'] == previous['freeze_sha256'], 'Resume reset deadline or freeze')
        count = launcher['inherited_stage_count']
        require(count == len(previous['stages']) and launcher['stages'][:count] == previous['stages'], 'Resume duplicated/changed inherited stages')
        result['launcher_provenance'] = dict(previous=previous_rec, inherited_stages=count,
                                           continuation_source=launcher['continuation_source'], reason=launcher['reason'])
    result.update(frozen_contract=freeze_rec, launcher=launcher_rec, cpu_check=checked_rec,
                  plan=checked['plan'], manifest=checked['manifest'])
    evaluations, benches, paths = {}, {}, {}
    for phase, target in [('evaluate', evaluations), ('bench', benches)]:
        for arm in ARMS:
            path = args.report_dir/f'E014_{phase}_{arm}.json'
            paths[phase, arm] = path
            report, _ = files.json(path)
            common_report(report, files, checked, checked['manifest'], checked['plan'])
            require(report['phase'] == phase and report['arm'] == arm, 'Report arm/phase mismatch')
            if phase == 'evaluate':
                require(tuple(c['id'] for c in report['cases']) == CASES and report['complete_dit_calls'] == 3, 'Evaluation case/call set differs')
            target[arm] = report
    result['cases'] = evaluate_audit(evaluations, manifest, checked, files)
    result['benchmark'] = benchmark_audit(benches, evaluations, files, launcher, paths)
    result['export'] = export_audit(manifest, files, checked)
    for arm in ('svd', 'plain'):
        export_record = result['export']['svd_manifest' if arm == 'svd' else 'manifest']
        for report in (evaluations[arm], benches[arm]):
            install = report['installation']
            require(install['target_count'] == install['exact_roundtrip_count'] == 200
                    and install['manifest_sha256'] == export_record['sha256'], 'Installed different quantized export')
            require({r['name'] for r in install['layers']} == TARGETS
                    and all(r['old_weight_parameter_cleared'] for r in install['layers']), 'Incomplete replacement/old weight retained')
    dominance = all(case['modalities'][m]['plain']['error_energy'] <= case['modalities'][m]['svd']['error_energy']
                    for case in result['cases'] for m in MODS)
    cost = result['benchmark']['arms']
    faster = cost['plain']['latency_ms']['host_forward_ms_excluding_checks']['median'] < cost['svd']['latency_ms']['host_forward_ms_excluding_checks']['median']
    result['decision_inputs'] = dict(plain_no_worse_all_six_numeric_endpoints=dominance,
                                    plain_lower_median_forward_latency=faster,
                                    plan_plain_replacement_condition=dominance and faster,
                                    scope='Fixed-state numerical/performance evidence only; no quality or generalization claim')
    result['files_verified'] = files.checked
    require(not torch.cuda.is_initialized(), 'Independent summary initialized CUDA')
    result.update(status='complete', cuda_initialized=False)


def self_test():
    r = torch.tensor([1., 2., 3.], dtype=torch.float64)
    g = r + torch.tensor([1., -1., 0.], dtype=torch.float64)
    m = metrics(g, r)
    require(m['error_energy'] == 2 and m['reference_energy'] == 14 and m['nmse'] == 1/7
            and m['changed_elements'] == 2 and m['max_abs'] == 1, 'FP64 test failed')
    require(metrics(r, r)['cosine'] == 1 and metrics(r, r)['error_l2'] == 0, 'Identity metric failed')
    require(metrics(torch.ones(2), torch.zeros(2))['nmse'] is None, 'Zero denominator must be undefined')
    require(error_alignment(g, 2*r-g, r)['cosine'] == -1, 'Signed error alignment failed')
    require(tensor_record(torch.tensor(0.5))['shape'] == [], 'Scalar SHA regression')
    v = torch.arange(1*24*2*4*6).reshape(1, 24, 2, 4, 6)
    rows = video_rows(v)
    require(rows.shape == (12, 96) and rows[0, :4].tolist() == [0, 1, 6, 7], 'Video packing order')
    a = torch.arange(2*32*3).reshape(2, 32, 3)
    require(audio_rows(a)[1, :3].tolist() == [1, 4, 7], 'Audio packing order')
    require(not torch.cuda.is_initialized(), 'CPU test initialized CUDA')
    return dict(status='complete', tests=['known_fp64_energy', 'identity', 'zero_reference', 'negative_error_cosine',
                                         'scalar_hash', 'video_row_order', 'audio_row_order'], cuda_initialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/research/E014')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E014/independent_summary.json')
    parser.add_argument('--launcher', type=Path, help='Completed launcher; defaults to preserved resume1 if present')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(6)
    require(not torch.cuda.is_initialized(), 'CPU-only startup required')
    if args.self_test:
        print(json.dumps({**self_test(), 'source_sha256': file_sha(__file__)}, indent=2, allow_nan=False))
        return
    require(not args.output.exists(), f'Refusing to overwrite {args.output}')
    result = dict(experiment='E014', status='running', audit_source=dict(file=str(Path(__file__).resolve()), sha256=file_sha(__file__)),
                  numerical_source='Original raw tensor artifacts, not runner-computed metrics', torch=torch.__version__)
    start = time.monotonic()
    try:
        run(args, result)
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        result['seconds_total'] = time.monotonic()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output), seconds=result['seconds_total']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
