#!/usr/bin/env python3
"""E022: fresh main-weight QAD, balanced four-step accumulation, saved native dev selection."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E022'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E022/train')
MODEL = Path('/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer')
PLAN = ROOT / 'research_state/06_experiments/E022_expanded_qad_plan.md'
CHECKPOINT_STEPS = (0, 32, 64, 128)
SHUFFLE_SEED = 20261005


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def file_record(path, digest=True):
    path = Path(path)
    return dict(file=str(path), bytes=path.stat().st_size, **({'sha256': sha(path)} if digest else {}))


def verify_file(row):
    path = Path(row['file'])
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], f'Artifact changed: {path}'
    return path


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def tree(value, device):
    if torch.is_tensor(value): return value.to(device)
    if isinstance(value, dict): return {k: tree(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(tree(v, device) for v in value)
    return value


def identity(row):
    return {k: row[k] for k in ('prompt_id', 'seed', 'replica', 'step')}


def build_micro_schedule(rows, seed=SHUFFLE_SEED):
    """Two independently shuffled epochs for each step; each update has t=0,1,2,3."""
    assert len(rows) == 256
    by_step = {s: [i for i, row in enumerate(rows) if row['step'] == s] for s in range(4)}
    assert all(len(indices) == 64 for indices in by_step.values())
    schedule = []
    for epoch in range(2):
        shuffled = {}
        for step, indices in by_step.items():
            indices = list(indices)
            random.Random(seed + 1009 * step + 1000003 * epoch).shuffle(indices)
            shuffled[step] = indices
        schedule.extend([[shuffled[s][position] for s in range(4)] for position in range(64)])
    assert len(schedule) == 128 and Counter(i for update in schedule for i in update) == Counter({i: 2 for i in range(256)})
    assert all([rows[i]['step'] for i in update] == [0, 1, 2, 3] for update in schedule)
    return schedule


def load_records(inventory_path):
    inventory = json.loads(inventory_path.read_text()); assert inventory['status'] == 'complete'
    collection_path = verify_file(inventory['collection_report'])
    collection = json.loads(collection_path.read_text()); assert collection['status'] == 'complete'
    manifest_path = verify_file(inventory['manifest']); manifest = json.loads(manifest_path.read_text())
    assert collection['manifest'] == inventory['manifest']
    assert collection['complete_dit_calls'] == inventory['complete_dit_calls'] == 288
    assert collection['actual_total_calls'] == inventory['actual_total_calls'] == dict(sdpa=17280, native_mm=0)
    assert inventory['model'] == manifest['model'] and Path(inventory['model']['explicit_rcm_transformer']) == MODEL
    weight_path = MODEL / 'diffusion_pytorch_model.safetensors'; weight_stat = weight_path.stat()
    assert manifest['asset_metadata'][str(weight_path)] == dict(bytes=weight_stat.st_size, mtime_ns=weight_stat.st_mtime_ns)
    manifest_cases = {r['prompt_id']: r for r in manifest['cases']}; assert len(manifest_cases) == 36
    records, checks = {}, {}
    for split, expected, prompt_count in [('train', 256, 32), ('validation', 32, 4)]:
        rows = inventory[split]; assert len(rows) == expected
        assert len({r['prompt_id'] for r in rows}) == prompt_count
        records[split] = []; seen = set()
        for row in rows:
            path = verify_file(row)
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            assert row['split'] == split and payload['filename'] == row['prompt_id']
            assert all(payload[k] == row[k] for k in ('step', 'seed', 'replica'))
            assert payload['guidance'] == 0 and isinstance(payload['prompt'], str) and payload['prompt']
            case = manifest_cases[row['prompt_id']]
            assert case['split'] == split and case['prompt'] == payload['prompt']
            assert case['seeds'][row['replica']] == row['seed']
            key = (row['prompt_id'], row['replica'], row['step']); assert key not in seen; seen.add(key)
            assert row['replica'] in (0, 1) and row['step'] in range(4)
            assert len(payload['input_args']) == len(payload['outputs']) == 1
            kwargs = payload['input_kwargs']
            assert set(kwargs) == {'timestep', 'encoder_hidden_states', 'return_dict'} and kwargs['return_dict'] is False
            tensors = [(payload['input_args'][0], (1, 16, 20, 60, 104)),
                       (payload['outputs'][0], (1, 16, 20, 60, 104)),
                       (kwargs['encoder_hidden_states'], (1, 512, 4096)), (kwargs['timestep'], (1,))]
            for value, shape in tensors:
                assert value.shape == shape and value.dtype == torch.bfloat16 and torch.isfinite(value).all()
            assert payload['outputs'][0].float().square().sum() > 0
            assert float(kwargs['timestep'].item()) == row['timestep']
            records[split].append(dict(**row, payload=payload, target=payload['outputs'][0]))
        expected_keys = {(pid, replica, step) for pid in {r['prompt_id'] for r in rows} for replica in range(2) for step in range(4)}
        assert seen == expected_keys
        checks[split] = dict(records=expected, prompts=prompt_count, replicas=2, steps=[0, 1, 2, 3],
                            all_file_hashes_valid=True, all_targets_and_inputs_finite=True)
    assert not ({r['prompt_id'] for r in records['train']} & {r['prompt_id'] for r in records['validation']})
    return records, dict(inventory=file_record(inventory_path), collection_report=file_record(collection_path),
        manifest=file_record(manifest_path), collection_status=collection['status'], payload_checks=checks,
        target_provenance='Use fresh BF16 collector outputs[0]; no repeated teacher forward in trainer')


def metric(out, target):
    x, y = out.double(), target.double()
    assert torch.isfinite(x).all()
    err2, ref2 = float((x - y).square().sum()), float(y.square().sum())
    assert ref2 > 0
    return dict(err2=err2, ref2=ref2, nmse=err2 / ref2, elements=y.numel())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, default=RD / 'data_inventory.json')
    p.add_argument('--data-dir', type=Path, default=DATA)
    p.add_argument('--output', type=Path)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    output = args.output or RD / ('trainer_check.json' if args.check_only else 'train_run.json')
    assert not output.exists(), f'Preserve existing output {output}'
    if args.check_only: assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    else: assert args.deadline_unix and args.deadline_unix > time.time(), 'Root must provide training deadline'
    started = time.time(); deadline = min(args.deadline_unix or float('inf'), started + 5400)
    report = dict(experiment='E022', status='running', settings=dict(steps=128, lr=3e-6, weight_decay=0.0,
        grad_clip=1.0, accumulation=4, shuffle_seed=SHUFFLE_SEED, initialization='fresh original BF16',
        inventory=str(args.inventory), data_dir=str(args.data_dir), deadline_unix=args.deadline_unix,
        evaluation_steps=list(CHECKPOINT_STEPS), selection='min native development pooled NMSE; exact tie -> earlier step'),
        updates=[], evaluations=[], checkpoints=[], exports=[], dit_attempts=0, complete_dit_calls=0,
        backward_calls=0, optimizer_calls=0, micro_steps=0)
    def log():
        report['seconds_total'] = time.time() - started; save_json(output, report)
    def budget():
        if time.time() > deadline: raise TimeoutError('Original E022 training deadline exceeded')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > 60 * 1024**3:
            raise MemoryError('E022 peak allocated exceeds 60GiB at phase check')
    try:
        torch.set_num_threads(6)
        report['sources'] = {str(f): sha(f) for f in [Path(__file__), PLAN, ROOT / 'scripts/research/wan_mainweight_qad.py',
            ROOT / 'scripts/research/wan_nvfp4_fastpack.py', ROOT / 'scripts/research/wan_native_nvfp4.py', MODEL / 'config.json']}
        old_path = ROOT / 'results/research/E020/train_run.json'; old = json.loads(old_path.read_text())
        assert old['status'] == 'complete'
        for name in ('wan_mainweight_qad.py', 'wan_nvfp4_fastpack.py', 'wan_native_nvfp4.py'):
            file = str(ROOT / 'scripts/research' / name)
            assert report['sources'][file] == old['sources'][file], f'Frozen QAD implementation drift: {name}'
        report['reused_numeric_contract'] = file_record(old_path)
        fixture = [dict(step=s, prompt_id=f'p{i // 2}', replica=i % 2) for i in range(64) for s in range(4)]
        schedule = build_micro_schedule(fixture)
        report['sampling_fixture'] = dict(updates=len(schedule), micro_steps=512, all_records_used_twice=True,
            every_update_steps=[0, 1, 2, 3], independent_seed_formula='20261005 + 1009*timestep + 1000003*epoch')
        records = None
        if args.inventory.exists():
            records, report['data_reference'] = load_records(args.inventory)
            schedule = build_micro_schedule(records['train'])
            report['record_counts'] = {k: len(v) for k, v in records.items()}
        report['data_ready'] = records is not None
        if args.check_only:
            assert not torch.cuda.is_initialized()
            report.update(status='complete', cuda_initialized=False)
            return
        assert records is not None, 'Collect and finalize the fresh BF16 data inventory first'
        budget(); assert os.environ.get('CUDA_VISIBLE_DEVICES') == '5', 'Root schedules physical GPU5'
        assert torch.cuda.get_device_capability() == (12, 0)
        import torch.nn.functional as F
        from diffusers import WanTransformer3DModel
        import wan_mainweight_qad as qad
        from wan_nvfp4_fastpack import collect_fastpack_checks
        args.data_dir.mkdir(parents=True, exist_ok=False)
        report['micro_schedule'] = schedule
        torch.manual_seed(SHUFFLE_SEED); torch.backends.cuda.matmul.allow_tf32 = False
        report['environment'] = dict(python=os.sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            attention='BF16 torch FLASH_ATTENTION, context includes all training backward calls')
        wpath = MODEL / 'diffusion_pytorch_model.safetensors'; st = wpath.stat()
        report['original_model'] = dict(file=str(wpath), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        model = WanTransformer3DModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
        modules = qad.install_qad(model); params = [m.weight_master for m in modules.values()]
        assert len(params) == 300 and sum(p.numel() for p in params) == 1391984640
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in params}
        report.update(trainable_tensors=300, trainable_parameters=sum(p.numel() for p in params))
        model.enable_gradient_checkpointing(); log()

        def call(current, record):
            budget(); report['dit_attempts'] += 1
            payload = record['payload']
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = current(*tree(payload['input_args'], 'cuda'), **tree(payload['input_kwargs'], 'cuda'))
            out = output[0] if isinstance(output, (tuple, list)) else output.sample
            assert out.shape == (1, 16, 20, 60, 104) and out.dtype == torch.bfloat16
            report['complete_dit_calls'] += 1
            return out

        def evaluate(current, label, step):
            current.eval(); rows, outputs = [], []; tick = time.time()
            for record in records['validation']:
                budget()
                with torch.no_grad(), collect_fastpack_checks() as flags:
                    out = call(current, record).detach().cpu()
                expected_flags = 300 if label == 'native' else 600
                assert len(flags) == expected_flags
                rows.append(dict(**identity(record), **metric(out, record['target']), validated_flags=len(flags)))
                outputs.append(out)
            path = args.data_dir / f'{label}_validation_step{step:04d}.pt'; torch.save(outputs, path)
            item = dict(label=label, optimizer_step=step, split='validation', rows=rows,
                aggregate_nmse=sum(r['err2'] for r in rows) / sum(r['ref2'] for r in rows),
                mean_row_nmse=sum(r['nmse'] for r in rows) / len(rows), output=file_record(path), seconds=time.time() - tick)
            report['evaluations'].append(item); log()
            print('eval', label, step, item['aggregate_nmse'], flush=True)
            return item

        def export_and_evaluate(step):
            budget(); tick = time.time()
            artifact = qad.export_packed(model)
            path = args.data_dir / f'packed_step{step:04d}.pt'; torch.save(artifact, path)
            native = WanTransformer3DModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
            installed = qad.install_packed(native, artifact); del artifact
            assert len(installed) == 300 and not any('weight_master' in n for n, _ in native.named_parameters())
            assert not any(m._forward_hooks or m._forward_pre_hooks for m in native.modules())
            actual_mm = []; original_mm = F.scaled_mm
            def audited_mm(a, b, *pos, **kw):
                assert a.dtype == b.dtype == torch.float4_e2m1fn_x2
                value = original_mm(a, b, *pos, **kw); actual_mm.append(1)
                return value
            F.scaled_mm = audited_mm
            try: evaluation = evaluate(native, 'native', step)
            finally: F.scaled_mm = original_mm
            calls = sum(m.native_calls for m in installed.values())
            assert calls == len(actual_mm) == 300 * 32
            report['exports'].append(dict(step=step, **file_record(path), native_calls=calls,
                actual_scaled_mm_calls=len(actual_mm), module_count=300, online_lowrank=False,
                native_dev_pooled_nmse=evaluation['aggregate_nmse'], seconds=time.time() - tick))
            del installed, native; gc.collect(); torch.cuda.empty_cache(); log(); budget()

        evaluate(model, 'qdq', 0); export_and_evaluate(0)
        optimizer = torch.optim.AdamW(params, lr=3e-6, weight_decay=0.0, foreach=False)
        sampled_before = [p.detach().flatten()[::max(1, p.numel() // 256)][:256].clone() for p in params]
        for step, indices in enumerate(schedule, 1):
            budget(); model.train(); optimizer.zero_grad(set_to_none=True); tick = time.time(); micro = []
            report['in_progress_update'] = dict(optimizer_step=step, planned_record_indices=indices, completed_micro=micro)
            for index in indices:
                record = records['train'][index]
                with collect_fastpack_checks() as flags, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = call(model, record).float(); target = record['target'].to('cuda').float()
                    nmse = (out - target).square().sum() / target.square().sum().clamp_min(1e-20)
                    if not torch.isfinite(nmse): raise FloatingPointError('Nonfinite micro loss')
                    (nmse / 4).backward(); report['backward_calls'] += 1
                report['micro_steps'] += 1
                micro.append(dict(record_index=index, **identity(record), nmse=float(nmse.detach()),
                                  loss_multiplier=0.25, validated_pack_flags=len(flags)))
                del out, target, nmse; log(); budget()
            if step == 1:
                assert all(p.grad is not None for p in params)
                norms = torch.stack([p.grad.detach().float().norm() for p in params])
                assert torch.isfinite(norms).all() and (norms > 0).all()
                report['first_accumulated_gradient_norms'] = {name: float(v) for name, v in zip(modules, norms.cpu())}
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step(); report['optimizer_calls'] += 1
            torch.cuda.synchronize()
            if step == 1:
                changed = [int((p.detach().flatten()[::max(1, p.numel() // 256)][:256] != b).sum()) for p, b in zip(params, sampled_before)]
                assert sum(changed) > 0
                report['first_update_sampled_changed_counts'] = dict(zip(modules, changed)); del sampled_before
            report['updates'].append(dict(optimizer_step=step, micro=micro,
                mean_micro_nmse=sum(m['nmse'] for m in micro) / 4, gradient_norm=float(norm),
                seconds=time.time() - tick, allocated_gib=torch.cuda.memory_allocated() / 1024**3,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3))
            report.pop('in_progress_update')
            log(); print('update', step, report['updates'][-1]['mean_micro_nmse'], flush=True)
            if step in CHECKPOINT_STEPS:
                optimizer.zero_grad(set_to_none=True)
                evaluate(model, 'qdq', step)
                checkpoint = dict(step=step, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    settings=report['settings'], inventory_sha256=report['data_reference']['inventory']['sha256'], sources=report['sources'])
                path = args.data_dir / f'master_step{step:04d}.pt'; torch.save(checkpoint, path); del checkpoint
                report['checkpoints'].append(dict(step=step, **file_record(path, digest=False))); log()
                export_and_evaluate(step)
        optimizer.zero_grad(set_to_none=True)
        path = args.data_dir / 'optimizer_step0128.pt'; torch.save(optimizer.state_dict(), path)
        report['optimizer_checkpoint'] = file_record(path, digest=False)
        assert report['optimizer_calls'] == 128 and report['micro_steps'] == report['backward_calls'] == 512
        assert report['dit_attempts'] == report['complete_dit_calls'] == 512 + 8 * 32
        assert [r['step'] for r in report['exports']] == list(CHECKPOINT_STEPS)
        candidate = min(report['exports'], key=lambda r: (r['native_dev_pooled_nmse'], r['step']))
        selection = dict(experiment='E022', status='complete', rule=report['settings']['selection'],
            candidates=[{k: r[k] for k in ('step', 'file', 'sha256', 'native_dev_pooled_nmse')} for r in report['exports']],
            selected_step=candidate['step'], selected_packed_file=candidate['file'], selected_packed_sha256=candidate['sha256'],
            limitation='Development tensor NMSE selects a deployment candidate only; no independent video quality conclusion.',
            inventory_sha256=report['data_reference']['inventory']['sha256'], trainer_sha256=sha(Path(__file__)))
        path = args.data_dir / 'candidate_selection.json'; assert not path.exists(); save_json(path, selection)
        report['selection'] = dict(**selection, artifact=file_record(path))
        report['loss_by_timestep'] = {str(s): dict(count=128, mean_nmse=sum(m['nmse'] for u in report['updates'] for m in u['micro'] if m['step'] == s) / 128) for s in range(4)}
        budget(); report['status'] = 'complete'
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc()); raise
    finally:
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if torch.cuda.is_initialized(): report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        log(); print('E022', report['status'], report['seconds_total'], flush=True)


if __name__ == '__main__': main()
