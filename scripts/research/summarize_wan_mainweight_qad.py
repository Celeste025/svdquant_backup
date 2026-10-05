#!/usr/bin/env python3
"""Independent CPU reduction of E020's fixed development run; no model import."""
import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = ('attn1.to_q', 'attn1.to_k', 'attn1.to_v', 'attn1.to_out.0',
            'attn2.to_q', 'attn2.to_k', 'attn2.to_v', 'attn2.to_out.0',
            'ffn.net.0.proj', 'ffn.net.2')
NAMES = tuple(f'blocks.{i}.{s}' for i in range(30) for s in SUFFIXES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def file_record(path, hash_file=True):
    p = Path(path)
    return dict(file=str(p), bytes=p.stat().st_size,
                **({'sha256': sha(p)} if hash_file else {}))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def bits(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def metrics(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype == torch.bfloat16
    x, y = x.double(), y.double()
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    e = x - y
    err2, ref2, x2 = [float(v.square().sum()) for v in (e, y, x)]
    assert ref2 > 0 and x2 > 0
    return dict(err2=err2, ref2=ref2, nmse=err2 / ref2,
                max_abs=float(e.abs().max()), cosine=float((x * y).sum()) / math.sqrt(x2 * ref2),
                elements=y.numel())


def aggregate(rows):
    return dict(aggregate_nmse=sum(r['err2'] for r in rows) / sum(r['ref2'] for r in rows),
                mean_row_nmse=sum(r['nmse'] for r in rows) / len(rows))


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-12), (a, b)


def run(report_dir, result):
    launcher_path, train_path = report_dir / 'launcher.json', report_dir / 'train_run.json'
    launcher, train = [json.loads(p.read_text()) for p in (launcher_path, train_path)]
    result['references'] = [file_record(p) for p in (launcher_path, train_path)]
    if launcher['status'] == 'failed_stop' or train['status'] == 'failed_stop':
        result.update(status='upstream_failed_stop', upstream_status={
            'launcher': launcher['status'], 'train': train['status']},
            saved_updates=len(train.get('updates', [])), error=train.get('error', launcher.get('error')))
        return
    assert launcher['status'] == train['status'] == 'complete', 'Wait for both upstream reports to complete'
    stage = next(s for s in launcher['stages'] if s['name'] == 'train')
    assert stage['status'] == 'complete' and stage['returncode'] == 0
    assert stage['report_sha256'] == sha(train_path)
    data = Path(train['settings']['data_dir'])
    inventory_path = Path(train['settings']['inventory'])
    inv = json.loads(inventory_path.read_text())
    assert sha(inventory_path) == train['sources'][str(inventory_path)]
    result['references'].append(file_record(inventory_path))
    assert train['record_counts'] == {'train': 16, 'validation': 8}
    assert train['settings']['steps'] == train['backward_steps'] == 64
    assert train['trainable_tensors'] == 300 and train['trainable_parameters'] == 1391984640
    assert set(train['first_gradient_norms']) == set(NAMES)
    assert all(math.isfinite(x) and x > 0 for x in train['first_gradient_norms'].values())
    assert len(train['updates']) == 64 and train['complete_dit_calls'] == 168
    order = []
    for epoch in range(4):
        ids = list(range(16)); random.Random(train['settings']['seed'] + epoch).shuffle(ids); order.extend(ids)
    for step, (u, idx) in enumerate(zip(train['updates'], order), 1):
        r = inv['train'][idx]
        assert (u['optimizer_step'], u['record_index'], u['prompt_id'], u['denoise_step']) == (step, idx, r['prompt_id'], r['step'])
        assert math.isfinite(u['online_nmse']) and u['online_nmse'] >= 0
        assert math.isfinite(u['gradient_norm']) and u['gradient_norm'] > 0
    targets_path = data / 'teacher_targets.pt'; target_rows = load(targets_path)
    assert len(target_rows) == 24
    targets = {}
    for row, expected in zip(target_rows, [('train', r) for r in inv['train']] + [('validation', r) for r in inv['validation']]):
        split, r = expected
        assert (row['split'], row['file'], row['prompt_id'], row['step']) == (split, r['file'], r['prompt_id'], r['step'])
        assert row['target'].shape == (1, 16, 20, 60, 104) and row['target'].dtype == torch.bfloat16
        targets[(split, r['prompt_id'], r['step'])] = row['target']
    result['references'].append(file_record(targets_path))
    evaluations, raw = [], {}
    for arm, steps in [('qdq', (0, 16, 32, 64)), ('native', (0, 64))]:
        for step in steps:
            path = data / f'{arm}_validation_step{step:04d}.pt'; outputs = load(path)
            assert len(outputs) == 8
            saved = next(e for e in train['evaluations'] if (e['label'], e['optimizer_step'], e['split']) == (arm, step, 'validation'))
            rows = []
            for output, r, old in zip(outputs, inv['validation'], saved['rows']):
                assert (old['prompt_id'], old['step']) == (r['prompt_id'], r['step'])
                row = dict(prompt_id=r['prompt_id'], step=r['step'], **metrics(output, targets[('validation', r['prompt_id'], r['step'])]))
                close(row['nmse'], old['nmse']); rows.append(row)
            totals = aggregate(rows); close(totals['aggregate_nmse'], saved['aggregate_nmse'])
            evaluations.append(dict(arm=arm, optimizer_step=step, rows=rows, **totals))
            raw[(arm, step)] = outputs; result['references'].append(file_record(path))
    result['validation'] = evaluations
    result['native_vs_qdq'] = []
    for step in (0, 64):
        rows = [dict(prompt_id=r['prompt_id'], step=r['step'], **metrics(n, q))
                for r, n, q in zip(inv['validation'], raw[('native', step)], raw[('qdq', step)])]
        result['native_vs_qdq'].append(dict(optimizer_step=step, normalization='QDQ output energy', rows=rows, **aggregate(rows)))
    result['train_nmse_reported_only'] = [e for e in train['evaluations'] if e['split'] == 'train']
    exports = sorted(train['exports'], key=lambda x: x['step']); assert [x['step'] for x in exports] == [0, 64]
    artifacts = []
    for record in exports:
        p = Path(record['file']); assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256']
        assert record['native_calls'] == 2400 and record['module_count'] == 300 and not record['online_lowrank']
        a = load(p); assert set(a) == {'format_version', 'recipe', 'target_count', 'layers', 'non_target_state'}
        assert a['format_version'] == 1 and a['target_count'] == 300 and set(a['layers']) == set(NAMES)
        assert a['recipe']['low_rank'] is False and a['recipe']['smoothing'] is False
        assert not any('lowrank' in k.lower() or 'weight_master' in k for k in a['non_target_state'])
        artifacts.append(a); result['references'].append(record)
    a0, a1 = artifacts; assert a0['recipe'] == a1['recipe']
    assert set(a0['non_target_state']) == set(a1['non_target_state'])
    assert all(exact(v, a1['non_target_state'][k]) for k, v in a0['non_target_state'].items())
    changes = []
    for name in NAMES:
        a, b = a0['layers'][name], a1['layers'][name]
        assert set(a) == set(b) == {'packed', 'scales', 'global_scale', 'bias', 'shape'}
        assert a['shape'] == b['shape']; n, k = a['shape']
        for x in (a, b):
            assert x['packed'].shape == (n, k // 2) and x['packed'].dtype == torch.uint8
            assert x['scales'].dtype == torch.float8_e4m3fn
            assert x['scales'].numel() == ((n + 127) // 128) * 128 * (((k // 16 + 3) // 4) * 4)
            assert x['global_scale'].shape == (1,) and x['global_scale'].dtype == torch.float32
            assert torch.isfinite(x['global_scale']).all() and (x['global_scale'] > 0).all()
        assert (a['bias'] is None and b['bias'] is None) or exact(a['bias'], b['bias'])
        xor = a['packed'] ^ b['packed']
        changes.append(dict(name=name, code_elements=n * k,
            changed_codes=int(((xor & 15) != 0).sum()) + int(((xor >> 4) != 0).sum()),
            scale_elements=a['scales'].numel(), changed_scale_bytes=int((bits(a['scales']) != bits(b['scales'])).sum()),
            global_changed=not exact(a['global_scale'], b['global_scale']),
            global_initial=float(a['global_scale'].item()), global_final=float(b['global_scale'].item())))
    result['packed_changes'] = dict(layers=changes, changed_code_layers=sum(r['changed_codes'] > 0 for r in changes),
        changed_scale_layers=sum(r['changed_scale_bytes'] > 0 for r in changes),
        changed_global_layers=sum(r['global_changed'] for r in changes),
        changed_codes=sum(r['changed_codes'] for r in changes), total_codes=sum(r['code_elements'] for r in changes),
        changed_scale_bytes=sum(r['changed_scale_bytes'] for r in changes), total_scale_elements=sum(r['scale_elements'] for r in changes),
        non_target_state_exact=True, target_bias_exact=True, no_online_lowrank_in_artifacts=True)
    checkpoints = []
    assert [c['step'] for c in train['checkpoints']] == [16, 32, 64]
    for record in train['checkpoints']:
        path = Path(record['file']); assert path.stat().st_size == record['bytes']
        c = load(path); assert c['step'] == record['step'] and c['inventory_sha256'] == sha(inventory_path)
        state = c['model']; expected = {name + '.weight_master' for name in NAMES}
        assert {key for key in state if key.endswith('.weight_master')} == expected
        other = set(a0['non_target_state']) | {name + '.bias' for name in NAMES if a0['layers'][name]['bias'] is not None}
        assert set(state) == expected | other
        for name in NAMES:
            assert list(state[name + '.weight_master'].shape) == a0['layers'][name]['shape']
            assert state[name + '.weight_master'].dtype == torch.float32
            bias = a0['layers'][name]['bias']
            if bias is not None: assert exact(state[name + '.bias'], bias)
        assert all(exact(state[k], v) for k, v in a0['non_target_state'].items())
        checkpoints.append(dict(**record, master_tensors=300, non_target_and_bias_exact_to_initial_export=True))
        del c, state
    optrec = train['optimizer_checkpoint']; assert Path(optrec['file']).stat().st_size == optrec['bytes']
    opt = load(optrec['file']); assert len(opt['state']) == 300 and len(opt['param_groups']) == 1
    assert opt['param_groups'][0]['params'] == list(range(300))
    assert all(int(s['step'].item()) == 64 for s in opt['state'].values())
    for i, name in enumerate(NAMES):
        for key in ('exp_avg', 'exp_avg_sq'):
            assert list(opt['state'][i][key].shape) == a0['layers'][name]['shape']
            assert opt['state'][i][key].dtype == torch.float32
    result['training_checks'] = dict(updates=64, backward_steps=64, complete_dit_calls=168,
        first_backward_positive_finite_matrix_gradients=300, optimizer_states_at_step64=300,
        every_update_finite_nonzero_global_gradient_norm=True, sampled_order_exact=True,
        checkpoints=checkpoints, optimizer_checkpoint=optrec,
        limits='Per-matrix gradient norms were saved at first update only; later updates record the combined norm. Checkpoint master values and optimizer moments are not exhaustively rescanned or rehashed.')
    result['reported_resources_not_benchmark'] = {k: train[k] for k in ('seconds_total', 'peak_allocated_gib', 'teacher_seconds')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot([r['optimizer_step'] for r in train['updates']], [r['online_nmse'] for r in train['updates']], color='#59718c', lw=1)
    axes[0].set(title='Online training loss (different cached state each update)', xlabel='Optimizer update', ylabel='Single-state NMSE')
    for arm, marker in [('qdq', 'o'), ('native', 's')]:
        es = [e for e in evaluations if e['arm'] == arm]
        axes[1].plot([e['optimizer_step'] for e in es], [e['aggregate_nmse'] for e in es], marker=marker, label=arm.upper())
    axes[1].set(title='Same 8 development states, teacher-energy pooled', xlabel='Optimizer update', ylabel='Validation NMSE')
    axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('E020 main-weight QAD — development diagnostics, not video quality')
    image = data / 'learning_curve.png'; assert not image.exists(); fig.savefig(image, dpi=170); plt.close(fig)
    result['learning_curve'] = file_record(image)
    result['status'] = 'complete'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', type=Path, default=ROOT / 'results/research/E020')
    p.add_argument('--output', type=Path)
    args = p.parse_args(); output = args.output or args.report_dir / 'summary.json'
    assert not output.exists(), f'Preserve prior result: {output}'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run with CUDA_VISIBLE_DEVICES empty'
    torch.set_num_threads(6); started = time.time()
    result = dict(experiment='E020', status='running', analyzer=file_record(Path(__file__)),
        method='Independent CPU FP64 output arithmetic; metrics recomputed from saved BF16 tensors. Eight rows are correlated development states, not independent tests or video quality.',
        limits='No model/source-tree rehash. Packed files bind to trainer SHA; raw target/output hashes are established here. Online/train-endpoint scalar logs have no saved outputs and are explicitly reported-only.')
    try:
        run(args.report_dir, result)
        assert not torch.cuda.is_initialized()
    except BaseException:
        result.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        result.update(cuda_initialized=torch.cuda.is_initialized(), cpu_seconds=time.time() - started)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(status=result['status'], file=str(output), sha256=sha(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
