#!/usr/bin/env python3
"""CPU E022 snapshots/final learning curves; independent saved-output FP64 metrics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E022'
IDENTITY = ('prompt_id', 'replica', 'seed', 'step')
EVAL_STEPS = (0, 32, 64, 128)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def record(path):
    p = Path(path); return dict(file=str(p), bytes=p.stat().st_size, sha256=sha(p))


def verify(row):
    p = Path(row['file']); assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
    return p


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def metric(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype == torch.bfloat16
    x, y = x.double(), y.double()
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    e = x - y; err2 = float(e.square().sum()); ref2 = float(y.square().sum()); assert ref2 > 0
    return dict(err2=err2, ref2=ref2, nmse=err2 / ref2, error_l2=math.sqrt(err2),
                reference_l2=math.sqrt(ref2), max_abs=float(e.abs().max()), elements=y.numel())


def pooled(rows):
    return dict(row_count=len(rows), err2=sum(r['err2'] for r in rows), ref2=sum(r['ref2'] for r in rows),
        pooled_nmse=sum(r['err2'] for r in rows) / sum(r['ref2'] for r in rows),
        mean_row_nmse=sum(r['nmse'] for r in rows) / len(rows))


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys, key), **pooled(values)) for key, values in sorted(groups.items())]


def summaries(rows):
    return dict(rows=rows, **pooled(rows), by_prompt=grouped(rows, ('prompt_id',)),
        by_replica=grouped(rows, ('replica',)), by_timestep=grouped(rows, ('step',)),
        by_trajectory=grouped(rows, ('prompt_id', 'replica', 'seed')))


def close(x, y):
    assert math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-12), (x, y)


def expected_schedule(rows):
    by_step = {s: [i for i, row in enumerate(rows) if row['step'] == s] for s in range(4)}
    assert all(len(v) == 64 for v in by_step.values())
    orders = []
    for epoch in range(2):
        pools = {}
        for step, indices in by_step.items():
            values = list(indices); random.Random(20261005 + 1009 * step + 1000003 * epoch).shuffle(values); pools[step] = values
        orders.extend([[pools[s][j] for s in range(4)] for j in range(64)])
    return orders


def analyze(train, result, final):
    for file, digest in train['sources'].items(): assert sha(file) == digest, f'Training source drift: {file}'
    result['training_sources'] = train['sources']
    for reference in ('inventory', 'collection_report', 'manifest'): verify(train['data_reference'][reference])
    inventory = json.loads(Path(train['data_reference']['inventory']['file']).read_text())
    assert inventory['status'] == 'complete' and len(inventory['train']) == 256 and len(inventory['validation']) == 32
    result['data_reference'] = train['data_reference']
    expected = expected_schedule(inventory['train']); assert train['micro_schedule'] == expected
    completed = train['updates']; assert len(completed) <= 128
    online = []
    for number, update in enumerate(completed, 1):
        assert update['optimizer_step'] == number and len(update['micro']) == 4
        assert [m['record_index'] for m in update['micro']] == expected[number - 1]
        assert [m['step'] for m in update['micro']] == [0, 1, 2, 3]
        for micro in update['micro']:
            row = inventory['train'][micro['record_index']]
            assert all(row[k] == micro[k] for k in IDENTITY) and micro['loss_multiplier'] == .25
            assert math.isfinite(micro['nmse']) and micro['nmse'] >= 0
            online.append(dict(optimizer_step=number, **micro))
        close(update['mean_micro_nmse'], sum(m['nmse'] for m in update['micro']) / 4)
    result['online_reported_only'] = dict(rows=online, completed_updates=len(completed), completed_micro=len(online),
        per_timestep=[dict(step=s, count=sum(r['step'] == s for r in online),
            mean_nmse=sum(r['nmse'] for r in online if r['step'] == s) / len(completed) if completed else None) for s in range(4)],
        note='Loss scalars from training log, not independently reconstructed predictions; different teacher states across updates.')
    result['reported_execution'] = {k: train[k] for k in ('optimizer_calls', 'backward_calls', 'micro_steps', 'dit_attempts', 'complete_dit_calls')}
    result['in_progress_update_excluded_from_curve'] = train.get('in_progress_update')
    result['upstream_error'] = train.get('error')
    targets, dev = [], inventory['validation']
    assert len({r['prompt_id'] for r in dev}) == 4 and len({(r['prompt_id'], r['replica']) for r in dev}) == 8
    assert {(r['prompt_id'], r['replica'], r['step']) for r in dev} == {(p, replica, step) for p in {r['prompt_id'] for r in dev} for replica in range(2) for step in range(4)}
    result['target_verification'] = []
    for row in dev:
        p = Path(row['file']); assert p.stat().st_size == row['bytes']
        payload = load(p)
        assert payload['filename'] == row['prompt_id'] and all(payload[k] == row[k] for k in ('seed', 'replica', 'step'))
        target = payload['outputs'][0]; assert target.shape == (1, 16, 20, 60, 104) and target.dtype == torch.bfloat16
        assert torch.isfinite(target).all() and tensor_sha(target) == row['teacher_output_sha256']
        targets.append(target)
        result['target_verification'].append(dict(**{k: row[k] for k in IDENTITY}, file=str(p), teacher_output_sha256=row['teacher_output_sha256']))
    evaluations, raw, seen = [], {}, set()
    for saved in train['evaluations']:
        arm, step = saved['label'], saved['optimizer_step']; key = (arm, step)
        assert arm in ('qdq', 'native') and step in EVAL_STEPS and saved['split'] == 'validation' and key not in seen
        seen.add(key); output_path = verify(saved['output']); outputs = load(output_path)
        assert len(outputs) == len(saved['rows']) == 32
        rows = []
        for r, out, teacher, logged in zip(dev, outputs, targets, saved['rows']):
            assert all(logged[k] == r[k] for k in IDENTITY)
            score = metric(out, teacher); close(score['nmse'], logged['nmse'])
            assert logged['validated_flags'] == (300 if arm == 'native' else 600)
            rows.append(dict(**{k: r[k] for k in IDENTITY}, **score))
        evaluation = dict(arm=arm, optimizer_step=step, output=saved['output'], **summaries(rows))
        close(evaluation['pooled_nmse'], saved['aggregate_nmse']); close(evaluation['mean_row_nmse'], saved['mean_row_nmse'])
        evaluations.append(evaluation); raw[key] = outputs
    result['validation'] = evaluations
    result['native_qdq_gap'] = []
    for step in EVAL_STEPS:
        if ('native', step) not in raw or ('qdq', step) not in raw: continue
        rows = []
        for r, n, q, teacher in zip(dev, raw[('native', step)], raw[('qdq', step)], targets):
            score = metric(n, q); teacher2 = float(teacher.double().square().sum())
            rows.append(dict(**{k: r[k] for k in IDENTITY}, **score,
                teacher_energy=teacher2, gap_over_teacher_energy=score['err2'] / teacher2))
        result['native_qdq_gap'].append(dict(optimizer_step=step, normalization='QDQ output energy; absolute vector difference, not subtraction of two NMSEs',
            gap_over_teacher_energy=sum(r['err2'] for r in rows) / sum(r['teacher_energy'] for r in rows), **summaries(rows)))
    result['completed_exports_metadata'] = train['exports']
    for export in train['exports']:
        assert Path(export['file']).stat().st_size == export['bytes']
        assert export['native_calls'] == export['actual_scaled_mm_calls'] == 9600 and export['module_count'] == 300 and not export['online_lowrank']
        computed = next(e for e in evaluations if e['arm'] == 'native' and e['optimizer_step'] == export['step'])
        close(computed['pooled_nmse'], export['native_dev_pooled_nmse'])
    # Never open packed/master/optimizer checkpoints for snapshots. Final only
    # checks completed checkpoint metadata and verifies the selected packed SHA.
    if final:
        assert train['status'] == 'complete' and len(completed) == 128 and len(online) == 512
        assert train['optimizer_calls'] == 128 and train['backward_calls'] == train['micro_steps'] == 512
        assert train['complete_dit_calls'] == train['dit_attempts'] == 768
        assert [e['step'] for e in train['exports']] == list(EVAL_STEPS)
        assert Counter(r['record_index'] for r in online) == Counter({i: 2 for i in range(256)})
        assert seen == {(arm, step) for arm in ('qdq', 'native') for step in EVAL_STEPS}
        assert len(train['first_accumulated_gradient_norms']) == 300
        assert all(math.isfinite(v) and v > 0 for v in train['first_accumulated_gradient_norms'].values())
        assert [c['step'] for c in train['checkpoints']] == [32, 64, 128]
        for c in train['checkpoints'] + [train['optimizer_checkpoint']]: assert Path(c['file']).stat().st_size == c['bytes']
        selection = train['selection']; selection_file = verify(selection['artifact'])
        saved_selection = json.loads(selection_file.read_text())
        assert saved_selection == {k: v for k, v in selection.items() if k != 'artifact'}
        native = [e for e in evaluations if e['arm'] == 'native']
        best = min(native, key=lambda e: (e['pooled_nmse'], e['optimizer_step']))
        assert selection['selected_step'] == best['optimizer_step']
        assert [c['step'] for c in selection['candidates']] == list(EVAL_STEPS)
        for candidate in selection['candidates']:
            e = next(e for e in native if e['optimizer_step'] == candidate['step'])
            close(e['pooled_nmse'], candidate['native_dev_pooled_nmse'])
            export = next(e for e in train['exports'] if e['step'] == candidate['step'])
            assert (candidate['file'], candidate['sha256']) == (export['file'], export['sha256'])
        chosen = next(e for e in train['exports'] if e['step'] == best['optimizer_step'])
        assert (selection['selected_packed_file'], selection['selected_packed_sha256']) == (chosen['file'], chosen['sha256'])
        verify(chosen)
        result['selection_verified'] = dict(step=best['optimizer_step'], native_pooled_nmse=best['pooled_nmse'],
            packed_file=chosen['file'], packed_sha256=chosen['sha256'], rule='Minimum of all four native development pooled NMSEs; exact ties use earlier step')
    result['reported_resources'] = {k: train[k] for k in ('seconds_total', 'peak_allocated_gib') if k in train}


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rows = result['online_reported_only']['rows']
    for s in range(4):
        values = [r for r in rows if r['step'] == s]
        axes[0, 0].plot([r['optimizer_step'] for r in values], [r['nmse'] for r in values], lw=.9, alpha=.8, label=f'timestep {s}')
    axes[0, 0].set(title='Online loss by timestep (reported scalars)', ylabel='Single-state NMSE')
    for arm, marker in [('qdq', 'o'), ('native', 's')]:
        es = sorted((e for e in result['validation'] if e['arm'] == arm), key=lambda e: e['optimizer_step'])
        axes[0, 1].plot([e['optimizer_step'] for e in es], [e['pooled_nmse'] for e in es], marker=marker, label=arm.upper())
    axes[0, 1].set(title='All 32 development states: pooled error', ylabel='NMSE vs BF16 teacher')
    native = sorted((e for e in result['validation'] if e['arm'] == 'native'), key=lambda e: e['optimizer_step'])
    for s in range(4):
        axes[1, 0].plot([e['optimizer_step'] for e in native],
            [next(v['pooled_nmse'] for v in e['by_timestep'] if v['step'] == s) for e in native], marker='o', label=f'timestep {s}')
    axes[1, 0].set(title='Native development error by timestep', ylabel='NMSE vs BF16 teacher')
    gaps = result['native_qdq_gap']
    axes[1, 1].plot([g['optimizer_step'] for g in gaps], [g['pooled_nmse'] for g in gaps], marker='o', label='Native − QDQ')
    axes[1, 1].set(title='Deployment/training arithmetic difference', ylabel='Difference energy / QDQ output energy')
    for ax in axes.flat: ax.set_xlabel('Optimizer update'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"E022 {result['mode']} — 4 development prompts × 2 trajectories × 4 steps; correlated states, not video quality")
    assert not path.exists(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--report-dir', type=Path, default=RD)
    p.add_argument('--output', type=Path)
    args = p.parse_args(); assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation required'
    torch.set_num_threads(6)
    if args.check_only:
        x, y = torch.tensor([1, 3], dtype=torch.bfloat16), torch.tensor([1, 2], dtype=torch.bfloat16)
        m = metric(x, y); assert (m['err2'], m['ref2'], m['nmse']) == (1, 5, .2)
        rs = [dict(prompt_id='a', replica=0, seed=1, step=0, **m), dict(prompt_id='a', replica=0, seed=1, step=1, **metric(y, y))]
        assert pooled(rs)['pooled_nmse'] == .1 and len(summaries(rs)['by_trajectory']) == 1
        fixture = [dict(step=s) for i in range(64) for s in range(4)]
        order = expected_schedule(fixture); assert Counter(i for u in order for i in u) == Counter({i: 2 for i in range(256)})
        assert not torch.cuda.is_initialized()
        output = args.output or args.report_dir / 'summary_cpucheck.json'; assert not output.exists()
        output.write_text(json.dumps(dict(status='complete', analyzer=record(Path(__file__)), cuda_initialized=False,
            tests=['FP64 hand-computed error', 'pooled vs per-row grouping', '128x4 balanced schedule; each record twice']), indent=2) + '\n')
        print(str(output)); return
    # Read one atomic report version and hash those same bytes, never reread the
    # mutable pathname to compute its snapshot hash.
    source = args.report_dir / 'train_run.json'; blob = source.read_bytes(); train = json.loads(blob)
    final = not args.snapshot
    if final:
        assert train['status'] == 'complete', 'Final summary waits for terminal complete; use --snapshot for partial evidence'
        launcher = json.loads((args.report_dir / 'train_launcher.json').read_text())
        assert launcher['status'] == 'complete' and launcher['result_sha256'] == hashlib.sha256(blob).hexdigest()
    step = len(train['updates'])
    base = 'finalsummary' if final else f'snapshot_step{step:04d}'
    output = args.output or args.report_dir / f'{base}.json'
    if not args.output and args.snapshot:
        revision = 1
        while output.exists(): revision += 1; output = args.report_dir / f'{base}_r{revision:02d}.json'
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(experiment='E022', status='running', mode='final' if final else 'snapshot',
        upstream_status=train['status'], captured_completed_update=step, analyzer=record(Path(__file__)),
        source_report=dict(file=str(source), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()),
        methods='Independent FP64 arithmetic from saved output tensors and 32 collector target tensors. Sources and output files bound by SHA; target tensors by their collector SHA. No model/full teacher-tree rehash or unfinished checkpoint reads.',
        limitations='4 development prompts, 8 trajectories, 32 correlated timestep states; no independent-sample significance or video-quality claim. Native−QDQ is a vector difference, not a difference between two error scores.')
    if args.snapshot:
        capture = output.with_suffix('.train_report.json'); assert not capture.exists(); capture.write_bytes(blob)
        result['immutable_captured_report'] = record(capture)
    started = time.time()
    try:
        analyze(train, result, final)
        path = Path(train['settings']['data_dir']).parent / 'learning_curves' / (output.stem + '.png')
        plot(result, path); result['learning_curve'] = record(path)
        assert not torch.cuda.is_initialized(); result['status'] = 'complete'
    except BaseException:
        result.update(status='failed_stop', error=traceback.format_exc()); raise
    finally:
        result.update(cuda_initialized=torch.cuda.is_initialized(), cpu_seconds=time.time() - started)
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(dict(status=result['status'], output=str(output), sha256=sha(output))))


if __name__ == '__main__': main()
