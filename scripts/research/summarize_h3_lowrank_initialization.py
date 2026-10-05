#!/usr/bin/env python3
"""Independent CPU readout of E061 fixed-smooth initialization targets."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E061'
TEACHER = ('source_teacher', 'next_teacher')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(record):
    path = Path(record['file'])
    assert sha(path) == record['sha256'], str(path)
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def identical(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def center(x):
    return x-x.mean(tuple(range(2, x.ndim)), keepdim=True)


def main():
    out = RD/'summary.json'
    assert not out.exists()
    started = time.monotonic()
    torch.set_num_threads(4)
    ep = RD/'evaluate.json'
    evaluation = json.loads(ep.read_text())
    assert evaluation['status'] == 'complete'
    index = {(r['case_id'], r['position'], r['arm']): r for r in evaluation['cases']}
    assert len(index) == len(evaluation['cases'])
    e015_manifest = json.loads((ROOT/'research_state/06_experiments/E015_h3_crossmodal_propagation_manifest.json').read_text())
    e015_bf = json.loads((ROOT/'results/research/E015/evaluate_bf16.json').read_text())
    e015_plain = json.loads((ROOT/'results/research/E015/evaluate_plain.json').read_text())
    e057 = json.loads((ROOT/'results/research/E057/evaluate_v2.json').read_text())
    e059 = json.loads((ROOT/'results/research/E059/evaluate.json').read_text())
    zero = {(r['case_id'], r['position']): r for r in e059['cases'] if r['mode'] == 'zero'}
    assert len(zero) == 6 and all(r['historical_replay_exact'] for r in zero.values())
    replays = [r for r in evaluation['cases'] if r['arm'] == 'legacy_selected']
    assert len(replays) == 1
    replay = replays[0]
    replay_new = load(replay['artifact'])
    replay_old = load(zero[(replay['case_id'], replay['position'])]['artifact'])
    for field in ('actual_dit_inputs', 'raw_outputs', 'velocities'):
        assert identical(replay_new[field], replay_old[field]), field
    rows, sources = [], []
    for case in e015_manifest['cases']:
        cid = case['id']
        for pos in (*TEACHER, 'next_shifted'):
            if (cid, pos, 'error_init') not in index:
                assert pos == 'next_shifted' and (cid, pos, 'weight_init') not in index
                continue
            if pos == 'source_teacher':
                bf_ref = case['e014_outputs']['bf16']
                plain_ref = case['e014_outputs']['plain']
            elif pos == 'next_teacher':
                bf_ref = next(r['artifact'] for r in e015_bf['cases'] if r['id'] == cid and r['corner'] == 'BB')
                plain_ref = next(r['artifact'] for r in e015_plain['cases'] if r['id'] == cid and r['corner'] == 'BB')
            else:
                bf_ref = next(r['artifact'] for r in e057['cases'] if r['id'] == cid and r['source_arm'] == 'svd')
                plain_ref = None
            refs = {'bf16': bf_ref, 'legacy_selected': zero[(cid, pos)]['artifact']}
            if plain_ref:
                refs['plain'] = plain_ref
            refs.update({arm: index[(cid, pos, arm)]['artifact'] for arm in ('error_init', 'weight_init')})
            payloads = {arm: load(rec) for arm, rec in refs.items()}
            for arm in ('error_init', 'weight_init'):
                assert identical(payloads[arm]['actual_dit_inputs'], payloads['legacy_selected']['actual_dit_inputs'])
            stats = {}
            for modality in ('video', 'audio'):
                target = payloads['bf16']['velocities'][modality].double()
                detail = {}
                for name, transform in [('raw', lambda x: x), ('channel_centered', center)]:
                    errors = {arm: transform(payload['velocities'][modality].double()-target)
                              for arm, payload in payloads.items() if arm != 'bf16'}
                    sse = {arm: float(e.square().sum()) for arm, e in errors.items()}
                    detail[name] = dict(error_sse=sse,
                        weight_over_legacy=sse['weight_init']/sse['legacy_selected'],
                        weight_over_error_init=sse['weight_init']/sse['error_init'])
                stats[modality] = detail
            p = stats['video']['raw']
            passed = p['weight_over_legacy'] <= 0.8 and p['weight_over_error_init'] <= 0.8
            rows.append(dict(case_id=cid, position=pos, modalities=stats,
                primary_teacher_gate=passed if pos in TEACHER else None,
                plain_same_input_reference_available=plain_ref is not None))
            sources.append(dict(case_id=cid, position=pos, records=refs))
    primary = [r for r in rows if r['position'] in TEACHER]
    assert len(primary) == 4
    gate = all(r['primary_teacher_gate'] for r in primary)
    assert len(rows) == (6 if gate else 4), 'Conditional shifted-call protocol differs'
    assert not torch.cuda.is_initialized()
    result = dict(experiment='E061', status='complete', seconds=time.monotonic()-started,
        cuda_initialized=False, new_model_forwards=0, script_sha256=sha(__file__),
        evaluation_sha256=sha(ep), primary_all_four_gate=gate, cases=rows, sources=sources,
        fresh_legacy_replay_exact=True,
        scope='Fixed-smooth initialization target, not carry-Q/full official PTQ or a novel method',
        limitation='Old diagnostic inputs; no new free trajectory, video, quality, or generalization evidence')
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', gate=gate, seconds=result['seconds'],
        rows=[dict(case=r['case_id'], position=r['position'], video=r['modalities']['video']['raw']) for r in rows]), indent=2))


if __name__ == '__main__':
    main()
