#!/usr/bin/env python3
"""E046 media bindings for the unchanged E043/E044 temporal and MJ algorithms."""
import argparse
import json
import os
from pathlib import Path
import sys

import vanilla_wan_evaluation as base

ROOT = base.ROOT
RD = ROOT/'results/research/E046'
PLAN = ROOT/'research_state/06_experiments/E046_wan_terminal_repair_manifest.json'
ARMS = ('native_full', 'bf16_last')
IDENTITY = base.IDENTITY
SCOPE = ('All eight same-run native_full/bf16_last pairs retained. Both variants get new evaluation; '
         'E043 BF16 and E044 SVD are historical references. Whole last-step deployed recipe replacement, '
         'not isolated FP4/LR attribution. No quality filter; MJ safety/bias raw only.')
require, record, read = base.require, base.temporal.file_record, lambda p: json.loads(Path(p).read_text())


def read_plan(path):
    plan, reference = read(path), base.read_plan(base.PLAN)
    require(plan['experiment'] == 'E046' and tuple(plan['variants']) == ARMS, 'Fixed E046 variants')
    require(plan['settings'] == reference['settings'] and len(plan['cases']) == 8, 'E043 sampling settings')
    require(all(all(a[k] == b[k] for k in IDENTITY)
                for a, b in zip(plan['cases'], reference['cases'], strict=True)), 'Original eight cases/order')
    return plan


def completed_workers(report_dir):
    path = report_dir/'launcher.json'
    launch = read(path)
    require(launch['status'] == 'complete', 'Generation must complete before evaluation')
    index, refs = {}, []
    for worker in launch['workers']:
        require(worker['status'] == 'complete' and worker['returncode'] == 0, 'Worker not cleanly complete')
        source = worker['result']
        require(record(Path(source['file'])) == source, 'Actual worker report changed')
        report = read(source['file'])
        require(report['status'] == 'complete' and report['experiment'] == 'E046', 'Generation report status')
        refs.append(source)
        for case in report['cases']:
            require(case['status'] == 'complete' and case['case_id'] not in index, 'Duplicate/incomplete case')
            index[case['case_id']] = case
    return index, refs, record(path)


def check(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for CPU structure check')
    plan = read_plan(args.plan)
    prior = read(ROOT/'results/research/E044/evaluation_check.json')
    require(prior['status'] == 'complete' and prior['sampling'] == base.SAMPLING, 'Inherited metric settings')
    for path in (base.PYTHON, base.MJ_MODEL/'model.safetensors', base.TOKENIZER/'tokenizer_config.json',
                 *base.temporal.WEIGHTS.values()):
        require(path.is_file(), f'Missing existing evaluator asset: {path}')
    require('torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized(), 'CUDA initialized')
    result = dict(experiment='E046', status='complete', phase='CPU_structure_check', cuda_initialized=False,
        source=record(Path(__file__)), plan=record(args.plan), reused_temporal=record(Path(base.temporal.__file__)),
        sampling=base.SAMPLING, variants=list(ARMS), cases=[r['case_id'] for r in plan['cases']],
        new_videos=16, reused_quality_references=['E043', 'E044'], model_loads=0, scope=SCOPE)
    base.save_new(args.output or args.report_dir/'evaluation_check.json', result)
    print(json.dumps(dict(status='complete', phase='check', cuda_initialized=False)))


def prepare(args):
    plan = read_plan(args.plan)
    generated, refs, launch = completed_workers(args.report_dir)
    require(set(generated) == {r['case_id'] for r in plan['cases']}, 'All eight paired generations required')
    rows = []
    for expected in plan['cases']:
        case = generated[expected['case_id']]
        require(all(case[k] == expected[k] for k in IDENTITY), 'Generation case identity')
        outputs = {row['variant']: row for row in case['outputs']}
        require(len(case['outputs']) == len(outputs) == 2 and set(outputs) == set(ARMS), 'Both actual branches required')
        for arm in ARMS:
            output = outputs[arm]
            require(output['status'] == 'complete', 'Incomplete branch output')
            video = record(Path(output['video']['file']))
            require(video == output['video'], 'Video changed')
            require(Path(video['file']).resolve() == (Path(plan['data_dir'])/arm/case['case_id']/'video.mp4').resolve(), 'Media layout')
            require(output['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0), '81-frame media contract')
            rows.append(dict(**{k: expected[k] for k in IDENTITY}, trajectory_id=expected['case_id'], arm=arm,
                video=video['file'], video_sha256=video['sha256'], video_bytes=video['bytes'],
                initial_noise=case['initial_noise'], embeddings=case['embeddings']))
    inherited = {name: record(ROOT/f'results/research/{experiment}/evaluation_summary.json')
                 for name, experiment in [('bf16', 'E043'), ('previous_native', 'E044')]}
    result = dict(experiment='E046', status='complete', phase='evaluation_binding', source_manifest=record(args.plan),
        generation_launcher=launch, worker_reports=refs, inherited=inherited, sampling=plan['settings'],
        metric_sampling=base.SAMPLING, cases=plan['cases'], rows=rows, scope=SCOPE)
    base.save_new(args.output or args.report_dir/'evaluation_manifest.json', result)
    for arm in ARMS:
        subset = dict(experiment='E046', status='complete', variant=arm, source_manifest=result['source_manifest'],
                      worker_reports=refs, cases=[r for r in rows if r['arm'] == arm], sampling=plan['settings'])
        base.save_new(args.report_dir/f'temporal_{arm}_manifest.json', subset)
    print(json.dumps(dict(status='complete', phase='prepare', videos=16)))


def metric_rows(path):
    manifest = read(path)
    require(manifest['status'] == 'complete' and manifest['experiment'] == 'E046'
            and manifest['variant'] in ARMS, 'Expected E046 arm manifest')
    plan = read_plan(Path(manifest['source_manifest']['file']))
    require(record(Path(manifest['source_manifest']['file'])) == manifest['source_manifest'], 'Plan binding')
    require(len(manifest['cases']) == 8, 'Eight cases retained')
    rows = []
    for case, reference in zip(manifest['cases'], plan['cases'], strict=True):
        require(all(case[k] == reference[k] for k in IDENTITY) and case['arm'] == manifest['variant'], 'Fixed case/arm')
        video = record(Path(case['video']))
        require(video['sha256'] == case['video_sha256'] and video['bytes'] == case['video_bytes'], 'Media identity')
        rows.append(dict(**{k: case[k] for k in (*IDENTITY, 'trajectory_id')}, arm=case['arm'], video=video, scores={}))
    return rows, record(path)


def run_temporal(args):
    require(args.arm in ARMS, '--arm required')
    temporal = base.temporal
    manifest = args.manifest or args.report_dir/f'temporal_{args.arm}_manifest.json'
    require(read(manifest)['variant'] == args.arm, 'CLI arm binding')
    old_rows, old_save, old_argv = temporal.manifest_rows, temporal.save, sys.argv

    def save(path, report):
        report.update(experiment='E046', arm=args.arm, policy=SCOPE)
        report['limits'] = [v for v in report['limits'] if 'FastWan product' not in v]
        report['adapter'] = dict(source=record(Path(__file__)), reused_loop=record(Path(temporal.__file__)),
            changes='Only case/media binding and report labels; inherited 81-frame metric formulas unchanged',
            generation_report_meaning='Eight-case arm evaluation manifest')
        old_save(path, report)

    suffix = 'cpucheck' if args.check_only else 'scores'
    output = args.output or args.report_dir/f'temporal_{args.arm}_{suffix}.json'
    command = [temporal.__file__, '--manifest', str(manifest), '--output', str(output)]
    if args.check_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for media check')
        command.append('--check-only')
    if args.deadline_unix is not None:
        command += ['--deadline-unix', str(args.deadline_unix)]
    try:
        temporal.manifest_rows, temporal.save, sys.argv = metric_rows, save, command
        temporal.main()
    finally:
        temporal.manifest_rows, temporal.save, sys.argv = old_rows, old_save, old_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['check', 'prepare', 'temporal'], required=True)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    {'check': check, 'prepare': prepare, 'temporal': run_temporal}[args.phase](args)


if __name__ == '__main__':
    main()
