#!/usr/bin/env python3
"""E044 bindings for unchanged E043 AMT/RAFT/DINO and generic MJ evaluators.

Prepare a 24-row comparison manifest, but evaluate only the 16 new videos.
No metric implementation, model download, or sample filtering is added.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import vanilla_wan_evaluation as old

ROOT = old.ROOT
RD = ROOT / 'results/research/E044'
PLAN = ROOT / 'research_state/06_experiments/E044_vanilla_wan_native_manifest.json'
ARMS = ('plain_nvfp4', 'svdquant_nvfp4')
BASE = 'bf16'
IDENTITY = old.IDENTITY
SCOPE = ('All eight E043 cases and their actual saved noise/text embeddings retained for both native arms. '
         'E043 BF16 scores reused without inference. Same recipe comparisons include each complete quantization recipe; '
         'no sample filtering, safety/bias ranking, or new composite score.')


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    old.require(value, message)


def record(path):
    return old.temporal.file_record(Path(path))


def read_plan(path):
    plan = read(path)
    reference = old.read_plan(old.PLAN)
    require(plan['experiment'] == 'E044' and len(plan['cases']) == 8, 'Fixed E044 cases')
    for case, ref in zip(plan['cases'], reference['cases'], strict=True):
        require(all(case[k] == ref[k] for k in IDENTITY), 'Keep E043 identities/order')
    for key in ('height', 'width', 'num_frames', 'fps', 'num_inference_steps', 'guidance_scale', 'flow_shift'):
        require(plan['settings'][key] == reference['settings'][key], 'Sampling differs: ' + key)
    return plan


def inherited():
    paths = dict(summary=old.RD / 'evaluation_summary.json', temporal=old.RD / 'temporal_scores.json',
                 mjvideo=old.RD / 'mjvideo_scores.json', manifest=old.EVAL_MANIFEST)
    require(read(paths['summary'])['status'] == 'complete', 'E043 reference not complete')
    return {key: record(path) for key, path in paths.items()}


def worker_rows(worker_dir):
    # Use the launcher's actual outputs when available; it also supports several
    # prompt IDs per worker. No filename assumption selects or drops a sample.
    launcher_path = worker_dir / 'launcher.json'
    if launcher_path.exists():
        launch = read(launcher_path)
        require(launch['status'] == 'complete', 'Generation launcher incomplete')
        workers = launch['workers']
        require(all(w['status'] == 'complete' and w['returncode'] == 0 for w in workers), 'Unclean worker completion')
        paths = [Path(w['result']['file']) for w in workers]
    else:
        paths = sorted(worker_dir.glob('worker_*.json'))
    require(paths, 'No completed worker reports')
    index, refs = {}, []
    for path in paths:
        report = read(path)
        require(report['experiment'] == 'E044' and report['status'] == 'complete', f'Worker incomplete: {path}')
        require(report['arm'] in ARMS, 'Unexpected native arm')
        refs.append(record(path))
        for row in report['cases']:
            key = (row['variant'], row['case_id'])
            require(row['variant'] == report['arm'] and row['status'] == 'complete' and key not in index,
                    'Duplicate/incomplete/wrong-arm case')
            index[key] = row
    return index, refs


def check(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for structure check')
    plan = read_plan(args.plan)
    prior = read(old.RD / 'evaluation_check.json')
    require(prior['status'] == 'complete' and prior['sampling'] == old.SAMPLING, 'Inherited evaluator readiness')
    for path in (old.PYTHON, old.MJ_MODEL / 'model.safetensors', old.TOKENIZER / 'tokenizer_config.json',
                 *old.temporal.WEIGHTS.values()):
        require(path.is_file(), f'Missing local asset: {path}')
    require('torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized(), 'Unexpected CUDA initialization')
    result = dict(experiment='E044', status='complete', mode='CPU_structure_check', cuda_initialized=False,
        source=record(__file__), plan=record(args.plan), inherited_evaluation_check=record(old.RD / 'evaluation_check.json'),
        reused_loop=record(old.temporal.__file__), sampling=old.SAMPLING, arms=[BASE, *ARMS],
        new_videos=16, reused_bf16_videos=8, case_ids=[r['case_id'] for r in plan['cases']],
        padding=prior['padding'], model_loads=0, scope=SCOPE)
    old.save_new(args.output or args.report_dir / 'evaluation_check.json', result)
    print(json.dumps(dict(status='complete', phase='check', cuda_initialized=False)))


def prepare(args):
    plan = read_plan(args.plan)
    native, refs = worker_rows(args.worker_dir or args.report_dir)
    ids = [r['case_id'] for r in plan['cases']]
    require(set(native) == {(arm, tid) for arm in ARMS for tid in ids}, 'Require all 16 native videos')
    baseline = {}
    for pid in (161, 192, 269, 316):
        report = read(old.RD / f'worker_{pid}.json')
        require(report['status'] == 'complete', 'Reference generation incomplete')
        baseline.update({r['case_id']: r for r in report['cases']})
    inherited_refs = inherited()
    rows = []
    for case in plan['cases']:
        tid = case['case_id']
        teacher = baseline[tid]
        for arm in (BASE, *ARMS):
            row = teacher if arm == BASE else native[(arm, tid)]
            require(all(row[k] == case[k] for k in IDENTITY), 'Generation identity')
            require(row['initial_noise'] == teacher['initial_noise'] and row['embeddings'] == teacher['embeddings'],
                    'Same actual E043 noise and embeddings required')
            actual = record(row['video']['file'])
            require(actual == row['video'], 'Media identity changed')
            if arm != BASE:
                wanted = Path(plan['data_dir']) / arm / tid / 'video.mp4'
                require(Path(actual['file']).resolve() == wanted.resolve(), 'Native media layout')
            require(row['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0), 'Actual media settings')
            rows.append(dict(**{k: case[k] for k in IDENTITY}, trajectory_id=tid, arm=arm,
                video=actual['file'], video_sha256=actual['sha256'], video_bytes=actual['bytes'],
                initial_noise=row['initial_noise'], embeddings=row['embeddings'],
                evaluation='reuse_E043' if arm == BASE else 'new_inference'))
    master = dict(experiment='E044', status='complete', phase='evaluation_binding',
        source_manifest=record(args.plan), worker_reports=refs, inherited=inherited_refs,
        sampling=plan['settings'], metric_sampling=old.SAMPLING, cases=plan['cases'], rows=rows, scope=SCOPE)
    old.save_new(args.output or args.report_dir / 'evaluation_manifest.json', master)
    for arm in ARMS:
        sub = dict(experiment='E044', status='complete', variant=arm, source_manifest=master['source_manifest'],
                   worker_reports=refs, sampling=master['sampling'], metric_sampling=old.SAMPLING,
                   cases=[r for r in rows if r['arm'] == arm], scope=SCOPE)
        old.save_new(args.report_dir / f'temporal_{arm}_manifest.json', sub)
    print(json.dumps(dict(status='complete', phase='prepare', rows=24, new_videos=16, inherited_videos=8)))


def metric_rows(path):
    manifest = read(path)
    require(manifest['status'] == 'complete' and manifest['experiment'] == 'E044'
            and manifest['variant'] in ARMS, 'Native arm manifest required')
    plan = read_plan(Path(manifest['source_manifest']['file']))
    require(record(manifest['source_manifest']['file']) == manifest['source_manifest'], 'Plan binding')
    require(len(manifest['cases']) == 8, 'Retain all eight arm cases')
    rows = []
    for case, reference in zip(manifest['cases'], plan['cases'], strict=True):
        require(all(case[k] == reference[k] for k in IDENTITY) and case['arm'] == manifest['variant'], 'Case/arm binding')
        video = record(case['video'])
        require(video['sha256'] == case['video_sha256'] and video['bytes'] == case['video_bytes'], 'Media binding')
        rows.append(dict(**{k: case[k] for k in (*IDENTITY, 'trajectory_id')}, arm=case['arm'], video=video, scores={}))
    return rows, record(path)


def run_temporal(args):
    require(args.arm in ARMS, '--arm required for temporal')
    temporal = old.temporal
    original_rows, original_save, original_argv = temporal.manifest_rows, temporal.save, sys.argv
    manifest = args.manifest or args.report_dir / f'temporal_{args.arm}_manifest.json'
    require(read(manifest)['variant'] == args.arm, 'CLI arm/manifest mismatch')

    def save(path, report):
        report.update(experiment='E044', arm=args.arm, policy=SCOPE)
        report['limits'] = [v for v in report['limits'] if 'FastWan product' not in v]
        report['adapter'] = dict(source=record(__file__), reused_metric_loop=record(temporal.__file__),
            changes='Only E044 case/media binding and report labels; inherited 81-frame metric algorithms unchanged',
            generation_report_meaning='The current eight-case arm manifest')
        original_save(path, report)

    suffix = 'cpucheck' if args.check_only else 'scores'
    output = args.output or args.report_dir / f'temporal_{args.arm}_{suffix}.json'
    command = [temporal.__file__, '--manifest', str(manifest), '--output', str(output)]
    if args.check_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for media check')
        command += ['--check-only']
    if args.deadline_unix is not None:
        command += ['--deadline-unix', str(args.deadline_unix)]
    try:
        temporal.manifest_rows, temporal.save, sys.argv = metric_rows, save, command
        temporal.main()
    finally:
        temporal.manifest_rows, temporal.save, sys.argv = original_rows, original_save, original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['check', 'prepare', 'temporal'], required=True)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--worker-dir', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    {'check': check, 'prepare': prepare, 'temporal': run_temporal}[args.phase](args)


if __name__ == '__main__':
    main()
