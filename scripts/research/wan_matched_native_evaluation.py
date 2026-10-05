#!/usr/bin/env python3
"""E048 bindings: score eight new matched-SVD videos with unchanged existing metrics."""
import argparse
import json
import os
from pathlib import Path
import sys

import vanilla_wan_evaluation as base

ROOT = base.ROOT
RD = ROOT/'results/research/E048'
PLAN = ROOT/'research_state/06_experiments/E048_matched_wan_native_manifest.json'
ARM = 'svdquant_nvfp4'
IDENTITY = base.IDENTITY
SCOPE = ('All eight new matched-calibration SVDQuant videos retained. E043 BF16, E044 old SVD, '
         'and E046 bf16_last media/scores are inherited without repeated inference. This is baseline '
         'completion, not a new method; these are previously inspected diagnostic cases. '
         'No quality filtering; MJ safety/bias criteria remain raw only.')
require, record = base.require, base.temporal.file_record


def read(path):
    return json.loads(Path(path).read_text())


def read_plan(path):
    plan, reference = read(path), base.read_plan(base.PLAN)
    require(plan['experiment'] == 'E048' and plan['arms'] == [ARM], 'Fixed E048 single arm')
    require(plan['settings'] == reference['settings'] and plan['cases'] == reference['cases'],
            'Keep original eight cases/order and sampling settings')
    return plan


def completed_workers(report_dir):
    path = report_dir/'launcher.json'; launch = read(path)
    require(launch['status'] == 'complete' and launch['experiment'] == 'E048', 'Generation must complete first')
    index, refs = {}, []
    for worker in launch['workers']:
        require(worker['status'] == 'complete' and worker['returncode'] == 0, 'Generation worker did not exit cleanly')
        source = worker['result']
        require(record(Path(source['file'])) == source, 'Generation worker report changed')
        value = read(source['file'])
        require(value['status'] == 'complete' and value['experiment'] == 'E048' and value['arm'] == ARM,
                'Expected completed E048 worker')
        refs.append(source)
        for case in value['cases']:
            require(case['status'] == 'complete' and case['variant'] == ARM and case['case_id'] not in index,
                    'Duplicate, incomplete, or wrong-arm video')
            index[case['case_id']] = case
    return index, refs, record(path)


def check(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for CPU structure check')
    plan = read_plan(args.plan)
    prior_path = ROOT/'results/research/E044/evaluation_check.json'; prior = read(prior_path)
    require(prior['status'] == 'complete' and prior['sampling'] == base.SAMPLING, 'Inherited sampling differs')
    for path in (base.PYTHON, base.MJ_MODEL/'model.safetensors', base.TOKENIZER/'tokenizer_config.json',
                 *base.temporal.WEIGHTS.values()):
        require(path.is_file(), f'Missing existing evaluation asset: {path}')
    require('torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized(), 'CUDA initialized')
    result = dict(experiment='E048', status='complete', phase='CPU_structure_check', cuda_initialized=False,
        source=record(Path(__file__)), plan=record(args.plan), inherited_check=record(prior_path),
        reused_temporal=record(Path(base.temporal.__file__)), sampling=base.SAMPLING, padding=prior['padding'],
        variants=[ARM], cases=[r['case_id'] for r in plan['cases']], new_videos=8,
        reused_quality_references=['E043', 'E044', 'E046'], model_loads=0,
        generation_files_required=False, scope=SCOPE)
    base.save_new(args.output or args.report_dir/'evaluation_check.json', result)
    print(json.dumps(dict(status='complete', phase='check', cuda_initialized=False, model_loads=0)))


def prepare(args):
    plan = read_plan(args.plan)
    generated, refs, launch = completed_workers(args.report_dir)
    require(set(generated) == {r['case_id'] for r in plan['cases']}, 'Require all eight completed new videos')
    teacher = {}
    for pid in sorted({r['prompt_id'] for r in plan['cases']}):
        value = read(Path(plan['reference_dir'])/f'worker_{pid}.json')
        require(value['status'] == 'complete', 'Original E043 reference incomplete')
        teacher.update({r['case_id']: r for r in value['cases']})
    rows = []
    for expected in plan['cases']:
        case = generated[expected['case_id']]; old = teacher[expected['case_id']]
        require(all(case[k] == expected[k] for k in IDENTITY), 'Generation case identity differs')
        require(case['initial_noise'] == old['initial_noise'] and case['embeddings'] == old['embeddings'],
                'Actual E043 noise and embeddings must be shared')
        video = record(Path(case['video']['file']))
        require(video == case['video'], 'Video changed')
        require(Path(video['file']).resolve() == (Path(plan['data_dir'])/ARM/case['case_id']/'video.mp4').resolve(),
                'Unexpected media layout')
        require(case['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0),
                '81-frame media contract differs')
        rows.append(dict(**{k: expected[k] for k in IDENTITY}, trajectory_id=expected['case_id'], arm=ARM,
                         video=video['file'], video_sha256=video['sha256'], video_bytes=video['bytes'],
                         initial_noise=case['initial_noise'], embeddings=case['embeddings']))
    inherited = {}
    for item in plan['inherited_comparisons']:
        path = Path(item['summary']); require(read(path)['status'] == 'complete', 'Historical summary incomplete')
        inherited[item['experiment']] = dict(variant=item['variant'], summary=record(path),
                                             data_dir=item['data_dir'], evaluation='reuse_existing')
    result = dict(experiment='E048', status='complete', phase='evaluation_binding', source_manifest=record(args.plan),
        generation_launcher=launch, worker_reports=refs, inherited=inherited, sampling=plan['settings'],
        metric_sampling=base.SAMPLING, cases=plan['cases'], rows=rows, scope=SCOPE)
    base.save_new(args.output or args.report_dir/'evaluation_manifest.json', result)
    subset = dict(experiment='E048', status='complete', variant=ARM, source_manifest=result['source_manifest'],
                  generation_launcher=launch, worker_reports=refs, cases=rows, sampling=plan['settings'])
    base.save_new(args.report_dir/f'temporal_{ARM}_manifest.json', subset)
    print(json.dumps(dict(status='complete', phase='prepare', new_videos=8, inherited_scoring_calls=0)))


def metric_rows(path):
    manifest = read(path)
    require(manifest['status'] == 'complete' and manifest['experiment'] == 'E048' and manifest['variant'] == ARM,
            'Expected E048 evaluation manifest')
    launch = manifest['generation_launcher']
    require(record(Path(launch['file'])) == launch and read(launch['file'])['status'] == 'complete',
            'Actual generation must remain complete')
    plan = read_plan(Path(manifest['source_manifest']['file']))
    require(record(Path(manifest['source_manifest']['file'])) == manifest['source_manifest'], 'Plan binding changed')
    require(len(manifest['cases']) == 8, 'Retain all eight cases')
    rows = []
    for case, expected in zip(manifest['cases'], plan['cases'], strict=True):
        require(all(case[k] == expected[k] for k in IDENTITY) and case['arm'] == ARM, 'Case/arm differs')
        video = record(Path(case['video']))
        require(video['sha256'] == case['video_sha256'] and video['bytes'] == case['video_bytes'], 'Media identity changed')
        rows.append(dict(**{k: case[k] for k in (*IDENTITY, 'trajectory_id')}, arm=ARM, video=video, scores={}))
    return rows, record(path)


def run_temporal(args):
    temporal = base.temporal
    manifest = args.manifest or args.report_dir/f'temporal_{ARM}_manifest.json'
    original_rows, original_save, original_argv = temporal.manifest_rows, temporal.save, sys.argv

    def save(path, report):
        report.update(experiment='E048', arm=ARM, policy=SCOPE)
        report['limits'] = [v for v in report['limits'] if 'FastWan product' not in v]
        report['adapter'] = dict(source=record(Path(__file__)), reused_loop=record(Path(temporal.__file__)),
            changes='Only E048 media binding and report labels; inherited 81-frame metric formulas unchanged',
            generation_report_meaning='Eight-case single-arm evaluation manifest')
        original_save(path, report)

    suffix = 'cpucheck' if args.check_only else 'scores'
    output = args.output or args.report_dir/f'temporal_{ARM}_{suffix}.json'
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
        temporal.manifest_rows, temporal.save, sys.argv = original_rows, original_save, original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'prepare', 'temporal'), required=True)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--arm', choices=(ARM,), default=ARM)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    {'check': check, 'prepare': prepare, 'temporal': run_temporal}[args.phase](args)


if __name__ == '__main__':
    main()
