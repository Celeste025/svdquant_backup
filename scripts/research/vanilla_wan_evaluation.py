#!/usr/bin/env python3
"""E043 evaluation binding; reuse the unchanged E024 81-frame metric loop.

check: CPU-only source/geometry readiness, before videos exist.
prepare: merge all four complete workers into the fixed eight-case manifest.
temporal [--check-only]: call the existing AMT/RAFT/DINO evaluator, replacing
only its E024 generation binding and report labels. MJ uses its existing CLI.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys

import evaluate_fastwan_qad_temporal as temporal

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E043'
PLAN = ROOT / 'research_state/06_experiments/E043_vanilla_wan_manifest.json'
EVAL_MANIFEST = RD / 'evaluation_manifest.json'
DATA = Path('/data1/models/svdquant-wjq')
PYTHON = DATA / 'conda-envs/mjvideo/bin/python'
MJ_REPO = DATA / 'third_party/MJ-Video'
MJ_MODEL = DATA / 'models/MJ-VIDEO-2B'
TOKENIZER = DATA / 'research/20261003/E021/tokenizer'
VARIANT = 'vanilla_wan_bf16'
IDENTITY = ('case_id', 'prompt_id', 'replica', 'prompt', 'seed')
SAMPLING = dict(mjvideo_indices=[0, 10, 20, 30, 40, 50, 60, 70],
                amt_input_frames=list(range(0, 81, 2)),
                amt_reference_frames=list(range(1, 81, 2)),
                raft_frames=list(range(0, 81, 2)), raft_iterations=20,
                dino_frames=list(range(81)))
SCOPE = ('Original non-distilled Wan reference; all eight fixed cases retained. '
         'Not a same-model quantization comparison with E024 or E038. '
         'MJ safety and bias criteria are retained raw only, not eligibility gates. '
         'Report total/alignment/fineness/coherence separately; no new composite.')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_plan(path):
    value = json.loads(path.read_text())
    rows = value['cases']
    require(value['experiment'] == 'E043' and len(rows) == 8, 'Expected E043 eight-case plan')
    require(len({r['case_id'] for r in rows}) == 8, 'Duplicate cases')
    require([r['prompt_id'] for r in rows] == [161, 161, 192, 192, 269, 269, 316, 316],
            'Keep the fixed prompt/replica order')
    require([r['replica'] for r in rows] == [0, 1] * 4, 'Keep both seeds per prompt')
    setting = value['settings']
    require(tuple(setting[k] for k in ('num_frames', 'height', 'width', 'fps')) == (81, 480, 832, 16),
            'This adapter is only for the existing 81-frame metric contract')
    return value


def save_new(path, value):
    if path.exists():
        require(json.loads(path.read_text()) == value, f'Keep existing output: {path}')
    else:
        temporal.save(path, value)


def check(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU check requires CUDA_VISIBLE_DEVICES empty')
    plan = read_plan(args.plan)
    # Execute only the upstream index function, not its model/data imports.
    import numpy as np
    tree = ast.parse((MJ_REPO / 'scripts/data_processor/data.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_index')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MJ_REPO / 'scripts/data_processor/data.py'), 'exec'), scope)
    indices = scope['get_index'](None, 16.0, 80, first_idx=0, num_segments=8).tolist()
    require(indices == SAMPLING['mjvideo_indices'], 'MJ local sampling contract changed')
    inherited = json.loads(temporal.INHERITED.read_text())
    require(inherited['status'] == 'complete', 'Local temporal model binding missing')
    refs = {r['file']: r for r in inherited['sources_and_models']}
    for path in (temporal.CONFIG, *temporal.WEIGHTS.values()):
        require(path.is_file() and path.stat().st_size == refs[str(path.resolve())]['bytes'],
                f'Missing/changed local checkpoint: {path}')
    for path in (PYTHON, MJ_MODEL / 'model.safetensors', TOKENIZER / 'tokenizer_config.json',
                 temporal.DINO_REPO / 'hubconf.py'):
        require(path.is_file(), f'Missing local asset: {path}')
    require('torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized(), 'Unexpected CUDA initialization')
    result = dict(experiment='E043', status='complete', mode='CPU_structure_check',
                  cuda_initialized=False, model_loads=0, generation_files_required=False,
                  manifest=temporal.file_record(args.plan), source=temporal.file_record(Path(__file__)),
                  temporal_loop=temporal.file_record(Path(temporal.__file__)),
                  case_ids=[r['case_id'] for r in plan['cases']], variant=VARIANT,
                  sampling=SAMPLING,
                  padding=dict(temporal_padding=False, amt_scale_1_spatial_padding=[0, 0, 0, 0],
                               raft_spatial_padding=[0, 0, 0, 0],
                               note='480x832 divisible by AMT16/RAFT8; AMT runtime scale is recorded by the inherited evaluator; no media padding or frame duplication.'),
                  dino_normalization='80 adjacent/first-frame cosine averages per video; pairs are the two real seeds of each prompt',
                  mjvideo_scope='8 sampled frames, not all 81; max_num=1, original resize/normalization unchanged',
                  scope=SCOPE)
    save_new(args.output or RD / 'evaluation_check.json', result)
    print(json.dumps(dict(status='complete', phase='check', cases=8, cuda_initialized=False)))


def prepare(args):
    plan = read_plan(args.plan)
    index = {}
    workers = []
    for prompt_id in (161, 192, 269, 316):
        path = args.worker_dir / f'worker_{prompt_id}.json'
        report = json.loads(path.read_text())
        require(report['experiment'] == 'E043' and report['status'] == 'complete', f'Worker incomplete: {path}')
        rows = report['cases']
        expected = [r for r in plan['cases'] if r['prompt_id'] == prompt_id]
        require(len(rows) == 2 and {r['case_id'] for r in rows} == {r['case_id'] for r in expected},
                f'Worker does not contain both fixed seeds: {path}')
        workers.append(temporal.file_record(path))
        for row in rows:
            require(row['status'] == 'complete' and row['variant'] == VARIANT, 'Incomplete/wrong variant')
            require(row['case_id'] not in index, 'Duplicate worker case')
            index[row['case_id']] = row
    cases = []
    for expected in plan['cases']:
        row = index[expected['case_id']]
        require(all(row[k] == expected[k] for k in IDENTITY), 'Worker identity differs from fixed manifest')
        video = Path(row['video']['file'])
        wanted = Path(plan['data_dir']) / row['case_id'] / 'video.mp4'
        require(video.is_absolute() and video.resolve() == wanted.resolve(), f'Unexpected media path: {video}')
        actual = temporal.file_record(video)
        require(actual['sha256'] == row['video']['sha256'] and actual['bytes'] == row['video']['bytes'], 'Media changed')
        require(row['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0), 'Worker media metadata differs')
        cases.append(dict(**{k: expected[k] for k in IDENTITY}, trajectory_id=expected['case_id'],
                          video=actual['file'], video_sha256=actual['sha256'], video_bytes=actual['bytes']))
    result = dict(experiment='E043', status='complete', phase='evaluation_binding', variant=VARIANT,
                  source_manifest=temporal.file_record(args.plan), worker_reports=workers,
                  sampling=plan['settings'], metric_sampling=SAMPLING, cases=cases, scope=SCOPE)
    target = args.output or args.manifest
    save_new(target, result)
    print(json.dumps(dict(status='complete', phase='prepare', manifest=str(target), videos=8, gpu_calls=0)))


def e043_rows(path):
    manifest = json.loads(path.read_text())
    require(manifest['experiment'] == 'E043' and manifest['status'] == 'complete' and manifest['variant'] == VARIANT,
            'Expected complete E043 evaluation binding')
    bound_plan = manifest['source_manifest']
    require(temporal.sha(bound_plan['file']) == bound_plan['sha256'], 'Source manifest changed')
    expected = read_plan(Path(bound_plan['file']))['cases']
    require(len(manifest['cases']) == len(expected) == 8, 'Evaluate all eight videos')
    for source in manifest['worker_reports']:
        require(temporal.sha(source['file']) == source['sha256'], 'Completed worker report changed')
    rows = []
    for row, reference in zip(manifest['cases'], expected, strict=True):
        require(all(row[k] == reference[k] for k in IDENTITY), 'Evaluation case changed/reordered')
        actual = temporal.file_record(Path(row['video']))
        require(actual['sha256'] == row['video_sha256'] and actual['bytes'] == row['video_bytes'], 'Media changed')
        rows.append(dict(**{k: row[k] for k in ('trajectory_id', *IDENTITY)}, video=actual, scores={}))
    require(len({r['video']['file'] for r in rows}) == 8, 'Distinct videos required')
    return rows, temporal.file_record(path)


def run_temporal(args):
    # Only provenance/report hooks change; the frozen metric loop executes as-is.
    original_rows, original_save, original_argv = temporal.manifest_rows, temporal.save, sys.argv

    def save_e043(path, report):
        report['experiment'] = 'E043'
        report['policy'] = SCOPE
        report['limits'] = [item for item in report['limits'] if 'FastWan product' not in item]
        report['limits'].append('Different models/seeds/resolution from E038: descriptive teacher readiness, not quantization causality.')
        # save is called after every case; keep this note unique.
        report['limits'] = list(dict.fromkeys(report['limits']))
        report['adapter'] = dict(source=temporal.file_record(Path(__file__)),
                                reused_metric_loop=str(Path(temporal.__file__).resolve()),
                                changes='E043 eight-worker-case binding and report labels only; same 81-frame AMT/RAFT/DINO formulas and local weights',
                                generation_report_meaning='Merged evaluation binding, not an E024 generation report')
        original_save(path, report)

    output = args.output or RD / ('temporal_cpucheck.json' if args.check_only else 'temporal_scores.json')
    command = [str(Path(temporal.__file__)), '--manifest', str(args.manifest), '--output', str(output)]
    if args.check_only:
        command.append('--check-only')
    if args.deadline_unix is not None:
        command += ['--deadline-unix', str(args.deadline_unix)]
    try:
        temporal.manifest_rows, temporal.save, sys.argv = e043_rows, save_e043, command
        temporal.main()
    finally:
        temporal.manifest_rows, temporal.save, sys.argv = original_rows, original_save, original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['check', 'prepare', 'temporal'], required=True)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--worker-dir', type=Path, default=RD)
    parser.add_argument('--manifest', type=Path, default=EVAL_MANIFEST)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true', help='CPU media validation for --phase temporal')
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if args.phase == 'check':
        check(args)
    elif args.phase == 'prepare':
        prepare(args)
    else:
        run_temporal(args)


if __name__ == '__main__':
    main()
