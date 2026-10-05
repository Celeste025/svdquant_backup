#!/usr/bin/env python3
"""Prepare the E025 same-latent decoder manifest or supervise one existing evaluator.

No metric implementations or model changes. Root starts this in a named tmux.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E025'
DATA = Path('/data1/models/svdquant-wjq')
PYTHON = DATA / 'conda-envs/mjvideo/bin/python'
VARIANT = 'fastwan_full_vae_fp32'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path).absolute()
    return dict(file=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args):
    generation_path = args.generation.absolute()
    generation = json.loads(generation_path.read_text())
    assert generation['status'] == 'complete' and generation['phase'] == 'suite'
    launcher_path = args.generation_launcher
    if launcher_path is None:
        assert '_run' in generation_path.stem, 'Pass --generation-launcher for an independently named report'
        launcher_path = generation_path.with_name(generation_path.name.replace('_run', '_launcher', 1))
    launcher = json.loads(launcher_path.read_text())
    assert launcher['status'] == 'complete' and launcher['returncode'] == 0
    assert Path(launcher['result']).resolve() == generation_path.resolve()
    generation_sha = sha(generation_path)
    assert launcher['result_sha256'] == generation_sha
    expected = {row['trajectory_id']: row for row in generation['trajectories']}
    rows = generation['cases']
    assert len(expected) == len(rows) == 16
    assert {r['trajectory_id'] for r in rows} == set(expected)
    keys = ('trajectory_id', 'prompt_id', 'prompt', 'seed', 'replica')
    cases = []
    for row in rows:
        assert row['status'] == 'complete'
        assert all(row[key] == expected[row['trajectory_id']][key] for key in keys)
        video = Path(row['video']['file'])
        assert video.is_absolute() and video.name == row['trajectory_id'] + '.mp4'
        assert video.is_file() and video.stat().st_size == row['video']['bytes']
        assert sha(video) == row['video']['sha256']
        cases.append(dict({key: row[key] for key in keys}, case_id=row['trajectory_id'],
                          video=str(video), video_sha256=row['video']['sha256']))
    parents = {Path(row['video']).parent for row in cases}
    assert len(parents) == 1 and len({r['video'] for r in cases}) == 16
    manifest = dict(experiment='E025', variant=VARIANT, generation_report=str(generation_path),
                    generation_sha256=generation_sha, generation_launcher=record(launcher_path),
                    sampling=generation['sampling'], cases=cases,
                    scope='All 16 same-latent full Wan VAE FP32 videos; paired decoder-stack plus dtype intervention versus E024 TAEHV FP16, no filtering.')
    assert manifest['sampling']['num_frames'] == 81 and manifest['sampling']['fps'] == 16
    if args.manifest.exists():
        assert json.loads(args.manifest.read_text()) == manifest, 'Preserve existing manifest; use --manifest for a new binding'
    else:
        write(args.manifest, manifest)
    return manifest, next(iter(parents))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=['temporal', 'mjvideo'], default='temporal')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--generation', type=Path, default=RD / 'decode_run.json')
    parser.add_argument('--generation-launcher', type=Path)
    parser.add_argument('--manifest', type=Path, default=RD / 'evaluation_manifest.json')
    parser.add_argument('--temporal-cpucheck', type=Path, default=RD / 'temporal_cpucheck.json')
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    assert args.attempt >= 1
    if args.prepare_only:
        manifest, samples = prepare(args)
        print(json.dumps(dict(status='prepared', manifest=record(args.manifest),
                              count=len(manifest['cases']), samples=str(samples), gpu_calls=0)))
        return
    suffix = '' if args.attempt == 1 else f'_attempt{args.attempt}'
    result = RD / f'{args.metric}{suffix}.json'
    output = RD / f'{args.metric}_launcher{suffix}.json'
    logfile = ROOT / 'results/logs' / f'E025_{args.metric}{suffix}.log'
    assert not any(p.exists() for p in (result, output, logfile)), 'Preserve prior attempt; pass a new --attempt'
    temporal = args.metric == 'temporal'
    gpu = 1 if temporal else 5
    start = time.time()
    budget = 600 if temporal else 300
    deadline = args.deadline_unix or start + budget
    assert 0 < deadline - start <= budget
    script = ROOT / 'scripts/research' / ('evaluate_fastwan_qad_temporal.py' if temporal else 'eval_mjvideo_e021.py')
    report = dict(experiment='E025', status='running', metric=args.metric, attempt=args.attempt,
                  gpu=gpu, start_epoch=start, deadline_epoch=deadline,
                  script=record(script), launcher=record(__file__), log=str(logfile))
    proc = None
    try:
        manifest, samples = prepare(args)
        report['manifest'] = record(args.manifest)
        report['generation_report'] = record(args.generation)
        report['generation_launcher'] = manifest['generation_launcher']
        if temporal:
            check = json.loads(args.temporal_cpucheck.read_text())
            assert check['status'] == 'complete' and check['mode'] == 'CPU_check'
            assert check['cuda_initialized'] is False and check['model_loads'] == 0
            assert check['manifest']['sha256'] == report['manifest']['sha256']
            assert check['generation_report']['sha256'] == report['generation_report']['sha256']
            assert len(check['rows']) == 16
            assert {r['trajectory_id'] for r in check['rows']} == {r['trajectory_id'] for r in manifest['cases']}
            bound = {r['file']: r['sha256'] for r in check['sources']}
            assert bound[str(script)] == report['script']['sha256']
            report['temporal_cpucheck'] = record(args.temporal_cpucheck)
        command = [str(PYTHON), '-u', str(script), '--manifest', str(args.manifest.absolute()), '--output', str(result)]
        if temporal:
            command += ['--deadline-unix', str(deadline)]
        else:
            command += ['--samples', str(samples), '--model', str(DATA / 'models/MJ-VIDEO-2B'),
                        '--tokenizer', str(DATA / 'research/20261003/E021/tokenizer'),
                        '--mjvideo-repo', str(DATA / 'third_party/MJ-Video'),
                        '--variants', VARIANT, '--video-template', '{case_id}.mp4', '--num-segments', '8']
        report['command'] = command
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_HOME='/usr/local/cuda',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   MASTER_ADDR='127.0.0.1', MASTER_PORT='29575', OMP_NUM_THREADS='6',
                   PYTHONUNBUFFERED='1', MAX_JOBS='4')
        cache = DATA / 'research/20261003/E025/evaluation_cache'
        for key, relative in dict(XDG_CACHE_HOME='.', HF_HOME='huggingface', TORCH_HOME='torch',
                                  TRITON_CACHE_DIR='triton', CUDA_CACHE_PATH='cuda',
                                  TORCH_EXTENSIONS_DIR='torch_extensions', TMPDIR='tmp').items():
            path = cache / relative
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        env['PATH'] = str(PYTHON.parent) + ':/usr/local/cuda/bin:' + env.get('PATH', '')
        from resume_h3_plain_baseline import idle_stable
        report['gpu_before'] = idle_stable(gpu)
        assert time.time() < deadline
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            report['pid'] = proc.pid
            write(output, report)
            try:
                code = proc.wait(timeout=max(.01, deadline - time.time()))
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                raise
        report['returncode'] = code
        assert code == 0, f'{args.metric} worker exited {code}; see retained log'
        value = json.loads(result.read_text())
        ids = {row['case_id'] for row in manifest['cases']}
        if temporal:
            assert value['status'] == 'complete' and len(value['rows']) == 16
            assert {r['case_id'] for r in value['rows']} == ids
            assert value['manifest']['sha256'] == report['manifest']['sha256']
        else:
            assert value['num_segments'] == 8 and set(value['results']) == ids
            for row in manifest['cases']:
                scored = value['results'][row['case_id']]
                assert scored['prompt'] == row['prompt'] and set(scored['variants']) == {VARIANT}
                score = scored['variants'][VARIANT]
                assert score['video'] == row['video'] and score['video_sha256'] == row['video_sha256']
                assert len(score['criteria']) == 28 and len(score['aspects']) == 5
        report.update(status='complete', result=record(result))
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        if proc is not None:
            report['returncode'] = proc.poll()
        if result.exists():
            report['result'] = record(result)
        report.update(end_epoch=time.time(), seconds=time.time() - start)
        write(output, report)
        print(json.dumps(dict(status=report['status'], output=str(output), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
