#!/usr/bin/env python3
"""E017: eight fixed videos in one frozen official VisionReward model process.

Reuses the E010 provenance, CPU fixtures and evaluator without editing them.
Large model SHA values are inherited only after size/mtime/header checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import evaluate_h3_paired_visionreward as original

MANIFEST = ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json'
ARMS = ('bf16', 'native', 'block_mean', 'global_mean')
PROMPTS = (30, 36)


def file_record(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': original.sha(path), 'bytes': path.stat().st_size}


def verify_file(record):
    path = Path(record['file'])
    if path.stat().st_size != record['bytes'] or original.sha(path) != record['sha256']:
        raise RuntimeError(f'Bound file changed: {path}')
    return path


def load_complete(path):
    value = json.loads(Path(path).read_text())
    if value['status'] != 'complete':
        raise RuntimeError(f'Require a completed source: {path}')
    return value


def check_sources(records):
    for path, row in records.items():
        if original.sha(path) != row['sha256']:
            raise RuntimeError(f'Bound source changed: {path}')


def prerequisites(args, report):
    manifest = json.loads(args.manifest.read_text())
    if tuple(manifest['arms']) != ARMS[2:] or tuple(c['prompt_id'] for c in manifest['cases']) != PROMPTS:
        raise RuntimeError('Fixed E017 case/arm list changed')
    if manifest['budget']['evaluation_questions'] != 232 or sys.executable != manifest['python_evaluate']:
        raise RuntimeError('Pinned evaluator environment or question allocation changed')
    plan = verify_file(manifest['plan'])
    old_manifest_path = verify_file(manifest['e010_manifest'])
    old_manifest = json.loads(old_manifest_path.read_text())
    if manifest['settings'] != old_manifest['settings'] or manifest['cases'] != old_manifest['cases']:
        raise RuntimeError('E017 prompt, seed or generation settings differ from E010')
    reference_path = verify_file(manifest['e010_visionreward'])
    reference = load_complete(reference_path)
    if reference['model_full_hash_complete'] is not True:
        raise RuntimeError('Historical VisionReward report lacks complete model SHA verification')
    check_sources(reference['sources'])
    old_decode = load_complete(verify_file(manifest['e010_decode']))
    check_sources(old_decode['sources'])
    # These are exactly the frozen checks, including every shard header/index.
    questions, weights = original.provenance(report, full_hash=False)
    if questions != reference['questions'] or weights.tolist() != reference['weights']:
        raise RuntimeError('Official questions or weights changed from E010')
    if report['sources'] != {p: {k: v for k, v in row.items() if k != 'snapshot'}
                             for p, row in reference['sources'].items()}:
        raise RuntimeError('Official evaluator source inventory changed from E010')
    current = report['model_files']
    if set(current) != set(reference['model_files']):
        raise RuntimeError('VisionReward model file set changed')
    for path, row in current.items():
        old = reference['model_files'][path]
        for key in ('bytes', 'mtime_ns', 'header_sha256', 'sha256'):
            if key in row and row[key] != old[key]:
                raise RuntimeError(f'VisionReward model metadata/hash changed: {path}, {key}')
        if 'sha256' not in old:
            raise RuntimeError('Historical model file has no complete SHA')
        row['sha256_checked_this_run'] = 'sha256' in row
        row['sha256'] = old['sha256']
        row['sha256_origin'] = 'current_full_hash' if row['sha256_checked_this_run'] else 'inherited_E010_full_hash'
    report['model_full_hash_complete'] = False
    report['model_full_hash_inherited'] = True
    report['model_hash_policy'] = 'No repeated 25 GB shard hash. Frozen official provenance rechecks shard headers/length/index; exact file size/mtime/header and all small-file SHA must match the completed E010 full-SHA inventory.'
    report['e010_visionreward'] = manifest['e010_visionreward']
    report['e010_decode'] = manifest['e010_decode']
    report['manifest'] = file_record(args.manifest)
    report['plan'] = file_record(plan)
    for path in (Path(__file__), args.manifest, plan):
        report['sources'][str(path.resolve())] = {'sha256': original.sha(path), 'bytes': path.stat().st_size}
    if manifest.get('source_files'):
        check_sources(manifest['source_files'])
    return manifest, reference, old_decode, questions, weights


def expected_specs(manifest, reference):
    old = {(c['prompt_id'], c['variant']): c for c in reference['results']}
    if set(old) != {(p, a) for p in PROMPTS for a in ARMS[:2]}:
        raise RuntimeError('Historical VisionReward coverage differs')
    specs = []
    # Replay the original four in their original order, then the four new ones.
    for arms in (ARMS[:2], ARMS[2:]):
        for prompt in PROMPTS:
            for arm in arms:
                path = (Path(old[(prompt, arm)]['case_json']) if arm in ARMS[:2] else
                        Path(manifest['data_dir'])/('decode_'+arm)/f'p{prompt:03d}_{arm}.json')
                specs.append({'prompt_id': prompt, 'variant': arm, 'path': path})
    return specs


def load_cases(paths, manifest, reference, old_decode, *, allow_missing=False):
    specs = expected_specs(manifest, reference)
    if list(map(str, paths)) != [str(s['path']) for s in specs]:
        raise RuntimeError('Require exactly the eight preregistered media paths in fixed order')
    prompt_rows = {c['prompt_id']: c for c in manifest['cases']}
    old_rows = {(c['prompt_id'], c['variant']): c for c in reference['results']}
    old_decode_rows = {(c['prompt_id'], c['variant']): c for c in old_decode['cases']}
    cases, inventory, decode_bindings = [], [], {}
    expected_media = {'frames': manifest['settings']['num_frames'], 'width': manifest['settings']['width'],
        'height': manifest['settings']['height'], 'fps': float(manifest['settings']['fps']),
        'audio_streams': 1, 'audio_sample_rate': manifest['settings']['audio_sample_rate']}
    for spec in specs:
        path, prompt, arm = spec['path'], spec['prompt_id'], spec['variant']
        item = {'prompt_id': prompt, 'variant': arm, 'case_json': str(path), 'present': path.is_file()}
        inventory.append(item)
        if not path.is_file():
            if allow_missing and arm in ARMS[2:]:
                item.update(validated=False, reason='New media not yet produced; required before GPU evaluation')
                continue
            raise FileNotFoundError(path)
        row = json.loads(path.read_text())
        source = prompt_rows[prompt]
        if (row['status'] != 'complete' or row['prompt_id'] != prompt or row['variant'] != arm or
                any(row[k] != source[k] for k in ('seed', 'prompt', 'prompt_sha256')) or
                row['settings'] != manifest['settings']):
            raise RuntimeError(f'Case identity/settings changed: {path}')
        if hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']:
            raise RuntimeError('Prompt text SHA mismatch')
        video = Path(row['video'])
        if not video.is_absolute() or video != path.with_suffix('.mp4'):
            raise RuntimeError('Media must have the exact absolute preregistered path')
        item['video_present'] = video.is_file()
        if not video.is_file():
            if allow_missing and arm in ARMS[2:]:
                item.update(present=False, validated=False, reason='New video not yet present; required before GPU evaluation')
                continue
            raise FileNotFoundError(video)
        if (original.sha(video) != row['video_sha256'] or
                {k: row['media'].get(k) for k in expected_media} != expected_media or
                ('audio_channels' in row['media'] and row['media']['audio_channels'] != 2)):
            raise RuntimeError(f'Video SHA or encoded metadata differs: {path}')
        if arm in ARMS[:2]:
            previous = old_rows[(prompt, arm)]
            if (original.sha(path) != previous['case_json_sha256'] or row != old_decode_rows[(prompt, arm)] or
                    row['video_sha256'] != previous['video_sha256']):
                raise RuntimeError('Historical media changed from E010 decode/evaluation')
        else:
            decode_path = Path(manifest['report_dir'])/f'E017_decode_{arm}.json'
            if decode_path.is_file():
                decoded = json.loads(decode_path.read_text())
                if decoded['status'] != 'complete' and not allow_missing:
                    raise RuntimeError('New decoder has not completed')
                candidates = [c for c in decoded.get('cases', []) if c['prompt_id'] == prompt and c['variant'] == arm]
                if candidates != [row]:
                    raise RuntimeError('New media JSON differs from its decoder report')
                if 'sources' in decoded:
                    check_sources(decoded['sources'])
                decode_bindings[arm] = file_record(decode_path)
                item['decode_report_complete'] = decoded['status'] == 'complete'
            elif not allow_missing:
                raise FileNotFoundError(decode_path)
            else:
                item['decode_report_complete'] = False
        item.update(validated=True, case_json_sha256=original.sha(path), video_sha256=row['video_sha256'])
        cases.append({**row, 'case_json': str(path.resolve()), 'case_json_sha256': item['case_json_sha256'],
            'case_id': f'p{prompt:03d}_seed{row["seed"]}_{arm}_{row["video_sha256"][:12]}'})
    if not allow_missing and len(cases) != 8:
        raise RuntimeError('Require all eight videos before loading the evaluator')
    return cases, inventory, decode_bindings


def compare_historical_answers(report, reference):
    previous = {(c['prompt_id'], c['variant']): c for c in reference['results']}
    current = {(c['prompt_id'], c['variant']): c for c in report.get('results', [])}
    compared = []
    raw_fields = ('first_token_id', 'first_token_decoded', 'generated_token_ids', 'full_generated_decoded')
    for key, old in previous.items():
        fresh = current.get(key, {})
        answers = {a['question_index']: a for a in fresh.get('answers', [])}
        rows = []
        for old_answer in old['answers']:
            index = old_answer['question_index']
            if index not in answers:
                rows.append({'question_index': index, 'present': False})
                continue
            new_answer = answers[index]
            equal = {field: new_answer[field] == old_answer[field] for field in raw_fields}
            rows.append({'question_index': index, 'present': True, 'raw_fields_equal': equal,
                'raw_answer_equal': all(equal.values()),
                'old_first_token_decoded': old_answer['first_token_decoded'],
                'new_first_token_decoded': new_answer['first_token_decoded'],
                'query_equal': new_answer['expanded_query'] == old_answer['expanded_query'],
                'frame_trace_equal': new_answer['frames'] == old_answer['frames'],
                'input_token_ids_equal': new_answer['input_token_ids'] == old_answer['input_token_ids']})
        compared.append({'prompt_id': key[0], 'variant': key[1], 'questions': rows,
            'compared_questions': len(answers),
            'changed_raw_question_indices': [r['question_index'] for r in rows if r['present'] and not r['raw_answer_equal']]})
    return {'reference': report['e010_visionreward'], 'cases': compared,
        'interpretation': 'Differences are recorded evaluator replay variation on identical historical media; never attributed to the new quantization arms and never used as a stop condition.'}


def deadline_guard(args):
    if args.deadline_unix is None or time.time() >= args.deadline_unix:
        raise TimeoutError('Shared E017 GPU deadline absent or expired; do not reset it')


@contextmanager
def bounded_evaluator(args, report, manifest, reference, old_decode):
    old_loader, old_functions, old_save = original.load_cases, original.official_functions, original.save
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError('Shared E017 absolute deadline reached')
    def checked_loader(paths):
        cases, inventory, bindings = load_cases(paths, manifest, reference, old_decode)
        report['media_inventory'], report['new_decode_reports'] = inventory, bindings
        return cases
    def checked_functions(namespace):
        ns = old_functions(namespace)
        inference = ns['inference']
        def guarded_inference(*pos, **kwargs):
            import torch
            deadline_guard(args)
            if torch.cuda.max_memory_allocated()/1024**3 > manifest['budget']['max_peak_allocated_gib']:
                raise RuntimeError('Evaluator exceeded the E017 historical peak allocation budget')
            result = inference(*pos, **kwargs)
            if torch.cuda.max_memory_allocated()/1024**3 > manifest['budget']['max_peak_allocated_gib']:
                raise RuntimeError('Evaluator exceeded the E017 historical peak allocation budget')
            deadline_guard(args)
            return result
        ns['inference'] = guarded_inference
        return ns
    def checked_save(value, path):
        value['historical_answer_replay'] = compare_historical_answers(value, reference)
        old_save(value, path)
    deadline_guard(args)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, args.deadline_unix-time.time()))
    original.load_cases, original.official_functions, original.save = checked_loader, checked_functions, checked_save
    try:
        yield
    finally:
        original.load_cases, original.official_functions, original.save = old_loader, old_functions, old_save
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    manifest_hint = json.loads(args.manifest.read_text())
    args.output = args.output or Path(manifest_hint['report_dir'])/('E017_visionreward_cpucheck.json' if args.check else 'E017_visionreward.json')
    if args.output.exists():
        raise FileExistsError(f'Preserve existing report: {args.output}')
    report = {'experiment': 'E017', 'status': 'running', 'check_only': args.check,
        'deadline_unix': args.deadline_unix, 'expected_question_count': 232,
        'scope': 'Two already inspected prompts, four variants; auxiliary video-only scores, no quality equivalence/generalization or audio claim.',
        'official_compatibility': 'Frozen E010 provenance/cpu_check/evaluate and official load_video/inference reused; only case coverage, deadline boundaries and provenance/replay reporting are extended.',
        'unknown_policy': 'Both usable scores null for unrecognized answers; frozen strict and normalized scoring unchanged.'}
    started = time.monotonic()
    reference = None
    try:
        manifest, reference, old_decode, questions, weights = prerequisites(args, report)
        original.cpu_check(report, weights)
        report['historical_versions'] = {'reference': reference['versions'], 'current': report['versions'],
                                         'exact': reference['versions'] == report['versions']}
        args.case_json = [s['path'] for s in expected_specs(manifest, reference)]
        report['expected_cases'] = list(map(str, args.case_json))
        cases, inventory, bindings = load_cases(args.case_json, manifest, reference, old_decode, allow_missing=args.check)
        report['media_inventory'], report['new_decode_reports'] = inventory, bindings
        report['case_files_present'] = {r['case_json']: r['present'] for r in inventory}
        if args.check:
            report.update(status='complete', available_cases=len(cases))
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != str(manifest['budget']['gpu']):
                raise RuntimeError('Evaluator must use only the preregistered GPU')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with bounded_evaluator(args, report, manifest, reference, old_decode):
                original.evaluate(args, report, questions, weights)
            if len(report['results']) != 8 or sum(len(r['answers']) for r in report['results']) != 232:
                raise RuntimeError('Evaluator did not execute all 232 fixed questions')
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if reference is not None:
            report['historical_answer_replay'] = compare_historical_answers(report, reference)
        report['seconds_total'] = time.monotonic()-started
        original.save(report, args.output)
        print(json.dumps({'status': report['status'], 'report': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
