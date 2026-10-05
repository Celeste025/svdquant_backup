#!/usr/bin/env python3
"""E024: original local E022 AMT/RAFT/DINO metrics, adapted to 81-frame media.

No composite score, filtering, model download, or change to upstream formulas.
CPU --check-only decodes all media and verifies provenance without model loads.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from evaluate_wan_qad_temporal import (
    ROOT, VBENCH, EXTRA_SITE, DINO_REPO, WEIGHTS, CONFIG,
    sha, file_record, save, local_checkpoint_load,
)

INHERITED = ROOT / 'results/research/E022/temporal_selected.json'
IDENTITY_KEYS = ('trajectory_id', 'prompt_id', 'prompt', 'seed', 'replica')


def manifest_rows(path):
    manifest = json.loads(path.read_text())
    generation_path = Path(manifest['generation_report'])
    assert generation_path.is_absolute() and generation_path.is_file()
    generation = json.loads(generation_path.read_text())
    assert generation['status'] == 'complete' and generation['phase'] == 'suite'
    generated = generation['cases']
    assert all(r['status'] == 'complete' for r in generated)
    index = {r['trajectory_id']: r for r in generated}
    assert len(index) == len(generated)
    rows = manifest['cases']
    assert len(rows) >= 2 and len(rows) % 2 == 0, 'DINO needs real pairs'
    assert len({r['case_id'] for r in rows}) == len(rows)
    assert {r['trajectory_id'] for r in rows} == set(index) and len(rows) == len(index), 'Full suite coverage required'
    result = []
    for row in rows:
        source = index[row['trajectory_id']]
        assert all(row[k] == source[k] for k in IDENTITY_KEYS), row['case_id']
        video = Path(row['video'])
        assert video.is_absolute() and video.is_file() and video.suffix == '.mp4'
        bound = source['video']
        assert video.resolve() == Path(bound['file']).resolve()
        actual = file_record(video)
        assert actual['sha256'] == row['video_sha256'] == bound['sha256']
        assert actual['bytes'] == bound['bytes']
        result.append({**{k: row[k] for k in ('case_id', *IDENTITY_KEYS)}, 'video': actual, 'scores': {}})
    assert len({r['video']['file'] for r in result}) == len(result), 'Distinct real videos required'
    return result, file_record(generation_path)


def media_check_81(path, cv2, budget):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    fps = float(cap.get(cv2.CAP_PROP_FPS)); declared = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    try:
        while True:
            budget(); ok, frame = cap.read()
            if not ok: break
            assert frame.shape == (480, 832, 3), (path, count, frame.shape)
            count += 1
    finally: cap.release()
    assert count == declared == 81 and math.isclose(fps, 16, abs_tol=1e-8), (path, count, declared, fps)
    return dict(decoded_frames=count, declared_frames=declared, fps=fps, width=832, height=480)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--deadline-unix', type=float)
    args = p.parse_args()
    assert not args.output.exists(), f'Preserve existing output: {args.output}'
    if args.check_only: assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    else: assert args.deadline_unix and args.deadline_unix > time.time(), 'GPU evaluation needs an absolute deadline'
    started = time.time()
    def budget():
        if args.deadline_unix and time.time() >= args.deadline_unix: raise TimeoutError('Caller deadline exceeded')
    report = dict(experiment='E024', status='running', mode='CPU_check' if args.check_only else 'GPU_evaluation', rows=[],
        policy='Every manifest/suite video, manifest order, no filtering or composite score.',
        limits=['AMT measures interpolation agreement, not overall quality.',
                'RAFT dynamic_degree is a binary motion decision, not a monotonic quality score.',
                'DINO consistency can favor static content and is not semantic correctness.',
                '81-frame FastWan product differs from 77-frame rCM; do not interpret this as a same-weight single-factor quantization comparison.'],
        sampling=dict(amt_input_frames=list(range(0, 81, 2)), amt_reference_frames=list(range(1, 81, 2)),
                      raft_frames=list(range(0, 81, 2)), raft_iterations=20, dino_frames=list(range(81))),
        checkpoint_policy='Inherit fixed E022 local AMT/RAFT/DINO checkpoints and original formulas; no model/source-tree reaudit. Stored prior hashes plus current file lengths, not a new full checkpoint hash verification.')
    def progress():
        report['seconds'] = time.time() - started; save(args.output, report)
    try:
        budget(); report['rows'], report['generation_report'] = manifest_rows(args.manifest)
        report['manifest'] = file_record(args.manifest)
        inherited = json.loads(INHERITED.read_text()); assert inherited['status'] == 'complete'
        references = {r['file']: r for r in inherited['sources_and_models']}
        report['inherited_e022'] = file_record(INHERITED)
        report['inherited_models'] = []
        for path in (CONFIG, *WEIGHTS.values()):
            ref = references[str(path.resolve())]
            assert path.is_file() and path.stat().st_size == ref['bytes'], path
            report['inherited_models'].append(ref)
        report['sources'] = [file_record(Path(__file__)), file_record(Path(__file__).with_name('evaluate_wan_qad_temporal.py'))]
        import cv2
        report['environment'] = dict(python=sys.executable, cv2=cv2.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        for row in report['rows']:
            row['media'] = media_check_81(row['video']['file'], cv2, budget)
        report['media_count'] = len(report['rows'])
        if args.check_only:
            assert 'torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized()
            report.update(status='complete', cuda_initialized=False, model_loads=0)
            return
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        sys.path.insert(0, str(VBENCH)); sys.path.append(str(EXTRA_SITE))
        import torch
        import numpy as np
        from easydict import EasyDict
        from vbench.motion_smoothness import MotionSmoothness
        from vbench.dynamic_degree import DynamicDegree
        from vbench.subject_consistency import subject_consistency
        torch.set_num_threads(6); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        report['environment'].update(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), extra_site=str(EXTRA_SITE))
        report['model_loads'] = 0; torch.cuda.reset_peak_memory_stats(); budget()
        with local_checkpoint_load(torch): motion = MotionSmoothness(str(CONFIG), str(WEIGHTS['amt']), 'cuda')
        report['model_loads'] += 1
        scale = min(1, motion.anchor_resolution / (480 * 832) * np.sqrt((motion.vram_avail - motion.anchor_memory_bias) / motion.anchor_memory))
        report['amt_actual_scale_factor'] = float(16 / np.floor(16 / np.sqrt(scale)))
        for row in report['rows']:
            budget(); errors = []; original = motion.get_diff
            def capture(a, b):
                value = original(a, b); errors.append(float(value)); return value
            motion.get_diff = capture
            try: score = float(motion.motion_score(row['video']['file']))
            finally: motion.get_diff = original
            assert len(errors) == 40 and all(math.isfinite(v) for v in errors) and math.isfinite(score)
            assert math.isclose(score, 1 - sum(errors) / 40 / 255, abs_tol=1e-7)
            row['scores']['motion_smoothness'] = score
            row['amt'] = dict(interpolation_reconstruction_absdiff=errors, input_even_frames=41, reference_odd_frames=40,
                              meaning='Original predicted odd vs actual odd frame absolute pixel error, not adjacent-frame MAE')
            progress()
        del motion; gc.collect(); torch.cuda.empty_cache(); budget()
        with local_checkpoint_load(torch):
            dynamic = DynamicDegree(EasyDict(model=str(WEIGHTS['raft']), small=False, mixed_precision=False, alternate_corr=False), 'cuda')
        report['model_loads'] += 1
        for row in report['rows']:
            budget(); flows = []; actual_frames = {}; original = dynamic.get_score; original_frames = dynamic.get_frames
            def capture_score(img, flo):
                value = original(img, flo); flows.append(float(value)); return value
            def capture_frames(path):
                value = original_frames(path); actual_frames.update(count=len(value), shape=list(value[0].shape)); return value
            dynamic.get_score = capture_score; dynamic.get_frames = capture_frames
            try: score = bool(dynamic.infer(row['video']['file']))
            finally: dynamic.get_score = original; dynamic.get_frames = original_frames
            assert actual_frames['count'] == 41 and len(flows) == 40 and all(math.isfinite(v) for v in flows)
            threshold = float(dynamic.params['thres']); required = int(dynamic.params['count_num'])
            assert threshold == 6 * min(actual_frames['shape'][-2:]) / 256
            assert required == round(4 * (actual_frames['count'] / 16.0))
            count = sum(v > threshold for v in flows); assert score == (count >= required)
            row['scores']['dynamic_degree'] = score
            row['raft'] = dict(top5percent_mean_flow_magnitudes=flows, sampled_frames=actual_frames['count'], actual_flows=len(flows),
                threshold=threshold, required_count=required, above_threshold_count=count,
                required_count_formula='Original local set_params: round(4*(sampled_frame_count/16.0)); not ceil or a hardcoded constant')
            progress()
        del dynamic; gc.collect(); torch.cuda.empty_cache(); budget()
        dino = torch.hub.load(str(DINO_REPO), 'dino_vitb16', source='local', pretrained=False)
        with local_checkpoint_load(torch): dino.load_state_dict(torch.load(WEIGHTS['dino']), strict=True)
        dino.to('cuda').eval(); report['model_loads'] += 1; report['dino_real_pairs'] = []
        for index in range(0, len(report['rows']), 2):
            budget(); rows = report['rows'][index:index + 2]
            assert len(rows) == 2 and rows[0]['video']['file'] != rows[1]['video']['file']
            score, raw = subject_consistency(dino, [r['video']['file'] for r in rows], 'cuda', read_frame=False)
            assert len(raw) == 2
            for row, value in zip(rows, raw):
                assert value['video_path'] == row['video']['file']
                total = float(value['video_results']); normalized = total / 80
                assert math.isfinite(normalized)
                row['scores']['subject_consistency'] = normalized
                row['dino'] = dict(original_video_results_sum=total, frame_pairs=80, normalization='Original sum of 80 clipped adjacent/first-frame cosine averages divided by 80')
            assert math.isclose(float(score), sum(r['scores']['subject_consistency'] for r in rows) / 2, abs_tol=1e-10)
            report['dino_real_pairs'].append(dict(case_ids=[r['case_id'] for r in rows], original_pair_score=float(score)))
            progress()
        del dino; gc.collect(); torch.cuda.synchronize(); budget()
        assert all(set(r['scores']) == {'motion_smoothness', 'dynamic_degree', 'subject_consistency'} for r in report['rows'])
        report.update(status='complete', peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, cuda_initialized=torch.cuda.is_initialized())
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc()); raise
    finally:
        report['seconds'] = time.time() - started; save(args.output, report)
        print(json.dumps(dict(status=report['status'], output=str(args.output), sha256=sha(args.output))))


if __name__ == '__main__': main()
