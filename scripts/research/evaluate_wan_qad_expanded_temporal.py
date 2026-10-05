#!/usr/bin/env python3
"""E022: unchanged local VBench AMT/RAFT/DINO for every completed phase video.

Only input identities/counts change from E021. No score filtering or composite
quality score. CPU media/model preflight requires a complete generation phase.
"""
import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from evaluate_wan_qad_temporal import (
    ROOT, VBENCH, EXTRA_SITE, DINO_REPO, WEIGHTS, CONFIG,
    sha, file_record, save, local_checkpoint_load, media_check,
)
from generate_wan_qad_expanded_comparison import protocol, BASELINE_ARMS, SELECTED_ARM

RD = ROOT / 'results/research/E022'
PHASE_ARMS = {'baselines': BASELINE_ARMS, 'selected': (SELECTED_ARM,)}


def completed_phase(generation, manifest, trajectories, phase):
    """Validate identities before any media is opened; no model imports here."""
    assert generation['status'] == 'complete', 'Generation phase must be complete before media checking'
    assert generation['experiment'] == 'E022_video_evaluation' and generation['phase'] == phase
    assert generation['manifest'] == manifest, 'Generation embeds a different manifest'
    arms = PHASE_ARMS[phase]
    expected_totals = (dict(dit_calls=192, native_mm_calls=38400, sdpa_calls=11520, videos=48)
                       if phase == 'baselines' else
                       dict(dit_calls=64, native_mm_calls=19200, sdpa_calls=3840, videos=16))
    assert generation['actual_totals'] == expected_totals
    assert set(generation['arms']) == set(arms), 'Incomplete or extra arms'
    identities = {t['trajectory_id'] for t in trajectories}
    rows = []
    for arm in arms:
        group = generation['arms'][arm]
        assert group['status'] == 'complete' and set(group['trajectories']) == identities
        for trajectory in trajectories:
            tid = trajectory['trajectory_id']; generated = group['trajectories'][tid]
            assert generated['status'] == 'complete'
            assert all(generated[k] == v for k, v in trajectory.items()), (arm, tid, 'Identity mismatch')
            video = generated['video']
            assert {k: video[k] for k in ('frames', 'height', 'width', 'fps')} == dict(frames=77, height=480, width=832, fps=16)
            rows.append(dict(case_id=f'{tid}_{arm}', trajectory_id=tid, arm=arm,
                             **{k: trajectory[k] for k in ('prompt_id', 'manifest_index', 'replica', 'seed', 'prompt')},
                             generated_video=video, scores={}))
    assert len(rows) == expected_totals['videos'] and len({r['case_id'] for r in rows}) == len(rows)
    assert len({r['generated_video']['path'] for r in rows}) == len(rows), 'Each row needs its own generated video'
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--phase', choices=tuple(PHASE_ARMS), required=True)
    p.add_argument('--generation', type=Path)
    p.add_argument('--manifest', type=Path, default=RD / 'video_test_manifest.json')
    p.add_argument('--check-report', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--deadline-unix', type=float)
    args = p.parse_args()
    args.generation = args.generation or RD / f'video_{args.phase}.json'
    args.check_report = args.check_report or RD / f'temporal_{args.phase}_cpucheck.json'
    output = args.output or (args.check_report if args.check_only else RD / f'temporal_{args.phase}.json')
    assert not output.exists(), f'Preserve existing report: {output}'
    if args.check_only: assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    else: assert args.deadline_unix and args.deadline_unix > time.time(), 'Explicit wall deadline required'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(VBENCH)); sys.path.append(str(EXTRA_SITE))
    started = time.time()
    report = dict(experiment='E022', phase=args.phase, status='running', mode='CPU_check' if args.check_only else 'GPU_evaluation',
        rows=[], policy='All 8 fixed prompts x 2 seeds x every phase arm; no filtering or composite overall score.',
        aggregation_policy='Downstream reports first average the two seeds within prompt, then average the eight prompts. Rows are not independent samples.',
        limits=['Fixed local held-out text split, not globally unseen or a broad independent benchmark.',
                'AMT interpolation agreement is not overall video quality; static or blurry clips may score highly.',
                'RAFT dynamic_degree is the original binary movement decision, not a monotonic quality score.',
                'DINO consistency may reward static content and does not establish semantic correctness.'],
        sampling=dict(amt_input_frames=list(range(0, 77, 2)), amt_reference_frames=list(range(1, 77, 2)),
                      raft_frames=list(range(0, 77, 2)), raft_iterations=20,
                      dino_frames=list(range(77))),
        checkpoint_policy='Explicit local checkpoint paths; scoped torch.load(map_location=cpu, weights_only=False) for trusted legacy pickles. No init_submodules, network download, or upstream source change.')
    try:
        # Reject incomplete phases before loading models or opening any video.
        generation = json.loads(args.generation.read_text())
        manifest, trajectories = protocol(args.manifest)
        report['rows'] = completed_phase(generation, manifest, trajectories, args.phase)
        report['generation'] = file_record(args.generation)
        report['manifest'] = file_record(args.manifest)
        expected_manifest = generation['manifest_reference']
        assert expected_manifest['sha256'] == report['manifest']['sha256'] and expected_manifest['bytes'] == report['manifest']['bytes']
        report['generation_status'] = generation['status']
        if args.phase == 'selected': report['checkpoint_binding'] = generation['checkpoint_binding']
        else: report['plain_step0000_binding'] = generation['plain_step0000_binding']
        import torch
        import cv2
        import numpy as np
        from easydict import EasyDict
        from vbench.motion_smoothness import MotionSmoothness
        from vbench.dynamic_degree import DynamicDegree
        from vbench.subject_consistency import subject_consistency
        torch.set_num_threads(6); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['environment'] = dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), extra_site=str(EXTRA_SITE), imports={})
        for name in ('torchvision', 'cv2', 'numpy', 'decord', 'omegaconf', 'easydict'):
            m = importlib.import_module(name)
            report['environment']['imports'][name] = dict(file=m.__file__, version=getattr(m, '__version__', None))
        files = [Path(__file__), Path(__file__).with_name('evaluate_wan_qad_temporal.py'),
                 Path(__file__).with_name('generate_wan_qad_expanded_comparison.py'),
                 Path(__file__).with_name('generate_wan_qad_comparison.py'), CONFIG, *WEIGHTS.values(), DINO_REPO / 'hubconf.py', DINO_REPO / 'vision_transformer.py']
        files += [VBENCH / 'vbench' / f'{n}.py' for n in ('utils', 'motion_smoothness', 'dynamic_degree', 'subject_consistency')]
        report['sources_and_models'] = [file_record(f) for f in files]
        if not args.check_only:
            check = json.loads(args.check_report.read_text())
            assert check['status'] == 'complete' and not check['cuda_initialized'] and check['phase'] == args.phase
            assert report['sources_and_models'] == check['sources_and_models'], 'Evaluator/model source changed since CPU check'
            assert report['generation'] == check['generation'] and report['manifest'] == check['manifest']
            report['cpucheck_reference'] = file_record(args.check_report)
        for row in report['rows']:
            video = row.pop('generated_video'); path = Path(video['path'])
            assert path.stat().st_size == video['bytes'] and sha(path) == video['sha256']
            row.update(video=file_record(path), media=media_check(path, cv2))
        report['media_count'] = len(report['rows'])
        if not args.check_only:
            assert [{k: r[k] for k in ('case_id', 'video', 'media')} for r in report['rows']] == [
                {k: r[k] for k in ('case_id', 'video', 'media')} for r in check['rows']]
        device = 'cpu' if args.check_only else 'cuda'
        if not args.check_only:
            report['environment']['device'] = torch.cuda.get_device_name()
            torch.cuda.reset_peak_memory_stats()
        def budget():
            if args.deadline_unix and time.time() > args.deadline_unix: raise TimeoutError('Caller deadline exceeded')
        def progress():
            report['seconds'] = time.time() - started; save(output, report)
        with local_checkpoint_load(torch):
            motion = MotionSmoothness(str(CONFIG), str(WEIGHTS['amt']), device)
        report.setdefault('model_load_checks', {})['amt'] = 'strict state_dict loaded'
        if not args.check_only:
            scale = motion.anchor_resolution / (480 * 832) * np.sqrt((motion.vram_avail - motion.anchor_memory_bias) / motion.anchor_memory)
            scale = min(1, scale); scale = float(16 / np.floor(16 / np.sqrt(scale)))
            report['amt_actual_scale_factor'] = scale
            for row in report['rows']:
                budget(); errors = []; original = motion.get_diff
                def capture(a, b):
                    v = original(a, b); errors.append(float(v)); return v
                motion.get_diff = capture
                try: score = float(motion.motion_score(row['video']['file']))
                finally: motion.get_diff = original
                assert len(errors) == 38 and math.isfinite(score)
                assert math.isclose(score, 1 - sum(errors) / len(errors) / 255, abs_tol=1e-7)
                row['scores']['motion_smoothness'] = score
                row['amt'] = dict(interpolation_reconstruction_absdiff=errors, meaning='Original AMT predicted odd frame vs actual odd frame, not adjacent-frame MAE')
                progress()
        del motion; gc.collect()
        if not args.check_only: torch.cuda.empty_cache()
        with local_checkpoint_load(torch):
            dynamic = DynamicDegree(EasyDict(model=str(WEIGHTS['raft']), small=False,
                mixed_precision=False, alternate_corr=False), device)
        report['model_load_checks']['raft'] = 'strict DataParallel state_dict loaded; original full RAFT'
        if not args.check_only:
            for row in report['rows']:
                budget(); flows = []; original = dynamic.get_score
                def capture(img, flo):
                    v = original(img, flo); flows.append(float(v)); return v
                dynamic.get_score = capture
                try: score = bool(dynamic.infer(row['video']['file']))
                finally: dynamic.get_score = original
                assert len(flows) == 38 and all(math.isfinite(v) for v in flows)
                count = sum(v > dynamic.params['thres'] for v in flows)
                assert score == (count >= dynamic.params['count_num'])
                row['scores']['dynamic_degree'] = score
                row['raft'] = dict(top5percent_mean_flow_magnitudes=flows, threshold=float(dynamic.params['thres']),
                                   required_count=int(dynamic.params['count_num']), above_threshold_count=count)
                progress()
        del dynamic; gc.collect()
        if not args.check_only: torch.cuda.empty_cache()
        dino = torch.hub.load(str(DINO_REPO), 'dino_vitb16', source='local', pretrained=False)
        with local_checkpoint_load(torch): dino.load_state_dict(torch.load(WEIGHTS['dino']), strict=True)
        dino.to(device).eval(); report['model_load_checks']['dino'] = 'local hub pretrained=False + explicit strict checkpoint'
        if not args.check_only:
            # Upstream's unused sim_per_video divides by len(video_list)-1: use
            # real pairs, and normalize its returned raw per-video sums by 76.
            assert len(report['rows']) % 2 == 0
            for index in range(0, len(report['rows']), 2):
                budget(); rows = report['rows'][index:index + 2]
                assert len(rows) == 2 and rows[0]['video']['file'] != rows[1]['video']['file']
                score, raw = subject_consistency(dino, [r['video']['file'] for r in rows], device, read_frame=False)
                assert len(raw) == 2
                for row, value in zip(rows, raw):
                    assert value['video_path'] == row['video']['file']
                    normalized = float(value['video_results']) / 76
                    assert math.isfinite(normalized)
                    row['scores']['subject_consistency'] = normalized
                    row['dino'] = dict(original_video_results_sum=float(value['video_results']), frame_pairs=76,
                                       normalization='Same mean over 76 terms as original sim_per_frame')
                assert math.isclose(float(score), sum(r['scores']['subject_consistency'] for r in rows) / 2, abs_tol=1e-10)
                progress()
            assert all(set(r['scores']) == {'motion_smoothness', 'dynamic_degree', 'subject_consistency'} for r in report['rows'])
        del dino; gc.collect()
        # Capture actual imported local model implementation, not a large tree audit.
        local_sources = set()
        for module in list(sys.modules.values()):
            f = getattr(module, '__file__', None)
            if f and str(f).endswith('.py') and (str(f).startswith(str(VBENCH)) or str(f).startswith(str(DINO_REPO))):
                local_sources.add(Path(f).resolve())
        report['imported_implementation'] = [file_record(p) for p in sorted(local_sources)]
        if not args.check_only:
            assert report['imported_implementation'] == check['imported_implementation']
            torch.cuda.synchronize(); report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        else: assert not torch.cuda.is_initialized()
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc()); raise
    finally:
        report.update(seconds=time.time() - started,
            cuda_initialized=bool('torch' in locals() and torch.cuda.is_initialized()))
        save(output, report); print(json.dumps(dict(status=report['status'], output=str(output), sha256=sha(output))))


if __name__ == '__main__': main()
