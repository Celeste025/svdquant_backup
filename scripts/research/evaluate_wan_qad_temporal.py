#!/usr/bin/env python3
"""E021: unchanged local VBench AMT/RAFT/DINO scores for all sixteen videos."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E021'
VBENCH = Path('/home/wjq/workspace/ViDiT-Q/eval/video/Vbench')
EXTRA_SITE = Path('/home/wjq/.venvs/vbench/lib/python3.12/site-packages')
DINO_REPO = Path('/home/wjq/.cache/torch/hub/facebookresearch_dino_main')
WEIGHTS = dict(amt=Path('/home/wjq/.cache/vbench/amt_model/amt-s.pth'),
               raft=Path('/home/wjq/.cache/vbench/raft_model/models/raft-things.pth'),
               dino=Path('/home/wjq/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth'))
CONFIG = VBENCH / 'vbench/third_party/amt/cfgs/AMT-S.yaml'
ARMS = ('bf16', 'packed_step0000', 'packed_step0064', 'svd_lr')
CASES = ('vbench_066', 'vbench_091', 'vbench_182', 'vbench_067')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def file_record(path):
    p = Path(path).resolve()
    return dict(file=str(p), bytes=p.stat().st_size, sha256=sha(p))


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


@contextmanager
def local_checkpoint_load(torch):
    """Only trusted explicit local checkpoints: compatibility with old pickles."""
    original = torch.load
    allowed = {str(p.resolve()) for p in WEIGHTS.values()}
    def reader(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and str(Path(path).resolve()) in allowed:
            kwargs.update(map_location='cpu', weights_only=False)
        return original(path, *args, **kwargs)
    torch.load = reader
    try: yield
    finally: torch.load = original


def media_check(path, cv2):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    fps = cap.get(cv2.CAP_PROP_FPS); declared = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        assert frame.shape == (480, 832, 3), (path, frame.shape)
        count += 1
    cap.release()
    assert count == declared == 77 and math.isclose(fps, 16, abs_tol=1e-8), (path, count, declared, fps)
    return dict(decoded_frames=count, fps=fps, width=832, height=480)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--generation', type=Path, default=RD / 'generation_run.json')
    p.add_argument('--check-report', type=Path, default=RD / 'temporal_cpucheck.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--deadline-unix', type=float)
    args = p.parse_args()
    output = args.output or (args.check_report if args.check_only else RD / 'temporal_scores.json')
    assert not output.exists(), f'Preserve existing report: {output}'
    if args.check_only: assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    else: assert args.deadline_unix and args.deadline_unix > time.time(), 'Explicit wall deadline required'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(VBENCH)); sys.path.append(str(EXTRA_SITE))
    started = time.time()
    report = dict(experiment='E021', status='running', mode='CPU_check' if args.check_only else 'GPU_evaluation',
        rows=[], policy='All 4 fixed prompts x 4 deployed arms; separate diagnostics, no synthetic overall score.',
        limits=['Four development prompts and one seed, not an independent benchmark.',
                'AMT interpolation agreement is not overall video quality; static or blurry clips may score highly.',
                'RAFT dynamic_degree is the original binary movement decision, not a monotonic quality score.',
                'DINO consistency may reward static content and does not establish semantic correctness.'],
        sampling=dict(amt_input_frames=list(range(0, 77, 2)), amt_reference_frames=list(range(1, 77, 2)),
                      raft_frames=list(range(0, 77, 2)), raft_iterations=20,
                      dino_frames=list(range(77))),
        checkpoint_policy='Explicit local checkpoint paths; scoped torch.load(map_location=cpu, weights_only=False) for trusted legacy pickles. No init_submodules, network download, or upstream source change.')
    try:
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
        files = [Path(__file__), CONFIG, *WEIGHTS.values(), DINO_REPO / 'hubconf.py', DINO_REPO / 'vision_transformer.py']
        files += [VBENCH / 'vbench' / f'{n}.py' for n in ('utils', 'motion_smoothness', 'dynamic_degree', 'subject_consistency')]
        report['sources_and_models'] = [file_record(f) for f in files]
        generation = json.loads(args.generation.read_text()); manifest = generation['manifest']
        assert tuple(c['case_id'] for c in manifest['cases']) == CASES
        assert tuple(a['name'] for a in manifest['arms']) == ARMS
        assert (manifest['settings']['frames'], manifest['settings']['fps']) == (77, 16)
        report['generation'] = file_record(args.generation)
        report['generation_status'] = generation['status']
        if not args.check_only:
            assert generation['status'] == 'complete' and generation['actual_totals']['videos'] == 16
            check = json.loads(args.check_report.read_text()); assert check['status'] == 'complete' and not check['cuda_initialized']
            assert report['sources_and_models'] == check['sources_and_models'], 'Evaluator/model source changed since CPU check'
            report['cpucheck_reference'] = file_record(args.check_report)
        for arm in ARMS:
            for case in manifest['cases']:
                cid = case['case_id']; generated = generation.get('arms', {}).get(arm, {}).get('cases', {}).get(cid, {})
                video = generated.get('video')
                row = dict(case_id=cid, arm=arm, prompt=case['prompt'], seed=case['seed'], scores={})
                if video:
                    path = Path(video['path']); assert path.stat().st_size == video['bytes'] and sha(path) == video['sha256']
                    assert generated['prompt'] == case['prompt'] and generated['seed'] == case['seed']
                    row.update(video=file_record(path), media=media_check(path, cv2))
                else:
                    assert args.check_only, f'Missing video {arm}/{cid}'
                    row['video_pending'] = True
                report['rows'].append(row)
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
            for index in range(0, 16, 2):
                budget(); rows = report['rows'][index:index + 2]
                score, raw = subject_consistency(dino, [r['video']['file'] for r in rows], device, read_frame=False)
                for row, value in zip(rows, raw):
                    assert value['video_path'] == row['video']['file']
                    normalized = float(value['video_results']) / 76
                    assert math.isfinite(normalized)
                    row['scores']['subject_consistency'] = normalized
                    row['dino'] = dict(original_video_results_sum=float(value['video_results']), frame_pairs=76,
                                       normalization='Same mean over 76 terms as original sim_per_frame')
                assert math.isclose(float(score), sum(r['scores']['subject_consistency'] for r in rows) / 2, abs_tol=1e-10)
                progress()
            report['arm_means_descriptive_only'] = {a: {metric: sum(r['scores'][metric] for r in report['rows'] if r['arm'] == a) / 4
                for metric in ('motion_smoothness', 'dynamic_degree', 'subject_consistency')} for a in ARMS}
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
