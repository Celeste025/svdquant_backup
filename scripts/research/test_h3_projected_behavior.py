#!/usr/bin/env python3
"""Deterministic CPU geometry fixtures for E013; no generated-model video input."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import evaluate_h3_projected_behavior as metric


def fixture(name):
    width, height = 640, 360
    for index in range(metric.EXPECTED_FRAMES):
        t = index/(metric.EXPECTED_FRAMES-1)
        red = np.array([140.+150*t, 180.])
        blue = np.array([450., 180.])
        radius, draw_blue = 20., True
        if name == 'away': red[0] = 290.-150*t
        if name in ('static', 'translation', 'zoom', 'inflation', 'double_red', 'cropping', 'conflict'):
            red[0] = 140.
        if name == 'translation':
            red[0] += 80*t
            blue[0] += 80*t
        if name == 'zoom':
            origin = np.array([320., 180.])
            scale = 1+.1*t
            red, blue = origin+(red-origin)*scale, origin+(blue-origin)*scale
            radius *= scale
        if name == 'inflation': radius *= 1+.5*t
        if name == 'missing' and 40 <= index < 70: draw_blue = False
        if name == 'cropping': red[0] = 5.
        if name == 'jump' and index == 62: red[1] += 120.
        if name == 'conflict':
            # Distances increase while common diameters increase faster, without
            # violating the 20% size guard: ratios must be classified unknown.
            blue[0] += 15*t
            radius *= 1+.15*t
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.circle(frame, tuple(np.rint(red).astype(int)), round(radius), (255, 0, 0), -1)
        if draw_blue: cv2.circle(frame, tuple(np.rint(blue).astype(int)), round(radius), (0, 0, 255), -1)
        if name == 'double_red': cv2.circle(frame, (300, 100), 18, (255, 0, 0), -1)
        yield frame, index/24


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
        default=Path(__file__).resolve().parents[2]/'results/research/E013_readout_cpu_tests')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = [('toward', 'toward', 'behavior_pass'), ('away', 'away', 'behavior_pass'),
        ('static', 'toward', 'behavior_fail'), ('translation', 'toward', 'behavior_fail'),
        # Rasterized circle radii change in whole pixels: this shared 10% zoom
        # exceeds the fixed 1% conflict guard after endpoint medians. It must
        # remain unknown, never be scored as object separation success.
        ('zoom', 'away', 'unknown'), ('double_red', 'toward', 'unknown'),
        ('missing', 'toward', 'unknown'), ('cropping', 'toward', 'unknown'),
        ('inflation', 'toward', 'unknown'), ('jump', 'toward', 'unknown'),
        ('conflict', 'away', 'unknown')]
    summary = {'status': 'running', 'scope': 'CPU synthetic geometry, not model behavior or quality',
        'versions': {'executable': sys.executable, 'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__},
        'sources': {str(p.resolve()): metric.sha(p) for p in (Path(__file__), Path(metric.__file__))},
        'thresholds': metric.THRESHOLDS, 'cases': []}
    started = time.monotonic()
    try:
        for name, direction, expected in cases:
            directory = args.output_dir/name
            directory.mkdir()
            report, contacts = metric.evaluate_frames(fixture(name), direction)
            report['fixture_name'] = name
            report['expected_status'] = expected
            report['visuals'] = metric.write_visuals(report, contacts, directory)
            metric.save(report, directory/'report.json')
            row = {'fixture': name, 'expected': expected, 'actual': report['status'],
                   'valid_fraction': report['valid_fraction'], 'reasons': report['reasons'],
                   'report': str((directory/'report.json').resolve())}
            summary['cases'].append(row)
            metric.save(summary, args.output_dir/'summary.json')
            assert report['status'] == expected, row
            assert len(report['frames']) == 124
            assert report['requires_manual_review'] is True and report['manual_review_status'] == 'pending'
            if name in ('toward', 'away', 'static', 'translation'):
                assert report['valid_fraction'] == 1.
            if name == 'double_red':
                assert all('red_multiple_regions' in r['reasons'] for r in report['frames'])
            if name == 'jump': assert len(report['jumps']) == 2
            print(json.dumps(row), flush=True)
        # Changing the requested direction cannot convert a true toward fixture
        # into success; this is a separate falsification of the amplitude logic.
        reverse, _ = metric.evaluate_frames(fixture('toward'), 'away')
        assert reverse['status'] == 'behavior_fail'
        summary['opposite_direction_fixture'] = reverse['status']
        summary['status'] = 'complete'
    except BaseException as exc:
        summary.update(status='failed_stop', error=repr(exc))
        raise
    finally:
        summary['seconds_total'] = time.monotonic()-started
        metric.save(summary, args.output_dir/'summary.json')


if __name__ == '__main__':
    main()
