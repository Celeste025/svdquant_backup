#!/usr/bin/env python3
"""CPU-only E013 red/blue projected-separation readout with fixed guards.

Automatic pass is a numeric result, never a substitute for semantic inspection.
Thresholds have no CLI overrides and must be frozen before inspecting outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)
EXPECTED_FRAMES = 124
THRESHOLDS = {
    'red_hue': [[0, 10], [170, 179]], 'blue_hue': [[100, 130]],
    'saturation_min': 120, 'value_min': 60,
    'main_share_min': .90, 'area_fraction_range': [.002, .08],
    'valid_fraction_min': .90, 'endpoint_fraction': .10, 'endpoint_valid_min': .80,
    'rho_min': 1.5, 'diameter_relative_tolerance': .20,
    'diameter_stable_fraction_min': .90, 'jump_diagonal_fraction_max': .10,
    'toward_rho_ratio_max': .80, 'toward_distance_ratio_max': .90,
    'away_rho_ratio_min': 1.25, 'away_distance_ratio_min': 1.10,
    'direction_conflict_min_deviation': .01,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def save(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def detect_color(hsv, color):
    height, width = hsv.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for low, high in THRESHOLDS[color+'_hue']:
        part = cv2.inRange(hsv, (low, THRESHOLDS['saturation_min'], THRESHOLDS['value_min']), (high, 255, 255))
        mask = cv2.bitwise_or(mask, part)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    row = {'components': count-1, 'colored_pixels': int(np.count_nonzero(mask)),
           'centroid': None, 'diameter': None, 'area': 0, 'area_fraction': 0.,
           'main_share': 0., 'bbox': None, 'touches_border': False, 'reasons': []}
    if count == 1:
        row['reasons'] = [color+'_missing']
        return row, []
    index = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = map(int, stats[index])
    row.update(centroid=list(map(float, centers[index])), diameter=2*math.sqrt(area/math.pi),
        area=area, area_fraction=area/(height*width), main_share=area/row['colored_pixels'],
        bbox=[x, y, w, h], touches_border=x == 0 or y == 0 or x+w == width or y+h == height)
    if row['main_share'] < THRESHOLDS['main_share_min']: row['reasons'].append(color+'_multiple_regions')
    if not THRESHOLDS['area_fraction_range'][0] <= row['area_fraction'] <= THRESHOLDS['area_fraction_range'][1]:
        row['reasons'].append(color+'_area_out_of_range')
    if row['touches_border']: row['reasons'].append(color+'_touches_border')
    principal = np.asarray(labels == index, dtype=np.uint8)*255
    contours, _ = cv2.findContours(principal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return row, contours


def analyze_frame(rgb, index, timestamp=None):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected CPU uint8 RGB HxWx3 image')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    objects, contours, reasons = {}, {}, []
    for color in ('red', 'blue'):
        objects[color], contours[color] = detect_color(hsv, color)
        reasons.extend(objects[color]['reasons'])
    distance, rho, diameter_ratio = None, None, None
    if all(objects[c]['centroid'] is not None for c in objects):
        red, blue = objects['red'], objects['blue']
        distance = float(np.linalg.norm(np.asarray(red['centroid'])-blue['centroid']))
        rho = distance/((red['diameter']+blue['diameter'])/2)
        diameter_ratio = red['diameter']/blue['diameter']
        if rho < THRESHOLDS['rho_min']: reasons.append('separation_below_rho_min')
    return {'frame_index': index, 'timestamp_seconds': timestamp, 'objects': objects,
        'distance_pixels': distance, 'rho': rho, 'red_blue_diameter_ratio': diameter_ratio,
        'valid': not reasons, 'reasons': reasons}, contours


def annotate(rgb, row, contours):
    image = rgb.copy()
    for color, outline in (('red', (255, 255, 0)), ('blue', (0, 255, 255))):
        cv2.drawContours(image, contours[color], -1, outline, 2)
        obj = row['objects'][color]
        if obj['centroid'] is not None:
            center = tuple(int(round(v)) for v in obj['centroid'])
            cv2.drawMarker(image, center, outline, cv2.MARKER_CROSS, 12, 2)
            cv2.putText(image, color, (center[0]+6, center[1]-6), cv2.FONT_HERSHEY_SIMPLEX, .55, outline, 2)
    text = f'f{row["frame_index"]:03d} '+('VALID' if row['valid'] else 'INVALID')
    if row['rho'] is not None: text += f'  D={row["distance_pixels"]:.1f} rho={row["rho"]:.2f}'
    cv2.rectangle(image, (0, 0), (image.shape[1], 52), (20, 20, 20), -1)
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
    cv2.putText(image, ','.join(row['reasons'])[:110], (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 200, 200), 1)
    return image


def summarize_frames(rows, width, height, expected_direction):
    n = len(rows)
    result = {'expected_direction': expected_direction, 'frame_count': n, 'width': width, 'height': height,
              'status': 'unknown', 'reasons': [], 'requires_manual_review': True,
              'manual_review_status': 'pending', 'frames': rows}
    if n != EXPECTED_FRAMES: result['reasons'].append('frame_count_not_124')
    if n == 0:
        result['reasons'].append('no_frames')
        return result
    valid = np.array([r['valid'] for r in rows])
    endpoint_n = max(1, math.ceil(n*THRESHOLDS['endpoint_fraction']))
    result['detection_valid_fraction'] = float(valid.mean())
    result['valid_fraction'] = float(valid.mean())
    result['endpoint_frames'] = endpoint_n
    result['endpoint_valid_fraction'] = {'start': float(valid[:endpoint_n].mean()), 'end': float(valid[-endpoint_n:].mean())}
    if result['valid_fraction'] < THRESHOLDS['valid_fraction_min']: result['reasons'].append('insufficient_valid_frames')
    if any(v < THRESHOLDS['endpoint_valid_min'] for v in result['endpoint_valid_fraction'].values()):
        result['reasons'].append('insufficient_valid_endpoint_frames')
    # Consecutive detected centers are checked even if another guard invalidated
    # either frame. Missing frames are never interpolated to invent a trajectory.
    jump_limit = math.hypot(width, height)*THRESHOLDS['jump_diagonal_fraction_max']
    jumps = []
    for prev, current in zip(rows, rows[1:]):
        for color in ('red', 'blue'):
            a, b = prev['objects'][color]['centroid'], current['objects'][color]['centroid']
            if a is not None and b is not None:
                displacement = float(np.linalg.norm(np.asarray(b)-a))
                if displacement > jump_limit:
                    jumps.append({'from_frame': prev['frame_index'], 'to_frame': current['frame_index'],
                                  'color': color, 'displacement_pixels': displacement})
    result['jumps'], result['jump_limit_pixels'] = jumps, jump_limit
    if jumps: result['reasons'].append('centroid_jump')
    start_rows = [r for r in rows[:endpoint_n] if r['valid']]
    end_rows = [r for r in rows[-endpoint_n:] if r['valid']]
    result['diameter_stability'] = {}
    if start_rows:
        getters = {'red_diameter': lambda r: r['objects']['red']['diameter'],
                   'blue_diameter': lambda r: r['objects']['blue']['diameter'],
                   'red_blue_diameter_ratio': lambda r: r['red_blue_diameter_ratio']}
        for name, getter in getters.items():
            baseline = float(np.median([getter(r) for r in start_rows]))
            stable = []
            for row in rows:
                value = getter(row)
                okay = bool(row['valid'] and value is not None and
                    abs(value/baseline-1) <= THRESHOLDS['diameter_relative_tolerance'])
                stable.append(okay)
                row.setdefault('size_guards', {})[name] = okay
                if row['valid'] and not okay: row['reasons'].append(name+'_unstable')
            fraction = sum(stable)/n
            result['diameter_stability'][name] = {'first_window_median': baseline, 'stable_fraction': fraction}
            if fraction < THRESHOLDS['diameter_stable_fraction_min']:
                result['reasons'].append(name+'_unstable')
    else:
        result['reasons'].append('no_valid_start_reference')
    for row in rows:
        row['detection_valid'] = row.pop('valid')
        row['valid'] = row['detection_valid'] and all(row.get('size_guards', {}).values())
    final_valid = np.array([r['valid'] for r in rows])
    result['valid_fraction'] = float(final_valid.mean())
    result['endpoint_valid_fraction'] = {'start': float(final_valid[:endpoint_n].mean()),
                                        'end': float(final_valid[-endpoint_n:].mean())}
    if result['valid_fraction'] < THRESHOLDS['valid_fraction_min']:
        result['reasons'].append('insufficient_valid_frames_after_size_guards')
    if any(v < THRESHOLDS['endpoint_valid_min'] for v in result['endpoint_valid_fraction'].values()):
        result['reasons'].append('insufficient_valid_endpoint_frames_after_size_guards')
    start_rows = [r for r in rows[:endpoint_n] if r['valid']]
    end_rows = [r for r in rows[-endpoint_n:] if r['valid']]
    result['endpoint_metrics'] = None
    if start_rows and end_rows:
        values = {key: {'start': float(np.median([r[key] for r in start_rows])),
                        'end': float(np.median([r[key] for r in end_rows]))} for key in ('rho', 'distance_pixels')}
        for value in values.values(): value['ratio'] = value['end']/value['start']
        result['endpoint_metrics'] = values
        rr, dr = values['rho']['ratio'], values['distance_pixels']['ratio']
        minimum = THRESHOLDS['direction_conflict_min_deviation']
        if (rr-1)*(dr-1) < 0 and abs(rr-1) >= minimum and abs(dr-1) >= minimum:
            result['reasons'].append('rho_and_distance_direction_conflict')
        amplitude_pass = ((rr <= THRESHOLDS['toward_rho_ratio_max'] and dr <= THRESHOLDS['toward_distance_ratio_max'])
            if expected_direction == 'toward' else
            (rr >= THRESHOLDS['away_rho_ratio_min'] and dr >= THRESHOLDS['away_distance_ratio_min']))
        result['amplitude_pass'] = amplitude_pass
        if not result['reasons']:
            result['status'] = 'behavior_pass' if amplitude_pass else 'behavior_fail'
            if not amplitude_pass: result['reasons'].append('valid_detection_but_direction_or_amplitude_failed')
    else:
        result['reasons'].append('endpoint_metric_unavailable')
    return result


def evaluate_frames(frames, expected_direction):
    if expected_direction not in ('toward', 'away'): raise ValueError('Direction must be toward or away')
    rows, contacts, width, height = [], {}, None, None
    selected = set(np.linspace(0, EXPECTED_FRAMES-1, 12, dtype=int).tolist())
    for index, item in enumerate(frames):
        rgb, timestamp = item if isinstance(item, tuple) else (item, None)
        h, w = rgb.shape[:2]
        if width is None: width, height = w, h
        if (w, h) != (width, height): raise ValueError('Frame dimensions changed')
        row, contours = analyze_frame(rgb, index, timestamp)
        rows.append(row)
        if index in selected: contacts[index] = (rgb.copy(), contours)
    report = summarize_frames(rows, width or 0, height or 0, expected_direction)
    contacts = {i: annotate(rgb, report['frames'][i], contours) for i, (rgb, contours) in contacts.items()}
    report['thresholds'] = THRESHOLDS
    report['scope'] = 'Projected red/blue region separation, not 3D distance, exclusive red motion, video quality, or confirmed object identity'
    return report, contacts


def write_visuals(report, contacts, directory):
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tile_width = 384
    tile_height = round(tile_width*report['height']/report['width'])
    sheet = Image.new('RGB', (tile_width*3, tile_height*4+48), 'white')
    ImageDraw.Draw(sheet).text((8, 8), f'{report["status"]}; expected={report["expected_direction"]}; manual review REQUIRED', fill='black')
    for slot, (index, array) in enumerate(sorted(contacts.items())):
        tile = Image.fromarray(array).resize((tile_width, tile_height))
        sheet.paste(tile, ((slot%3)*tile_width, 48+(slot//3)*tile_height))
    contact_path = directory/'contactsheet.png'
    sheet.save(contact_path)
    rows = report['frames']
    x = [r['frame_index'] for r in rows]
    number = lambda value: np.nan if value is None else value
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    axes[0].plot(x, [number(r['distance_pixels']) for r in rows], label='Center distance D (pixels)')
    axes[1].plot(x, [number(r['rho']) for r in rows], label='D / mean equivalent diameter')
    for color in ('red', 'blue'):
        axes[2].plot(x, [number(r['objects'][color]['diameter']) for r in rows], color=color, label=f'{color} diameter')
        axes[3].plot(x, [np.nan if r['objects'][color]['centroid'] is None else r['objects'][color]['centroid'][0] for r in rows], color=color, label=f'{color} centroid x')
        axes[3].plot(x, [np.nan if r['objects'][color]['centroid'] is None else r['objects'][color]['centroid'][1] for r in rows], color=color, ls='--', label=f'{color} centroid y')
    for axis in axes:
        for row in rows:
            if not row['valid']: axis.axvspan(row['frame_index']-.5, row['frame_index']+.5, color='gray', alpha=.12)
        axis.legend(loc='best'); axis.grid(alpha=.25)
    axes[-1].set_xlabel('Decoded frame index; gray = invalid detection/size guard')
    fig.suptitle(f'{report["status"]}; projected behavior only; manual inspection required')
    plot_path = directory/'trajectory.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return {'contactsheet': {'path': str(contact_path.resolve()), 'sha256': sha(contact_path)},
            'trajectory_plot': {'path': str(plot_path.resolve()), 'sha256': sha(plot_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--case-json', type=Path)
    parser.add_argument('--expected-direction', required=True, choices=('toward', 'away'))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir/'report.json'
    report = {'status': 'failed_stop', 'requires_manual_review': True, 'thresholds': THRESHOLDS,
        'source': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}
    started = time.monotonic()
    try:
        import av
        import PIL
        case = json.loads(args.case_json.read_text()) if args.case_json else None
        video = Path(case['video']) if case else args.video
        video = video.resolve()
        if case and case.get('direction', case.get('expected_direction', args.expected_direction)) != args.expected_direction:
            raise RuntimeError('Case direction differs from explicit expected direction')
        video_sha = sha(video)
        if case and case.get('video_sha256', video_sha) != video_sha: raise RuntimeError('Video SHA changed')
        report.update(video=str(video), video_sha256=video_sha, case=case,
            case_json=None if args.case_json is None else {'path': str(args.case_json.resolve()), 'sha256': sha(args.case_json)},
            versions={'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
                      'av': av.__version__, 'pillow': PIL.__version__})
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            report['container'] = {'fps': float(stream.average_rate), 'width': stream.width, 'height': stream.height}
            frames = ((frame.to_ndarray(format='rgb24'), None if frame.time is None else float(frame.time))
                      for frame in container.decode(video=0))
            metrics, contacts = evaluate_frames(frames, args.expected_direction)
        report.update(metrics)
        save(report, result_path)
        if report['frame_count']:
            report['visuals'] = write_visuals(report, contacts, args.output_dir)
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        save(report, result_path)
        print(json.dumps({'status': report['status'], 'report': str(result_path.resolve())}), flush=True)


if __name__ == '__main__':
    main()
