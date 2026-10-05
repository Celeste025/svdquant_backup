#!/usr/bin/env python3
"""E042 fixed CPU PCM readout; no peak detector, offset fitting, or semantic score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'research_state/06_experiments/E042_existing_audio_readout_plan.md'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E042')
OUTPUT = ROOT / 'results/research/E042/readout.json'
ARMS = ('bf16', 'global_mean', 'coarse16')


def record(path):
    path = Path(path)
    return dict(file=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def normalize_visual(document, arm):
    """Map the two existing E041 schemas; preserve unknown/contact uncertainty."""
    arms = document['arms']
    row = next(x for x in arms if x['arm'] == arm) if isinstance(arms, list) else arms[arm]
    source_events = row.get('cycles', row.get('events', []))
    events = []
    for event in source_events:
        lo, hi = event.get('closure_appearance_interval', event.get('closure_interval'))
        assert 0 <= lo <= hi < 124 and event['contact'] == 'uncertain'
        events.append(dict(frames=[lo, hi], seconds=[lo / 24, (hi + 1) / 24],
                           open_before_frame=event.get('open_before', event.get('open_before_frame')),
                           open_after_frame=event.get('open_after', event.get('open_after_frame')),
                           contact=event['contact']))
    unknown = list(row.get('unreadable_for_complete_two_hand_event_count', row.get('uncertain_intervals', [])))
    if 'incomplete_tail' in row:
        unknown.append(row['incomplete_tail'])
    unknown = [dict(frames=x['frames'], seconds=[x['frames'][0]/24, (x['frames'][1]+1)/24],
                    reason=x['reason']) for x in unknown]
    return dict(source_video=row['source_video'], closure_intervals=events,
                uncountable_or_incomplete=unknown,
                missing_annotation_means='unknown, never absence of contact')


def read_curves(waveform, sample_rate):
    x = waveform.astype(np.float64)
    hop, rw, hw = 64, 320, 640
    rms_windows = np.lib.stride_tricks.sliding_window_view(x, rw, axis=-1)[:, ::hop, :]
    rms = np.sqrt(np.mean(rms_windows * rms_windows, axis=(0, 2)))
    rms_time = (np.arange(rms.size) * hop + (rw-1)/2) / sample_rate
    hf_windows = np.lib.stride_tricks.sliding_window_view(x, hw, axis=-1)[:, ::hop, :]
    spectrum = np.fft.rfft(hf_windows * np.hanning(hw), axis=-1)
    frequencies = np.fft.rfftfreq(hw, 1/sample_rate)
    selected = frequencies >= 1000
    # Exact protocol: mean over stereo channels and selected rFFT bins, no PSD scaling.
    hf = np.sqrt(np.mean(np.abs(spectrum[..., selected])**2, axis=(0, 2)))
    hf_time = (np.arange(hf.size) * hop + (hw-1)/2) / sample_rate
    rise = np.maximum(np.diff(hf, prepend=hf[0]), 0.)
    arrays = dict(waveform=waveform, sample_rate=np.int64(sample_rate),
                  rms_time_seconds=rms_time, rms=rms,
                  hf_time_seconds=hf_time, hf_sqrt_mean_power=hf,
                  hf_positive_first_difference=rise,
                  hf_selected_frequencies_hz=frequencies[selected])
    maxima = dict(rms=float(rms.max()), hf_sqrt_mean_power=float(hf.max()),
                  hf_positive_first_difference=float(rise.max()))
    for name, maximum in maxima.items():
        arrays[name+'_normalized'] = arrays[name]/maximum if maximum > 0 else np.zeros_like(arrays[name])
    assert all(np.isfinite(a).all() for a in arrays.values())
    return arrays, maxima


def plot_seed(rows, arrays, destination, limit):
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
    for axis, row in zip(axes, rows, strict=True):
        a = arrays[row['arm']]
        for item in row['visual']['uncountable_or_incomplete']:
            axis.axvspan(*item['seconds'], color='#9d9d9d', alpha=.20, zorder=0)
        for item in row['visual']['closure_intervals']:
            axis.axvspan(*item['seconds'], color='#2ca02c', alpha=.13, zorder=0)
        axis.plot(a['rms_time_seconds'], a['rms_normalized'], label='Stereo RMS (10ms)', lw=1.15, color='#1f77b4')
        axis.plot(a['hf_time_seconds'], a['hf_sqrt_mean_power_normalized'], label='HF energy (20ms Hann, >=1kHz)', lw=.9, color='#ff7f0e')
        axis.plot(a['hf_time_seconds'], a['hf_positive_first_difference_normalized'], label='Positive HF difference', lw=.75, alpha=.8, color='#9467bd')
        axis.set_title(f"{row['case_id']} / {row['arm']} / own E041 closure intervals: {len(row['visual']['closure_intervals'])}", loc='left', fontsize=11)
        axis.set_ylim(-.02, 1.06); axis.set_xlim(*limit); axis.set_ylabel('Own-max normalized')
        axis.grid(axis='x', alpha=.22)
    axes[-1].set_xlabel('Time from original zero start (seconds)')
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Patch(facecolor='#2ca02c', alpha=.25), Patch(facecolor='#9d9d9d', alpha=.35)]
    labels += ['E041 closure appearance (contact uncertain)', 'E041 uncountable / incomplete']
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('E042: fixed signal readout, not detected claps or synchronization scores\nEach curve uses its OWN full-track maximum; do not compare loudness. Unannotated does not mean no contact.', fontsize=12)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA explicitly'
    assert not torch.cuda.is_initialized()
    if args.output.exists() or args.data_dir.exists():
        raise FileExistsError('Refusing to overwrite an executed readout or artifact directory')
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=False)
    report = dict(experiment='E042', status='running', cpu_only=True, model_calls=0, gpu_calls=0,
                  source=record(__file__), plan=record(PLAN), cases=[], plots=[],
                  versions=dict(torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__),
                  recipe=dict(rms_window_samples=320, hf_window_samples=640, hop_samples=64,
                              padding=False, hann='numpy.hanning: symmetric, endpoints zero',
                              fft='numpy unnormalized rFFT', hf_band_hz=[1000,16000],
                              hf_reduction='sqrt(mean(abs(rFFT(x * Hann))^2)) over both channels and selected frequency bins',
                              positive_difference='max(hf[t]-hf[t-1],0), first entry0; no division by hop',
                              timestamps='(window_start + (window_samples-1)/2)/32000',
                              normalization='Each curve divided by its own entire-track maximum; zero stays zero',
                              peak_detection=False, fitted_time_offset=False, semantic_audio_labels=False),
                  limitations='Unheard PCM energy structures, not clap detections, contact evidence, loudness comparison, or sync scores.')
    try:
        documents = {}
        for replica in (0, 1):
            p = ROOT/f'results/research/E041/replica{replica}_observations.json'
            documents[replica] = json.loads(p.read_text())
        report['visual_sources'] = [record(ROOT/f'results/research/E041/replica{r}_observations.json') for r in (0,1)]
        decode = {a: json.loads((ROOT/f'results/research/E038/decode_{a}.json').read_text()) for a in ARMS}
        report['decode_sources'] = [record(ROOT/f'results/research/E038/decode_{a}.json') for a in ARMS]
        for replica in (0, 1):
            cache, seed_rows = {}, []
            for arm in ARMS:
                case_id = f'vbench0161_r{replica}'
                row = next(x for x in decode[arm]['cases'] if x['case_id'] == case_id)
                pcm_file = record(row['audio_pcm_file']['file'])
                assert pcm_file == row['audio_pcm_file'] and row['status'] == 'complete'
                payload = torch.load(pcm_file['file'], map_location='cpu', weights_only=False)
                x = payload['waveform']
                assert set(payload) == {'waveform','sample_rate'} and payload['sample_rate'] == 32000
                assert x.dtype == torch.float32 and tuple(x.shape) == (2,165600) and bool(x.isfinite().all())
                waveform = x.contiguous().numpy()
                assert hashlib.sha256(waveform.tobytes()).hexdigest() == row['audio_pcm']['sha256']
                visual = normalize_visual(documents[replica], arm)
                assert visual['source_video'] == row['video']
                a, maxima = read_curves(waveform, 32000)
                target = args.data_dir/f'{case_id}_{arm}.npz'
                np.savez_compressed(target, **a)
                item = dict(case_id=case_id, replica=replica, arm=arm, seed=row['seed'],
                            pcm_file=pcm_file, pcm_tensor=row['audio_pcm'], source_video=row['video'],
                            duration_seconds=165600/32000, video_duration_seconds=124/24,
                            visual=visual, arrays=record(target), maxima=maxima,
                            rms_windows=len(a['rms']), hf_windows=len(a['hf_sqrt_mean_power']),
                            array_shapes={k:list(v.shape) for k,v in a.items()}, finite=True)
                report['cases'].append(item); seed_rows.append(item); cache[arm]=a
            for label, limit in [('full',(0,165600/32000)),('zoom_0_1p4',(0,1.4))]:
                p = args.data_dir/f'vbench0161_r{replica}_{label}.png'
                plot_seed(seed_rows, cache, p, limit)
                report['plots'].append(dict(replica=replica, range_seconds=list(limit), artifact=record(p)))
            print(f'Completed replica {replica}: 3 PCM arrays and 2 figures', flush=True)
        assert len(report['cases']) == 6 and len(report['plots']) == 4
        assert not torch.cuda.is_initialized()
        report.update(status='complete', cuda_initialized=False, seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(status='failed_stop', error=repr(error), traceback=traceback.format_exc(),
                      cuda_initialized=torch.cuda.is_initialized(), seconds=time.monotonic()-started)
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
