#!/usr/bin/env python3
"""CPU E022 video readout: fixed 8 prompts x 2 seeds, all phase arms.

Reuse E021 scalar math; verify saved arrays and media identities without model
inference, decoding again, training-weight hashes, or a new composite score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from summarize_wan_qad_videos import (
    ROOT, INDICES, ASPECTS, CRITERIA, require, sha, finite, close, mean, difference,
)

RD = ROOT / 'results/research/E022'
BASELINE_ARMS = ('bf16', 'plain_step0000', 'svd_lr')
SELECTED = 'qad_native_dev_selected'
SAMPLING = dict(amt_input_frames=list(range(0, 77, 2)), amt_reference_frames=list(range(1, 77, 2)),
                raft_frames=list(range(0, 77, 2)), raft_iterations=20, dino_frames=list(range(77)))
PRIMARY = ['motion_smoothness', 'dynamic_degree', 'subject_consistency',
           'mj_total', 'mj_alignment', 'mj_fineness', 'mj_coherence_consistency']


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def recompute(row, mj, label):
    errors = row['amt']['interpolation_reconstruction_absdiff']
    flows = row['raft']['top5percent_mean_flow_magnitudes']
    require(len(errors) == len(flows) == 38 and all(finite(x) for x in errors + flows), f'{label}: AMT/RAFT arrays')
    smooth = 1 - mean(errors) / 255
    close(smooth, row['scores']['motion_smoothness'], f'{label}: AMT', atol=1e-7)
    threshold, required = row['raft']['threshold'], row['raft']['required_count']
    require(threshold == 11.25 and required == 10, f'{label}: original RAFT decision parameters')
    count = sum(v > threshold for v in flows)
    dynamic = count >= required
    require(count == row['raft']['above_threshold_count'] and isinstance(row['scores']['dynamic_degree'], bool)
            and dynamic == row['scores']['dynamic_degree'], f'{label}: RAFT decision')
    dino_sum, pairs = row['dino']['original_video_results_sum'], row['dino']['frame_pairs']
    require(finite(dino_sum) and pairs == 76, f'{label}: DINO sum/pairs')
    consistency = dino_sum / pairs
    close(consistency, row['scores']['subject_consistency'], f'{label}: DINO')
    require(set(mj['criteria']) == set(CRITERIA) and set(mj['aspects']) == set(ASPECTS), f'{label}: MJ keys')
    require(all(finite(x) for x in [mj['score'], *mj['criteria'].values(), *mj['aspects'].values()]), f'{label}: finite MJ')
    require(mj['sampled_indices'] == [INDICES] and mj['patches'] == [1] * 8, f'{label}: actual MJ frames/patches')
    metrics = dict(motion_smoothness=smooth, dynamic_degree=dynamic, subject_consistency=consistency,
                   amt_mean_absdiff_0_255=mean(errors), raft_mean_top5percent_flow=mean(flows),
                   raft_above_threshold_count=count, mj_total=mj['score'], mj_alignment=mj['aspects']['alignment'],
                   mj_fineness=mj['aspects']['fineness'], mj_coherence_consistency=mj['aspects']['coherence_consistency'])
    return dict(metrics=metrics,
                temporal_raw=dict(amt=errors, raft=flows, raft_threshold=threshold, raft_required_count=required,
                                  dino_saved_sum=dino_sum, dino_terms=pairs, dino_per_frame_array_available=False),
                mjvideo_raw=dict(total=mj['score'], aspects=mj['aspects'], criteria=mj['criteria'],
                                 sampled_indices=mj['sampled_indices'], patches=mj['patches']))


def average_metrics(entries):
    keys = set(entries[0])
    require(all(set(e) == keys for e in entries), 'Metric keys differ')
    return {k: mean([e[k] for e in entries]) for k in entries[0]}


def add_differences(arms):
    pairs = dict(svd_minus_plain='svd_lr')
    if SELECTED in arms:
        pairs = dict(qad_minus_plain=SELECTED, **pairs)
    return {name: difference(arms[arm]['metrics'], arms['plain_step0000']['metrics']) for name, arm in pairs.items()}


def summarize(args):
    references = {}
    def load(name):
        path = args.report_dir / name
        value = json.loads(path.read_text())
        references[name] = record(path)
        return value

    manifest = load('video_test_manifest.json')
    cases, trajectories = manifest['cases'], manifest['trajectories']
    tids = [t['trajectory_id'] for t in trajectories]
    prompts = {c['prompt_id']: c for c in cases}
    require(len(prompts) == len(cases) == 8 and len(set(tids)) == len(tids) == 16, 'Expected fixed 8x2 protocol')
    expected_tids = []
    for case in cases:
        rows = [t for t in trajectories if t['prompt_id'] == case['prompt_id']]
        require([t['replica'] for t in rows] == [0, 1] and [t['seed'] for t in rows] == case['seeds'], 'Replica/seed mapping')
        expected_tids += [t['trajectory_id'] for t in rows]
    require(tids == expected_tids, 'Trajectory ordering changed')
    phases = ('baselines',) if args.phase == 'baselines' else ('baselines', 'selected')
    arms = BASELINE_ARMS if args.phase == 'baselines' else (*BASELINE_ARMS, SELECTED)
    result = dict(experiment='E022_video_readout', phase=args.phase, status='running', cpu_only=True,
                  primary_statistics=PRIMARY, per_trajectory={}, per_prompt={}, evaluator_runtime={},
                  protocol=dict(prompt_order=list(prompts), trajectory_order=tids, arm_order=list(arms),
                                temporal_sampling=SAMPLING, mj_sampled_indices=INDICES),
                  policy='All rows retained. First average the two seeds within each prompt, then equally average eight prompts. No composite score, confidence interval, or sample filtering.',
                  limits=['Sixteen trajectories are grouped within eight fixed local held-out prompts, not independent broad-benchmark samples.',
                          'AMT may reward static or blurry clips; original binary RAFT dynamic_degree is not a monotonic quality score.',
                          'DINO only saves the sum of 76 terms; sum/76 is verified without inferring semantics or unavailable per-frame similarities.',
                          'MJ uses eight frames ending at frame66; all 28 criteria and 5 aspects remain raw. Its learned total is not an arithmetic mean of them.',
                          'All current videos use the shared BF16 VAE configuration, whose numerical behavior is under investigation after visible BF16-arm stripes. The root cause is unconfirmed; these scores cannot establish quantization quality effects. Original media and every case remain retained.',
                          'SVD versus plain compares entire recipes, not the isolated effect of low-rank branches.',
                          'QAD checkpoint selection uses development native NMSE, not these videos or an asserted quality optimum.'])
    for t in trajectories:
        result['per_trajectory'][t['trajectory_id']] = dict(prompt_id=t['prompt_id'], replica=t['replica'], seed=t['seed'],
                                                          prompt=prompts[t['prompt_id']]['prompt'], arms={})
    base = None
    checked = 0
    for phase in phases:
        gen = load(f'video_{phase}.json'); valid = load(f'video_{phase}_validation.json')
        temporal = load(f'temporal_{phase}.json'); mj = load(f'mjvideo_{phase}.json')
        require(gen['status'] == valid['status'] == temporal['status'] == 'complete', f'{phase}: incomplete upstream')
        require(gen['phase'] == valid['phase'] == temporal['phase'] == phase and valid['cuda_initialized'] is False, 'Phase/CPU validation')
        gh = references[f'video_{phase}.json']['sha256']
        require(valid['generation']['sha256'] == temporal['generation']['sha256'] == gh, 'Generation identity differs')
        require(gen['manifest'] == manifest and gen['manifest_reference']['sha256'] == references['video_test_manifest.json']['sha256'], 'Manifest differs')
        phase_arms = BASELINE_ARMS if phase == 'baselines' else (SELECTED,)
        total = dict(dit_calls=192, native_mm_calls=38400, sdpa_calls=11520, videos=48) if phase == 'baselines' else dict(dit_calls=64, native_mm_calls=19200, sdpa_calls=3840, videos=16)
        require(gen['actual_totals'] == valid['recomputed_totals'] == total, 'Validated execution counts differ')
        for metric in ('temporal', 'mjvideo'):
            launch = load(f'{metric}_{phase}_launcher.json')
            require(launch['status'] == 'complete' and launch['returncode'] == 0, f'{metric}/{phase}: launcher incomplete')
            require(launch['result_sha256'] == references[f'{metric}_{phase}.json']['sha256'] and launch['generation_sha256'] == gh
                    and launch['validation_sha256'] == references[f'video_{phase}_validation.json']['sha256'], 'Scoring source binding')
        if phase == 'baselines':
            base = gen
            reference_mj_runtime = mj['runtime']
            reference_temporal_sources = temporal['sources_and_models']
        else:
            require(gen['shared_inputs'] == base['shared_inputs'] and gen['schedule'] == base['schedule'], 'Selected phase changed paired inputs')
            require(valid['selected_shared_inputs_identical_to_baselines'] is True and valid['baseline_reference']['sha256'] == references['video_baselines.json']['sha256'], 'Selected CPU paired-input validation')
            require(gen['checkpoint_binding']['baseline_report']['sha256'] == references['video_baselines.json']['sha256'], 'Selected checkpoint baseline binding')
            require(mj['runtime'] == reference_mj_runtime and temporal['sources_and_models'] == reference_temporal_sources, 'Evaluator implementation/runtime changed across phases')
            result['selected_checkpoint_binding'] = gen['checkpoint_binding']
        require(temporal['sampling'] == SAMPLING and mj['num_segments'] == 8 and set(mj['results']) == set(tids), 'Sampling/MJ trajectory set')
        keyed = {(r['trajectory_id'], r['arm']): r for r in temporal['rows']}
        require(len(keyed) == len(temporal['rows']) == total['videos'] and set(keyed) == {(t, a) for t in tids for a in phase_arms}, 'Temporal row set')
        validation_files = {r['path']: r for r in valid['files']}
        result['evaluator_runtime'][phase] = dict(mjvideo=mj['runtime'], temporal_seconds=temporal.get('seconds'))
        for t in trajectories:
            tid = t['trajectory_id']; entry = result['per_trajectory'][tid]
            require(mj['results'][tid]['prompt'] == entry['prompt'] and set(mj['results'][tid]['variants']) == set(phase_arms), f'{tid}: MJ identity')
            for arm in phase_arms:
                generated = gen['arms'][arm]['trajectories'][tid]; row = keyed[(tid, arm)]; mr = mj['results'][tid]['variants'][arm]
                require(generated['status'] == 'complete' and all(generated[k] == v for k, v in t.items()), f'{tid}/{arm}: generation identity')
                require(all(row[k] == entry[k] for k in ('prompt_id', 'replica', 'seed', 'prompt')) and generated['prompt'] == entry['prompt'], 'Temporal identity differs')
                video = generated['video']; path = Path(video['path']); actual = record(path)
                require(actual['bytes'] == video['bytes'] and actual['sha256'] == video['sha256'] == row['video']['sha256'] == mr['video_sha256'] == validation_files[str(path)]['sha256'], f'{tid}/{arm}: media SHA/size')
                require(row['video']['file'] == mr['video'] == str(path), 'Media path differs')
                require(row['media'] == dict(decoded_frames=77, fps=16, width=832, height=480), 'Temporal media geometry')
                require(valid['arms'][arm][tid]['video_sha256'] == actual['sha256'], 'Validated video differs')
                entry['arms'][arm] = dict(video=str(path), video_sha256=actual['sha256'], **recompute(row, mr, f'{tid}/{arm}'))
                checked += 1
    for entry in result['per_trajectory'].values():
        entry['paired_differences'] = add_differences(entry['arms'])
    for pid, case in prompts.items():
        ids = [t['trajectory_id'] for t in trajectories if t['prompt_id'] == pid]
        prompt_arms = {a: dict(metrics=average_metrics([result['per_trajectory'][t]['arms'][a]['metrics'] for t in ids])) for a in arms}
        result['per_prompt'][pid] = dict(prompt=case['prompt'], trajectory_ids=ids, seeds=case['seeds'], arms=prompt_arms,
                                        paired_differences=add_differences(prompt_arms))
    overall = {a: dict(metrics=average_metrics([result['per_prompt'][p]['arms'][a]['metrics'] for p in prompts])) for a in arms}
    result['arm_means_descriptive_only'] = {a: overall[a]['metrics'] for a in arms}
    result['paired_mean_differences'] = add_differences(overall)
    for name, metrics in result['paired_mean_differences'].items():
        independently_averaged = average_metrics([result['per_prompt'][p]['paired_differences'][name] for p in prompts])
        for key, value in metrics.items():
            close(value, independently_averaged[key], f'{name}/{key}: paired hierarchy', atol=1e-7)
    result['dynamic_trajectory_counts'] = {a: dict(dynamic=sum(result['per_trajectory'][t]['arms'][a]['metrics']['dynamic_degree'] for t in tids), total=16) for a in arms}
    result.update(status='complete', input_reports=references, verification=dict(media_files_checked=checked, prompts=8, trajectories=16,
                  mj_rows=checked, criteria_per_mj_row=28, aspects_per_mj_row=5, all_recomputed_readouts_match=True,
                  inherited_full_decode=True, no_model_weights_or_source_trees_rehashed=True))
    require('torch' not in sys.modules, 'Summary must not import Torch/CUDA')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('baselines', 'complete'), required=True)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or args.report_dir / ('video_baselines_summary.json' if args.phase == 'baselines' else 'video_summary.json')
    require(not args.output.exists(), f'Preserve existing summary: {args.output}')
    sources = [record(Path(__file__)), record(Path(__file__).with_name('summarize_wan_qad_videos.py'))]
    try:
        result = summarize(args)
    except BaseException:
        failed = args.output.with_name(args.output.stem + '_failed.json')
        require(not failed.exists(), f'Preserve prior failure: {failed}')
        failed.write_text(json.dumps(dict(status='failed_stop', phase=args.phase, sources=sources, error=traceback.format_exc()), indent=2) + '\n')
        raise
    result['sources'] = sources
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output), sha256=sha(args.output),
                         arm_means=result['arm_means_descriptive_only'], paired_mean_differences=result['paired_mean_differences']), indent=2))


if __name__ == '__main__':
    main()
