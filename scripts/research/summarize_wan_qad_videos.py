#!/usr/bin/env python3
"""CPU-only E021 summary of all four cases, four arms and separate readouts.

Rehash only the sixteen videos and four small input reports. Recompute AMT and
RAFT from saved arrays, DINO from its saved sum, and retain all MJ raw scores.
No evaluator inference, source-tree freeze, sample filtering, or overall index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E021'
ARMS = ('bf16', 'packed_step0000', 'packed_step0064', 'svd_lr')
CASES = ('vbench_066', 'vbench_091', 'vbench_182', 'vbench_067')
INDICES = [0, 9, 19, 28, 38, 47, 57, 66]
ASPECTS = ('alignment', 'safety', 'fineness', 'coherence_consistency', 'bias_fairness')
CRITERIA = (
    'object', 'attribute', 'actions', 'count', 'location',
    'crime', 'shocking', 'disgust', 'nsfw_evasive', 'nsfw_subtle', 'political_sensitivity',
    'human_face_distortion', 'human_limb_distortion', 'object_distortion', 'defocused_blur', 'motion_blur',
    'spatial_consistency', 'action_continuity', 'object_disappearance', 'abrupt_background_changes',
    'inconsistent_lighting_shadows', 'frame_flickering', 'object_drift',
    'race_bias', 'age_bias', 'education_bias', 'job_bias', 'gender_bias',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, label, atol=1e-10):
    require(finite(actual) and finite(expected) and math.isclose(actual, expected, rel_tol=0, abs_tol=atol),
            f'{label}: recomputed {actual} != recorded {expected}')


def mean(values):
    return sum(values)/len(values)


def difference(left, right):
    require(set(left) == set(right), 'Metric keys differ')
    return {k: float(left[k])-float(right[k]) for k in left}


def summarize(args):
    paths = dict(generation=args.generation, generation_validation=args.validation,
                 temporal=args.temporal, mjvideo=args.mjvideo)
    inputs = {k: json.loads(p.read_text()) for k, p in paths.items()}
    hashes = {k: sha(p) for k, p in paths.items()}
    gen, valid, temporal, mj = (inputs[k] for k in paths)
    require(gen['status'] == valid['status'] == temporal['status'] == 'complete', 'Incomplete upstream report')
    require(not valid['failures'] and valid['cuda_initialized'] is False, 'Generation CPU validation did not pass')
    require(valid['report']['sha256'] == temporal['generation']['sha256'] == hashes['generation'], 'Generation report identity mismatch')
    require(gen['actual_totals'] == valid['recomputed_totals'] ==
            dict(native_mm_calls=14400, sdpa_calls=3840, dit_calls=64, videos=16), 'Generation totals changed')
    require(tuple(c['case_id'] for c in gen['manifest']['cases']) == CASES, 'Case manifest changed')
    require(tuple(a['name'] for a in gen['manifest']['arms']) == ARMS, 'Arm manifest changed')
    require(set(mj['results']) == set(CASES) and mj['num_segments'] == 8, 'MJ needs all four fixed cases and eight segments')
    expected_sampling = dict(amt_input_frames=list(range(0, 77, 2)), amt_reference_frames=list(range(1, 77, 2)),
                            raft_frames=list(range(0, 77, 2)), raft_iterations=20, dino_frames=list(range(77)))
    require(temporal['sampling'] == expected_sampling, 'Temporal sampling changed')
    keyed = {(r['case_id'], r['arm']): r for r in temporal['rows']}
    expected_keys = {(c, a) for c in CASES for a in ARMS}
    require(len(temporal['rows']) == len(keyed) == 16 and set(keyed) == expected_keys, 'Temporal rows incomplete or duplicated')
    validation_files = {r['path']: r for r in valid['files']}
    result = dict(experiment='E021', status='complete', cpu_only=True,
        input_reports={k: dict(path=str(paths[k]), sha256=hashes[k]) for k in paths},
        policy='All four predetermined development cases retained. Equal-weight descriptive means; no composite quality score or improvement claim.',
        protocol=dict(case_order=list(CASES), arm_order=list(ARMS), seed=20261004,
                      temporal_sampling=expected_sampling, mj_sampled_indices=INDICES,
                      mj_primary_statistics=['total', 'alignment', 'fineness', 'coherence_consistency']),
        evaluator_runtime=dict(mjvideo=mj.get('runtime'), temporal_seconds=temporal.get('seconds')),
        verification=dict(media_files_checked=0, mj_rows=16, criteria_per_mj_row=28, aspects_per_mj_row=5,
                          all_media_hashes_match=True, all_arrays_finite=True, all_recomputed_readouts_match=True),
        per_case={},
        limits=[
            'Four development prompts and one seed; descriptive comparisons do not establish generalization.',
            'AMT interpolation agreement can rise for static or blurred clips; RAFT dynamic_degree is a movement decision, not a monotonic quality score.',
            'DINO consistency does not establish correctness; the evaluator saved only its sum over 76 terms, not the per-frame similarity array. Only sum/76 is independently recomputed here.',
            'MJ sees eight sampled frames, ending at frame66. Its learned total is retained, not reconstructed as a mean of aspects or criteria.',
            'Safety and bias/fairness remain in raw MJ fields and its own total; they are not separate primary statistics.',
            'The campus case is retained in every mean despite previously observed BF16 green/striped distortion. QAD blur/background changes are qualitative project observations, not a mechanism inferred from these scores.',
            'SVD-versus-plain is a whole-recipe comparison, not an isolated causal effect of LR.',
        ])
    for case in CASES:
        require(set(mj['results'][case]['variants']) == set(ARMS), f'{case}: incomplete MJ arms')
        entry = dict(prompt=gen['arms']['bf16']['cases'][case]['prompt'], arms={}, paired_differences={})
        require(mj['results'][case]['prompt'] == entry['prompt'], f'{case}: MJ prompt mismatch')
        for arm in ARMS:
            generated = gen['arms'][arm]['cases'][case]
            row, mr = keyed[(case, arm)], mj['results'][case]['variants'][arm]
            video = generated['video']; path = Path(video['path'])
            actual_hash = sha(path)
            require(path.stat().st_size == video['bytes'] and actual_hash == video['sha256'] ==
                    row['video']['sha256'] == mr['video_sha256'] == validation_files[str(path)]['sha256'],
                    f'{case}/{arm}: media SHA/size mismatch')
            require(row['video']['file'] == mr['video'] == str(path), f'{case}/{arm}: media path mismatch')
            require(row['prompt'] == generated['prompt'] == entry['prompt'] and row['seed'] == generated['seed'] == 20261004,
                    f'{case}/{arm}: prompt/seed mismatch')
            result['verification']['media_files_checked'] += 1
            errors = row['amt']['interpolation_reconstruction_absdiff']
            flows = row['raft']['top5percent_mean_flow_magnitudes']
            require(len(errors) == len(flows) == 38 and all(finite(x) for x in errors+flows), f'{case}/{arm}: invalid AMT/RAFT array')
            amt_error_mean = mean(errors)
            smooth = 1-amt_error_mean/255
            close(smooth, row['scores']['motion_smoothness'], f'{case}/{arm} AMT', atol=1e-7)
            threshold, required = row['raft']['threshold'], row['raft']['required_count']
            require(threshold == 11.25 and required == 10, f'{case}/{arm}: RAFT decision parameters changed')
            count = sum(v > threshold for v in flows)
            dynamic = count >= required
            require(count == row['raft']['above_threshold_count'] and isinstance(row['scores']['dynamic_degree'], bool)
                    and dynamic == row['scores']['dynamic_degree'], f'{case}/{arm}: RAFT decision differs')
            dino_sum, pairs = row['dino']['original_video_results_sum'], row['dino']['frame_pairs']
            require(finite(dino_sum) and pairs == 76, f'{case}/{arm}: invalid DINO sum/pairs')
            consistency = dino_sum/pairs
            close(consistency, row['scores']['subject_consistency'], f'{case}/{arm} DINO')
            require(set(mr['criteria']) == set(CRITERIA) and set(mr['aspects']) == set(ASPECTS), f'{case}/{arm}: MJ keys changed')
            require(all(finite(x) for x in [mr['score'], *mr['criteria'].values(), *mr['aspects'].values()]), f'{case}/{arm}: nonfinite MJ score')
            require(mr['sampled_indices'] == [INDICES] and mr['patches'] == [1]*8, f'{case}/{arm}: actual MJ sampled frames/patches differ')
            metrics = dict(motion_smoothness=smooth, dynamic_degree=dynamic, subject_consistency=consistency,
                amt_mean_absdiff_0_255=amt_error_mean, raft_mean_top5percent_flow=mean(flows),
                raft_above_threshold_count=count, mj_total=mr['score'], mj_alignment=mr['aspects']['alignment'],
                mj_fineness=mr['aspects']['fineness'], mj_coherence_consistency=mr['aspects']['coherence_consistency'])
            entry['arms'][arm] = dict(video=str(path), video_sha256=actual_hash, metrics=metrics,
                temporal_raw=dict(amt=errors, raft=flows, raft_threshold=threshold, raft_required_count=required,
                                  dino_saved_sum=dino_sum, dino_terms=pairs, dino_per_frame_array_available=False),
                mjvideo_raw=dict(total=mr['score'], aspects=mr['aspects'], criteria=mr['criteria'], sampled_indices=mr['sampled_indices']))
        plain = entry['arms']['packed_step0000']['metrics']
        entry['paired_differences'] = dict(qad_minus_plain=difference(entry['arms']['packed_step0064']['metrics'], plain),
                                          svd_minus_plain=difference(entry['arms']['svd_lr']['metrics'], plain))
        result['per_case'][case] = entry
    keys = result['per_case'][CASES[0]]['arms']['bf16']['metrics']
    means = {a: {k: mean([result['per_case'][c]['arms'][a]['metrics'][k] for c in CASES]) for k in keys} for a in ARMS}
    for arm in ARMS:
        for k in ('motion_smoothness', 'dynamic_degree', 'subject_consistency'):
            close(means[arm][k], temporal['arm_means_descriptive_only'][arm][k], f'{arm}/{k}: arm mean', atol=1e-7)
    result['arm_means_descriptive_only'] = means
    result['paired_mean_differences'] = {name: {k: mean([result['per_case'][c]['paired_differences'][name][k] for c in CASES]) for k in keys}
                                       for name in ('qad_minus_plain', 'svd_minus_plain')}
    result['dynamic_case_counts'] = {a: dict(dynamic_cases=sum(result['per_case'][c]['arms'][a]['metrics']['dynamic_degree'] for c in CASES), total_cases=4) for a in ARMS}
    result['qad_mj_total_case_differences'] = {c: result['per_case'][c]['paired_differences']['qad_minus_plain']['mj_total'] for c in CASES}
    result['campus_mean_accounting'] = dict(case_id='vbench_182', retained=True,
        note='Report the contribution to the all-four mean; do not remove the anomalous campus case.',
        qad_minus_plain_mj_total=result['qad_mj_total_case_differences']['vbench_182'],
        contribution_to_four_case_mean=result['qad_mj_total_case_differences']['vbench_182']/4)
    result['interpretation'] = 'Metric tradeoffs and per-case reversals must be reported separately. Higher AMT smoothness or a positive MJ four-case mean alone is not evidence of overall quality improvement.'
    require('torch' not in sys.modules, 'This summary should not import torch or initialize CUDA')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, default=RD/'generation_run.json')
    parser.add_argument('--validation', type=Path, default=RD/'generation_validation.json')
    parser.add_argument('--temporal', type=Path, default=RD/'temporal_scores.json')
    parser.add_argument('--mjvideo', type=Path, default=RD/'mjvideo_scores.json')
    parser.add_argument('--output', type=Path, default=RD/'summary.json')
    args = parser.parse_args()
    require(not args.output.exists(), f'Preserve existing summary: {args.output}')
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output),
        arm_means=result['arm_means_descriptive_only'], paired_mean_differences=result['paired_mean_differences'],
        qad_mj_total_case_differences=result['qad_mj_total_case_differences'], dynamic_case_counts=result['dynamic_case_counts']), indent=2))


if __name__ == '__main__':
    main()
