#!/usr/bin/env python3
"""Independent CPU readout of the complete E024 official-product suite.

Recompute saved scalar arrays, retain learned MJ outputs without inventing a
formula for its gated total, and aggregate two seeds then eight prompts.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

from summarize_wan_qad_videos import ROOT, ASPECTS, CRITERIA, require, sha, finite, close, mean, difference

RD = ROOT / 'results/research/E024'
ARM = 'fastwan_qad_official'
INDICES = list(range(0, 80, 10))
IDENTITY = ('trajectory_id', 'prompt_id', 'prompt', 'seed', 'replica')


def record(path):
    path = Path(path).absolute()
    return dict(file=str(path), bytes=path.stat().st_size, sha256=sha(path))


def average(rows):
    require(all(set(row) == set(rows[0]) for row in rows), 'Metric keys differ')
    return {key: mean([row[key] for row in rows]) for key in rows[0]}


def recompute(row, mj):
    errors = row['amt']['interpolation_reconstruction_absdiff']
    flows = row['raft']['top5percent_mean_flow_magnitudes']
    require(len(errors) == len(flows) == 40 and all(finite(x) for x in errors + flows), 'AMT/RAFT arrays')
    smooth = 1 - mean(errors) / 255
    close(smooth, row['scores']['motion_smoothness'], 'AMT', atol=1e-7)
    raft = row['raft']
    require(raft['sampled_frames'] == 41 and raft['actual_flows'] == 40, 'RAFT sampled frames')
    threshold, required = raft['threshold'], raft['required_count']
    require(threshold == 6 * min(row['media']['height'], row['media']['width']) / 256
            and required == round(4 * (41 / 16)), 'Original RAFT parameters')
    count = sum(value > threshold for value in flows)
    dynamic = count >= required
    require(count == raft['above_threshold_count'] and isinstance(row['scores']['dynamic_degree'], bool)
            and dynamic == row['scores']['dynamic_degree'], 'RAFT decision')
    dino = row['dino']
    require(dino['frame_pairs'] == 80 and finite(dino['original_video_results_sum']), 'DINO saved sum')
    consistency = dino['original_video_results_sum'] / 80
    close(consistency, row['scores']['subject_consistency'], 'DINO normalization')
    require(set(mj['criteria']) == set(CRITERIA) and set(mj['aspects']) == set(ASPECTS), 'MJ 28/5 schema')
    require(all(finite(v) for v in [mj['score'], *mj['criteria'].values(), *mj['aspects'].values()]), 'MJ finite outputs')
    require(mj['sampled_indices'] == [INDICES] and mj['patches'] == [1] * 8, 'Actual eight-frame MJ sampling')
    metrics = dict(motion_smoothness=smooth, dynamic_degree=dynamic, subject_consistency=consistency,
                   amt_mean_absdiff_0_255=mean(errors), raft_mean_top5percent_flow=mean(flows),
                   raft_above_threshold_count=count, mj_total=mj['score'],
                   mj_alignment=mj['aspects']['alignment'], mj_fineness=mj['aspects']['fineness'],
                   mj_coherence_consistency=mj['aspects']['coherence_consistency'])
    return dict(metrics=metrics, mj_aspects=mj['aspects'], mj_criteria=mj['criteria'],
                mj_sampled_indices=mj['sampled_indices'],
                temporal_raw=dict(amt=errors, raft=flows, threshold=threshold, required_count=required,
                                  dino_saved_sum=dino['original_video_results_sum'], dino_terms=80))


def summarize(args):
    references = {}

    def load(path):
        path = Path(path)
        references[str(path)] = record(path)
        return json.loads(path.read_text())

    manifest_path = args.report_dir / 'evaluation_manifest.json'
    manifest = load(manifest_path)
    generation = load(manifest['generation_report'])
    require(generation['status'] == 'complete' and generation['phase'] == 'suite', 'Suite incomplete')
    require(sha(manifest['generation_report']) == manifest['generation_sha256'], 'Suite SHA')
    launch = load(manifest['generation_launcher']['file'])
    require(sha(manifest['generation_launcher']['file']) == manifest['generation_launcher']['sha256']
            and launch['status'] == 'complete' and launch['returncode'] == 0, 'Generation launcher')
    require(launch['result_sha256'] == manifest['generation_sha256'], 'Generation launcher output binding')
    evaluated = {}
    for metric in ('temporal', 'mjvideo'):
        launcher_path = args.report_dir / (metric + '_launcher.json')
        launched = load(launcher_path)
        require(launched['status'] == 'complete' and launched['returncode'] == 0, f'{metric}: launcher incomplete/failed')
        require(launched['manifest']['sha256'] == sha(manifest_path)
                and launched['generation_report']['sha256'] == manifest['generation_sha256'], 'Evaluator input binding')
        require(sha(launched['script']['file']) == launched['script']['sha256'], 'Executed evaluator source binding')
        result_path = Path(launched['result']['file'])
        require(sha(result_path) == launched['result']['sha256'], 'Evaluator result binding')
        evaluated[metric] = load(result_path)
    temporal, mj = evaluated['temporal'], evaluated['mjvideo']
    require(temporal['status'] == 'complete' and mj['num_segments'] == 8, 'Evaluator completion/settings')
    require(temporal['manifest']['sha256'] == sha(manifest_path)
            and temporal['generation_report']['sha256'] == manifest['generation_sha256'], 'Temporal provenance')
    cases = manifest['cases']
    tids = [row['case_id'] for row in cases]
    require(len(tids) == len(set(tids)) == len(generation['cases']) == len(temporal['rows']) == 16, 'Fixed 16 trajectories')
    require(set(mj['results']) == set(tids), 'MJ coverage')
    generated = {row['trajectory_id']: row for row in generation['cases']}
    rows = {row['case_id']: row for row in temporal['rows']}
    require(set(rows) == set(generated) == set(tids), 'Full identity coverage')
    result = dict(experiment='E024_product_readout', status='running', cpu_only=True,
                  per_trajectory={}, per_prompt={}, product_reference_table={},
                  policy='All 16 rows retained; mean two seeds within each prompt, then mean eight prompts equally. No composite score or significance claim.',
                  limits=[
                      'Complete product reference: weights, 81 vs 77 frames, UniPC 3 vs rCM 4 steps, attention and decoder differ. Same texts/seeds do not imply same noise.',
                      'FastWan uses official FP16 TAEHV; E022 inherited BF16 Wan VAE has a documented unresolved numerical/artifact concern. These scores cannot isolate quantization quality effects.',
                      'AMT may reward static/blurred media; binary RAFT dynamic_degree is not a monotonic quality measure; DINO consistency is not semantic correctness.',
                      'DINO sum/80 is independently normalized; individual similarities were not saved and are not reconstructed.',
                      'MJ learned total/aspects/criteria are saved model outputs; aggregation is independently recomputed, not their unavailable model forward or a mean-as-total surrogate.',
                      'MJ samples 0,10,...,70 here versus 0,9,19,28,38,47,57,66 in E022; temporal metric formulas remain original with 40/80 rather than 38/76 terms.',
                      'Eight prompts with two related seeds each are a local diagnostic set, not broad independent quality evidence.',
                      'Generation timing includes synchronized stage diagnostics and is not a compiled or steady-state performance benchmark.'])
    totals = dict(dit=0, nvfp4_gemm=0, fp4_attention=0, dense_cross_attention=0)
    for case in cases:
        tid = case['case_id']
        row, gen = rows[tid], generated[tid]
        require(gen['status'] == 'complete' and all(case[k] == row[k] == gen[k] for k in IDENTITY), 'Case identity')
        require(mj['results'][tid]['prompt'] == case['prompt']
                and set(mj['results'][tid]['variants']) == {ARM}, 'MJ identity/arm')
        m = mj['results'][tid]['variants'][ARM]
        actual = record(case['video'])
        require(actual['sha256'] == case['video_sha256'] == row['video']['sha256']
                == gen['video']['sha256'] == m['video_sha256'], 'Actual video SHA')
        require(actual['bytes'] == row['video']['bytes'] == gen['video']['bytes'], 'Video bytes')
        require(actual['file'] == row['video']['file'] == gen['video']['file'] == m['video'], 'Video path')
        require(row['media'] == dict(decoded_frames=81, declared_frames=81.0, fps=16.0, width=832, height=480), 'Decoded media contract')
        receipt = gen['backend_receipt']
        calls = dict(dit=len(receipt['dit_inputs']), nvfp4_gemm=receipt['gemm_count'],
                     fp4_attention=sum(n for kind, n in receipt['attention_counts'].items() if kind.endswith('.AttnQatInferImpl')),
                     dense_cross_attention=sum(n for kind, n in receipt['attention_counts'].items() if kind.endswith(('.FlashAttentionImpl', '.SDPAImpl'))))
        require(calls == gen['actual_calls'] == dict(dit=3, nvfp4_gemm=900, fp4_attention=90, dense_cross_attention=90), 'Observed call counts')
        require(len(receipt['linear_counts']) == 300 and set(receipt['linear_counts'].values()) == {3}
                and receipt['gemm_backends'] == {'cutlass': 900}, 'Actual linear execution')
        for key in totals:
            totals[key] += calls[key]
        result['per_trajectory'][tid] = dict(case, **recompute(row, m))
    for pair in temporal['dino_real_pairs']:
        close(pair['original_pair_score'], mean([result['per_trajectory'][tid]['metrics']['subject_consistency']
                                                for tid in pair['case_ids']]), 'DINO original pair aggregate')
    prompts = list(dict.fromkeys(case['prompt_id'] for case in cases))
    require(len(prompts) == 8, 'Eight prompts')
    for pid in prompts:
        items = [result['per_trajectory'][tid] for tid in tids if result['per_trajectory'][tid]['prompt_id'] == pid]
        require(len(items) == 2 and [r['replica'] for r in items] == [0, 1], 'Two replicas per prompt')
        result['per_prompt'][pid] = dict(prompt=items[0]['prompt'], seeds=[r['seed'] for r in items],
            trajectory_ids=[r['trajectory_id'] for r in items],
            **{key: average([r[key] for r in items]) for key in ('metrics', 'mj_aspects', 'mj_criteria')})
    overall = {key: average([row[key] for row in result['per_prompt'].values()])
               for key in ('metrics', 'mj_aspects', 'mj_criteria')}
    result['official_means'] = overall
    old = load(args.e022_summary)
    require(old['status'] == 'complete' and old['phase'] == 'complete', 'E022 inherited summary')
    require(set(old['per_prompt']) == set(prompts) and set(old['per_trajectory']) == set(tids), 'Product comparison set')
    result['product_reference_table'].update(old['arm_means_descriptive_only'])
    result['product_reference_table'][ARM] = overall['metrics']
    for tid in tids:
        require(all(result['per_trajectory'][tid][key] == old['per_trajectory'][tid][key]
                    for key in ('prompt_id', 'prompt', 'replica', 'seed')), 'E022 same text/seed binding')
    for pid in prompts:
        entry = result['per_prompt'][pid]
        entry['official_minus_e022_products'] = {
            arm: difference(entry['metrics'], row['metrics']) for arm, row in old['per_prompt'][pid]['arms'].items()}
    result['heterogeneity'] = {}
    for arm in old['arm_means_descriptive_only']:
        deltas = {pid: result['per_prompt'][pid]['official_minus_e022_products'][arm]['mj_total'] for pid in prompts}
        absolute_sum = sum(abs(v) for v in deltas.values())
        result['heterogeneity'][arm] = dict(mj_total_deltas=deltas, mean_delta=mean(list(deltas.values())),
            positive_prompts=sum(v > 0 for v in deltas.values()), negative_prompts=sum(v < 0 for v in deltas.values()),
            min_delta=min(deltas.values()), max_delta=max(deltas.values()),
            largest_absolute_delta_share=(max(abs(v) for v in deltas.values()) / absolute_sum if absolute_sum else None),
            leave_one_prompt_out_mean_range=[min((sum(deltas.values())-v)/7 for v in deltas.values()),
                                            max((sum(deltas.values())-v)/7 for v in deltas.values())])
    result.update(status='complete', observed_generation_totals=totals,
                  dynamic_videos=sum(r['metrics']['dynamic_degree'] for r in result['per_trajectory'].values()),
                  input_reports=references, sources=[record(__file__), record(Path(__file__).with_name('summarize_wan_qad_videos.py'))],
                  verification=dict(videos=16, prompts=8, amt_terms_per_video=40, raft_terms_per_video=40,
                                    dino_terms_per_video=80, mj_criteria=28, mj_aspects=5, mj_actual_indices=INDICES,
                                    video_sha_checked=True, inherited_media_decode=True, model_weights_rehashed=False))
    require('torch' not in sys.modules, 'CPU summary must not import torch')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--e022-summary', type=Path, default=ROOT / 'results/research/E022/video_summary.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report_dir / 'evaluation_summary.json'
    require(not output.exists(), 'Preserve prior output; pass --output for a new attempt')
    result = dict(experiment='E024_product_readout', status='failed_stop', source=record(__file__))
    try:
        result = summarize(args)
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(output)
        print(json.dumps(dict(status=result['status'], output=str(output), sha256=sha(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
