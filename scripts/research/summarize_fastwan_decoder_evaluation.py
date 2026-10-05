#!/usr/bin/env python3
"""CPU paired readout: all E024 final latents, TAEHV FP16 vs full Wan VAE FP32.

Reuse the independent 81-frame scalar formulas, not evaluator-produced means.
No model imports, new metrics, video filtering, or claim about quantization alone.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

from summarize_fastwan_qad_evaluation import record, average, recompute
from summarize_wan_qad_videos import ROOT, require, sha, close, difference

RD = ROOT / 'results/research/E025'
OLD = ROOT / 'results/research/E024'
ARMS = ('taehv_fp16', 'full_wan_vae_fp32')
VARIANTS = ('fastwan_qad_official', 'fastwan_full_vae_fp32')
IDENTITY = ('trajectory_id', 'prompt_id', 'prompt', 'seed', 'replica')


def summarize(args):
    refs = {}

    def load(path):
        path = Path(path)
        refs[str(path)] = record(path)
        return json.loads(path.read_text())

    all_rows, generations, protocols = {}, {}, {}
    for directory, arm, variant in zip((args.e024_dir, args.report_dir), ARMS, VARIANTS):
        mp = directory / 'evaluation_manifest.json'
        manifest = load(mp)
        generation = load(manifest['generation_report'])
        gh = sha(manifest['generation_report'])
        require(manifest['generation_sha256'] == gh, 'Generation/manifest SHA')
        require(generation['status'] == 'complete' and generation['phase'] == 'suite', 'Incomplete generation')
        launch_ref = manifest['generation_launcher']
        gl = load(launch_ref['file'])
        require(sha(launch_ref['file']) == launch_ref['sha256'] and gl['status'] == 'complete'
                and gl['returncode'] == 0 and gl['result_sha256'] == gh, 'Generation launcher binding')
        evaluations = {}
        for metric in ('temporal', 'mjvideo'):
            launched = load(directory / (metric + '_launcher.json'))
            require(launched['status'] == 'complete' and launched['returncode'] == 0, f'{arm}/{metric}: unfinished or failed')
            require(launched['manifest']['sha256'] == sha(mp)
                    and launched['generation_report']['sha256'] == gh, 'Evaluation input binding')
            require(sha(launched['script']['file']) == launched['script']['sha256'], 'Evaluator source changed')
            rp = Path(launched['result']['file'])
            require(sha(rp) == launched['result']['sha256'], 'Evaluation result SHA')
            evaluations[metric] = load(rp)
        temporal, mj = evaluations['temporal'], evaluations['mjvideo']
        require(temporal['status'] == 'complete' and mj['num_segments'] == 8, 'Evaluation contract')
        require(temporal['manifest']['sha256'] == sha(mp) and temporal['generation_report']['sha256'] == gh,
                'Temporal output binding')
        tr = {r['case_id']: r for r in temporal['rows']}
        generated = {r['trajectory_id']: r for r in generation['cases']}
        cases = manifest['cases']
        tids = [r['case_id'] for r in cases]
        require(len(tids) == len(set(tids)) == len(temporal['rows']) == len(generation['cases']) == 16, 'Full 16-case coverage')
        require(set(tids) == set(tr) == set(generated) == set(mj['results']), 'Evaluation row identities')
        rows = {}
        for case in cases:
            tid = case['case_id']
            t, g = tr[tid], generated[tid]
            require(case['case_id'] == case['trajectory_id'] and g['status'] == 'complete', 'Case completed')
            require(all(case[k] == t[k] == g[k] for k in IDENTITY), 'Actual identity mismatch')
            require(mj['results'][tid]['prompt'] == case['prompt']
                    and set(mj['results'][tid]['variants']) == {variant}, 'MJ prompt/variant')
            m = mj['results'][tid]['variants'][variant]
            video = record(case['video'])
            require(video['sha256'] == case['video_sha256'] == t['video']['sha256'] == g['video']['sha256']
                    == m['video_sha256'], 'Actual media SHA')
            require(video['file'] == t['video']['file'] == g['video']['file'] == m['video']
                    and video['bytes'] == t['video']['bytes'] == g['video']['bytes'], 'Actual media path/size')
            require(t['media'] == dict(decoded_frames=81, declared_frames=81.0, fps=16.0, width=832, height=480),
                    'All-frame CPU decoded media contract')
            rows[tid] = dict({k: case[k] for k in IDENTITY}, video=video, **recompute(t, m))
        for pair in temporal['dino_real_pairs']:
            close(pair['original_pair_score'], sum(rows[tid]['metrics']['subject_consistency']
                                                  for tid in pair['case_ids']) / 2, 'DINO raw pair mean')
        all_rows[arm], generations[arm] = rows, generation
        protocols[arm] = dict(temporal_sampling=temporal['sampling'], mj_runtime=mj['runtime'],
            temporal_sources=temporal['sources'], inherited_models=temporal['inherited_models'],
            amt_actual_scale_factor=temporal['amt_actual_scale_factor'], temporal_seconds=temporal['seconds'])
    old, new = (generations[a] for a in ARMS)
    require(new['actual_dit_calls'] == 0 and new['actual_vae_decode_calls'] == 16, 'Zero DiT decoder-only intervention')
    require(new['input_generation']['sha256'] == sha(args.e024_dir / 'suite_run.json')
            and Path(new['input_generation']['file']).resolve() == (args.e024_dir / 'suite_run.json').resolve(),
            'Original latent source generation')
    require(old['trajectories'] == new['trajectories'], 'Fixed whole suite identities/order')
    old_cases = {r['trajectory_id']: r for r in old['cases']}
    for r in new['cases']:
        source = old_cases[r['trajectory_id']]
        require(r['latents_artifact'] == source['latents_artifact'], 'Same immutable input latent artifact')
        require(r['tensors']['final_latents'] == source['tensors']['final_latents']
                and r['tensors']['final_latents']['finite'], 'Decoder CPU-verified input tensor record')
    for key in ('temporal_sampling', 'mj_runtime', 'temporal_sources', 'inherited_models'):
        require(protocols[ARMS[0]][key] == protocols[ARMS[1]][key], 'Metric implementation changed: ' + key)
    tids = list(all_rows[ARMS[0]])
    require(tids == list(all_rows[ARMS[1]]), 'Paired order')
    result = dict(experiment='E025_decoder_readout', status='running', cpu_only=True,
        per_trajectory={}, per_prompt={}, arm_means={}, mean_differences={}, protocol=protocols,
        policy='All 16 same-latent pairs; average two seeds per prompt, then equally average eight prompts. No composite metric or case filtering.',
        limits=[
            'Intervention changes decoder stack plus compute dtype: official TAEHV FP16 versus matching Wan VAE in Diffusers FP32, including their required latent normalization and output processing.',
            'This is not an isolated quantization, VAE-dtype, or decoder-architecture causal effect. Both arms share the already generated QAD latents and their unresolved semantic failures.',
            'AMT can favor static or blurry media, binary RAFT dynamic_degree is not motion correctness, and DINO consistency is not prompt alignment.',
            'MJ total/aspects/criteria are stored learned-model outputs; only their identities and aggregation are independently verified. No mean-of-criteria surrogate is introduced.',
            'DINO saves a sum of 80 terms, not the individual similarities. Sum normalization is checked without claiming unavailable per-frame verification.',
            'Final-latent equality inherits the decoder runner CPU tensor check and identical source artifact binding; this summary does not load or rehash model weights or re-run decoding.',
            'Sixteen related seed trajectories from eight observed prompts are descriptive diagnostics, not evidence of general quality equivalence.'])
    for tid in tids:
        before, after = (all_rows[a][tid] for a in ARMS)
        require(all(before[k] == after[k] for k in IDENTITY), 'Paired text/seed identity')
        result['per_trajectory'][tid] = dict({k: before[k] for k in IDENTITY},
            arms={a: all_rows[a][tid] for a in ARMS},
            full_minus_taehv={key: difference(after[key], before[key]) for key in ('metrics', 'mj_aspects', 'mj_criteria')})
    pids = list(dict.fromkeys(all_rows[ARMS[0]][t]['prompt_id'] for t in tids))
    require(len(pids) == 8, 'Eight prompts')
    for pid in pids:
        ids = [t for t in tids if all_rows[ARMS[0]][t]['prompt_id'] == pid]
        require(len(ids) == 2 and [all_rows[ARMS[0]][t]['replica'] for t in ids] == [0, 1], 'Both fixed seeds')
        arms = {a: {key: average([all_rows[a][t][key] for t in ids])
                    for key in ('metrics', 'mj_aspects', 'mj_criteria')} for a in ARMS}
        result['per_prompt'][pid] = dict(prompt=all_rows[ARMS[0]][ids[0]]['prompt'], trajectory_ids=ids,
            seeds=[all_rows[ARMS[0]][t]['seed'] for t in ids], arms=arms,
            full_minus_taehv={key: difference(arms[ARMS[1]][key], arms[ARMS[0]][key])
                              for key in ('metrics', 'mj_aspects', 'mj_criteria')})
    for a in ARMS:
        result['arm_means'][a] = {key: average([r['arms'][a][key] for r in result['per_prompt'].values()])
                                  for key in ('metrics', 'mj_aspects', 'mj_criteria')}
    result['mean_differences'] = {key: difference(result['arm_means'][ARMS[1]][key], result['arm_means'][ARMS[0]][key])
                                  for key in ('metrics', 'mj_aspects', 'mj_criteria')}
    for key in result['mean_differences']:
        paired = average([r['full_minus_taehv'][key] for r in result['per_prompt'].values()])
        for name, value in paired.items():
            close(value, result['mean_differences'][key][name], 'Two-seed/prompt paired hierarchy')
    inherited = load(args.e024_dir / 'evaluation_summary.json')
    require(inherited['status'] == 'complete', 'Prior independent summary')
    for key, value in result['arm_means'][ARMS[0]]['metrics'].items():
        close(value, inherited['official_means']['metrics'][key], 'E024 independent recomputation')
    transitions = {name: [] for name in ('static_to_static', 'static_to_dynamic', 'dynamic_to_static', 'dynamic_to_dynamic')}
    for tid in tids:
        flags = [all_rows[a][tid]['metrics']['dynamic_degree'] for a in ARMS]
        name = '_to_'.join('dynamic' if flag else 'static' for flag in flags)
        transitions[name].append(tid)
    result.update(status='complete', dynamic_transitions=transitions,
        dynamic_video_counts={a: sum(all_rows[a][t]['metrics']['dynamic_degree'] for t in tids) for a in ARMS},
        amt_internal_scale_equal=protocols[ARMS[0]]['amt_actual_scale_factor'] == protocols[ARMS[1]]['amt_actual_scale_factor'],
        input_reports=refs, sources=[record(__file__), record(Path(__file__).with_name('summarize_fastwan_qad_evaluation.py'))],
        verification=dict(pairs=16, prompts=8, actual_dit_calls=0, actual_vae_decode_calls=16,
            amt_terms=40, raft_terms=40, dino_terms=80, mj_actual_indices=list(range(0, 80, 10)),
            same_final_latent_bindings=True, source_media_sha_checked=True, model_weights_rehashed=False))
    require('torch' not in sys.modules, 'Pure CPU, no torch import')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--e024-dir', type=Path, default=OLD)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report_dir / 'evaluation_summary.json'
    require(not output.exists(), 'Preserve old result; use --output for another attempt')
    result = dict(experiment='E025_decoder_readout', status='failed_stop', source=record(__file__))
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
        print(json.dumps(dict(status=result['status'], output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
