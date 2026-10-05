#!/usr/bin/env python3
"""CPU-only paired E044 readout, inheriting the completed E043 BF16 scores."""
import argparse
import json
import os
from pathlib import Path
import traceback

import vanilla_wan_native_evaluation as binding
from summarize_fastwan_qad_evaluation import record, recompute, average
from summarize_vanilla_wan_reference import saved_tensor, step_records, PRIMARY
from summarize_wan_qad_videos import require, close, difference, mean, sha

RD, ARMS, BASE = binding.RD, binding.ARMS, binding.BASE
EXPECTED = dict(dit=100, native_gemm=30000, dit_sdpa=6000,
                scheduler=50, public_vae_decode=1, text_encoder=0, fastpack_checks=30000)


def summarize(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for CPU summary')
    import torch
    torch.set_num_threads(4)
    refs = {}

    def load(path):
        path = Path(path)
        refs[str(path)] = record(path)
        return json.loads(path.read_text())

    manifest_path = args.report_dir / 'evaluation_manifest.json'
    manifest = load(manifest_path)
    require(manifest['status'] == 'complete' and manifest['experiment'] == 'E044', 'Evaluation binding incomplete')
    plan = load(manifest['source_manifest']['file'])
    require(refs[manifest['source_manifest']['file']] == manifest['source_manifest'], 'Plan binding')
    reference = load(manifest['inherited']['summary']['file'])
    require(refs[manifest['inherited']['summary']['file']] == manifest['inherited']['summary']
            and reference['status'] == 'complete', 'Completed E043 reference binding')
    launch = load(args.report_dir / 'launcher.json')
    require(launch['status'] == 'complete' and all(w['status'] == 'complete' and w['returncode'] == 0
                for w in launch['workers']), 'Generation did not complete cleanly')
    launched = {w['result']['file']: w['result'] for w in launch['workers']}
    require(set(launched) == {r['file'] for r in manifest['worker_reports']}, 'Actual worker coverage')
    cases = plan['cases']
    tids = [r['case_id'] for r in cases]
    require(len(tids) == len(set(tids)) == 8 and set(reference['per_case']) == set(tids), 'Eight reference cases')
    rows = {(r['arm'], r['case_id']): r for r in manifest['rows']}
    require(len(manifest['rows']) == len(rows) == 24
            and set(rows) == {(arm, tid) for arm in (BASE, *ARMS) for tid in tids}, 'All 24 media rows')
    generated, schedules, receipts = {}, {}, []
    native_targets = None
    totals = {arm: {} for arm in ARMS}
    for source in manifest['worker_reports']:
        worker = load(source['file'])
        require(refs[source['file']] == source == launched[source['file']], 'Actual worker report binding')
        require(worker['status'] == 'complete' and worker['arm'] in ARMS, 'Native worker incomplete')
        arm = worker['arm']
        require(worker['settings'] == plan['settings'], 'Worker recipe settings')
        require(worker['manifest'] == manifest['source_manifest'], 'Native worker plan binding')
        targets = worker['native_targets']
        require(len(targets) == len(set(targets)) == worker['conversion']['target_count'] == 300, 'Native target coverage')
        if native_targets is None:
            native_targets = set(targets)
        require(set(targets) == native_targets, 'Same target modules across workers/arms')
        summed = {k: sum(r['actual_counts'][k] for r in worker['cases']) for k in worker['actual_counts']}
        require(summed == worker['actual_counts'], 'Worker actual counts versus cases')
        for key, value in summed.items():
            totals[arm][key] = totals[arm].get(key, 0) + value
        receipts.append(dict(arm=arm, report=source, actual_counts=summed, native_target_count=len(targets),
                             case_ids=[r['case_id'] for r in worker['cases']]))
        for row in worker['cases']:
            key = (arm, row['case_id'])
            require(key not in generated and row['status'] == 'complete' and row['variant'] == arm, 'Duplicate/invalid native case')
            require(all(row['actual_counts'][k] == v for k, v in EXPECTED.items()), 'Per-case actual execution counts')
            require(len(row['dit_calls']) == 100 and all(call == dict(native_gemm=300, dit_sdpa=60,
                    fastpack_checks=300, invalid_flags=0) for call in row['dit_calls']), 'Actual per-DiT receipts')
            require(row['reference_case_id'] == row['case_id'] and row['reference_worker']
                    == worker['references'][str(row['prompt_id'])], 'Reference worker identity')
            generated[key] = row
            schedules[key] = reference['per_case'][row['case_id']]['step_chain']['schedule']
    require(set(generated) == {(arm, tid) for arm in ARMS for tid in tids}, 'All 16 actual generations')
    for arm in ARMS:
        require(all(totals[arm][k] == 8*v for k, v in EXPECTED.items()), 'Actual arm totals')
    mj = load(args.report_dir / 'mjvideo_scores.json')
    require(mj['num_segments'] == 8 and set(mj['results']) == set(tids), 'MJ complete fixed cases')
    temporal = {}
    for arm in ARMS:
        path = args.report_dir / f'temporal_{arm}_manifest.json'
        sub = load(path)
        out = load(args.report_dir / f'temporal_{arm}_scores.json')
        require(out['status'] == 'complete' and out['arm'] == sub['variant'] == arm, 'Temporal completion/arm')
        require(out['manifest']['sha256'] == out['generation_report']['sha256'] == refs[str(path)]['sha256'], 'Temporal input binding')
        require(out['sampling'] == {k: v for k, v in binding.old.SAMPLING.items() if k != 'mjvideo_indices'}, 'Inherited temporal sampling')
        index = {r['case_id']: r for r in out['rows']}
        require(len(out['rows']) == len(index) == 8 and set(index) == set(tids), 'Temporal all-eight coverage')
        temporal[arm] = (out, index)
    result = dict(experiment='E044', status='running', cpu_only=True,
        per_case={}, per_prompt={}, actual_new_generation_totals=totals,
        reused_bf16_generation_totals=reference['observed_generation_totals'], worker_receipts=receipts,
        limits=[
            'All four prompts and both seeds retained, including E043 teacher failures; no metric-based eligibility rule.',
            'Paired complete deployed recipes: SVD uses old smooth/LR and other PTQ settings, plus an older calibration domain. Differences do not isolate the LR branch.',
            'Same actual noise/embedding artifacts are bound. Free-running final states can differ semantically; final-latent difference is not treated as local quantization noise.',
            'Initial/final artifact identities and all saved scalar step records are checked. Intermediate tensors were not saved; no numerical trajectory replay is claimed.',
            'Original E043 BF16 scores are reused without inference. New AMT/RAFT arrays are reaggregated; DINO saved sums are normalized, not reconstructed frame-level inference.',
            'MJ scores are raw learned outputs, not probabilities; safety/bias remain per-case raw only. AMT/RAFT/DINO do not establish action correctness.',
            'Full software media decoding is inherited from completed temporal evaluations. This reducer rehashes media, not model assets, and adds no second decode.',
            'Four purposively chosen actions with two seeds each are a local diagnostic set, not broad quality evidence.'])
    for case in cases:
        tid = case['case_id']
        ref = reference['per_case'][tid]
        require(all(case[k] == ref[k] for k in binding.IDENTITY), 'Reference case identity')
        require(mj['results'][tid]['prompt'] == case['prompt'] and set(mj['results'][tid]['variants']) == set(ARMS), 'MJ two native variants')
        arms = {BASE: {k: ref[k] for k in ('metrics', 'video', 'media', 'mj_aspects_raw', 'mj_criteria_raw',
                                          'mj_sampled_indices', 'temporal_raw')}}
        base_binding = rows[(BASE, tid)]
        require(record(base_binding['video']) == ref['video'], 'Inherited BF16 media identity')
        for arm in ARMS:
            gen, bound = generated[(arm, tid)], rows[(arm, tid)]
            out = temporal[arm][1][tid]
            m = mj['results'][tid]['variants'][arm]
            require(all(case[k] == gen[k] == bound[k] == out[k] for k in binding.IDENTITY), 'Paired case identity')
            require(gen['initial_noise'] == bound['initial_noise'] == base_binding['initial_noise']
                    and gen['initial_noise']['artifact'] == ref['initial_noise']['artifact']
                    and gen['initial_noise']['tensor'] == ref['initial_noise']['tensor'], 'Shared actual initial noise')
            require(gen['embeddings'] == bound['embeddings'] == base_binding['embeddings'], 'Shared actual text embeddings')
            video = record(bound['video'])
            require(video == gen['video'] == out['video'] and video['sha256'] == bound['video_sha256'] == m['video_sha256']
                    and video['bytes'] == bound['video_bytes'] and video['file'] == m['video'], 'Actual evaluated media identity')
            require(gen['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0)
                    and out['media'] == dict(decoded_frames=81, declared_frames=81., fps=16., width=832, height=480), '81-frame media')
            require(gen['raw_decoder']['dtype'] == 'torch.float32' and gen['raw_decoder']['finite'] is True
                    and gen['decoded_shape'] == [81, 480, 832, 3] and gen['decoded_finite'] is True, 'FP32 complete decode')
            final = saved_tensor(gen['final_latents'], torch)
            chain = step_records(gen, schedules[(arm, tid)])
            close(final['statistics']['min'], gen['steps'][-1]['min'], 'Final callback minimum')
            close(final['statistics']['max'], gen['steps'][-1]['max'], 'Final callback maximum')
            readout = recompute(out, m)
            arms[arm] = dict(metrics={k: readout['metrics'][k] for k in PRIMARY},
                video=video, media=out['media'], initial_noise=gen['initial_noise'], embeddings=gen['embeddings'],
                final_latents=final, scalar_steps_checked=chain['callbacks'], actual_counts=gen['actual_counts'],
                mj_aspects_raw=readout['mj_aspects'], mj_criteria_raw=readout['mj_criteria'],
                mj_sampled_indices=readout['mj_sampled_indices'], temporal_raw=readout['temporal_raw'])
        pairs = {f'{arm}_minus_bf16': difference(arms[arm]['metrics'], arms[BASE]['metrics']) for arm in ARMS}
        pairs['svdquant_minus_plain'] = difference(arms[ARMS[1]]['metrics'], arms[ARMS[0]]['metrics'])
        result['per_case'][tid] = dict(**case, arms=arms, paired_deltas=pairs)
    for arm, (out, index) in temporal.items():
        require(len(out['dino_real_pairs']) == 4, 'DINO pair coverage')
        for pair in out['dino_real_pairs']:
            close(pair['original_pair_score'], mean([result['per_case'][tid]['arms'][arm]['metrics']['subject_consistency']
                                                    for tid in pair['case_ids']]), 'DINO real-seed pair mean')
    for pid in dict.fromkeys(c['prompt_id'] for c in cases):
        pair = [r for r in result['per_case'].values() if r['prompt_id'] == pid]
        require([r['replica'] for r in pair] == [0, 1], 'Retain two seeds')
        result['per_prompt'][pid] = dict(prompt=pair[0]['prompt'], case_ids=[r['case_id'] for r in pair],
            arms={arm: average([r['arms'][arm]['metrics'] for r in pair]) for arm in (BASE, *ARMS)},
            paired_deltas={name: average([r['paired_deltas'][name] for r in pair]) for name in pair[0]['paired_deltas']})
    result.update(status='complete',
        arm_means={arm: average([r['arms'][arm] for r in result['per_prompt'].values()]) for arm in (BASE, *ARMS)},
        paired_mean_deltas={name: average([r['paired_deltas'][name] for r in result['per_prompt'].values()])
                            for name in next(iter(result['per_prompt'].values()))['paired_deltas']},
        dynamic_videos={arm: sum(r['arms'][arm]['metrics']['dynamic_degree'] for r in result['per_case'].values()) for arm in (BASE, *ARMS)},
        aggregation='Mean two seeds within each prompt, then mean four prompts equally; paired deltas are descriptive, not significance claims.',
        verification=dict(media=24, newly_evaluated=16, inherited_bf16=8, initial_noise_pairs=16,
            embedding_pairs=16, new_final_fp32_tensors=16, native_scalar_step_records=800,
            native_dit_receipts=1600, same_native_target_names=sorted(native_targets),
            intermediate_tensor_replay=False, rehashed_model_assets=False),
        input_reports=refs, sources=[record(__file__), record(binding.__file__)], cuda_initialized=torch.cuda.is_initialized())
    require(result['cuda_initialized'] is False, 'CPU summary initialized CUDA')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report_dir / 'evaluation_summary.json'
    require(not output.exists(), 'Preserve earlier result; use --output for a new attempt')
    result = dict(experiment='E044', status='failed_preserved', cpu_only=True)
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
