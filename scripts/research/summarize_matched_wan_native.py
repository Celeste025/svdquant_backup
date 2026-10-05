#!/usr/bin/env python3
"""Thin independent E048 readout, retaining eight paired diagnostic cases."""
import argparse
import json
import os
from pathlib import Path
import traceback

from summarize_fastwan_qad_evaluation import record, recompute, average
from summarize_vanilla_wan_reference import saved_tensor, step_records, PRIMARY
from summarize_vanilla_wan_native import EXPECTED
from summarize_wan_qad_videos import require, close, difference, mean, sha
from summarize_wan_matched_calibration import checked_tensor
from vanilla_wan_evaluation import SAMPLING, IDENTITY

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E048'
VARIANT = 'svdquant_nvfp4'
ARMS = ('bf16', 'old_svd', 'bf16_last', 'matched_svd')
HISTORICAL = ('E043', 'E044', 'E046')


def summarize(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for CPU summary')
    import torch
    torch.set_num_threads(4)
    refs = {}

    def load(path):
        path = Path(path).resolve()
        refs[str(path)] = record(path)
        return json.loads(path.read_text())

    def bound(item):
        value = load(item['file'])
        require(refs[str(Path(item['file']).resolve())] == item, 'Report file binding')
        return value

    manifest = load(args.report_dir/'evaluation_manifest.json')
    require(manifest['status'] == 'complete' and manifest['experiment'] == 'E048', 'E048 evaluation manifest')
    plan = bound(manifest['source_manifest'])
    launch = load(args.report_dir/'launcher.json')
    evaluation = load(args.report_dir/'evaluation_launcher.json')
    require(manifest['generation_launcher'] == evaluation['generation_launcher'] ==
            refs[str((args.report_dir/'launcher.json').resolve())], 'Generation-to-evaluation binding')
    for item in (launch, evaluation):
        require(item['status'] == 'complete' and all(w['status'] == 'complete' and w['returncode'] == 0
                for w in item['workers']), 'Generation/evaluation workers did not exit cleanly')
    tids = [r['case_id'] for r in plan['cases']]
    require(len(tids) == len(set(tids)) == 8, 'Eight fixed cases')
    rows = {r['case_id']: r for r in manifest['rows']}
    require(len(manifest['rows']) == len(rows) == 8 and set(rows) == set(tids)
            and all(r['arm'] == VARIANT for r in rows.values()), 'Eight new media bindings')
    historical = {}
    for experiment in HISTORICAL:
        item = manifest['inherited'][experiment]
        source = item['summary'] if 'summary' in item else item
        value = bound(source)
        require(value['status'] == 'complete' and value['experiment'] == experiment
                and set(value['per_case']) == set(tids), 'Complete same-case inherited summary')
        historical[experiment] = value
    mj = load(args.report_dir/'mjvideo_scores.json')
    sub_path = args.report_dir/f'temporal_{VARIANT}_manifest.json'
    sub = load(sub_path)
    temporal_path = args.report_dir/f'temporal_{VARIANT}_scores.json'
    temporal = load(temporal_path)
    require(mj['num_segments'] == 8 and set(mj['results']) == set(tids), 'MJ all eight cases')
    require(temporal['status'] == 'complete' and temporal['arm'] == sub['variant'] == VARIANT, 'Temporal completion')
    require(temporal['manifest']['sha256'] == temporal['generation_report']['sha256'] == refs[str(sub_path.resolve())]['sha256'],
            'Temporal input binding')
    require(temporal['sampling'] == {k: v for k, v in SAMPLING.items() if k != 'mjvideo_indices'}, 'Unchanged81-frame metrics')
    scored = {r['case_id']: r for r in temporal['rows']}
    require(len(temporal['rows']) == len(scored) == 8 and set(scored) == set(tids), 'Temporal coverage')
    for worker in evaluation['workers']:
        bound(worker['result'])
    launched = {w['result']['file']: w['result'] for w in launch['workers']}
    require(set(launched) == {r['file'] for r in manifest['worker_reports']}, 'Generation worker coverage')
    generated, receipts, totals, native_targets = {}, [], {k: 0 for k in EXPECTED}, None
    e043_workers = {}
    for source in manifest['worker_reports']:
        worker = bound(source)
        require(source == launched[source['file']] and worker['status'] == 'complete'
                and worker['experiment'] == 'E048' and worker['arm'] == VARIANT, 'Complete E048 worker')
        require(worker['manifest'] == manifest['source_manifest'] and worker['settings'] == plan['settings'], 'Worker recipe binding')
        target_names = set(worker['native_targets'])
        require(len(target_names) == len(worker['native_targets']) == worker['conversion']['target_count'] == 300,
                'All300 native modules')
        require(worker['conversion']['exact_roundtrip_count'] == 300, 'Inherited native conversion receipt')
        if native_targets is None:
            native_targets = target_names
        require(native_targets == target_names, 'Same targets across workers')
        ptq = bound(worker['matched_ptq_run'])
        require(ptq['status'] == 'complete' and ptq['experiment'] == 'E047' and
                ptq['checkpoint_identity'] == worker['matched_checkpoint_identity'], 'Completed matched PTQ binding')
        identity = bound(worker['matched_checkpoint_identity'])
        require(identity['status'] == 'complete' and worker['checkpoint_identity'] == worker['matched_checkpoint_identity'],
                'Actual loader matched checkpoint identity')
        count = {k: sum(row['actual_counts'][k] for row in worker['cases']) for k in EXPECTED}
        require(count == worker['actual_counts'], 'Worker case-count sum')
        for key in totals:
            totals[key] += count[key]
        receipts.append(dict(report=source, actual_counts=count, matched_ptq=worker['matched_ptq_run'],
            matched_checkpoint_identity=worker['matched_checkpoint_identity'], seconds_total=worker['seconds_total'],
            conversion_seconds=worker['conversion_seconds'], resident_allocated_bytes=worker['resident_allocated_bytes'],
            device=worker['device'], sources=worker['sources']))
        for row in worker['cases']:
            tid = row['case_id']
            require(tid not in generated and row['status'] == 'complete' and row['variant'] == VARIANT, 'Generation identity')
            require(row['actual_counts'] == EXPECTED and len(row['dit_calls']) == 100
                    and all(c == dict(native_gemm=300, dit_sdpa=60, fastpack_checks=300, invalid_flags=0)
                            for c in row['dit_calls']), 'Per-case real call receipts')
            require(row['reference_case_id'] == tid and row['reference_worker'] == worker['references'][str(row['prompt_id'])],
                    'Same E043 reference worker')
            if row['prompt_id'] not in e043_workers:
                e043_workers[row['prompt_id']] = bound(row['reference_worker'])
            generated[tid] = row
    require(set(generated) == set(tids) and all(totals[k] == 8 * v for k, v in EXPECTED.items()), 'Complete8 generations/totals')
    result = dict(experiment='E048', status='running', cpu_only=True, per_case={}, per_prompt={},
        actual_new_generation_totals=totals, worker_receipts=receipts,
        arm_definitions=dict(bf16='E043 original BASE BF16', old_svd='E044 old33-frame SVD calibration',
            bf16_last='E046 old SVD free trajectory with original BF16 final update',
            matched_svd='E048 E047 matched81-frame/shift8 SVD calibration'),
        limits=[
            'All eight previously observed diagnostic cases are retained; no held-out generalization, significance or quality-equivalence claim.',
            'Calibration frame count and scheduler trajectory changed together. This completes a stronger baseline and does not identify a single causal factor.',
            'E043/E044/E046 scores are inherited, not re-evaluated. Only the eight E048 temporal arrays and MJ outputs are independently reaggregated.',
            'Saved actual initial noise/embedding/final tensors and scalar schedules are checked. No intermediate numerical trajectory or model output is recomputed.',
            'MJ learned scores, AMT smoothness, RAFT dynamic degree and DINO consistency are descriptive and do not by themselves establish completed actions.',
            'Generation costs include loading, conversion, diagnostics and decode; no optimized serving-speed claim is made.'])
    checked_embeddings = {}
    for case in plan['cases']:
        tid = case['case_id']; gen, media, out = generated[tid], rows[tid], scored[tid]
        refs_case = {ex: historical[ex]['per_case'][tid] for ex in HISTORICAL}
        require(all(all(r[k] == case[k] for k in IDENTITY) for r in (gen, media, out, *refs_case.values())), 'Paired identities')
        teacher_worker = e043_workers[case['prompt_id']]
        teacher = next(r for r in teacher_worker['cases'] if r['case_id'] == tid)
        require(gen['initial_noise'] == media['initial_noise'] == teacher['initial_noise'], 'Same actual initial noise')
        noise = saved_tensor(gen['initial_noise'], torch)
        require(noise['artifact'] == refs_case['E043']['initial_noise']['artifact'] and
                noise['tensor'] == refs_case['E043']['initial_noise']['tensor'], 'Inherited verified initial tensor')
        require(gen['embeddings'] == media['embeddings'] == teacher_worker['embeddings']['artifact'], 'Same actual embeddings')
        if case['prompt_id'] not in checked_embeddings:
            require(record(gen['embeddings']['file']) == gen['embeddings'], 'Embedding artifact')
            tensors = torch.load(gen['embeddings']['file'], map_location='cpu', weights_only=True)
            for key in ('prompt_embeds', 'negative_prompt_embeds'):
                checked_tensor(tensors[key], teacher_worker['embeddings'][key], torch,
                               shape=(1, 512, 4096), dtype=torch.bfloat16)
            checked_embeddings[case['prompt_id']] = gen['embeddings']
            del tensors
        video = record(media['video']); m = mj['results'][tid]
        require(m['prompt'] == case['prompt'] and set(m['variants']) == {VARIANT}, 'MJ identity/variant')
        m = m['variants'][VARIANT]
        require(video == gen['video'] == out['video'] and video['sha256'] == media['video_sha256'] == m['video_sha256']
                and video['bytes'] == media['video_bytes'] and video['file'] == m['video'], 'Actual media evaluated')
        require(gen['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0)
                and out['media'] == dict(decoded_frames=81, declared_frames=81., fps=16., width=832, height=480), 'Full81-frame geometry')
        require(gen['raw_decoder']['dtype'] == 'torch.float32' and gen['raw_decoder']['finite'] is True
                and gen['decoded_shape'] == [81, 480, 832, 3] and gen['decoded_finite'] is True, 'FP32 finite decode receipt')
        final = saved_tensor(gen['final_latents'], torch)
        chain = step_records(gen, refs_case['E043']['step_chain']['schedule'])
        close(final['statistics']['min'], gen['steps'][-1]['min'], 'Final callback min')
        close(final['statistics']['max'], gen['steps'][-1]['max'], 'Final callback max')
        values = recompute(out, m)
        arms = dict(bf16={k: refs_case['E043'][k] for k in ('metrics', 'video', 'media')},
            old_svd={k: refs_case['E044']['arms'][VARIANT][k] for k in ('metrics', 'video', 'media')},
            bf16_last={k: refs_case['E046']['arms']['bf16_last'][k] for k in ('metrics', 'video', 'media')})
        arms['matched_svd'] = dict(metrics={k: values['metrics'][k] for k in PRIMARY}, video=video, media=out['media'],
            initial_noise=noise, embeddings=gen['embeddings'], final_latents=final,
            scalar_steps_checked=chain['callbacks'], actual_counts=gen['actual_counts'],
            temporal_raw=values['temporal_raw'], mj_aspects_raw=values['mj_aspects'],
            mj_criteria_raw=values['mj_criteria'], mj_sampled_indices=values['mj_sampled_indices'],
            seconds_pipeline_including_diagnostics=gen['seconds_pipeline_including_diagnostics'],
            peak_allocated_bytes=gen['peak_allocated_bytes'], peak_reserved_bytes=gen['peak_reserved_bytes'])
        deltas = {f'matched_minus_{arm}': difference(arms['matched_svd']['metrics'], arms[arm]['metrics']) for arm in ARMS[:-1]}
        result['per_case'][tid] = dict(**case, arms=arms, paired_deltas=deltas)
    require(len(temporal['dino_real_pairs']) == 4, 'Four original DINO seed pairs')
    for pair in temporal['dino_real_pairs']:
        close(pair['original_pair_score'], mean([result['per_case'][tid]['arms']['matched_svd']['metrics']['subject_consistency']
                                              for tid in pair['case_ids']]), 'DINO pair aggregate')
    for pid in dict.fromkeys(c['prompt_id'] for c in plan['cases']):
        pair = [r for r in result['per_case'].values() if r['prompt_id'] == pid]
        require([r['replica'] for r in pair] == [0, 1], 'Both seeds retained')
        result['per_prompt'][pid] = dict(prompt=pair[0]['prompt'], case_ids=[r['case_id'] for r in pair],
            arms={arm: average([r['arms'][arm]['metrics'] for r in pair]) for arm in ARMS},
            paired_deltas={key: average([r['paired_deltas'][key] for r in pair]) for key in pair[0]['paired_deltas']},
            per_seed_deltas=[dict(case_id=r['case_id'], replica=r['replica'], values=r['paired_deltas']) for r in pair])
    result.update(status='complete', arm_means={arm: average([r['arms'][arm] for r in result['per_prompt'].values()]) for arm in ARMS},
        paired_mean_deltas={key: average([r['paired_deltas'][key] for r in result['per_prompt'].values()])
                           for key in next(iter(result['per_prompt'].values()))['paired_deltas']},
        dynamic_videos={arm: sum(r['arms'][arm]['metrics']['dynamic_degree'] for r in result['per_case'].values()) for arm in ARMS},
        dynamic_transitions_vs_old_svd={f'{old}_to_{new}': sum(
            r['arms']['old_svd']['metrics']['dynamic_degree'] == old and r['arms']['matched_svd']['metrics']['dynamic_degree'] == new
            for r in result['per_case'].values()) for old in (0, 1) for new in (0, 1)},
        aggregation='Mean both seeds within each prompt, then mean all four prompts equally; no sample exclusions or significance claim.',
        costs=dict(generation_launcher_wall_seconds=launch['seconds'], evaluation_wall_seconds=evaluation['seconds']),
        verification=dict(new_media=8, inherited_score_sets=3, initial_noise_pairs=8, embedding_pairs=8,
            new_final_fp32_tensors=8, native_scalar_step_records=400, native_dit_receipts=800,
            same_native_target_names=sorted(native_targets), intermediate_tensor_replay=False, rehashed_model_assets=False),
        input_reports=refs, sources=[record(__file__), record(ROOT/'scripts/research/summarize_vanilla_wan_native.py'),
            record(ROOT/'scripts/research/summarize_wan_matched_calibration.py'),
            record(ROOT/'scripts/research/wan_matched_native_evaluation.py')],
        cuda_initialized=torch.cuda.is_initialized())
    require(result['cuda_initialized'] is False, 'CPU summary initialized CUDA')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report_dir/'evaluation_summary.json'
    require(not output.exists(), 'Preserve prior result; explicit --output required for retry')
    result = dict(experiment='E048', status='failed_preserved', cpu_only=True)
    try:
        result = summarize(args)
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)
        print(json.dumps(dict(status=result['status'], output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
