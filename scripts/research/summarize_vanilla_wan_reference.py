#!/usr/bin/env python3
"""Thin CPU readout of all eight E043 original-Wan reference videos.

Check saved initial/final tensors and the 50-step scalar/schedule records, then
reaggregate existing temporal/MJ outputs. No inference or cross-model deltas.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from summarize_fastwan_qad_evaluation import record, average, recompute
from summarize_wan_qad_videos import ROOT, require, finite, close, sha, mean

RD = ROOT / 'results/research/E043'
PLAN = ROOT / 'research_state/06_experiments/E043_vanilla_wan_manifest.json'
ARM = 'vanilla_wan_bf16'
IDENTITY = ('case_id', 'prompt_id', 'replica', 'prompt', 'seed')
PRIMARY = ('mj_total', 'mj_alignment', 'mj_fineness', 'mj_coherence_consistency',
           'motion_smoothness', 'dynamic_degree', 'subject_consistency')


def saved_tensor(entry, torch):
    actual_file = record(entry['artifact']['file'])
    require(actual_file == entry['artifact'], 'Saved tensor artifact identity')
    x = torch.load(actual_file['file'], map_location='cpu', weights_only=True)
    require(isinstance(x, torch.Tensor) and x.device.type == 'cpu', 'Expected saved CPU tensor')
    x = x.contiguous()
    actual = dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(x.isfinite().all()),
                  sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    require(actual == entry['tensor'] and actual['finite'], 'Saved tensor contents')
    require(x.dtype == torch.float32 and list(x.shape) == [1, 16, 21, 60, 104], 'FP32 latent shape')
    return dict(artifact=actual_file, tensor=actual,
                statistics=dict(min=float(x.min()), max=float(x.max()),
                                rms=float(x.double().square().mean().sqrt())))


def step_records(row, cpu_schedule):
    schedule, steps = row['schedule'], row['steps']
    times, sigmas = schedule['timesteps'], schedule['sigmas']
    require(len(times) == len(steps) == 50 and len(sigmas) == 51, 'Complete 50-step records')
    require(all(finite(v) for v in times + sigmas), 'Finite schedule')
    require(all(a > b for a, b in zip(times, times[1:]))
            and all(a >= b for a, b in zip(sigmas, sigmas[1:])), 'Descending schedule')
    require(len(cpu_schedule['timesteps']) == 50 and len(cpu_schedule['sigmas']) == 51, 'CPU schedule')
    for values, expected in ((times, cpu_schedule['timesteps']), (sigmas, cpu_schedule['sigmas'])):
        for value, reference in zip(values, expected, strict=True):
            close(value, reference, 'Actual versus CPU schedule', atol=1e-6)
    for i, step in enumerate(steps):
        require(step['index'] == i and step['dtype'] == 'torch.float32' and step['finite'] is True,
                'Consecutive finite FP32 callback records')
        require(all(finite(step[k]) for k in ('rms', 'min', 'max'))
                and step['rms'] >= 0 and step['min'] <= step['max'], 'Step scalar statistics')
        for key, value in (('timestep', times[i]), ('sigma', sigmas[i]), ('next_sigma', sigmas[i+1])):
            close(step[key], value, 'Callback ' + key, atol=1e-6)
        if i:
            close(steps[i-1]['next_sigma'], step['sigma'], 'Consecutive sigma chain', atol=1e-6)
    return dict(callbacks=50, schedule=schedule, records=steps,
                coverage='All post-update scalar and schedule records; intermediate tensors were not saved, so this is not numerical trajectory replay.')


def summarize(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run with CUDA_VISIBLE_DEVICES empty')
    import torch
    torch.set_num_threads(4)
    references = {}

    def load(path):
        path = Path(path)
        references[str(path)] = record(path)
        return json.loads(path.read_text())

    plan = load(args.plan)
    manifest_path = args.report_dir / 'evaluation_manifest.json'
    manifest = load(manifest_path)
    launcher = load(args.report_dir / 'launcher.json')
    temporal = load(args.temporal or args.report_dir / 'temporal_scores.json')
    mj = load(args.mjvideo or args.report_dir / 'mjvideo_scores.json')
    require(plan['experiment'] == manifest['experiment'] == 'E043', 'Experiment identity')
    require(manifest['status'] == launcher['status'] == temporal['status'] == 'complete', 'Upstream incomplete')
    require(manifest['variant'] == ARM and manifest['source_manifest'] == references[str(args.plan)], 'Evaluation plan binding')
    require(launcher['manifest'] == manifest['source_manifest'], 'Launcher plan binding')
    require(manifest['sampling'] == plan['settings'], 'Settings binding')
    require(temporal['manifest']['sha256'] == temporal['generation_report']['sha256']
            == references[str(manifest_path)]['sha256'], 'Temporal binding')
    require(mj['num_segments'] == 8, 'MJ sampling')
    cases = plan['cases']
    tids = [r['case_id'] for r in cases]
    require(len(tids) == len(set(tids)) == 8 and len({r['seed'] for r in cases}) == 8, 'Eight fixed cases/seeds')
    require([r['case_id'] for r in manifest['cases']] == tids, 'Fixed evaluation order')
    require(set(mj['results']) == set(tids), 'MJ all-eight coverage')
    evaluated = {r['case_id']: r for r in temporal['rows']}
    require(len(temporal['rows']) == len(evaluated) == 8 and set(evaluated) == set(tids), 'Temporal coverage')
    require(temporal['sampling'] == dict(amt_input_frames=list(range(0, 81, 2)),
                amt_reference_frames=list(range(1, 81, 2)), raft_frames=list(range(0, 81, 2)),
                raft_iterations=20, dino_frames=list(range(81))), 'Original temporal readout')
    workers = launcher['workers']
    require(len(workers) == 4 and {r['prompt_id'] for r in workers} == {161, 192, 269, 316}, 'Four workers')
    generated, schedules, worker_readout = {}, {}, []
    totals = dict(dit=0, scheduler=0, public_vae_decode=0, text_encoder=0)
    bound_workers = {r['file']: r for r in manifest['worker_reports']}
    require(len(bound_workers) == 4, 'Four bound worker reports')
    for worker in workers:
        require(worker['status'] == 'complete' and worker['returncode'] == 0, 'Worker not cleanly complete')
        path = Path(worker['result']['file'])
        report = load(path)
        require(references[str(path)] == worker['result'] == bound_workers[str(path)], 'Worker report binding')
        require(report['status'] == 'complete' and report['phase'] == 'generate'
                and report['prompt_id'] == worker['prompt_id'], 'Worker identity')
        require(report['settings'] == plan['settings'] and report['manifest'] == manifest['source_manifest'], 'Worker settings')
        require(report['actual_configs']['scheduler']['flow_shift'] == 8., 'Actual flow shift')
        expected = [r for r in cases if r['prompt_id'] == worker['prompt_id']]
        require([r['case_id'] for r in report['cases']] == [r['case_id'] for r in expected], 'Worker fixed cases')
        summed = {key: sum(r['actual_counts'][key] for r in report['cases']) for key in totals}
        summed['text_encoder'] += report['embeddings']['actual_text_encoder_calls']
        require(summed == report['actual_counts'], 'Worker observed call accounting')
        for key in totals:
            totals[key] += summed[key]
        worker_readout.append(dict(prompt_id=worker['prompt_id'], pid=worker['pid'], actual_counts=summed,
                                   seconds=report['seconds_total'], report=references[str(path)]))
        for row in report['cases']:
            generated[row['case_id']] = row
            schedules[row['case_id']] = report['cpu_schedule']
    require(set(generated) == set(tids) and {k: totals[k] for k in ('dit', 'scheduler', 'public_vae_decode')}
            == dict(dit=800, scheduler=400, public_vae_decode=8), 'Actual total calls')
    result = dict(experiment='E043', status='running', cpu_only=True, variant=ARM,
                  settings=plan['settings'], observed_generation_totals=totals, workers=worker_readout,
                  per_case={}, per_prompt={}, limits=[
        'All eight fixed videos retained; four purposively chosen actions with two seeds each. No score threshold or sample filtering.',
        'This is original-Wan reference readiness, not a causal comparison with another model or a quantization quality claim.',
        'Only initial and final tensors were saved. The 50-step check covers scalar records, order and sigma continuity, not intermediate tensor replay.',
        'Full media decoding is inherited from the temporal evaluator; this reducer independently checks each media SHA, not a second full decode.',
        'AMT/RAFT arrays are independently reaggregated. DINO is normalized from its saved sum; unsaved individual similarities are not reconstructed.',
        'MJ total/aspects/criteria are learned outputs, not recomputed inference. Safety and bias scores remain raw per-case only.',
        'AMT may reward static or blurred video; RAFT dynamic degree is a binary motion readout, not monotonic quality; DINO consistency is not semantic correctness.'])
    noise_hashes = []
    for case, binding in zip(cases, manifest['cases'], strict=True):
        tid = case['case_id']
        gen, row = generated[tid], evaluated[tid]
        require(gen['status'] == 'complete' and gen['variant'] == ARM, 'Generation case incomplete')
        require(all(case[k] == binding[k] == gen[k] == row[k] for k in IDENTITY), 'Case identity')
        require(gen['actual_counts'] == dict(dit=100, scheduler=50, public_vae_decode=1, text_encoder=0), 'Per-case calls')
        require(mj['results'][tid]['prompt'] == case['prompt'] and set(mj['results'][tid]['variants']) == {ARM}, 'MJ identity')
        m = mj['results'][tid]['variants'][ARM]
        video = record(binding['video'])
        require(video == gen['video'] == row['video'] and video['sha256'] == binding['video_sha256'] == m['video_sha256']
                and video['bytes'] == binding['video_bytes'] and video['file'] == m['video'], 'Media identity')
        require(gen['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0)
                and row['media'] == dict(decoded_frames=81, declared_frames=81., fps=16., width=832, height=480), 'Actual 81-frame media')
        require(gen['decoded_shape'] == [81, 480, 832, 3] and gen['decoded_finite'] is True
                and gen['raw_decoder']['dtype'] == 'torch.float32' and gen['raw_decoder']['finite'] is True, 'FP32 decode receipts')
        initial, final = (saved_tensor(gen[k], torch) for k in ('initial_noise', 'final_latents'))
        noise_hashes.append(initial['tensor']['sha256'])
        chain = step_records(gen, schedules[tid])
        close(final['statistics']['min'], gen['steps'][-1]['min'], 'Final callback minimum')
        close(final['statistics']['max'], gen['steps'][-1]['max'], 'Final callback maximum')
        readout = recompute(row, m)
        result['per_case'][tid] = dict(**case, video=video, media=row['media'],
            initial_noise=initial, final_latents=final, step_chain=chain, actual_counts=gen['actual_counts'],
            metrics={k: readout['metrics'][k] for k in PRIMARY},
            temporal_raw=readout['temporal_raw'], mj_aspects_raw=readout['mj_aspects'],
            mj_criteria_raw=readout['mj_criteria'], mj_sampled_indices=readout['mj_sampled_indices'])
    require(len(set(noise_hashes)) == 8, 'Eight actual initial noise tensors must differ')
    prompts = list(dict.fromkeys(r['prompt_id'] for r in cases))
    for pid in prompts:
        pair = [r for r in result['per_case'].values() if r['prompt_id'] == pid]
        require([r['replica'] for r in pair] == [0, 1], 'Two replicas per prompt')
        result['per_prompt'][pid] = dict(prompt=pair[0]['prompt'], case_ids=[r['case_id'] for r in pair],
                                        metrics=average([r['metrics'] for r in pair]))
    dino_pairs = temporal['dino_real_pairs']
    require(len(dino_pairs) == 4 and {tuple(p['case_ids']) for p in dino_pairs}
            == {tuple(p['case_ids']) for p in result['per_prompt'].values()}, 'DINO real-seed pair coverage')
    for pair in dino_pairs:
        close(pair['original_pair_score'], mean([result['per_case'][tid]['metrics']['subject_consistency']
                                               for tid in pair['case_ids']]), 'DINO pair aggregate')
    result.update(status='complete', mean_metrics=average([p['metrics'] for p in result['per_prompt'].values()]),
        dynamic_videos=sum(p['metrics']['dynamic_degree'] for p in result['per_case'].values()),
        aggregation='Mean both seeds within each prompt, then mean the four prompts equally; no composite score.',
        verification=dict(videos=8, prompts=4, actual_unique_initial_noise_hashes=8,
            initial_final_fp32_tensors=16, scalar_step_records=400, intermediate_tensor_replay=False,
            independently_rehashed_media=8, full_media_decode_inherited=True, model_assets_rehashed=False,
            amt_terms_per_video=40, raft_terms_per_video=40, dino_terms_per_video=80,
            mj_criteria=28, mj_aspects=5, mj_actual_indices=list(range(0, 80, 10))),
        input_reports=references, sources=[record(__file__), record(Path(__file__).with_name('summarize_fastwan_qad_evaluation.py'))],
        cuda_initialized=torch.cuda.is_initialized())
    require(result['cuda_initialized'] is False, 'CPU summary initialized CUDA')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--temporal', type=Path)
    parser.add_argument('--mjvideo', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report_dir / 'evaluation_summary.json'
    require(not output.exists(), 'Preserve earlier output; use --output for another attempt')
    result = dict(experiment='E043', status='failed_preserved', cpu_only=True)
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
