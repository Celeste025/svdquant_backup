#!/usr/bin/env python3
"""E049 CPU reference readout: actual endpoints/receipts and unchanged metric formulas."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from summarize_fastwan_qad_evaluation import record, recompute, average
from summarize_vanilla_wan_reference import saved_tensor, step_records, PRIMARY
from summarize_wan_matched_calibration import checked_tensor
from summarize_wan_qad_videos import require, close, mean, sha
from wan14b_reference_evaluation import ARM, IDENTITY, PLAN, RD, ROOT, read_plan
from vanilla_wan_evaluation import SAMPLING

EXPECTED = dict(dit=100, dit_sdpa=8000, scheduler=50, public_vae_decode=1, text_encoder=0, native_gemm=0)


def summarize(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for CPU readout')
    import torch
    torch.set_num_threads(4)
    refs, checked_sources = {}, {}

    def load(path):
        path = Path(path).resolve(); refs[str(path)] = record(path)
        return json.loads(path.read_text())

    def bound(item):
        value = load(item['file'])
        require(all(refs[str(Path(item['file']).resolve())][k] == item[k]
                    for k in ('file', 'bytes', 'sha256')), 'Report identity')
        return value

    manifest = load(args.report_dir/'evaluation_manifest.json')
    plan = bound(manifest['source_manifest'])
    require(plan == read_plan(Path(manifest['source_manifest']['file'])), 'E049 recipe')
    launch, evaluation = (load(args.report_dir/name) for name in ('launcher.json', 'evaluation_launcher.json'))
    require(manifest['experiment'] == launch['experiment'] == evaluation['experiment'] == 'E049'
            and manifest['status'] == launch['status'] == evaluation['status'] == 'complete', 'Completed E049 inputs')
    launch_record = refs[str((args.report_dir/'launcher.json').resolve())]
    require(manifest['generation_launcher'] == evaluation['generation_launcher'] == launch_record, 'Generation binding')
    require(manifest['sampling'] == plan['settings'] and manifest['metric_sampling'] == SAMPLING, 'Sampling binding')
    require(launch['manifest'] == manifest['source_manifest'], 'Actual launch plan')
    for report in (launch, evaluation):
        require(all(w['status'] == 'complete' and w['returncode'] == 0 for w in report['workers']), 'Clean worker completion')
    require(len(launch['workers']) == 4 and len(evaluation['workers']) == 2, 'Four generation and two scoring workers')
    tids = [r['case_id'] for r in plan['cases']]
    rows = {r['case_id']: r for r in manifest['rows']}
    require(len(rows) == len(manifest['rows']) == 8 and set(rows) == set(tids), 'All eight media')
    subpath = args.report_dir/f'temporal_{ARM}_manifest.json'
    sub = load(subpath)
    temporal = load(args.report_dir/f'temporal_{ARM}_scores.json')
    mj = load(args.report_dir/'mjvideo_scores.json')
    require(temporal['status'] == 'complete' and temporal['arm'] == sub['variant'] == ARM, 'Temporal completion')
    require(temporal['manifest']['sha256'] == temporal['generation_report']['sha256']
            == refs[str(subpath.resolve())]['sha256'], 'Temporal media binding')
    require(temporal['sampling'] == {k: v for k, v in SAMPLING.items() if k != 'mjvideo_indices'}, 'Original temporal formula')
    require(mj['num_segments'] == 8 and set(mj['results']) == set(tids), 'MJ complete eight cases')
    scored = {r['case_id']: r for r in temporal['rows']}
    require(len(scored) == len(temporal['rows']) == 8 and set(scored) == set(tids), 'Temporal complete eight cases')
    for worker in evaluation['workers']:
        bound(worker['result'])
    launched = {w['result']['file']: w['result'] for w in launch['workers']}
    launch_rows = {w['result']['file']: w for w in launch['workers']}
    require(len(manifest['worker_reports']) == 4 and set(launched) == {r['file'] for r in manifest['worker_reports']}, 'Generation coverage')
    generated, schedules, receipts, e043, totals = {}, {}, [], {}, dict.fromkeys(EXPECTED, 0)
    for source in manifest['worker_reports']:
        worker = bound(source)
        require(source == launched[source['file']] and worker['status'] == 'complete'
                and worker['experiment'] == 'E049' and worker['arm'] == ARM, 'Completed 14B worker')
        require(worker['settings'] == plan['settings'] and worker['manifest'] == manifest['source_manifest'], 'Worker recipe')
        cfg = worker['actual_configs']
        require(cfg['transformer']['num_layers'] == 40 and cfg['scheduler']['flow_shift'] == 3., 'Actual 40-layer / shift3 model')
        require(worker['cpu_schedule']['config']['flow_shift'] == 3., 'CPU schedule uses E049 shift')
        model_cfg = bound(next(x for x in plan['model_metadata'] if x['file'].endswith('/transformer/config.json')))
        require(all(cfg['transformer'][k] == v for k, v in model_cfg.items() if not k.startswith('_')), 'Actual transformer matches declared14B configuration')
        # Sources are small code/config records; model weights are never rehashed here.
        for item in worker['sources']:
            key = item['file']
            if key not in checked_sources:
                require(Path(key).stat().st_size < 5_000_000, 'Unexpected large source record')
                checked_sources[key] = record(key)
            require(all(checked_sources[key][k] == item[k] for k in ('file','bytes','sha256')), 'Executed source binding')
        for key in ('asset_download', 'asset_supervisor', 'component_reuse'):
            receipt = bound(worker[key])
            if key != 'component_reuse':
                require(receipt['status'] == 'complete' and worker[key]['file'] == plan[key], 'Completed model asset provenance')
            if key == 'asset_download':
                require(receipt['repo'] == plan['model_repo'] and receipt['revision'] == plan['model_revision']
                        and Path(receipt['target']).resolve() == Path(plan['model_dir']).resolve(), 'Actual14B asset identity')
            elif key == 'asset_supervisor':
                require(receipt['returncode'] == 0, 'Asset supervisor rc0')
            else:
                require(worker[key] == plan[key] and receipt['all_identity_match'], 'Shared component identity')
        pid = launch_rows[source['file']]['prompt_id']
        reference = worker['reference_workers'][str(pid)]
        require(reference == plan['reference_workers'][str(pid)], 'E043 reference binding')
        e043[pid] = bound(reference)
        require(e043[pid]['status'] == 'complete', 'E043 original inputs complete')
        expected = [r['case_id'] for r in plan['cases'] if r['prompt_id'] == pid]
        require([r['case_id'] for r in worker['cases']] == expected
                and [r['case_id'] for r in worker['selected_cases']] == expected, 'Both fixed seeds in worker')
        count = {k: sum(r['actual_counts'][k] for r in worker['cases']) for k in EXPECTED}
        require(count == worker['actual_counts'], 'Worker count sum')
        for key in totals: totals[key] += count[key]
        receipts.append(dict(report=source, actual_counts=count, seconds=worker['seconds_total'],
            resident_allocated_bytes=worker['resident_allocated_bytes'], resident_storage_bytes=worker['resident_storage_bytes'],
            model_load_and_h2d_seconds=worker['model_load_and_h2d_seconds'], device=worker['device'],
            asset_download=worker['asset_download'], asset_supervisor=worker['asset_supervisor'],
            component_reuse=worker['component_reuse'], actual_transformer_config=cfg['transformer']))
        for row in worker['cases']:
            require(row['case_id'] not in generated and row['status'] == 'complete' and row['variant'] == ARM, 'Case identity')
            require(row['actual_counts'] == EXPECTED and len(row['dit_calls']) == 100
                    and all(call['index'] == i and call['dit_sdpa'] == 80 and call['native_gemm'] == 0
                            and call['input_dtype'] == 'torch.bfloat16' and call['input_shape'] == [1,16,21,60,104]
                            for i, call in enumerate(row['dit_calls'])), 'Actual40-layer BF16 calls')
            generated[row['case_id']], schedules[row['case_id']] = row, worker['cpu_schedule']
    require(set(generated) == set(tids) and totals == {k: 8*v for k,v in EXPECTED.items()}, 'Actual eight-case totals')
    result = dict(experiment='E049', status='running', cpu_only=True, variant=ARM, settings=plan['settings'],
        observed_generation_totals=totals, workers=receipts, per_case={}, per_prompt={}, limits=[
            'All eight previously observed diagnostic cases retained; four prompts with two related seeds. No held-out generalization, semantic eligibility threshold or significance claim.',
            'Wan14B uses CFG5/shift3 whereas E043 uses CFG6/shift8. Shared actual noise/embeddings do not isolate model size or quantization causally. No historical videos are rescored.',
            'Saved initial/embedding/final tensors and all scalar callbacks are checked; intermediate states and DiT predictions are not numerically replayed.',
            'Media SHA is independently checked; full81-frame decoding is inherited from temporal evaluation, not repeated here.',
            'AMT can reward static/blurred content; RAFT motion is not action completion; DINO consistency is not semantic accuracy. No composite score.',
            'MJ learned outputs are reaggregated, not model-forward recomputation. All28 criteria/five aspects retained, including safety/bias as raw only.',
            'DINO is recomputed from saved sum/80; unsaved individual similarities cannot be recovered.',
            'Loading/generation/diagnostics/decode wall and peaks are descriptive costs, not serving benchmarks.'])
    checked_embeddings, noise_hashes = {}, []
    for case in plan['cases']:
        tid = case['case_id']; gen, binding, out = generated[tid], rows[tid], scored[tid]
        require(all(all(r[k] == case[k] for k in IDENTITY) for r in (gen,binding,out)), 'Paired case identity')
        teacher_worker = e043[case['prompt_id']]
        teacher = next(r for r in teacher_worker['cases'] if r['case_id'] == tid)
        require(gen['initial_noise'] == binding['initial_noise'] == teacher['initial_noise'], 'Actual shared E043 noise')
        require(gen['embeddings'] == binding['embeddings'] == teacher_worker['embeddings']['artifact'], 'Actual shared E043 embeddings')
        require(gen['reference_worker'] == plan['reference_workers'][str(case['prompt_id'])]
                and gen['reference_case_id'] == tid, 'Generation original-case reference')
        require(gen['actual_inputs'] == dict(initial_noise=teacher['initial_noise']['tensor'],
                prompt_embeds=teacher_worker['embeddings']['prompt_embeds'],
                negative_prompt_embeds=teacher_worker['embeddings']['negative_prompt_embeds']), 'Actually passed E043 tensor signatures')
        initial, final = (saved_tensor(gen[k], torch) for k in ('initial_noise','final_latents'))
        noise_hashes.append(initial['tensor']['sha256'])
        if case['prompt_id'] not in checked_embeddings:
            require(record(gen['embeddings']['file']) == gen['embeddings'], 'Embedding file binding')
            values = torch.load(gen['embeddings']['file'], map_location='cpu', weights_only=True)
            for key in ('prompt_embeds','negative_prompt_embeds'):
                checked_tensor(values[key], teacher_worker['embeddings'][key], torch, shape=(1,512,4096), dtype=torch.bfloat16)
            checked_embeddings[case['prompt_id']] = gen['embeddings']; del values
        chain = step_records(gen, schedules[tid])
        close(final['statistics']['min'], gen['steps'][-1]['min'], 'Final callback min')
        close(final['statistics']['max'], gen['steps'][-1]['max'], 'Final callback max')
        video = record(binding['video']); item = mj['results'][tid]
        require(item['prompt'] == case['prompt'] and set(item['variants']) == {ARM}, 'MJ identity')
        item = item['variants'][ARM]
        require(video == gen['video'] == out['video'] and video['sha256'] == binding['video_sha256'] == item['video_sha256']
                and video['bytes'] == binding['video_bytes'] and video['file'] == item['video'], 'Actual evaluated media')
        require(gen['media'] == dict(frames=81,width=832,height=480,fps=16.,audio_streams=0)
                and out['media'] == dict(decoded_frames=81,declared_frames=81.,fps=16.,width=832,height=480), '81-frame media contract')
        require(gen['decoded_shape'] == [81,480,832,3] and gen['decoded_finite'] is True
                and gen['raw_decoder']['dtype'] == 'torch.float32' and gen['raw_decoder']['finite'] is True, 'Finite FP32 decode')
        readout = recompute(out,item)
        result['per_case'][tid] = dict(**case, video=video, media=out['media'], initial_noise=initial,
            embeddings=gen['embeddings'], final_latents=final, step_chain=chain, actual_counts=gen['actual_counts'],
            metrics={k: readout['metrics'][k] for k in PRIMARY}, temporal_raw=readout['temporal_raw'],
            mj_aspects_raw=readout['mj_aspects'], mj_criteria_raw=readout['mj_criteria'], mj_sampled_indices=readout['mj_sampled_indices'],
            seconds_pipeline_including_diagnostics=gen['seconds_pipeline_including_diagnostics'],
            peak_allocated_bytes=gen['peak_allocated_bytes'], peak_reserved_bytes=gen['peak_reserved_bytes'])
    require(len(set(noise_hashes)) == 8, 'Eight distinct actual E043 noises')
    for pid in dict.fromkeys(c['prompt_id'] for c in plan['cases']):
        pair = [r for r in result['per_case'].values() if r['prompt_id'] == pid]
        require([r['replica'] for r in pair] == [0,1], 'Both seeds retained')
        result['per_prompt'][pid] = dict(prompt=pair[0]['prompt'], case_ids=[r['case_id'] for r in pair],
            metrics=average([r['metrics'] for r in pair]),
            mj_aspects_raw=average([r['mj_aspects_raw'] for r in pair]),
            mj_criteria_raw=average([r['mj_criteria_raw'] for r in pair]))
    pairs = temporal['dino_real_pairs']
    require(len(pairs) == 4 and {tuple(p['case_ids']) for p in pairs}
            == {tuple(p['case_ids']) for p in result['per_prompt'].values()}, 'Four DINO seed pairs')
    for pair in pairs:
        close(pair['original_pair_score'], mean([result['per_case'][tid]['metrics']['subject_consistency'] for tid in pair['case_ids']]), 'DINO pair aggregate')
    result.update(status='complete', mean_metrics=average([r['metrics'] for r in result['per_prompt'].values()]),
        mean_mj_aspects_raw=average([r['mj_aspects_raw'] for r in result['per_prompt'].values()]),
        mean_mj_criteria_raw=average([r['mj_criteria_raw'] for r in result['per_prompt'].values()]),
        dynamic_videos=sum(r['metrics']['dynamic_degree'] for r in result['per_case'].values()),
        aggregation='Average two seeds per prompt, then four prompts equally; no filtering or composite score.',
        costs=dict(generation_wall_seconds=launch['seconds'], evaluation_wall_seconds=evaluation['seconds']),
        verification=dict(shared_e043_noises=8, shared_e043_embedding_pairs=8, actual_fp32_endpoints=8,
            callback_records=400, dit_receipts=800, model_layers=40, media=8,
            intermediate_trajectory_replay=False, model_weights_rehashed=False),
        input_reports=refs, sources=[record(__file__),record(ROOT/'scripts/research/summarize_fastwan_qad_evaluation.py'),
            record(ROOT/'scripts/research/summarize_vanilla_wan_reference.py'),
            record(ROOT/'scripts/research/summarize_wan_matched_calibration.py'),
            record(ROOT/'scripts/research/wan14b_reference_evaluation.py')], cuda_initialized=torch.cuda.is_initialized())
    require(result['cuda_initialized'] is False, 'CPU summary initialized CUDA')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); output = args.output or args.report_dir/'evaluation_summary.json'
    require(not output.exists(), 'Preserve prior result; explicit different --output required')
    started = time.monotonic(); result = dict(experiment='E049', status='failed_preserved', cpu_only=True)
    try:
        result = summarize(args)
    except BaseException:
        result['error'] = traceback.format_exc(); raise
    finally:
        result['seconds_cpu'] = time.monotonic()-started
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); tmp.replace(output)
        print(json.dumps(dict(status=result['status'], output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
