#!/usr/bin/env python3
"""Independent CPU paired-quality and execution readout for E046."""
import argparse
import json
import os
from pathlib import Path
import traceback

import wan_terminal_repair_evaluation as binding
from summarize_fastwan_qad_evaluation import record, recompute, average
from summarize_vanilla_wan_reference import PRIMARY, saved_tensor, step_records
from summarize_wan_terminal_intervention import pair_curve, tensor_signature
from summarize_wan_qad_videos import require, close, difference, mean, sha

RD, ARMS = binding.RD, binding.ARMS


def quality_reports(report_dir, manifest, load, refs):
    mj = load(report_dir/'mjvideo_scores.json')
    tids = [r['case_id'] for r in manifest['cases']]
    require(mj['num_segments'] == 8 and set(mj['results']) == set(tids), 'MJ eight cases')
    temporal = {}
    for arm in ARMS:
        path = report_dir/f'temporal_{arm}_manifest.json'
        sub = load(path)
        report = load(report_dir/f'temporal_{arm}_scores.json')
        require(report['status'] == 'complete' and report['arm'] == sub['variant'] == arm, 'Temporal arm complete')
        require(report['manifest']['sha256'] == report['generation_report']['sha256'] == refs[str(path)]['sha256'], 'Temporal binding')
        require(report['sampling'] == {k: v for k, v in binding.base.SAMPLING.items() if k != 'mjvideo_indices'}, 'Temporal sampling')
        rows = {r['case_id']: r for r in report['rows']}
        require(len(report['rows']) == len(rows) == 8 and set(rows) == set(tids), 'Temporal complete coverage')
        temporal[arm] = (report, rows)
    return mj, temporal


def summarize(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for CPU summary')
    import torch
    torch.set_num_threads(4)
    refs = {}

    def load(path):
        path = Path(path)
        refs[str(path)] = record(path)
        return json.loads(path.read_text())

    manifest_path = args.report_dir/'evaluation_manifest.json'
    manifest = load(manifest_path)
    require(manifest['status'] == 'complete' and manifest['experiment'] == 'E046', 'Prepared media binding')
    plan = load(manifest['source_manifest']['file'])
    require(refs[manifest['source_manifest']['file']] == manifest['source_manifest'], 'Frozen plan binding')
    cases = plan['cases']; tids = [r['case_id'] for r in cases]
    launch = load(args.report_dir/'launcher.json')
    require(launch['status'] == 'complete' and all(w['status'] == 'complete' and w['returncode'] == 0
                for w in launch['workers']), 'Generation complete')
    require(refs[str(args.report_dir/'launcher.json')] == manifest['generation_launcher'], 'Generation launcher binding')
    launch_refs = {w['result']['file']: w['result'] for w in launch['workers']}
    require(set(launch_refs) == {r['file'] for r in manifest['worker_reports']}, 'Actual worker coverage')
    for name in ('bf16', 'previous_native'):
        item = manifest['inherited'][name]
        value = load(item['file'])
        require(refs[item['file']] == item and value['status'] == 'complete', 'Historical result identity')
        if name == 'bf16':
            reference = value
        else:
            previous = value
    require(set(reference['per_case']) == set(previous['per_case']) == set(tids) and len(tids) == 8, 'Fixed historical case coverage')
    generated, workers = {}, []
    for item in manifest['worker_reports']:
        report = load(item['file'])
        require(refs[item['file']] == item == launch_refs[item['file']], 'Worker file identity')
        require(report['status'] == 'complete' and report['settings'] == plan['settings'], 'Worker complete/settings')
        workers.append((report, item))
        for case in report['cases']:
            require(case['status'] == 'complete' and case['case_id'] not in generated, 'Distinct complete generation cases')
            generated[case['case_id']] = (case, report)
    require(set(generated) == set(tids), 'All eight generated pairs')
    media_rows = {(r['arm'], r['case_id']): r for r in manifest['rows']}
    require(len(manifest['rows']) == len(media_rows) == 16
            and set(media_rows) == {(arm, tid) for arm in ARMS for tid in tids}, 'All 16 new media rows')
    mj, temporal = quality_reports(args.report_dir, manifest, load, refs)
    result = dict(experiment='E046', status='running', cpu_only=True, per_case={}, per_prompt={},
        limits=[
            'All eight already-observed diagnostic pairs retained; no held-out generalization or sample filtering claim.',
            'The paired intervention replaces the whole final deployed recipe with an independent original BASE BF16 transformer, not only its FP4 or LR branch.',
            'Same-run native_full is the comparison baseline. E044 SVD is a historical drift reference and never replaces a newly generated output.',
            'E043 BF16 is quality context. Different free-running final-state MSE is not a quality measure; no native-versus-BF16 trajectory NMSE is computed.',
            'MJ total is learned, not a probability or an average of aspects. Safety and bias remain raw per-case only.',
            'AMT can reward static/blurred video, RAFT dynamic degree is not action success, and DINO consistency is not semantic correctness.',
            'Full media decoding is inherited from complete temporal evaluation; actual media SHA is independently checked.',
            'Saved conditions/history and branch receipts establish the execution binding; this CPU reducer does not rerun DiT or the scheduler.',
            'Model loading, H2D, storage and memory records are observed instrumentation costs, not optimized deployment benchmarks. Two added DiT calls do not imply 2% latency or zero memory cost.'])
    # Worker execution/state verification is intentionally isolated from scoring.
    result['execution'] = execution_readout(workers, generated, reference, previous, plan, load, refs, torch)
    for case in cases:
        tid = case['case_id']; gen, worker = generated[tid]
        require(all(case[k] == gen[k] == reference['per_case'][tid][k] == previous['per_case'][tid][k]
                    for k in binding.IDENTITY), 'Paired identities')
        outputs = {r['variant']: r for r in gen['outputs']}
        require(len(gen['outputs']) == len(outputs) == 2 and set(outputs) == set(ARMS), 'Two actual branches')
        require(mj['results'][tid]['prompt'] == case['prompt'] and set(mj['results'][tid]['variants']) == set(ARMS), 'MJ variant coverage')
        scored = {}
        for arm in ARMS:
            output, bound = outputs[arm], media_rows[(arm, tid)]
            row, m = temporal[arm][1][tid], mj['results'][tid]['variants'][arm]
            require(output['status'] == 'complete' and all(row[k] == bound[k] == case[k] for k in binding.IDENTITY), 'Case/media identities')
            video = record(bound['video'])
            require(video == output['video'] == row['video'] and video['sha256'] == bound['video_sha256'] == m['video_sha256']
                    and video['bytes'] == bound['video_bytes'] and video['file'] == m['video'], 'Actual media identity')
            require(output['media'] == dict(frames=81, width=832, height=480, fps=16., audio_streams=0)
                    and row['media'] == dict(decoded_frames=81, declared_frames=81., fps=16., width=832, height=480), 'Actual full media geometry')
            readout = recompute(row, m)
            scored[arm] = dict(metrics={k: readout['metrics'][k] for k in PRIMARY}, video=video,
                media=row['media'], temporal_raw=readout['temporal_raw'], mj_aspects_raw=readout['mj_aspects'],
                mj_criteria_raw=readout['mj_criteria'], mj_sampled_indices=readout['mj_sampled_indices'])
        historical = dict(bf16=reference['per_case'][tid]['metrics'],
                          previous_native=previous['per_case'][tid]['arms']['svdquant_nvfp4']['metrics'])
        result['per_case'][tid] = dict(**case, arms=scored, historical_context=historical,
            paired_delta_bf16_last_minus_native=difference(scored['bf16_last']['metrics'], scored['native_full']['metrics']),
            minus_historical_bf16={arm: difference(scored[arm]['metrics'], historical['bf16']) for arm in ARMS},
            native_score_drift_from_e044=difference(scored['native_full']['metrics'], historical['previous_native']))
    for arm, (out, index) in temporal.items():
        require(len(out['dino_real_pairs']) == 4, 'Four DINO actual-seed pairs')
        for pair in out['dino_real_pairs']:
            close(pair['original_pair_score'], mean([result['per_case'][tid]['arms'][arm]['metrics']['subject_consistency']
                                                    for tid in pair['case_ids']]), 'DINO pair aggregate')
    for pid in dict.fromkeys(r['prompt_id'] for r in cases):
        pair = [r for r in result['per_case'].values() if r['prompt_id'] == pid]
        require([r['replica'] for r in pair] == [0, 1], 'Both seeds in manifest order')
        result['per_prompt'][pid] = dict(prompt=pair[0]['prompt'], case_ids=[r['case_id'] for r in pair],
            arms={arm: average([r['arms'][arm]['metrics'] for r in pair]) for arm in ARMS},
            paired_delta_bf16_last_minus_native=average([r['paired_delta_bf16_last_minus_native'] for r in pair]),
            historical_context={arm: average([r['historical_context'][arm] for r in pair]) for arm in ('bf16', 'previous_native')})
    result.update(status='complete',
        arm_means={arm: average([r['arms'][arm] for r in result['per_prompt'].values()]) for arm in ARMS},
        paired_mean_delta=average([r['paired_delta_bf16_last_minus_native'] for r in result['per_prompt'].values()]),
        historical_context_means={arm: average([r['historical_context'][arm] for r in result['per_prompt'].values()]) for arm in ('bf16', 'previous_native')},
        dynamic_videos={arm: sum(r['arms'][arm]['metrics']['dynamic_degree'] for r in result['per_case'].values()) for arm in ARMS},
        aggregation='Two seeds per prompt, then four prompts equally; all descriptive paired differences, no significance/quality thresholds.',
        input_reports=refs, source=record(__file__), cuda_initialized=torch.cuda.is_initialized())
    require(result['cuda_initialized'] is False, 'CPU reducer initialized CUDA')
    return result


def execution_readout(workers, generated, reference, previous, plan, load, refs, torch):
    expected = dict(native_dit=100, bf16_dit=2, native_gemm=30000, dit_sdpa=6120,
                    fastpack_checks=30000, native_scheduler=50, repair_scheduler=1,
                    public_vae_decode=2, text_encoder=0)
    totals = {k: 0 for k in expected}
    resources, details, bound = [], {}, {}

    def bind(entry):
        if entry['file'] not in bound:
            bound[entry['file']] = record(entry['file'])
        require(bound[entry['file']] == entry, 'Actual artifact changed')
        return entry['file']

    def cpu(entry):
        return torch.load(bind(entry), map_location='cpu', weights_only=False, mmap=True)

    targets = None
    for worker, source in workers:
        require(worker['manifest'] == refs[worker['manifest']['file']], 'Worker plan binding')
        names = set(worker['native_targets'])
        require(len(names) == len(worker['native_targets']) == worker['conversion']['target_count'] == 300, '300 native modules')
        if targets is None:
            targets = names
        require(names == targets, 'Same native module coverage')
        counts = {k: sum(c['actual_counts'][k] for c in worker['cases']) for k in expected}
        require(worker['actual_counts'] == counts, 'Worker/case count agreement')
        for key in totals:
            totals[key] += counts[key]
        teacher = worker['independent_teacher']
        require(teacher['source'] == 'original BASE transformer; no PTQ state/hooks', 'Independent BASE teacher source')
        for kind in ('allocated', 'reserved'):
            require(teacher[f'additional_{kind}_bytes'] == teacher[f'{kind}_after_bytes']-teacher[f'{kind}_before_bytes'],
                    'Recorded teacher memory increment')
        resources.append(dict(prompt_id=worker['prompt_id'], worker=source,
            independent_teacher=teacher, capture_position=worker['capture_position'],
            native_target_count=len(names), worker_seconds_including_all_diagnostics=worker['seconds_total']))
    require(totals == {key: 8*value for key, value in expected.items()}, 'Full eight-pair execution budget')
    for tid, (case, worker) in generated.items():
        require(case['actual_counts'] == expected, 'Per-case actual call budget')
        require(len(case['dit_calls']) == 102 and
            case['dit_calls'][:100] == [dict(mode='native', native_gemm=300, dit_sdpa=60, fastpack_checks=300, invalid_flags=0)]*100 and
            case['dit_calls'][100:] == [dict(mode='bf16', native_gemm=0, dit_sdpa=60, fastpack_checks=0, invalid_flags=0)]*2,
            'Actual full-native and two independent BF16 call receipts')
        e043 = load(bind(case['reference_worker']))
        old = next(r for r in e043['cases'] if r['case_id'] == tid)
        require(case['initial_noise'] == old['initial_noise']
                and case['initial_noise']['artifact'] == reference['per_case'][tid]['initial_noise']['artifact']
                and case['initial_noise']['tensor'] == reference['per_case'][tid]['initial_noise']['tensor'], 'Actual original noise')
        require(case['embeddings'] == e043['embeddings']['artifact'], 'Actual original text embeddings')
        chain = step_records(case, e043['cpu_schedule'])
        capture = cpu(case['terminal_capture'])
        require(len(capture['calls']) == 2 and capture['history']['_step_index'] == 49, 'Pre-last two-branch capture/history')
        sample = capture['sample']
        require(sample.dtype == torch.float32 and list(sample.shape) == [1, 16, 21, 60, 104]
                and bool(sample.isfinite().all()), 'Actual FP32 pre-step state')
        close(float(capture['timestep']), float(case['schedule']['timesteps'][-1]), 'Captured terminal timestep')
        close(float(capture['history']['sigmas'][49]), case['steps'][-1]['sigma'], 'Captured pre-step sigma')
        close(float(capture['history']['sigmas'][50]), case['steps'][-1]['next_sigma'], 'Captured post-step sigma')
        embeddings = cpu(case['embeddings'])
        input_checks = {}
        for branch, call, emb_key in zip(('cond', 'uncond'), capture['calls'], ('prompt_embeds', 'negative_prompt_embeds'), strict=True):
            kw = call['kwargs']
            input_checks[branch] = dict(
                latent_is_original_pre_step_cast=torch.equal(kw['hidden_states'], sample.to(torch.bfloat16)),
                actual_original_embedding=torch.equal(kw['encoder_hidden_states'], embeddings[emb_key]),
                same_timestep=bool((kw['timestep'] == capture['timestep']).all()))
            require(all(input_checks[branch].values()), 'Captured pre-quantization branch input identity')
        predictions = cpu(case['bf16_outputs']['artifact'])
        native_predictions = dict(cond=capture['calls'][0]['output'], uncond=capture['calls'][1]['output'], cfg=capture['cfg_output'])
        cfg_checks = {}
        for name, values in (('native', native_predictions), ('bf16', predictions)):
            require(all(t.dtype == torch.bfloat16 and list(t.shape) == list(sample.shape)
                        and bool(t.isfinite().all()) for t in values.values()), 'Actual BF16 predictions')
            cfg_checks[name] = pair_curve(values['uncond']+6.*(values['cond']-values['uncond']), values['cfg'], torch)
        for name, tensor in predictions.items():
            sig = tensor_signature(tensor, torch)
            require(all(sig[key] == case['bf16_outputs']['tensors'][name][key] for key in sig), 'Saved independent BF16 output')
        outputs = {r['variant']: r for r in case['outputs']}
        finals = {}
        for arm, entry in (('native_full', case['native_final']), ('bf16_last', case['repair_final'])):
            require(outputs[arm]['final_latents'] == entry, 'Actual branch final used for decode')
            finals[arm] = saved_tensor(entry, torch)
            decoder = outputs[arm]['raw_decoder']
            require(decoder['dtype'] == 'torch.float32' and decoder['finite'] is True
                    and decoder['shape'] == [1, 3, 81, 480, 832], 'Same complete FP32 VAE output contract')
        close(finals['native_full']['statistics']['min'], case['steps'][-1]['min'], 'Actual native final callback min')
        close(finals['native_full']['statistics']['max'], case['steps'][-1]['max'], 'Actual native final callback max')
        older = case['native_reference']['final_latents']
        require(older['artifact'] == previous['per_case'][tid]['arms']['svdquant_nvfp4']['final_latents']['artifact']
                and older['tensor'] == previous['per_case'][tid]['arms']['svdquant_nvfp4']['final_latents']['tensor'], 'Actual E044 native final reference')
        old_tensor = cpu(older['artifact'])
        native_tensor = cpu(case['native_final']['artifact'])
        drift = pair_curve(native_tensor, old_tensor, torch)
        details[tid] = dict(actual_counts=case['actual_counts'], original_noise=case['initial_noise'], embeddings=case['embeddings'],
            scalar_step_records=chain['callbacks'], terminal_capture=case['terminal_capture'],
            branch_input_identity=input_checks, cfg_cpu_replay=cfg_checks,
            scheduler=dict(pre_step_index=49, timestep=float(capture['timestep']),
                sigma=float(capture['history']['sigmas'][49]), next_sigma=float(capture['history']['sigmas'][50]),
                config=capture['scheduler_config'],
                evidence='Saved actual pre-step history/sample; reviewed worker reuses the captured inputs and copies that history for BF16 class.step. No independent second-copy history artifact or CPU scheduler replay.'),
            final_latents=finals, native_vs_e044_saved_final=drift, worker_native_vs_e044=case['native_vs_e044'],
            native_phase=case['native_phase'], bf16_phase=case['bf16_phase'],
            decode_phases={arm: {k: outputs[arm][k] for k in ('seconds_decode_media', 'peak_allocated_bytes', 'peak_reserved_bytes')} for arm in ARMS})
        del capture, predictions, native_predictions, sample, embeddings, old_tensor, native_tensor
    totals.update(dit=totals['native_dit']+totals['bf16_dit'], scheduler=totals['native_scheduler']+totals['repair_scheduler'])
    return dict(actual_totals=totals, workers=resources, cases=details, actual_artifact_bindings=bound,
                native_vs_e044_purpose='Numerical reproduction/drift only; no scientific byte gate or quality interpretation',
                native_vs_bf16_full_trajectory_mse_computed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report_dir/'evaluation_summary.json'
    require(not output.exists(), 'Preserve previous output; use --output for a new attempt')
    result = dict(experiment='E046', status='failed_preserved', cpu_only=True)
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
