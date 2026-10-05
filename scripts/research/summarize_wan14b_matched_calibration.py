#!/usr/bin/env python3
"""E050 independent CPU cache audit, reusing E047 tensor/schedule reduction helpers."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback

from summarize_wan_matched_calibration import (
    ROOT, SHAPE, require, file_record, checked_file, checked_tensor,
    load_tensor_file, signature, check_steps,
)
RD = ROOT/'results/research/E050'


def summarize(args, result):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for this CPU-only summary')
    import torch
    import yaml
    torch.set_num_threads(4)
    refs = {}

    def read(path):
        path = Path(path).resolve()
        refs[str(path)] = file_record(path)
        return json.loads(path.read_text())

    manifest = read(args.manifest)
    selection = read(args.selection)
    launcher = read(args.report_dir/'collect_launcher.json')
    reference = read(args.reference)
    require(launcher['status'] == reference['status'] == 'complete', 'Collection/reference not complete')
    require(launcher['manifest'] == refs[str(args.manifest.resolve())], 'Current collection manifest binding')
    require(selection['status'] == 'complete', 'Selection audit incomplete')
    require(manifest['experiment'] == launcher['experiment'] == 'E050', 'E050 collection identity')
    old_manifest = read(checked_file(manifest['selection_manifest']))
    model_reference = read(checked_file(manifest['reference_manifest']))
    require(old_manifest['experiment'] == 'E047' and model_reference['experiment'] == 'E049', 'Inherited reference identity')
    require(reference['experiment'] == 'E049' and reference['manifest'] == manifest['reference_manifest'], 'Real14B schedule source')
    require(reference['actual_configs']['transformer']['num_layers'] == 40 and
            reference['actual_configs']['scheduler']['flow_shift'] == 3., 'E049 actual model/schedule')
    require(manifest['settings']['guidance_scale'] == 5. and manifest['settings']['flow_shift'] == 3., 'E050 recipe')
    require(manifest['prompts'] == old_manifest['prompts'] and manifest['selected_caches'] == old_manifest['selected_caches'],
            'Actual E047 selection/text/seed reused unchanged')
    reuse = read(checked_file(model_reference['component_reuse']))
    require(reuse['all_identity_match'] and reuse['revision'] == model_reference['model_revision'], 'Shared TE/tokenizer identity')
    collection = read(checked_file(manifest['collection_reference']))
    require(collection['status'] == 'complete' and collection['manifest'] == manifest['selection_manifest'], 'E047 collection complete')
    old_embeddings = {}
    for job in collection['workers']:
        require(job['status'] == 'complete' and job['returncode'] == 0, 'E047 source worker rc0')
        worker = read(checked_file(job['result']))
        require(worker['status'] == 'complete' and worker['manifest'] == manifest['selection_manifest'], 'E047 source identity')
        for row in worker['prompts']:
            require(row['name'] not in old_embeddings and row['status'] == 'complete', 'Unique completed embedding source')
            old_embeddings[row['name']] = dict(row=row, worker=job['result'])
    require(len(old_embeddings) == 14, 'Fourteen inherited embedding pairs')
    expected_selection = manifest['selected_caches']
    previous_selection = selection['selected_records']
    require([r['filename'] for r in expected_selection] == [r['filename'] for r in previous_selection],
            'Selection changed from original Random0 receipt')
    require(manifest['virtual_prompt_names'] == selection['virtual_prompt_names'], 'Virtual prompt set changed')
    for new, old in zip(expected_selection, previous_selection, strict=True):
        require(all(new[k] == old[k] for k in ('name', 'step', 'guidance', 'branch')), 'Selected record identity')
    source_prompts = yaml.safe_load(checked_file(selection['prompt_source']).read_text())
    planned = {r['name']: r for r in manifest['prompts']}
    require(len(planned) == len(manifest['prompts']) == 14, 'Fourteen unique prompts')
    for name, prompt in planned.items():
        seed = 0
        for char in name + '-0':
            seed = (seed * 31 + ord(char)) % (10**9 + 7)
        require(prompt['prompt'] == source_prompts[name] and prompt['seed'] == seed, 'Prompt/seed provenance')
    schedule = reference['cases'][0]['schedule']
    require(len(schedule['timesteps']) == 50 and len(schedule['sigmas']) == 51, 'E049 reference schedule')
    require(all(case['schedule'] == schedule for case in reference['cases']), 'E049 schedule consistency')
    result.update(selection_receipt=refs[str(args.selection.resolve())],
        selection_audit_manifest=selection['e047_manifest'], current_manifest=launcher['manifest'],
        manifest_revision_note='Selection receipt is the actual E047 audit, not a newly claimed E050 audit. '
            'The original selected filenames/branch/step, virtual prompts, text and seeds are compared directly. '
            'E050 changes the model/CFG/schedule and reuses only identity-matched text embeddings, not old trajectory states.',
        schedule_reference=refs[str(args.reference.resolve())], expected_schedule=schedule,
        prompts=[], caches=[], workers=[], input_reports=refs)
    expected_counts = dict(dit=1400, dit_sdpa=112000, scheduler=700, text_encoder=0,
                           public_vae_decode=0, saved_caches=64)
    totals = Counter()
    observed_names, observed_caches, artifact_paths = set(), {}, set()
    require(len(launcher['workers']) == 4, 'Four complete worker processes')
    for job in launcher['workers']:
        require(job['status'] == 'complete' and job['returncode'] == 0, 'Worker did not exit successfully')
        worker_path = checked_file(job['result'])
        worker = read(worker_path)
        require(worker['status'] == 'complete' and worker['manifest'] == launcher['manifest'], 'Worker state/manifest')
        require(worker['settings'] == manifest['settings'] and worker['experiment'] == 'E050', 'Actual14B settings')
        require(worker['reference_manifest'] == manifest['reference_manifest'] and
                worker['selection_manifest'] == manifest['selection_manifest'] and
                worker['collection_reference'] == manifest['collection_reference'], 'Worker inherited contracts')
        require(worker['component_reuse'] == model_reference['component_reuse'] and
                worker['actual_configs']['transformer']['num_layers'] == 40 and
                worker['actual_configs']['scheduler']['flow_shift'] == 3., 'Actual14B model and shared TE identity')
        require(worker['text_encoder_loaded'] is False and worker['tokenizer_loaded'] is False, 'No TE/tokenizer load')
        require(any(s == launcher['runner'] for s in worker['sources']), 'Worker collector source binding')
        require(worker['cpu_schedule']['timesteps'] == schedule['timesteps'] and
                worker['cpu_schedule']['sigmas'] == schedule['sigmas'], 'CPU schedule versus E049')
        require(set(job['prompt_names']) == {p['name'] for p in worker['prompts']}, 'Worker prompt allocation')
        worker_count = Counter()
        worker_caches = []
        for row in worker['prompts']:
            name = row['name']
            require(name not in observed_names and name in planned and row['status'] == 'complete', 'Prompt coverage')
            observed_names.add(name)
            require(all(row[k] == planned[name][k] for k in ('name', 'prompt', 'prompt_sha256', 'seed')), 'Prompt identity')
            planned_calls = [r for r in expected_selection if r['name'] == name]
            count = dict(dit=100, dit_sdpa=8000, scheduler=50, text_encoder=0,
                         public_vae_decode=0, saved_caches=len(planned_calls))
            require(row['actual_counts'] == count, 'Per-prompt execution receipts')
            worker_count.update(count)
            step_check = check_steps(row, schedule)
            noise = load_tensor_file(row['initial_noise']['artifact'], torch)
            noise_record = checked_tensor(noise, row['initial_noise']['tensor'], torch,
                                          shape=SHAPE, dtype=torch.float32)
            replay = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(row['seed']),
                                 dtype=torch.float32)
            require(torch.equal(noise, replay), 'Independent CPU seed/noise replay differs')
            final = load_tensor_file(row['final_latents']['artifact'], torch)
            final_record = checked_tensor(final, row['final_latents']['tensor'], torch,
                                          shape=SHAPE, dtype=torch.float32)
            inherited = old_embeddings[name]
            require(all(inherited['row'][k] == row[k] for k in ('name','prompt','prompt_sha256','seed')),
                    'Actual embedding prompt/seed provenance')
            require(row['embedding_source_worker'] == inherited['worker'] and
                    row['embeddings'] == inherited['row']['embeddings'] and
                    row['actual_embeddings'] == row['embeddings']['tensors'], 'Shared actual E047 embeddings')
            embeddings = load_tensor_file(row['embeddings']['artifact'], torch)
            require(set(embeddings) == {'prompt_embeds', 'negative_prompt_embeds'}, 'Two embedding branches')
            for key in embeddings:
                checked_tensor(embeddings[key], row['embeddings']['tensors'][key], torch,
                               shape=(1, 512, 4096), dtype=torch.bfloat16)
            for entry in row['caches']:
                filename = entry['filename']
                require(filename not in observed_caches and entry['name'] == name, 'Unique selected cache')
                planned_entry = next(p for p in planned_calls if p['filename'] == filename)
                require(all(entry[k] == v for k, v in planned_entry.items()), 'Cache selection identity')
                require(entry['transformer_call_index'] == 2 * entry['step'] + entry['guidance'], 'Actual CFG call order')
                require(entry['inputs_copied_before_forward'] and entry['output_is_actual_forward'], 'Capture semantics receipt')
                payload = load_tensor_file(entry['artifact'], torch)
                require(signature(payload, torch) == entry['tensors'], 'Actual cache tensor/metadata receipts')
                require(payload['filename'] == entry['cache_filename'] and
                        payload['step'] == entry['step'] and payload['guidance'] == entry['guidance'], 'Cache payload identity')
                require(len(payload['input_args']) == len(payload['outputs']) == 1, 'Single batch model I/O')
                x, y = payload['input_args'][0], payload['outputs'][0]
                xr = checked_tensor(x, None, torch, shape=SHAPE, dtype=torch.bfloat16)
                yr = checked_tensor(y, None, torch, shape=SHAPE, dtype=torch.bfloat16)
                kw = payload['input_kwargs']; t = kw['timestep']
                checked_tensor(t, None, torch, shape=(1,), dtype=torch.int64)
                require(float(t.item()) == schedule['timesteps'][entry['step']] and
                        entry['actual_timestep'] == t.tolist(), 'Actual model timestep versus E049')
                embedding_key = 'prompt_embeds' if entry['guidance'] == 0 else 'negative_prompt_embeds'
                require(torch.equal(kw['encoder_hidden_states'], embeddings[embedding_key]), 'Actual CFG embedding branch')
                require(kw['return_dict'] is False, 'Actual model output container contract')
                observed_caches[filename] = entry
                artifact_paths.add(str(Path(entry['artifact']['file']).resolve()))
                worker_caches.append(entry)
                result['caches'].append(dict(name=name, filename=filename, step=entry['step'],
                    guidance=entry['guidance'], branch=entry['branch'], artifact=entry['artifact'],
                    actual_timestep=t.item(), input=xr, output=yr, embedding_key=embedding_key,
                    embedding_exact=True, actual_tensor_receipts_verified=True))
                del payload, x, y, kw, t
            require({r['filename'] for r in row['caches']} == {r['filename'] for r in planned_calls}, 'Prompt selected cache coverage')
            result['prompts'].append(dict(**planned[name], initial_noise=row['initial_noise']['artifact'],
                independent_seed_replay_exact=True, initial_tensor=noise_record,
                final_latents=row['final_latents']['artifact'], final_tensor=final_record,
                embeddings=row['embeddings'], embedding_source_worker=row['embedding_source_worker'],
                actual_embedding_signatures_verified=True, step_check=step_check, actual_counts=count,
                seconds_including_capture_and_diagnostics=row['seconds_including_capture_and_diagnostics'],
                peak_allocated_bytes=row['peak_allocated_bytes'], peak_reserved_bytes=row['peak_reserved_bytes']))
            del embeddings, noise, replay, final
        require(dict(worker_count) == worker['actual_counts'] == job['actual_counts'], 'Worker call totals')
        require(sorted(worker_caches, key=lambda r: r['filename']) ==
                sorted(worker['caches'], key=lambda r: r['filename']), 'Worker cache table matches prompts')
        totals.update(worker_count)
        result['workers'].append(dict(group=job['group'], pid=worker['pid'], returncode=job['returncode'],
            report=job['result'], planned_gpu=job.get('planned_gpu', job['gpu']), actual_gpu=job['gpu'],
            device=worker['device'], actual_counts=dict(worker_count),
            seconds=worker['seconds'], model_load_seconds=worker['model_load_seconds'],
            resident_allocated_bytes=worker['resident_allocated_bytes'], sources=worker['sources']))
    require(observed_names == set(planned), 'Complete14 prompt coverage')
    require(set(observed_caches) == {r['filename'] for r in expected_selection}, 'Complete64 cache coverage')
    require(dict(totals) == expected_counts == launcher['actual_counts'], 'Total execution receipts')
    require(sorted(observed_caches.values(), key=lambda r: r['filename']) ==
            sorted(launcher['caches'], key=lambda r: r['filename']), 'Launcher cache table')
    actual_paths = {str(p.resolve()) for p in (Path(manifest['data_dir'])/'calibration/caches').glob('*.pt')}
    require(actual_paths == artifact_paths, 'Exactly64 selected .pt files in calibration directory')
    checked_file(launcher['runner'])
    result.update(status='complete', actual_counts=dict(totals), prompt_count=len(observed_names),
        cache_count=len(observed_caches), coverage=selection['coverage'], collector_source=launcher['runner'],
        operational_amendment=launcher.get('operational_amendment'), launcher_source=launcher['launcher'],
        source=file_record(__file__), inherited_helpers=file_record(ROOT/'scripts/research/summarize_wan_matched_calibration.py'),
        embeddings_reused=14, new_text_encoder_calls=0, old_trajectory_states_loaded=False, costs=dict(launcher_wall_seconds=launcher['seconds'],
            sum_worker_seconds=sum(w['seconds'] for w in result['workers']),
            sum_prompt_seconds=sum(p['seconds_including_capture_and_diagnostics'] for p in result['prompts']),
            max_prompt_peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in result['prompts']),
            max_prompt_peak_reserved_bytes=max(p['peak_reserved_bytes'] for p in result['prompts']),
            interpretation='Observed model loading/capture/hash/synchronization costs; parallel worker durations are not wall time or an optimized generation benchmark.'),
        cuda_initialized=torch.cuda.is_initialized())
    require(result['cuda_initialized'] is False, 'CUDA initialized in CPU reducer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--manifest', type=Path, default=ROOT/'research_state/06_experiments/E050_wan14b_matched_calibration_manifest.json')
    parser.add_argument('--reference', type=Path, default=ROOT/'results/research/E049/worker_161.json')
    parser.add_argument('--selection', type=Path, default=ROOT/'results/research/E047/selection_audit.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.report_dir/'collection_summary.json'
    require(not output.exists(), 'Preserve previous summary; use --output for an explicit retry')
    result = dict(experiment='E050', status='running', cpu_only=True,
        limitations=['Saved inputs, embeddings, output shapes/finite values and receipts are audited; DiT outputs are not recomputed.',
            'No per-step latent trajectory was saved. The50 scalar records and schedule are checked, not numerical trajectory replay.',
            'The64 caches are correlated calls from14 calibration trajectories, not64 independent prompts or held-out quality evidence.',
            'Original fast Random0 coverage gaps are retained; this is not a comparison of sampling policies.',
            'These are new14B trajectories with CFG5/shift3. Only TE embeddings and selection are reused; no model-size-only or other single-factor causal claim.'])
    started = time.monotonic()
    try:
        summarize(args, result)
    except BaseException:
        result.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        result['seconds_cpu'] = time.monotonic()-started
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix+'.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temp.replace(output)
        print(json.dumps(dict(status=result['status'], output=str(output), seconds=result['seconds_cpu'])))


if __name__ == '__main__':
    main()
