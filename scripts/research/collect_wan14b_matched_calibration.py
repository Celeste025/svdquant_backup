#!/usr/bin/env python3
"""E050: actual Wan14B CFG5/shift3 teacher caches, fixed E047 selection; no decode."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback
import sys

if '--check' in sys.argv and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CPU --check requires CUDA_VISIBLE_DEVICES empty before imports')

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel

from run_vanilla_wan_reference import record, save_json, config_dict, require

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E050_wan14b_matched_calibration_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E050_wan14b_matched_calibration_plan.md'
SHAPE = (1, 16, 21, 60, 104)
COUNT_KEYS = ('dit', 'dit_sdpa', 'scheduler', 'text_encoder', 'public_vae_decode', 'saved_caches')


# Keep the existing cache serialization and scalar tensor handling unchanged.
from collect_wan_matched_calibration import tensor_record, cpu_clone, signature


def bound(item):
    actual = record(item['file'])
    require(all(actual[k] == item[k] for k in ('file', 'bytes', 'sha256')), 'Bound small input changed')
    return Path(actual['file'])


def budget(args):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix,
            'E050 worker deadline reached')


def prerequisites(args, report):
    m = json.loads(args.manifest.read_text())
    require(m['experiment'] == 'E050', 'Expected E050 protocol')
    old = json.loads(bound(m['selection_manifest']).read_text())
    reference = json.loads(bound(m['reference_manifest']).read_text())
    require(old['experiment'] == 'E047' and reference['experiment'] == 'E049', 'Selection/model reference identity')
    require(m['prompts'] == old['prompts'] and m['selected_caches'] == old['selected_caches'],
            'Keep the exact original Random0 fourteen prompts/64 records')
    require(m['negative_prompt'] == old['negative_prompt'] == reference['negative_prompt']
            and m['negative_prompt_sha256'] == old['negative_prompt_sha256'], 'Negative prompt identity')
    prompts = {p['name']: p for p in m['prompts']}
    require(len(prompts) == len(m['prompts']) == 14 and len(m['selected_caches']) == 64, 'Fixed selection size')
    for prompt in prompts.values():
        require(hashlib.sha256(prompt['prompt'].encode()).hexdigest() == prompt['prompt_sha256'], 'Prompt identity')
    groups = m['worker_groups']; names = [n for g in groups for n in g['prompt_names']]
    require(len(names) == len(set(names)) == 14 and set(names) == set(prompts), 'Group coverage')
    require(len({str(g['group']) for g in groups}) == len(groups) == 4, 'Four distinct worker groups')
    require(Path(m['data_dir']).resolve() != Path(old['data_dir']).resolve(), 'Do not overwrite old calibration')
    settings = m['settings']
    for key in ('height','width','num_frames','num_inference_steps','guidance_scale','flow_shift',
                'transformer_dtype','latent_update_dtype','max_sequence_length','compile','offload'):
        require(settings[key] == reference['settings'][key], 'Use E049 official sampling: '+key)
    require(m['model_dir'] == reference['model_dir'] ==
            '/data1/models/svdquant-wjq/models/Wan2.1-T2V-14B-Diffusers-38ec498c', 'Fixed14B model target')
    require(m['component_dir'] == reference['component_dir'], 'Shared official components')
    require(reference['model_revision'] == '38ec498cb3208fb688890f8cc7e94ede2cbd7f68', 'Fixed14B revision')
    for item in reference['model_metadata']:
        bound(item)
    download_path, supervisor_path = Path(reference['asset_download']), Path(reference['asset_supervisor'])
    download, supervisor = json.loads(download_path.read_text()), json.loads(supervisor_path.read_text())
    require(download['status'] == supervisor['status'] == 'complete' and supervisor['returncode'] == 0,
            '14B asset download not complete')
    require(download['revision'] == reference['model_revision'] and download['repo'] == reference['model_repo']
            and Path(download['target']).resolve() == Path(m['model_dir']).resolve(), 'Official model provenance')
    require(download['header_validation']['status'] == 'complete' and download['header_validation']['index_exact'],
            'Official transformer index not verified')
    for entry in download['files'].values():
        require(entry['status'] == 'verified' and Path(entry['file']).stat().st_size == entry['bytes'], 'Asset receipt/stat')
    reuse = json.loads(bound(reference['component_reuse']).read_text())
    require(reuse['all_identity_match'] and reuse['revision'] == reference['model_revision'], 'TE/tokenizer/VAE identity')
    require(Path(old['model_dir']).resolve() == Path(m['component_dir']).resolve(), 'E047 embeddings came from shared components')
    configs = dict(transformer=json.loads((Path(m['model_dir'])/'transformer/config.json').read_text()),
                   scheduler=json.loads((Path(m['model_dir'])/'scheduler/scheduler_config.json').read_text()),
                   vae=json.loads((Path(m['component_dir'])/'vae/config.json').read_text()))
    require([configs['transformer'][k] for k in ('num_layers','num_attention_heads','attention_head_dim','ffn_dim')]
            == [40,40,128,13824], 'Expected original14B geometry')
    schedule = UniPCMultistepScheduler.from_config(configs['scheduler'])
    schedule.set_timesteps(50, device='cpu')
    require(schedule.config.flow_shift == 3. and len(schedule.timesteps) == 50, 'Original14B schedule')
    collection = json.loads(bound(m['collection_reference']).read_text())
    require(collection['status'] == 'complete' and collection['manifest'] == m['selection_manifest'], 'Completed E047 collection')
    inherited, worker_sources = {}, []
    for worker in collection['workers']:
        require(worker['status'] == 'complete' and worker['returncode'] == 0, 'E047 worker exit')
        source = worker['result']; value = json.loads(bound(source).read_text()); worker_sources.append(source)
        require(value['status'] == 'complete' and value['manifest'] == m['selection_manifest'], 'E047 worker provenance')
        for row in value['prompts']:
            name = row['name']; require(name not in inherited and row['status'] == 'complete', 'Embedding prompt uniqueness')
            require(all(row[k] == prompts[name][k] for k in ('name','prompt','prompt_sha256','seed')), 'Embedding text/seed identity')
            inherited[name] = dict(embeddings=row['embeddings'], source_worker=source)
    require(set(inherited) == set(prompts), 'All fourteen original embeddings')
    if args.group is None:
        require(args.check, '--group is required for GPU collection'); chosen = m['prompts']
    else:
        selected = [g for g in groups if str(g['group']) == args.group]
        require(len(selected) == 1, 'Unknown group'); chosen = [prompts[n] for n in selected[0]['prompt_names']]
    prepared = {}
    for prompt in chosen:
        item = inherited[prompt['name']]
        tensors = torch.load(bound(item['embeddings']['artifact']), map_location='cpu', weights_only=True)
        require(signature(tensors) == item['embeddings']['tensors'], 'Actual E047 embedding signature')
        for key in ('prompt_embeds','negative_prompt_embeds'):
            x = tensors[key]
            require(x.dtype == torch.bfloat16 and tuple(x.shape) == (1,512,4096) and bool(x.isfinite().all()), 'Embedding shape/dtype/finite')
        prepared[prompt['name']] = dict(**item, tensors=tensors)
    import diffusers
    paths = [Path(__file__), args.manifest, PLAN, Path(__file__).with_name('collect_wan_matched_calibration.py'),
             Path(__file__).with_name('run_vanilla_wan_reference.py')] + [Path(inspect.getsourcefile(cls)) for cls in
             (WanPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler)]
    report.update(manifest=record(args.manifest), sources=[record(p) for p in paths], settings=settings,
        disk_configs=configs, latent_shape=list(SHAPE), reference_manifest=m['reference_manifest'],
        selection_manifest=m['selection_manifest'], collection_reference=m['collection_reference'],
        asset_download=record(download_path), asset_supervisor=record(supervisor_path), component_reuse=reference['component_reuse'],
        inherited_embedding_workers=worker_sources,
        versions=dict(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__),
        cpu_schedule=dict(config=dict(schedule.config), timesteps=schedule.timesteps.tolist(), sigmas=schedule.sigmas.tolist()),
        selected_caches=m['selected_caches'], selected_prompts=chosen, worker_groups=groups,
        cache_contract='Unchanged E047 actual pre-forward input_args/input_kwargs and post-forward outputs; CPU tensors; filename/step/guidance.',
        initial_noise_contract='Fresh CPU torch.randn from each fixed seed. No old1.3B trajectory state, output or final latent loaded.',
        embeddings_reused=[dict(name=p['name'], embeddings=inherited[p['name']]['embeddings'],
            source_worker=inherited[p['name']]['source_worker']) for p in chosen])
    return m, chosen, prepared


@torch.inference_mode()
def run(args, m, prompts, prepared, report):
    budget(args)
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    root = Path(m['data_dir'])/'calibration'
    caches_dir = root/'caches'; caches_dir.mkdir(parents=True, exist_ok=True)
    (root/'prompts').mkdir(parents=True, exist_ok=True)
    selected = {(s['name'], s['step'], s['guidance']): s for s in m['selected_caches']}
    counts = report['actual_counts']
    loaded = time.monotonic()
    transformer = WanTransformer3DModel.from_pretrained(m['model_dir'], subfolder='transformer',
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(m['component_dir'], subfolder='vae',
        torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True)
    scheduler = UniPCMultistepScheduler.from_pretrained(m['model_dir'], subfolder='scheduler', local_files_only=True)
    pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=scheduler, transformer=transformer)
    pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
    require(pipe.transformer.dtype == torch.bfloat16 and pipe.vae.dtype == torch.float32 and
            pipe.text_encoder is None and pipe.tokenizer is None, 'Model/component dtype contract')
    require(len(pipe.transformer.blocks) == 40 and pipe.scheduler.config.flow_shift == 3., 'Actual14B recipe')
    torch.cuda.synchronize()
    report.update(model_load_seconds=time.monotonic()-loaded,
        actual_configs={k: config_dict(getattr(pipe,k).config) for k in ('transformer','vae','scheduler')},
        pipeline_config=dict(pipe.config), resident_allocated_bytes=torch.cuda.memory_allocated(),
        attention_backend='torch SDPA FLASH_ATTENTION inside every DiT; no compile/offload',
        sampler_input_dtype='torch.float32', transformer_input_dtype='torch.bfloat16', output_type='latent',
        text_encoder_loaded=False, tokenizer_loaded=False)
    original_dit = pipe.transformer.forward
    original_step, original_sdpa = pipe.scheduler.step, F.scaled_dot_product_attention
    forward_signature = inspect.signature(original_dit)
    active = {}

    def sdpa(query, key, value, *a, **kw):
        if active.get('inside_dit', False):
            require(query.dtype == key.dtype == value.dtype == torch.bfloat16, 'BF16 DiT attention')
        output = original_sdpa(query, key, value, *a, **kw)
        if active.get('inside_dit', False):
            counts['dit_sdpa'] += 1
        return output

    def dit(*a, **kw):
        budget(args)
        row = active['row']; before = active['before']
        index = counts['dit']-before['dit']; step_index, guidance = divmod(index, 2)
        require(step_index == counts['scheduler']-before['scheduler'], 'DiT CFG/update ordering differs')
        actual = dict(forward_signature.bind(*a, **kw).arguments)
        hidden = actual.pop('hidden_states')
        require(tuple(hidden.shape) == SHAPE and hidden.dtype == torch.bfloat16, 'DiT input shape/dtype differs')
        selection = selected.get((row['name'], step_index, guidance))
        cached = None
        if selection is not None:
            cached = dict(input_args=[cpu_clone(hidden)], input_kwargs=cpu_clone(actual))
            require(bool(torch.equal(cached['input_kwargs']['encoder_hidden_states'],
                                     active['embedding_cpu']['prompt_embeds' if guidance == 0 else 'negative_prompt_embeds'])),
                    'Actual selected CFG embedding does not match branch')
        active['inside_dit'] = True
        sdpa_before = counts['dit_sdpa']
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = original_dit(*a, **kw)
        finally:
            active['inside_dit'] = False
        counts['dit'] += 1
        require(counts['dit_sdpa']-sdpa_before == 80, 'Expected80 SDPA calls per40-layer teacher DiT')
        require(isinstance(output, tuple) and len(output) == 1 and output[0].dtype == torch.bfloat16 and
                tuple(output[0].shape) == SHAPE, 'Actual DiT output differs')
        if cached is not None:
            cached.update(outputs=(cpu_clone(output[0]),), filename=selection['cache_filename'],
                          step=step_index, guidance=guidance)
            destination = caches_dir/selection['filename']
            require(not destination.exists(), 'Preserve existing calibration cache')
            torch.save(cached, destination)
            entry = dict(**selection, artifact=record(destination), tensors=signature(cached),
                         actual_timestep=cached['input_kwargs']['timestep'].reshape(-1).tolist(),
                         transformer_call_index=index, dtype='torch.bfloat16', shape=list(SHAPE),
                         inputs_copied_before_forward=True, output_is_actual_forward=True)
            row['caches'].append(entry); report['caches'].append(entry); counts['saved_caches'] += 1
            save_json(report, args.output)
        return output

    def step(*a, **kw):
        budget(args); output = original_step(*a, **kw); counts['scheduler'] += 1
        return output

    def forbidden_decode(*a, **kw):
        counts['public_vae_decode'] += 1
        raise RuntimeError('E050 is latent-only; VAE decode must never run')

    pipe.transformer.forward = dit
    pipe.scheduler.step, pipe.vae.decode = step, forbidden_decode
    F.scaled_dot_product_attention = sdpa
    try:
        for prompt in prompts:
            budget(args)
            case_dir = root/'prompts'/prompt['name']; case_dir.mkdir(exist_ok=False)
            row = dict(**prompt, status='running', steps=[], caches=[])
            report['prompts'].append(row)
            before = counts.copy(); active.update(row=row, before=before)
            began = time.monotonic()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            source = prepared[prompt['name']]
            embedding_cpu = source['tensors']
            positive, negative = (embedding_cpu[k].to('cuda') for k in ('prompt_embeds','negative_prompt_embeds'))
            active['embedding_cpu'] = embedding_cpu
            row['embeddings'] = source['embeddings']
            row['embedding_source_worker'] = source['source_worker']
            row['actual_embeddings'] = signature(dict(prompt_embeds=positive, negative_prompt_embeds=negative))
            require(row['actual_embeddings'] == row['embeddings']['tensors'], 'Actually passed shared embedding')
            initial = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(prompt['seed']), dtype=torch.float32)
            initial_path = case_dir/'initial_noise.pt'; torch.save(initial, initial_path)
            row['initial_noise'] = dict(artifact=record(initial_path), tensor=tensor_record(initial))

            def callback(pipeline, index, t, values):
                budget(args)
                x = values['latents']
                require(x.dtype == torch.float32 and bool(x.isfinite().all()), 'Nonfinite/non-FP32 sampler state')
                row['steps'].append(dict(index=index, timestep=float(t), sigma=float(pipeline.scheduler.sigmas[index]),
                    next_sigma=float(pipeline.scheduler.sigmas[index+1]), dtype=str(x.dtype), finite=True,
                    rms=float(x.square().mean().sqrt()), min=float(x.min()), max=float(x.max())))
                if (index+1) % 10 == 0:
                    save_json(report, args.output)
                    print(f"{prompt['name']} {index+1}/50 updates, {len(row['caches'])} selected caches", flush=True)
                return values

            result = pipe(prompt_embeds=positive, negative_prompt_embeds=negative, latents=initial,
                height=480, width=832, num_frames=81, num_inference_steps=50, guidance_scale=5.,
                output_type='latent', callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'],
                max_sequence_length=512).frames
            torch.cuda.synchronize()
            require(isinstance(result, torch.Tensor) and tuple(result.shape) == SHAPE and
                    result.dtype == torch.float32 and bool(result.isfinite().all()), 'Final latent contract differs')
            final = cpu_clone(result); final_path = case_dir/'final_latents.pt'; torch.save(final, final_path)
            row.update(final_latents=dict(artifact=record(final_path), tensor=tensor_record(final)),
                       seconds_including_capture_and_diagnostics=time.monotonic()-began,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       actual_counts={k: counts[k]-before[k] for k in counts},
                       schedule=dict(timesteps=pipe.scheduler.timesteps.cpu().tolist(), sigmas=pipe.scheduler.sigmas.cpu().tolist()))
            expected_saved = sum(s['name'] == prompt['name'] for s in m['selected_caches'])
            require(row['actual_counts'] == dict(dit=100, dit_sdpa=8000, scheduler=50, text_encoder=0,
                                                 public_vae_decode=0, saved_caches=expected_saved), 'Prompt counts differ')
            require(len(row['steps']) == 50, 'Incomplete callback chain')
            row['status'] = 'complete'; save_json(report, args.output)
            del positive, negative, embedding_cpu, initial, result, final
            active.pop('embedding_cpu')
    finally:
        F.scaled_dot_product_attention = original_sdpa
        pipe.transformer.forward, pipe.scheduler.step = original_dit, original_step
    expected_names = {s['filename'] for s in m['selected_caches'] if s['name'] in {p['name'] for p in prompts}}
    require({s['filename'] for s in report['caches']} == expected_names, 'Worker cache coverage differs')
    budget(args)
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--group')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    if args.output is None:
        args.output = Path(m['report_dir'])/('collect_check.json' if args.check else f'collect_worker_{args.group}.json')
    require(not args.output.exists(), 'Preserve prior check/worker output')
    start = time.time()
    report = dict(experiment='E050', status='checking' if args.check else 'running', group=args.group,
                  pid=os.getpid(), start_epoch=start, deadline_epoch=args.deadline_unix,
                  actual_counts={k: 0 for k in COUNT_KEYS}, prompts=[], caches=[])
    try:
        m, prompts, prepared = prerequisites(args, report)
        if args.check:
            require(not torch.cuda.is_initialized(), 'CPU check unexpectedly initialized CUDA')
            report.update(status='complete', cuda_initialized=False, check_only=True, model_weights_loaded=False,
                          selected_count=len(m['selected_caches']), prompt_count=len(m['prompts']))
        else:
            save_json(report, args.output)
            run(args, m, prompts, prepared, report)
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.time()-start
        save_json(report, args.output)
        print(json.dumps(dict(status=report['status'], output=str(args.output), seconds=report['seconds'],
                              actual_counts=report['actual_counts'])), flush=True)


if __name__ == '__main__':
    main()
