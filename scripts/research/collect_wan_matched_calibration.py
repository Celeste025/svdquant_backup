#!/usr/bin/env python3
"""E047: selected actual Wan teacher I/O, with E043 sampling semantics; no decode."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel

from run_vanilla_wan_reference import record, save_json, config_dict, require

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E047_wan_matched_calibration_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E047_wan_matched_calibration_plan.md'
SHAPE = (1, 16, 21, 60, 104)
COUNT_KEYS = ('dit', 'dit_sdpa', 'scheduler', 'text_encoder', 'public_vae_decode', 'saved_caches')


def tensor_record(value):
    # A one-element expanded timestep may have stride zero even after contiguous().
    x = value.detach().cpu()
    flat = torch.empty(x.numel(), dtype=x.dtype, device='cpu')
    flat.copy_(x.reshape(-1))
    return dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(flat.isfinite().all()),
                sha256=hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest())


def cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_clone(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_clone(v) for v in value)
    if isinstance(value, list):
        return [cpu_clone(v) for v in value]
    return value


def signature(value):
    if isinstance(value, torch.Tensor):
        return tensor_record(value)
    if isinstance(value, dict):
        return {k: signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [signature(v) for v in value]
    return value


def budget(args):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix,
            'E047 worker deadline reached')


def prerequisites(args, report):
    m = json.loads(args.manifest.read_text())
    prompts = {p['name']: p for p in m['prompts']}
    require(len(prompts) == len(m['prompts']) == 14, 'Expected 14 unique selected prompts')
    for p in prompts.values():
        require(hashlib.sha256(p['prompt'].encode()).hexdigest() == p['prompt_sha256'], 'Prompt SHA mismatch')
    require(hashlib.sha256(m['negative_prompt'].encode()).hexdigest() == m['negative_prompt_sha256'],
            'Negative prompt SHA mismatch')
    names = [name for g in m['worker_groups'] for name in g['prompt_names']]
    require(len(names) == len(set(names)) == 14 and set(names) == set(prompts), 'Worker prompt coverage differs')
    require(len({str(g['group']) for g in m['worker_groups']}) == 4, 'Expected four worker groups')
    selected = m['selected_caches']
    require(len(selected) == len({s['filename'] for s in selected}) == 64, 'Expected 64 unique caches')
    for s in selected:
        require(s['name'] in prompts and 0 <= s['step'] < 50 and s['guidance'] in (0, 1), 'Invalid selection')
        require(s['branch'] == ('cond' if s['guidance'] == 0 else 'uncond'), 'Branch mapping differs')
        require(s['cache_filename'] == s['name']+'-0' and
                s['filename'] == f"{s['cache_filename']}-{s['step']:05d}-{s['guidance']}.pt", 'Cache naming differs')
    settings = m['settings']
    for key, expected in dict(height=480, width=832, num_frames=81, num_inference_steps=50,
                              guidance_scale=6, flow_shift=8, max_sequence_length=512).items():
        require(settings[key] == expected, f'Fixed setting differs: {key}')
    base = Path(m['model_dir'])
    configs = {key: json.loads((base/key).read_text()) for key in
               ('model_index.json', 'transformer/config.json', 'vae/config.json',
                'text_encoder/config.json', 'scheduler/scheduler_config.json')}
    require(configs['model_index.json']['_class_name'] == 'WanPipeline' and
            configs['transformer/config.json']['num_layers'] == 30, 'Expected base Wan 1.3B')
    require(configs['text_encoder/config.json']['model_type'] == 'umt5', 'Expected UMT5 encoder')
    schedule = UniPCMultistepScheduler.from_config(configs['scheduler/scheduler_config.json'], flow_shift=8.)
    schedule.set_timesteps(50, device='cpu')
    require(len(schedule.timesteps) == 50 and schedule.config.flow_shift == 8., 'Schedule contract differs')
    forward_parameters = inspect.signature(WanTransformer3DModel.forward).parameters
    require(all(k in forward_parameters for k in ('hidden_states', 'timestep', 'encoder_hidden_states')),
            'Wan forward API differs')
    source_paths = [Path(__file__), args.manifest, PLAN, ROOT/'scripts/research/run_vanilla_wan_reference.py',
                    Path(inspect.getsourcefile(WanPipeline)), Path(inspect.getsourcefile(WanTransformer3DModel)),
                    Path(inspect.getsourcefile(UniPCMultistepScheduler)),
                    ROOT/'third_party/deepcompressor/deepcompressor/app/diffusion/dataset/collect/utils.py',
                    ROOT/'third_party/deepcompressor/deepcompressor/app/diffusion/dataset/collect/calib.py']
    import diffusers
    report.update(manifest=record(args.manifest), sources=[record(p) for p in source_paths],
                  settings=settings, disk_configs=configs, latent_shape=list(SHAPE),
                  versions=dict(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__),
                  cpu_schedule=dict(config=dict(schedule.config), timesteps=schedule.timesteps.tolist(),
                                    sigmas=schedule.sigmas.tolist()),
                  selected_caches=selected, worker_groups=m['worker_groups'],
                  cache_contract='Actual forward inputs CPU-cloned before forward; actual output CPU-cloned afterward; '
                                 'batch-one CollectHook input_args/input_kwargs/outputs and filename/step/guidance.')
    if args.group is None:
        require(args.check, '--group is required for GPU collection')
        chosen = m['prompts']
    else:
        groups = [g for g in m['worker_groups'] if str(g['group']) == args.group]
        require(len(groups) == 1, 'Unknown group')
        chosen = [prompts[name] for name in groups[0]['prompt_names']]
    report['selected_prompts'] = chosen
    return m, chosen


@torch.inference_mode()
def run(args, m, prompts, report):
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
    vae = AutoencoderKLWan.from_pretrained(m['model_dir'], subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(m['model_dir'], vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.)
    pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
    require(pipe.transformer.dtype == torch.bfloat16 and pipe.text_encoder.dtype == torch.bfloat16 and
            pipe.vae.dtype == torch.float32, 'Model dtype contract differs')
    require(not pipe.vae.use_tiling and not pipe.vae.use_slicing, 'Unexpected VAE tiling/slicing')
    torch.cuda.synchronize()
    report.update(model_load_seconds=time.monotonic()-loaded,
                  actual_configs={k: config_dict(getattr(pipe, k).config) for k in ('transformer','vae','text_encoder','scheduler')},
                  pipeline_config=dict(pipe.config), resident_allocated_bytes=torch.cuda.memory_allocated(),
                  attention_backend='torch SDPA FLASH_ATTENTION inside every DiT; no compile/offload',
                  sampler_input_dtype='torch.float32', transformer_input_dtype='torch.bfloat16',
                  output_type='latent')
    original_dit, original_te = pipe.transformer.forward, pipe.text_encoder.forward
    original_step, original_sdpa = pipe.scheduler.step, F.scaled_dot_product_attention
    forward_signature = inspect.signature(original_dit)
    active = {}

    def sdpa(query, key, value, *a, **kw):
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
        require(counts['dit_sdpa']-sdpa_before == 60, 'Expected 60 SDPA calls per teacher DiT')
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

    def te(*a, **kw):
        budget(args); output = original_te(*a, **kw); counts['text_encoder'] += 1
        return output

    def step(*a, **kw):
        budget(args); output = original_step(*a, **kw); counts['scheduler'] += 1
        return output

    def forbidden_decode(*a, **kw):
        counts['public_vae_decode'] += 1
        raise RuntimeError('E047 is latent-only; VAE decode must never run')

    pipe.transformer.forward, pipe.text_encoder.forward = dit, te
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
            positive, negative = pipe.encode_prompt(prompt=prompt['prompt'], negative_prompt=m['negative_prompt'],
                do_classifier_free_guidance=True, device=torch.device('cuda'), max_sequence_length=512)
            require(bool(positive.isfinite().all() and negative.isfinite().all()), 'Nonfinite embeddings')
            embedding_cpu = cpu_clone(dict(prompt_embeds=positive, negative_prompt_embeds=negative))
            active['embedding_cpu'] = embedding_cpu
            embedding_path = case_dir/'embeddings.pt'; torch.save(embedding_cpu, embedding_path)
            row['embeddings'] = dict(artifact=record(embedding_path), tensors=signature(embedding_cpu))
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
                height=480, width=832, num_frames=81, num_inference_steps=50, guidance_scale=6.,
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
            require(row['actual_counts'] == dict(dit=100, dit_sdpa=6000, scheduler=50, text_encoder=2,
                                                 public_vae_decode=0, saved_caches=expected_saved), 'Prompt counts differ')
            require(len(row['steps']) == 50, 'Incomplete callback chain')
            row['status'] = 'complete'; save_json(report, args.output)
            del positive, negative, embedding_cpu, initial, result, final
            active.pop('embedding_cpu')
    finally:
        F.scaled_dot_product_attention = original_sdpa
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
    report = dict(experiment='E047', status='checking' if args.check else 'running', group=args.group,
                  pid=os.getpid(), start_epoch=start, deadline_epoch=args.deadline_unix,
                  actual_counts={k: 0 for k in COUNT_KEYS}, prompts=[], caches=[])
    try:
        m, prompts = prerequisites(args, report)
        if args.check:
            require(not torch.cuda.is_initialized(), 'CPU check unexpectedly initialized CUDA')
            report.update(status='complete', cuda_initialized=False, check_only=True,
                          selected_count=len(m['selected_caches']), prompt_count=len(m['prompts']))
        else:
            save_json(report, args.output)
            run(args, m, prompts, report)
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
