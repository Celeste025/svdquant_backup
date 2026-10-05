#!/usr/bin/env python3
"""E049: original Wan14B BF16, actual E043 inputs, no text-encoder execution."""
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
import run_vanilla_wan_reference as ref

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E049_wan14b_reference_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E049_wan14b_reference_plan.md'
require, record, save_json = ref.require, ref.record, ref.save_json
COUNTS = dict(dit=0, dit_sdpa=0, scheduler=0, public_vae_decode=0, text_encoder=0, native_gemm=0)
PER_CASE = dict(dit=100, dit_sdpa=8000, scheduler=50, public_vae_decode=1, text_encoder=0, native_gemm=0)


def tensor_record(value):
    # Fresh flat storage also handles scalar/size-one zero-stride scheduler tensors.
    x = value.detach().cpu()
    flat = torch.empty(x.numel(), dtype=x.dtype).copy_(x.reshape(-1))
    return dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(x.isfinite().all()),
                sha256=hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest())


def bound(rec):
    p = Path(rec['file'])
    require(p.is_file() and p.stat().st_size == rec['bytes'] and ref.sha(p) == rec['sha256'],
            f'Bound small input/source changed: {p}')
    return p


def budget(args):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix, 'Worker deadline reached')


def prerequisites(args, report):
    m = json.loads(args.manifest.read_text())
    require(m['experiment'] == 'E049' and m['arm'] == 'wan14b_bf16', 'Unexpected protocol')
    selected = set(args.prompt_ids)
    require(len(selected) == len(args.prompt_ids), 'Duplicate prompt ids')
    cases = [c for c in m['cases'] if c['prompt_id'] in selected]
    require(len(cases) == 2*len(selected) and {c['prompt_id'] for c in cases} == selected, 'Expected two seeds per prompt')
    s = m['settings']
    require([s[k] for k in ('num_frames','height','width','num_inference_steps','guidance_scale','flow_shift','fps')]
            == [81,480,832,50,5.,3.,16], 'Frozen sampling settings differ')
    download = json.loads(Path(m['asset_download']).read_text())
    supervisor = json.loads(Path(m['asset_supervisor']).read_text())
    require(download['status'] == supervisor['status'] == 'complete' and supervisor['returncode'] == 0,
            'Official14B assets are not complete; no partial asset check/run permitted')
    require(download['revision'] == m['model_revision'] and download['repo'] == m['model_repo'], 'Wrong model revision')
    require(Path(download['target']).resolve() == Path(m['model_dir']).resolve(), 'Model target differs')
    header = download['header_validation']
    require(header['status'] == 'complete' and header['index_exact'] and header['tensor_loaded'] is False,
            'Header/index validation incomplete')
    index_path = Path(m['model_dir'])/'transformer/diffusion_pytorch_model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    require(header['tensors'] == len(index['weight_map']) == 1095 and
            header['payload_bytes'] == index['metadata']['total_size'], 'Index totals differ')
    stat_rows = []
    for rel, entry in download['files'].items():
        p = Path(entry['file'])
        require(p.resolve() == (Path(m['model_dir'])/rel).resolve(), 'Unexpected asset path')
        require(entry['status'] == 'verified' and p.stat().st_size == entry['bytes'], f'Asset stat differs: {p}')
        if entry['hub_entry'].get('lfs'):
            require(entry['sha256'] == entry['hub_entry']['lfs']['sha256'], 'Unverified LFS receipt')
        st = p.stat()
        stat_rows.append(dict(file=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns, verified_sha256=entry['sha256']))
    require(len({v for v in index['weight_map'].values()}) == 12 and
            all('transformer/'+v in download['files'] for v in set(index['weight_map'].values())), 'Missing shard receipt')
    for item in m['model_metadata']:
        bound(item)
    reuse = json.loads(bound(m['component_reuse']).read_text())
    require(reuse['all_identity_match'] and reuse['revision'] == m['model_revision'], 'Component identity incomplete')
    component_stats = []
    for item in reuse['files']:
        p = Path(item['local_file']); st = p.stat()
        require(item['identity_match'] and st.st_size == item['bytes'], 'Reusable component stat differs')
        component_stats.append(dict(file=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    bound(m['reference_manifest'])
    refs, prepared = {}, {}
    for pid in sorted(selected):
        source = json.loads(bound(m['reference_workers'][str(pid)]).read_text())
        require(source['status'] == 'complete' and source['prompt_id'] == pid, 'Reference not complete')
        embeddings = torch.load(bound(source['embeddings']['artifact']), map_location='cpu', weights_only=True)
        for key in ('prompt_embeds', 'negative_prompt_embeds'):
            require(tensor_record(embeddings[key]) == source['embeddings'][key] and
                    embeddings[key].dtype == torch.bfloat16 and tuple(embeddings[key].shape) == (1,512,4096),
                    'Actual reference embedding differs')
        refs[str(pid)] = m['reference_workers'][str(pid)]
        for case in [c for c in cases if c['prompt_id'] == pid]:
            old = next(c for c in source['cases'] if c['case_id'] == case['case_id'])
            require(all(old[k] == case[k] for k in ('prompt_id','seed','replica','prompt','prompt_sha256')), 'Case identity differs')
            initial = torch.load(bound(old['initial_noise']['artifact']), map_location='cpu', weights_only=True)
            require(tensor_record(initial) == old['initial_noise']['tensor'] and initial.dtype == torch.float32 and
                    tuple(initial.shape) == (1,16,21,60,104), 'Actual reference noise differs')
            prepared[case['case_id']] = dict(initial=initial, embeddings=embeddings, old=old, source=source)
    cfg = json.loads((Path(m['model_dir'])/'transformer/config.json').read_text())
    require([cfg[k] for k in ('num_layers','num_attention_heads','attention_head_dim','ffn_dim')] == [40,40,128,13824],
            'Expected original14B geometry')
    scheduler = UniPCMultistepScheduler.from_pretrained(m['model_dir'], subfolder='scheduler', local_files_only=True)
    require(scheduler.config.flow_shift == 3., 'Official scheduler shift differs')
    scheduler.set_timesteps(50, device='cpu')
    import diffusers
    paths = [Path(__file__), Path(ref.__file__), args.manifest, PLAN] + [Path(inspect.getsourcefile(cls)) for cls in
            (WanPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler)]
    report.update(manifest=record(args.manifest), sources=[record(p) for p in paths],
        asset_download=record(m['asset_download']), asset_supervisor=record(m['asset_supervisor']),
        header_validation=header, component_reuse=m['component_reuse'], assets_stat=stat_rows,
        component_assets_stat=component_stats, reference_workers=refs, selected_cases=cases, settings=s,
        versions=dict(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__, diffusers_path=diffusers.__file__),
        cpu_schedule=dict(config=dict(scheduler.config), timesteps=scheduler.timesteps.tolist(), sigmas=scheduler.sigmas.tolist()),
        prepared_inputs=[dict(case_id=c['case_id'], initial_noise=prepared[c['case_id']]['old']['initial_noise'],
            embeddings=prepared[c['case_id']]['source']['embeddings'], reference_worker=refs[str(c['prompt_id'])]) for c in cases])
    return m, cases, prepared


@torch.inference_mode()
def run(args, m, cases, prepared, report):
    require(len(args.prompt_ids) == 1, 'Actual worker handles one prompt/two seeds')
    budget(args)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    started = time.monotonic()
    # Streaming shard loading avoids materializing the complete original F32 state_dict.
    dit = WanTransformer3DModel.from_pretrained(m['model_dir'], subfolder='transformer', torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True, local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(m['component_dir'], subfolder='vae', torch_dtype=torch.float32,
                                          low_cpu_mem_usage=True, local_files_only=True)
    scheduler = UniPCMultistepScheduler.from_pretrained(m['model_dir'], subfolder='scheduler', local_files_only=True)
    pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=scheduler, transformer=dit)
    pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    require(pipe.transformer.dtype == torch.bfloat16 and pipe.vae.dtype == torch.float32 and
            pipe.text_encoder is None and pipe.tokenizer is None, 'Execution dtype/component contract differs')
    require(len(pipe.transformer.blocks) == 40 and not vae.use_tiling and not vae.use_slicing, 'Architecture/VAE contract differs')
    storage = {}
    for module in (dit, vae):
        for tensor in list(module.parameters()) + list(module.buffers()):
            st = tensor.untyped_storage(); storage[(str(tensor.device), st.data_ptr())] = st.nbytes()
    report.update(model_load_and_h2d_seconds=time.monotonic()-started, resident_storage_bytes=sum(storage.values()),
        resident_allocated_bytes=torch.cuda.memory_allocated(), resident_reserved_bytes=torch.cuda.memory_reserved(),
        loading=dict(low_cpu_mem_usage=True, transformer_dtype='torch.bfloat16', text_encoder_loaded=False, tokenizer_loaded=False),
        actual_configs={k: ref.config_dict(getattr(pipe, k).config) for k in ('transformer','vae','scheduler')},
        pipeline_config=dict(pipe.config))
    counts = report['actual_counts']; active = {'dit': False, 'case': None}
    old_dit, old_step, old_decode, old_sdpa = dit.forward, scheduler.step, vae.decode, F.scaled_dot_product_attention
    old_mm = getattr(F, 'scaled_mm', None)

    def sdpa(query, key, value, *a, **kw):
        if active['dit']:
            require(query.dtype == key.dtype == value.dtype == torch.bfloat16, 'Non-BF16 DiT SDPA')
            counts['dit_sdpa'] += 1
        return old_sdpa(query, key, value, *a, **kw)

    def forward(*a, **kw):
        budget(args); before = counts['dit_sdpa']; counts['dit'] += 1
        require(kw['hidden_states'].dtype == torch.bfloat16, 'DiT input not BF16')
        active['dit'] = True
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = old_dit(*a, **kw)
        finally:
            active['dit'] = False
        calls = counts['dit_sdpa']-before
        require(calls == 80, 'Expected80 SDPA per40-layer DiT')
        active['case']['dit_calls'].append(dict(index=len(active['case']['dit_calls']), dit_sdpa=calls,
            native_gemm=0, input_dtype=str(kw['hidden_states'].dtype), input_shape=list(kw['hidden_states'].shape)))
        return output

    def step(*a, **kw):
        budget(args); counts['scheduler'] += 1
        return old_step(*a, **kw)

    def decode(*a, **kw):
        budget(args); counts['public_vae_decode'] += 1
        require(a[0].dtype == torch.float32, 'VAE input not FP32')
        output = old_decode(*a, **kw); x = output[0]
        active['case']['raw_decoder'] = dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(x.isfinite().all()),
                                              min=float(x.min()), max=float(x.max()))
        require(active['case']['raw_decoder']['finite'], 'Nonfinite VAE output')
        return output

    def scaled_mm(*a, **kw):
        counts['native_gemm'] += 1
        return old_mm(*a, **kw)

    dit.forward, scheduler.step, vae.decode, F.scaled_dot_product_attention = forward, step, decode, sdpa
    if old_mm is not None:
        F.scaled_mm = scaled_mm
    try:
        for case in cases:
            budget(args); entry = prepared[case['case_id']]
            folder = Path(m['data_dir'])/m['arm']/case['case_id']; folder.mkdir(parents=True, exist_ok=False)
            row = dict(**case, variant=m['arm'], status='running', steps=[], dit_calls=[],
                reference_worker=m['reference_workers'][str(case['prompt_id'])], reference_case_id=case['case_id'],
                initial_noise=entry['old']['initial_noise'], embeddings=entry['source']['embeddings']['artifact'])
            report['cases'].append(row); active['case'] = row
            initial = entry['initial'].clone()
            positive = entry['embeddings']['prompt_embeds'].to('cuda')
            negative = entry['embeddings']['negative_prompt_embeds'].to('cuda')
            row['actual_inputs'] = dict(initial_noise=tensor_record(initial), prompt_embeds=tensor_record(positive),
                                       negative_prompt_embeds=tensor_record(negative))
            last = {}; before = counts.copy()

            def callback(pipeline, index, t, values):
                budget(args); x = values['latents']
                require(x.dtype == torch.float32 and bool(x.isfinite().all()), 'Sampler state nonfinite/non-FP32')
                row['steps'].append(dict(index=index, timestep=float(t), sigma=float(pipeline.scheduler.sigmas[index]),
                    next_sigma=float(pipeline.scheduler.sigmas[index+1]), dtype=str(x.dtype), finite=True,
                    rms=float(x.square().mean().sqrt()), min=float(x.min()), max=float(x.max())))
                if index == 49:
                    last['latents'] = x.detach().cpu().clone()
                if (index+1) % 10 == 0:
                    print(f"{case['case_id']} {index+1}/50 steps", flush=True)
                return values

            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); began = time.monotonic()
            frames = pipe(prompt_embeds=positive, negative_prompt_embeds=negative, latents=initial,
                height=480, width=832, num_frames=81, num_inference_steps=50, guidance_scale=5., output_type='np',
                callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'], max_sequence_length=512).frames[0]
            torch.cuda.synchronize()
            row.update(seconds_pipeline_including_diagnostics=time.monotonic()-began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                actual_counts={k: counts[k]-before[k] for k in counts},
                schedule=dict(timesteps=pipe.scheduler.timesteps.cpu().tolist(), sigmas=pipe.scheduler.sigmas.cpu().tolist()))
            require(row['actual_counts'] == PER_CASE and len(row['steps']) == 50 and len(row['dit_calls']) == 100,
                    'Incomplete actual call chain')
            final = folder/'final_latents.pt'; torch.save(last['latents'], final)
            row['final_latents'] = dict(artifact=record(final), tensor=tensor_record(last['latents']))
            budget(args); row.update(ref.write_media(frames, folder, m['settings']['fps']))
            row['status'] = 'complete'; save_json(report, args.output)
            del frames, last, initial, positive, negative
        require(counts == {k: 2*v for k,v in PER_CASE.items()}, 'Worker counts differ')
        report['status'] = 'complete'
    finally:
        dit.forward, scheduler.step, vae.decode, F.scaled_dot_product_attention = old_dit, old_step, old_decode, old_sdpa
        if old_mm is not None:
            F.scaled_mm = old_mm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=MANIFEST)
    p.add_argument('--prompt-ids', type=int, nargs='+', required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args(); m = json.loads(args.manifest.read_text())
    args.output = args.output or Path(m['report_dir'])/('check_bf16.json' if args.check_only else f'worker_{args.prompt_ids[0]}.json')
    require(not args.output.exists(), 'Preserve previous result')
    report = dict(experiment='E049', arm=m['arm'], status='running', phase='check' if args.check_only else 'generate',
                  actual_counts=COUNTS.copy(), cases=[])
    started = time.monotonic()
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        m, cases, prepared = prerequisites(args, report)
        if args.check_only:
            require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CPU check requires hidden CUDA')
            report.update(status='complete', cuda_available=False, model_weights_loaded=False)
        else:
            run(args, m, cases, prepared, report)
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc()); raise
    finally:
        report['seconds_total'] = time.monotonic()-started; save_json(report, args.output)
        print(json.dumps(dict(status=report['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
