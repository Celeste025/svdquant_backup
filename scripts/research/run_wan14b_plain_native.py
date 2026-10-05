#!/usr/bin/env python3
"""E051: untrained plain native NVFP4 Wan14B, paired with actual E049 inputs.

Frozen legacy packers are reused; no QAD, smoothing or low-rank branch.
Only one temporary FP32 weight exists during layerwise conversion.
"""
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
import run_wan14b_reference as teacher_ref
import wan14b_plain_native as plain
import wan_mainweight_qad as qad
import wan_native_nvfp4 as native
import wan_nvfp4_fastpack as fast

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E051_wan14b_plain_native_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E051_wan14b_plain_native_plan.md'
require, record, save_json = ref.require, ref.record, ref.save_json
COUNTS = dict(dit=0, dit_sdpa=0, scheduler=0, public_vae_decode=0, text_encoder=0, native_gemm=0, activation_packs=0)
PER_CASE = dict(dit=100, dit_sdpa=8000, scheduler=50, public_vae_decode=1, text_encoder=0, native_gemm=40000, activation_packs=40000)


tensor_record = teacher_ref.tensor_record

def bound(rec):
    p = Path(rec['file'])
    require(p.is_file() and p.stat().st_size == rec['bytes'] and ref.sha(p) == rec['sha256'],
            f'Bound small input/source changed: {p}')
    return p


def budget(args):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix, 'Worker deadline reached')


def prerequisites(args, report):
    m = json.loads(args.manifest.read_text())
    require(m['experiment'] == 'E051' and m['arm'] == 'wan14b_plain_nvfp4', 'Unexpected protocol')
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
    reference_manifest = json.loads(bound(m['reference_manifest']).read_text())
    require(reference_manifest['experiment'] == 'E049' and reference_manifest['arm'] == 'wan14b_bf16' and
            reference_manifest['settings'] == s and reference_manifest['cases'] == m['cases'] and
            reference_manifest['negative_prompt_sha256'] == m['negative_prompt_sha256'], 'E049 contract differs')
    refs, prepared = {}, {}
    for pid in sorted(selected):
        source = json.loads(bound(m['reference_workers'][str(pid)]).read_text())
        require(source['status'] == 'complete' and source['experiment'] == 'E049' and
                source['arm'] == 'wan14b_bf16' and source['settings'] == s, 'E049 reference not complete/matched')
        source_cases = {c['case_id']: c for c in source['cases']}
        refs[str(pid)] = m['reference_workers'][str(pid)]
        for case in [c for c in cases if c['prompt_id'] == pid]:
            old = source_cases[case['case_id']]
            require(old['status'] == 'complete' and
                    all(old[k] == case[k] for k in ('prompt_id','seed','replica','prompt','prompt_sha256')), 'Case identity differs')
            embeddings = torch.load(bound(old['embeddings']), map_location='cpu', weights_only=True)
            for key in ('prompt_embeds', 'negative_prompt_embeds'):
                require(tensor_record(embeddings[key]) == old['actual_inputs'][key] and
                        embeddings[key].dtype == torch.bfloat16 and tuple(embeddings[key].shape) == (1,512,4096),
                        'Actual E049 reference embedding differs')
            initial = torch.load(bound(old['initial_noise']['artifact']), map_location='cpu', weights_only=True)
            require(tensor_record(initial) == old['initial_noise']['tensor'] == old['actual_inputs']['initial_noise'] and
                    initial.dtype == torch.float32 and tuple(initial.shape) == (1,16,21,60,104), 'Actual E049 noise differs')
            prepared[case['case_id']] = dict(initial=initial, embeddings=embeddings, old=old)
    cfg = json.loads((Path(m['model_dir'])/'transformer/config.json').read_text())
    require([cfg[k] for k in ('num_layers','num_attention_heads','attention_head_dim','ffn_dim')] == [40,40,128,13824],
            'Expected original14B geometry')
    scheduler = UniPCMultistepScheduler.from_pretrained(m['model_dir'], subfolder='scheduler', local_files_only=True)
    require(scheduler.config.flow_shift == 3., 'Official scheduler shift differs')
    scheduler.set_timesteps(50, device='cpu')
    import diffusers
    require(callable(getattr(F, 'scaled_mm', None)) and hasattr(F, 'ScalingType') and
            hasattr(F, 'SwizzleType'), 'Native NVFP4 torch API missing')
    paths = [Path(__file__), Path(ref.__file__), Path(teacher_ref.__file__), Path(plain.__file__),
             Path(qad.__file__), Path(native.__file__), Path(fast.__file__), args.manifest, PLAN] + [Path(inspect.getsourcefile(cls)) for cls in
            (WanPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler)]
    report.update(manifest=record(args.manifest), sources=[record(p) for p in paths],
        asset_download=record(m['asset_download']), asset_supervisor=record(m['asset_supervisor']),
        header_validation=header, component_reuse=m['component_reuse'], assets_stat=stat_rows,
        component_assets_stat=component_stats, reference_manifest=m['reference_manifest'],
        reference_workers=refs, selected_cases=cases, settings=s,
        versions=dict(torch=torch.__version__, cuda=torch.version.cuda, diffusers=diffusers.__version__, diffusers_path=diffusers.__file__),
        cpu_schedule=dict(config=dict(scheduler.config), timesteps=scheduler.timesteps.tolist(), sigmas=scheduler.sigmas.tolist()),
        prepared_inputs=[dict(case_id=c['case_id'], initial_noise=prepared[c['case_id']]['old']['initial_noise'],
            embeddings=prepared[c['case_id']]['old']['embeddings'],
            actual_inputs=prepared[c['case_id']]['old']['actual_inputs'], reference_worker=refs[str(c['prompt_id'])]) for c in cases])
    return m, cases, prepared


def native_contract(model):
    modules = [(n,m) for n,m in model.named_modules() if isinstance(m, qad.PlainPackedWanLinear)]
    require(len(modules) == 400, 'Expected all 400 native main linears')
    for name, module in modules:
        require(module.weight_global.dtype == torch.float32 and module.weight_global.shape == (1,) and
                module.weight_packed.dtype == torch.uint8 and
                module.weight_scales_swizzled.dtype == torch.float8_e4m3fn and
                (module.bias is None or module.bias.dtype == torch.bfloat16), f'Native ABI differs: {name}')
        require(not list(module.parameters()) and not module._forward_pre_hooks and not module._forward_hooks,
                f'Unexpected retained parameter/hook: {name}')
        require(module.execution_mode == 'native' and module.activation_packer is fast.pack_activation_fast,
                f'Unexpected native execution/packer: {name}')
    require(not any('weight_master' in name for name,_ in model.named_parameters()), 'Persistent master present')
    return dict(target_count=400, block_count=40, global_dtype='torch.float32',
                group_scale_dtype='torch.float8_e4m3fn', packed_dtype='torch.uint8', bias_dtype='torch.bfloat16',
                persistent_weight_master=False, original_target_parameters_retained=False,
                smoothing=False, low_rank=False, native_output_dtype='torch.bfloat16')


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
    loaded_seconds = time.monotonic()-started
    budget(args)
    report['conversion'] = plain.install_plain_native(dit)
    require(report['conversion']['target_count'] == report['conversion']['weight_pack_calls'] == 400,
            'Incomplete native conversion')
    report['native_contract'] = native_contract(dit)
    report['recipe_interpretation'] = dict(training_updates=0, smoothing=False, low_rank=False,
        original_weights='official F32 checkpoint cast to the E049 BF16 teacher values before quantization',
        temporary_weight_input='One BF16 weight matrix promoted to FP32 for packing, then released',
        persistent_fp32_master=False,
        inherited_recipe_metadata='The frozen recipe name/weight_input mention QAD/FP32 master historically; '
                                  'this worker performs no training and retains no master')
    torch.cuda.synchronize()
    storage = {}
    for module in (dit, vae):
        for tensor in list(module.parameters()) + list(module.buffers()):
            st = tensor.untyped_storage(); storage[(str(tensor.device), st.data_ptr())] = st.nbytes()
    report.update(model_load_and_h2d_seconds=loaded_seconds,
        model_load_and_conversion_seconds=time.monotonic()-started, resident_storage_bytes=sum(storage.values()),
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
        budget(args); before = counts.copy(); modules_before = plain.native_call_counts(dit)
        counts['dit'] += 1
        require(kw['hidden_states'].dtype == torch.bfloat16, 'DiT input not BF16')
        active['dit'] = True
        try:
            with fast.collect_fastpack_checks() as flags:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = old_dit(*a, **kw)
            # Scope exit above validates all flags once. No per-layer CPU scalar checks.
            counts['activation_packs'] += len(flags)
        finally:
            active['dit'] = False
        modules_after = plain.native_call_counts(dit)
        delta = [modules_after['per_layer'][name]-value for name,value in modules_before['per_layer'].items()]
        sdpa_calls, mm_calls = counts['dit_sdpa']-before['dit_sdpa'], counts['native_gemm']-before['native_gemm']
        require(sdpa_calls == 80 and mm_calls == len(flags) == len(delta) == 400 and
                min(delta) == max(delta) == 1, 'Expected 400 native/pack and 80 SDPA per DiT')
        active['case']['dit_calls'].append(dict(index=len(active['case']['dit_calls']), dit_sdpa=sdpa_calls,
            native_gemm=mm_calls, fastpack_checks=len(flags), invalid_flags=0,
            module_native_delta_min=min(delta), module_native_delta_max=max(delta),
            input_dtype=str(kw['hidden_states'].dtype), input_shape=list(kw['hidden_states'].shape)))
        del flags
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
        require(active['dit'], 'Unexpected native GEMM outside DiT')
        require(a[2][1].dtype == a[4][1].dtype == torch.float32 and
                a[2][0].dtype == a[4][0].dtype == torch.float8_e4m3fn and
                kw.get('output_dtype') == torch.bfloat16, 'Native scale/output dtype differs')
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
                initial_noise=entry['old']['initial_noise'], embeddings=entry['old']['embeddings'])
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
        report['native_contract_after_inference'] = native_contract(dit)
        report['module_native_calls'] = plain.native_call_counts(dit)
        require(report['module_native_calls']['total_native_calls'] == counts['native_gemm'], 'Module/global count mismatch')
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
    args.output = args.output or Path(m['report_dir'])/('check_plain.json' if args.check_only else f'worker_{args.prompt_ids[0]}.json')
    require(not args.output.exists(), 'Preserve previous result')
    report = dict(experiment='E051', arm=m['arm'], status='running', phase='check' if args.check_only else 'generate',
                  actual_counts=COUNTS.copy(), cases=[])
    started = time.monotonic()
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        m, cases, prepared = prerequisites(args, report)
        if args.check_only:
            require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CPU check requires hidden CUDA')
            report['cpu_structure'] = plain.cpu_structure_check(Path(m['model_dir'])/'transformer/config.json')
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
