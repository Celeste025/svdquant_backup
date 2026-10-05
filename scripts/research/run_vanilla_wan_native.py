#!/usr/bin/env python3
"""E044: native plain/SVDQuant Wan, reusing the actual E043 conditions and noise."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'third_party/deepcompressor')]
import run_vanilla_wan_reference as ref
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler
import wan_native_nvfp4 as native
import wan_nvfp4_fastpack as fast
import wan_mainweight_qad as plain

MANIFEST = ROOT/'research_state/06_experiments/E044_vanilla_wan_native_manifest.json'
require, record, save_json = ref.require, ref.record, ref.save_json


def verify_file(item):
    require(record(item['file']) == item, f"Artifact changed: {item['file']}")
    return Path(item['file'])


def load_bound(item):
    return torch.load(verify_file(item), map_location='cpu', weights_only=False)


def prerequisites(args, report):
    m = json.loads(args.manifest.read_text())
    e043 = json.loads(ref.MANIFEST.read_text())
    require(args.arm in m['arms'], 'Arm is outside manifest')
    for key in ('settings','cases','negative_prompt','model_dir'):
        require(m[key] == e043[key], f'E043 {key} changed')
    ids = list(dict.fromkeys(args.prompt_ids))
    require(ids == args.prompt_ids, 'Duplicate prompt IDs')
    cases = [c for c in m['cases'] if c['prompt_id'] in ids]
    require(len(cases) == 2*len(ids), 'Expected two fixed cases per prompt')
    prepared, refs = {}, {}
    for pid in ids:
        path = Path(m['reference_dir'])/f'worker_{pid}.json'
        source = json.loads(path.read_text())
        require(source['status'] == 'complete' and source['settings'] == m['settings'], 'Incomplete/different E043 source')
        refs[str(pid)] = record(path)
        embeddings = load_bound(source['embeddings']['artifact'])
        for key in ('prompt_embeds', 'negative_prompt_embeds'):
            require(ref.tensor_record(embeddings[key]) == source['embeddings'][key], 'Embedding tensor changed')
            require(embeddings[key].dtype == torch.bfloat16 and tuple(embeddings[key].shape) == (1,512,4096),
                    'Embedding shape/dtype differs')
        for case in [c for c in cases if c['prompt_id'] == pid]:
            old = next(r for r in source['cases'] if r['case_id'] == case['case_id'])
            require(old['status'] == 'complete' and all(old[k] == case[k] for k in case), 'Case identity differs')
            initial = load_bound(old['initial_noise']['artifact'])
            require(ref.tensor_record(initial) == old['initial_noise']['tensor'], 'Initial tensor changed')
            require(initial.dtype == torch.float32 and tuple(initial.shape) == (1,16,21,60,104), 'Noise shape/dtype differs')
            prepared[case['case_id']] = dict(embeddings=embeddings, initial=initial, old=old,
                embeddings_record=source['embeddings']['artifact'], source=refs[str(pid)])
    sources = [Path(__file__), Path(ref.__file__), args.manifest,
               ROOT/'research_state/06_experiments/E044_vanilla_wan_native_plan.md', Path(native.__file__), Path(fast.__file__),
               Path(plain.__file__), ROOT/'scripts/infer_rcm_wan_4step.py',
               Path(inspect.getsourcefile(WanPipeline)), Path(inspect.getsourcefile(WanTransformer3DModel)),
               Path(inspect.getsourcefile(AutoencoderKLWan)), Path(inspect.getsourcefile(UniPCMultistepScheduler))]
    import diffusers
    report.update(manifest=record(args.manifest), sources=[record(p) for p in sources], references=refs,
        settings=m['settings'], selected_cases=cases, inherited_asset_provenance=record(ref.TRANSFORMER_RECEIPT),
        inherited_te_vae_receipt=record(ref.RECEIPT),
        environment=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                         diffusers=diffusers.__version__, diffusers_path=diffusers.__file__,
                         PATH=os.environ.get('PATH'), TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),
                         TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST')))
    config = json.loads((Path(m['model_dir'])/'transformer/config.json').read_text())
    with torch.device('meta'):
        model = WanTransformer3DModel.from_config(config)
    require(all(type(model.get_submodule(name)) is torch.nn.Linear for name in plain.TARGET_NAMES), '300 module ABI changed')
    report['cpu_target_count'] = len(plain.TARGET_NAMES)
    if args.arm == 'svdquant_nvfp4':
        # Import the actual loader dependencies in the launcher environment.
        from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
        from deepcompressor.app.diffusion.nn.struct import DiffusionAttentionStruct, DiffusionModelStruct
        import deepcompressor.csrc.load as extension
        attn_type = type(model.blocks[0].attn1)
        if attn_type not in DiffusionAttentionStruct._factories:
            DiffusionAttentionStruct.register_factory(attn_type, DiffusionAttentionStruct._default_construct)
        structure = DiffusionModelStruct.construct(model)
        require(len(structure.block_structs) == 30, 'DeepCompressor 0.40 structural mapping changed')
        checkpoint = Path(m['svd_checkpoint'])
        identity = json.loads(Path(m['checkpoint_identity']).read_text())
        require(identity['status'] == 'complete', 'Checkpoint identity receipt incomplete')
        for item in identity['files']:
            asset = Path(item['file']); stat = asset.stat()
            require(asset.parent == checkpoint and str(asset.resolve()) == item['resolved'] and
                    stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns'],
                    'Checkpoint differs from existing identity receipt')
        report['checkpoint_identity'] = record(m['checkpoint_identity'])
        report['checkpoint_stat'] = [{**dict(file=str(checkpoint/f'{n}.pt'), resolved=str((checkpoint/f'{n}.pt').resolve())),
            'bytes': (checkpoint/f'{n}.pt').stat().st_size} for n in ('model','scale','wgts','branch','smooth')]
        wgts = torch.load(checkpoint/'wgts.pt',map_location='cpu',weights_only=False)
        require(set(wgts) == set(plain.TARGET_NAMES), 'Saved SVD target coverage differs')
        report['loader_import'] = dict(status='complete', extension=extension._C.__file__,
                                      config_class=DiffusionPtqRunConfig.__name__, blocks=30)
        report['sources'].extend(record(ROOT/'third_party/deepcompressor/deepcompressor'/name) for name in
            ('app/diffusion/ptq.py','app/diffusion/nn/struct.py','app/diffusion/quant/weight.py',
             'app/diffusion/quant/activation.py','csrc/load.py','csrc/quantize/quantize.cu'))
    del model
    return m, cases, prepared


@torch.inference_mode()
def install_model(pipe, args, m, report):
    started = time.monotonic()
    model = pipe.transformer
    if args.arm == 'plain_nvfp4':
        masters = plain.install_qad(model)
        artifact = plain.export_packed(model)
        installed = plain.install_packed(model, artifact)
        report['conversion'] = dict(arm=args.arm, target_count=len(installed), training_updates=0,
            recipe=artifact['recipe'], targets=[dict(name=n, shape=v['shape'], global_scale=float(v['global_scale']),
                global_dtype=str(v['global_scale'].dtype), scales_dtype=str(v['scales'].dtype))
                for n,v in artifact['layers'].items()], non_target_state_count=len(artifact['non_target_state']))
        del masters, artifact, installed
    else:
        os.environ['SVDQUANT_DATA_ROOT'] = str(Path(m['model_dir']).parents[1])
        os.environ['RCM_RUNS_ROOT'] = str(Path(m['data_dir'])/'loader_scratch'/('-'.join(map(str,args.prompt_ids))))
        from infer_rcm_wan_4step import load_quantized_transformer
        load_quantized_transformer(pipe, Path(m['svd_checkpoint']), Path(m['model_dir']))
        conversion = native.convert_wan_transformer_to_native(model, Path(m['svd_checkpoint']),
                                                              activation_packer='legacy', chunk_rows=1024)
        require(conversion['target_count'] == conversion['exact_roundtrip_count'] == 300, 'Saved packet roundtrip failed')
        for module in model.modules():
            if isinstance(module, native.NativeWanLinear):
                fast.validate_quantizer_contract(module.activation_quantizer)
                module.activation_packer = fast.pack_activation_fast
        report['conversion'] = conversion
    coverage = {n: v for n,v in model.named_modules() if isinstance(v,native.NativeWanLinear)}
    require(set(coverage) == set(plain.TARGET_NAMES), 'Native module coverage differs')
    require(all(v.weight_global.dtype == torch.float32 and v.weight_scales_swizzled.dtype == torch.float8_e4m3fn
                for v in coverage.values()), 'Native scale dtype changed')
    require(not any('weight_master' in n for n,_ in model.named_parameters()), 'Training masters remain registered')
    report['native_targets'] = list(coverage)
    del coverage
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    report['conversion_seconds'] = time.monotonic()-started


@torch.inference_mode()
def run(args, m, cases, prepared, report):
    ref.budget(args)
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report['device'] = dict(name=torch.cuda.get_device_name(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           capability=list(torch.cuda.get_device_capability()))
    vae = AutoencoderKLWan.from_pretrained(m['model_dir'], subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    # Stored embeddings are both supplied to encode_prompt; neither TE nor tokenizer is used.
    pipe = WanPipeline.from_pretrained(m['model_dir'], vae=vae, text_encoder=None, tokenizer=None,
                                      torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.)
    pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
    require(pipe.transformer.dtype == torch.bfloat16 and pipe.vae.dtype == torch.float32, 'Model dtype changed')
    require(pipe.text_encoder is None and not pipe.vae.use_tiling and not pipe.vae.use_slicing, 'Unexpected model components')
    report['actual_configs'] = {k:ref.config_dict(getattr(pipe,k).config) for k in ('transformer','vae','scheduler')}
    report['pipeline_config'] = dict(pipe.config)
    install_model(pipe,args,m,report)
    report['resident_allocated_bytes'] = torch.cuda.memory_allocated()
    counts = report['actual_counts']
    old_dit, old_step, old_decode = pipe.transformer.forward, pipe.scheduler.step, pipe.vae.decode
    old_mm, old_sdpa = F.scaled_mm, F.scaled_dot_product_attention
    inside_dit = False

    def mm(*a, **kw):
        require(inside_dit, 'Unexpected generation native GEMM outside DiT')
        counts['native_gemm'] += 1
        return old_mm(*a, **kw)

    def sdpa(q, k, v, *a, **kw):
        if inside_dit:
            require(q.dtype == k.dtype == v.dtype == torch.bfloat16, 'Non-BF16 DiT attention input')
            counts['dit_sdpa'] += 1
        return old_sdpa(q,k,v,*a,**kw)

    def dit(*a, **kw):
        nonlocal inside_dit
        ref.budget(args); counts['dit'] += 1
        require(kw['hidden_states'].dtype == torch.bfloat16, 'DiT input must be BF16')
        before = counts.copy(); inside_dit = True
        try:
            with fast.collect_fastpack_checks() as checks:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    result = old_dit(*a, **kw)
        finally:
            inside_dit = False
        counts['fastpack_checks'] += len(checks)
        call = dict(native_gemm=counts['native_gemm']-before['native_gemm'],
                    dit_sdpa=counts['dit_sdpa']-before['dit_sdpa'], fastpack_checks=len(checks), invalid_flags=0)
        require(call == dict(native_gemm=300,dit_sdpa=60,fastpack_checks=300,invalid_flags=0), 'Actual DiT kernel counts differ')
        report['cases'][-1]['dit_calls'].append(call)
        return result

    def step(*a, **kw):
        ref.budget(args); counts['scheduler'] += 1
        return old_step(*a, **kw)

    def decode(*a, **kw):
        ref.budget(args); counts['public_vae_decode'] += 1
        require(a[0].dtype == torch.float32, 'VAE input must be FP32')
        out = old_decode(*a, **kw); x=out[0]
        info=dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(x.isfinite().all()), min=float(x.min()), max=float(x.max()))
        report['cases'][-1]['raw_decoder'] = info
        require(info['finite'], 'Nonfinite decoded tensor')
        return out

    pipe.transformer.forward,pipe.scheduler.step,pipe.vae.decode=dit,step,decode
    F.scaled_mm,F.scaled_dot_product_attention=mm,sdpa
    try:
        for case in cases:
            ref.budget(args); entry=prepared[case['case_id']]
            destination=Path(m['data_dir'])/args.arm/case['case_id']
            destination.mkdir(parents=True,exist_ok=False)
            row=dict(**case,status='running',variant=args.arm,steps=[],dit_calls=[],
                reference_worker=entry['source'],reference_case_id=case['case_id'],
                initial_noise=entry['old']['initial_noise'],embeddings=entry['embeddings_record'])
            report['cases'].append(row)
            positive=entry['embeddings']['prompt_embeds'].cuda()
            negative=entry['embeddings']['negative_prompt_embeds'].cuda()
            before=counts.copy(); last={}

            def callback(pipeline,index,t,values):
                ref.budget(args); x=values['latents']
                require(x.dtype == torch.float32 and bool(x.isfinite().all()), 'Nonfinite/non-FP32 sampler state')
                row['steps'].append(dict(index=index,timestep=float(t),sigma=float(pipeline.scheduler.sigmas[index]),
                    next_sigma=float(pipeline.scheduler.sigmas[index+1]),dtype=str(x.dtype),finite=True,
                    rms=float(x.square().mean().sqrt()),min=float(x.min()),max=float(x.max())))
                if index==49: last['latents']=x.detach().cpu().clone()
                if (index+1)%10==0: print(f"{args.arm} {case['case_id']} {index+1}/50",flush=True)
                return values

            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            frames=pipe(prompt_embeds=positive,negative_prompt_embeds=negative,latents=entry['initial'],
                height=480,width=832,num_frames=81,num_inference_steps=50,guidance_scale=6.,output_type='np',
                callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'],max_sequence_length=512).frames[0]
            torch.cuda.synchronize()
            row.update(seconds_pipeline_including_diagnostics=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                actual_counts={k:counts[k]-before[k] for k in counts},
                schedule=dict(timesteps=pipe.scheduler.timesteps.cpu().tolist(),sigmas=pipe.scheduler.sigmas.cpu().tolist()))
            require(row['actual_counts']==dict(dit=100,native_gemm=30000,dit_sdpa=6000,scheduler=50,
                public_vae_decode=1,text_encoder=0,fastpack_checks=30000),'Per-video runtime counts differ')
            require(len(row['steps'])==50 and len(row['dit_calls'])==100,'Incomplete trajectory')
            final=destination/'final_latents.pt';torch.save(last['latents'],final)
            row['final_latents']=dict(artifact=record(final),tensor=ref.tensor_record(last['latents']))
            ref.budget(args);row.update(ref.write_media(frames,destination,16));row['status']='complete'
            save_json(report,args.output)
            del frames,last,positive,negative
        report['status']='complete';ref.budget(args)
    finally:
        F.scaled_mm,F.scaled_dot_product_attention=old_mm,old_sdpa


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=MANIFEST)
    p.add_argument('--arm',choices=('plain_nvfp4','svdquant_nvfp4'),required=True)
    p.add_argument('--prompt-ids',type=int,nargs='+',required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--deadline-unix',type=float)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    args.output=args.output or ROOT/f'results/research/E044/{"check" if args.check_only else "worker"}_{args.arm}_{"-".join(map(str,args.prompt_ids))}.json'
    require(not args.output.exists(),'Refusing to overwrite previous result')
    report=dict(experiment='E044',arm=args.arm,status='running',phase='check' if args.check_only else 'generate',cases=[],
        actual_counts=dict(dit=0,native_gemm=0,dit_sdpa=0,scheduler=0,public_vae_decode=0,text_encoder=0,fastpack_checks=0))
    began=time.monotonic()
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        m,cases,prepared=prerequisites(args,report)
        if args.check_only:
            require(not torch.cuda.is_available() and not torch.cuda.is_initialized(),'CPU check requires CUDA hidden')
            report.update(status='complete',cuda_available=False,model_weights_loaded=False,prepared_case_count=len(prepared))
        else: run(args,m,cases,prepared,report)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds_total']=time.monotonic()-began
        save_json(report,args.output);print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__': main()
