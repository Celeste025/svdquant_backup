#!/usr/bin/env python3
"""Fixed E021 development comparison: BF16, two E020 exports, frozen SVD+LR.

--prepare-only writes a CPU/stdlib-only manifest draft. Generation is a separate
invocation with an explicit deadline. Reuse E008 sampling and VAE semantics;
save actual shared noise/embeddings once and load those tensors in every arm.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq')
BASE = DATA/'models/Wan2.1-T2V-1.3B-Diffusers'
RCM = DATA/'models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer'
E020 = DATA/'research/20261003/E020'
SVD = DATA/'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16'
INVENTORY = ROOT/'results/research/E020/E020_cache_inventory.json'
MANIFEST = ROOT/'results/research/E021/generation_manifest.draft.json'
ARMS = ('bf16', 'packed_step0000', 'packed_step0064', 'svd_lr')
CASES = ('vbench_066', 'vbench_091', 'vbench_182', 'vbench_067')
SETTINGS = dict(seed=20261004, height=480, width=832, frames=77, fps=16,
                steps=4, sigma_max=80.0, guidance=0.0, max_sequence_length=512,
                attention='BF16_FLASH_ATTENTION', vae_core=128, vae_halo=0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def prepare_manifest(inventory_path):
    inventory = json.loads(inventory_path.read_text())
    candidates = inventory['later_generation_source']['candidates']
    if inventory['status'] != 'complete' or tuple(c['case_id'] for c in candidates) != CASES:
        raise ValueError('Expected the four fixed E020 generation candidates in inventory order')
    if any(c['known_calibration_text_exact_match'] for c in candidates):
        raise ValueError('Candidate text overlaps the known calibration text inventory')
    return dict(
        experiment='E021', status='draft_not_generated', settings=SETTINGS,
        inventory=dict(path=str(inventory_path.resolve()), sha256=sha256(inventory_path)),
        model=dict(base_assets=str(BASE), explicit_rcm_transformer=str(RCM)),
        cases=[dict(case_id=c['case_id'], prompt=c['prompt'], dimensions=c['dimensions'],
                    seed=SETTINGS['seed']) for c in candidates],
        arms=[dict(name='bf16', weights=str(RCM), native_calls_per_video=0),
              *[dict(name=name, weights=str(E020/(name+'.pt')), native_calls_per_video=1200)
                for name in ARMS[1:3]],
              dict(name='svd_lr', weights=str(SVD), native_calls_per_video=1200)],
        sampling=dict(schedule='rCM TrigFlow -> RectifiedFlow',
                      angles='[atan(80), 1.5, 1.4, 1.0, 0.0] in FP64',
                      latent_shape=[1, 16, 20, 60, 104],
                      shared='Per case: one saved FP64 initial latent, four saved FP32 update noises, one BF16 embedding; all arms load identical tensors',
                      rng='CUDA Generator reset to fixed seed for each prompt; initial FP32 draw then four FP32 draws, including last zero-coefficient update'),
        seed_history=dict(
            claim='Same integer seed used previously; no global unseen-seed claim. The four prompt/noise generation pairs are separate from the fixed E020 train/development cache records.',
            known_same_seed_record=str(ROOT/'results/vbench/rcm_lowrank_8prompt_e10_vbench20/generated/0004/run_config.json'),
            note='A seed integer also used for training shuffle is not by itself data leakage. Prompt selection comes from the E020 inventory, not scores.'),
        comparison_scope='Whole deployed recipe comparison. SVD includes its frozen smooth/LR and legacy non-target quantization recipe; differences from plain QAD are not attributed solely to LR.',
        limits=['Four development prompts, one seed; not a final held-out benchmark or a quality conclusion.',
                'Candidate prompts were already in a public/planned VBench manifest; absence of prior media was checked only in inventory-listed local roots.',
                'No E020 training/validation cache state is injected into these free rollouts.',
                'Wall times include diagnostics; this generator is not an unbiased latency benchmark.'],
        expected_full_run=dict(videos=16, dit_calls=64, native_fp4_calls=14400),
    )


def validate_manifest(manifest):
    inventory = Path(manifest['inventory']['path'])
    expected = prepare_manifest(inventory)
    if manifest != expected:
        raise ValueError('Manifest differs from the fixed candidate/settings draft; use a new explicit protocol for changes')
    for path in (BASE/'model_index.json', RCM/'config.json', BASE/'vae/config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)


def execute(args, manifest, report):
    # Keep --prepare-only and --help free of torch/CUDA imports.
    sys.path.insert(0, str(ROOT/'scripts'))
    sys.path.insert(0, str(ROOT/'third_party/deepcompressor'))
    os.environ.setdefault('SVDQUANT_DATA_ROOT', str(DATA))
    os.environ['RCM_RUNS_ROOT'] = str(args.artifact_dir/'svd_load_scratch')
    import torch
    import torch.nn.functional as F
    import diffusers
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers.utils import export_to_video
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from infer_rcm_wan_4step import decode_spatial_tiled, load_quantized_transformer
    import wan_mainweight_qad as qad
    import wan_native_nvfp4 as native
    from wan_nvfp4_fastpack import collect_fastpack_checks, pack_activation_fast, validate_quantizer_contract

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('This comparison targets SM120 native NVFP4')
    if diffusers.__version__ != '0.33.1' or torch.__version__ != '2.11.0+cu128':
        raise RuntimeError('Use the established torch2.11.0+cu128 / Diffusers0.33.1 environment')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    start = time.time()
    case_rows = [c for c in manifest['cases'] if c['case_id'] in args.cases]
    arm_rows = [a for a in manifest['arms'] if a['name'] in args.arms]
    for arm in arm_rows:
        path = Path(arm['weights'])
        if not path.exists():
            raise FileNotFoundError(path)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)

    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def record_file(path):
        return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))

    def tensor_file(path, value):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        return record_file(path)

    def budget():
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>60*1024**3:
            raise MemoryError('E021 full-shape 60GiB allocation guard exceeded')
        if time.time() > args.deadline_unix:
            raise TimeoutError('E021 caller-provided deadline expired; keep partial artifacts')

    def checkpoint():
        report['elapsed_seconds_not_benchmark'] = time.time()-start
        report['peak_allocated_gib_not_benchmark'] = torch.cuda.max_memory_allocated()/1024**3
        save_json(args.output, report)
        budget()

    report.update(environment=dict(torch=torch.__version__, diffusers=diffusers.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                  artifact_dir=str(args.artifact_dir), shared_inputs={}, arms={})
    sources = [Path(__file__), Path(qad.__file__), Path(native.__file__),
               ROOT/'scripts/research/wan_nvfp4_fastpack.py', ROOT/'scripts/infer_rcm_wan_4step.py',
               args.manifest, RCM/'config.json', BASE/'vae/config.json']
    report['sources'] = {str(p): sha256(p) for p in sources}
    report['packed_artifacts'] = {a['name']: record_file(Path(a['weights']))
                                  for a in arm_rows if a['name'].startswith('packed_step')}
    checkpoint()

    with torch.inference_mode():
        # Base supplies text/tokenizer/processor assets; no invalid rCM pipeline symlink.
        pipe = WanPipeline.from_pretrained(BASE, transformer=None, vae=None,
                    torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.text_encoder.to(device).eval()
        video_processor = pipe.video_processor
        angles = torch.tensor([math.atan(80.), 1.5, 1.4, 1., 0.], dtype=torch.float64, device=device)
        t_steps = angles.sin()/(angles.cos()+angles.sin())
        ones = torch.ones((1,), dtype=torch.float64, device=device)
        report['schedule'] = dict(t_steps=t_steps.cpu().tolist(),
                timesteps_bf16=[(t.float()*ones*1000).to(torch.bfloat16).item() for t in t_steps[:-1]])
        for case in case_rows:
            budget()
            embedding, _ = pipe.encode_prompt(prompt=case['prompt'], do_classifier_free_guidance=False,
                    max_sequence_length=512, device=device, dtype=torch.bfloat16)
            if embedding.shape != (1, 512, 4096) or embedding.dtype != torch.bfloat16:
                raise RuntimeError('Unexpected prompt embedding contract')
            generator = torch.Generator(device=device).manual_seed(case['seed'])
            initial_noise = pipe.prepare_latents(batch_size=1, num_channels_latents=16,
                    height=480, width=832, num_frames=77, dtype=torch.float32,
                    device=device, generator=generator)
            initial = initial_noise.to(torch.float64)*t_steps[0]
            if initial.shape != (1, 16, 20, 60, 104):
                raise RuntimeError('Unexpected latent shape')
            noises = [torch.randn(initial.shape, dtype=torch.float32, device=device, generator=generator)
                      for _ in range(4)]
            path = args.artifact_dir/'shared_inputs'/(case['case_id']+'.pt')
            payload = dict(case_id=case['case_id'], prompt=case['prompt'], seed=case['seed'],
                initial_latent_fp64=initial.cpu(), update_noises_fp32=[n.cpu() for n in noises],
                embedding_bf16=embedding.cpu(), t_steps_fp64=t_steps.cpu(),
                generator_state_after_draws=generator.get_state().cpu())
            report['shared_inputs'][case['case_id']] = dict(artifact=tensor_file(path, payload),
                initial_latent_sha256=tensor_sha(initial), initial_noise_sha256=tensor_sha(initial_noise),
                noise_sha256=[tensor_sha(n) for n in noises], embedding_sha256=tensor_sha(embedding))
            del embedding, initial_noise, initial, noises, payload, generator
            checkpoint()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        original_sdpa, original_mm = F.scaled_dot_product_attention, F.scaled_mm
        counts = dict(sdpa=0, native=0)

        def audited_sdpa(q, k, v, *pos, **kw):
            if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
                raise RuntimeError('DiT attention must remain BF16 in every arm')
            counts['sdpa'] += 1
            return original_sdpa(q, k, v, *pos, **kw)

        def audited_mm(a, b, *pos, **kw):
            if a.dtype != torch.float4_e2m1fn_x2 or b.dtype != torch.float4_e2m1fn_x2:
                raise RuntimeError('Expected actual packed FP4 inputs to scaled_mm')
            counts['native'] += 1
            return original_mm(a, b, *pos, **kw)

        F.scaled_dot_product_attention, F.scaled_mm = audited_sdpa, audited_mm
        try:
            for arm in arm_rows:
                budget()
                name = arm['name']
                model = WanTransformer3DModel.from_pretrained(RCM, torch_dtype=torch.bfloat16,
                                                             local_files_only=True).to(device).eval()
                installed = {}
                if name.startswith('packed_step'):
                    packed = torch.load(arm['weights'], map_location='cpu', weights_only=False)
                    installed = qad.install_packed(model, packed)
                    del packed
                    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
                        raise RuntimeError('Plain packed arm unexpectedly contains runtime hooks')
                    if any('weight_master' in n for n, _ in model.named_parameters()):
                        raise RuntimeError('Plain packed arm unexpectedly retains master weights')
                elif name == 'svd_lr':
                    svd_pipe = WanPipeline.from_pretrained(BASE, transformer=model, vae=None,
                        text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16, local_files_only=True)
                    load_quantized_transformer(svd_pipe, SVD, BASE)
                    model = svd_pipe.transformer.eval()
                    conversion = native.convert_wan_transformer_to_native(model, SVD,
                                                activation_packer='legacy', chunk_rows=1024)
                    if conversion['target_count'] != 300 or conversion['exact_roundtrip_count'] != 300:
                        raise RuntimeError('Frozen SVD native conversion is incomplete')
                    for module in model.modules():
                        if isinstance(module, native.NativeWanLinear):
                            validate_quantizer_contract(module.activation_quantizer)
                            module.activation_packer = pack_activation_fast
                    del module, svd_pipe
                native_modules = [m for m in model.modules() if isinstance(m, native.NativeWanLinear)]
                expected_modules = 0 if name == 'bf16' else 300
                if len(native_modules) != expected_modules:
                    raise RuntimeError(f'{name}: expected {expected_modules} native linears')
                if any(m.execution_mode != 'native' for m in native_modules):
                    raise RuntimeError('Hidden QDQ fallback is not allowed')
                report['arms'][name] = dict(status='running', native_modules=len(native_modules), cases={})
                gc.collect()
                torch.cuda.empty_cache()
                for case in case_rows:
                    case_id = case['case_id']
                    shared = report['shared_inputs'][case_id]
                    payload = torch.load(shared['artifact']['path'], map_location='cpu', weights_only=False)
                    latents = payload['initial_latent_fp64'].to(device).clone()
                    noises = [n.to(device) for n in payload['update_noises_fp32']]
                    embedding = payload['embedding_bf16'].to(device)
                    if tensor_sha(latents) != shared['initial_latent_sha256']:
                        raise RuntimeError('Shared initial latent changed')
                    if [tensor_sha(n) for n in noises] != shared['noise_sha256']:
                        raise RuntimeError('Shared update noises changed')
                    if tensor_sha(embedding) != shared['embedding_sha256']:
                        raise RuntimeError('Shared prompt embedding changed')
                    row = dict(status='running', prompt=case['prompt'], seed=case['seed'],
                        shared_inputs=shared['artifact']['path'],
                        initial_latent_sha256=shared['initial_latent_sha256'],
                        noise_sha256=shared['noise_sha256'], steps=[])
                    report['arms'][name]['cases'][case_id] = row
                    before_module_calls = sum(m.native_calls for m in installed.values())
                    for step, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                        budget()
                        timestep = (t_cur.float()*ones*1000).to(torch.bfloat16)
                        before = dict(counts)
                        with collect_fastpack_checks() if expected_modules else nullcontext([]) as checks:
                            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                                velocity = model(hidden_states=latents.to(torch.bfloat16), timestep=timestep,
                                    encoder_hidden_states=embedding, return_dict=False)[0]
                        actual = dict(native=counts['native']-before['native'], sdpa=counts['sdpa']-before['sdpa'])
                        if actual != dict(native=expected_modules, sdpa=60) or len(checks) != expected_modules:
                            raise RuntimeError(f'Unexpected actual execution counts: {name}/{case_id}/{step}: {actual}')
                        if velocity.dtype != torch.bfloat16 or not torch.isfinite(velocity).all():
                            raise RuntimeError('Invalid DiT velocity')
                        latents = (1-t_next)*(latents-t_cur*velocity.to(torch.float64))+t_next*noises[step]
                        if not torch.isfinite(latents).all():
                            raise RuntimeError('Nonfinite free-rollout latent')
                        row['steps'].append(dict(step=step, timestep=timestep.item(),
                            actual_native_mm_calls=actual['native'], actual_sdpa_calls=actual['sdpa'],
                            validated_fastpack_flags=len(checks), noise_sha256=shared['noise_sha256'][step],
                            latent_after_sha256=tensor_sha(latents)))
                        checkpoint()
                    row['actual_native_mm_calls'] = sum(s['actual_native_mm_calls'] for s in row['steps'])
                    row['actual_dit_calls'] = len(row['steps'])
                    if row['actual_native_mm_calls'] != arm['native_calls_per_video'] or row['actual_dit_calls'] != 4:
                        raise RuntimeError('Four-step native call contract failed')
                    if installed and sum(m.native_calls for m in installed.values())-before_module_calls != 1200:
                        raise RuntimeError('Packed module counters disagree with actual scaled_mm calls')
                    path = args.artifact_dir/name/case_id/'final_latent.pt'
                    row['final_latent'] = tensor_file(path, latents.cpu())
                    row['final_latent_sha256'] = tensor_sha(latents)
                    row['status'] = 'denoising_complete'
                    del payload, latents, noises, embedding, velocity
                    print(json.dumps(dict(arm=name, case=case_id, status=row['status'],
                                          native_calls=row['actual_native_mm_calls'])), flush=True)
                    checkpoint()
                report['arms'][name]['status'] = 'denoising_complete'
                del model, installed, native_modules
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            F.scaled_dot_product_attention, F.scaled_mm = original_sdpa, original_mm

        # One identical VAE instance, after all DiT graphs are released.
        vae = AutoencoderKLWan.from_pretrained(BASE/'vae', torch_dtype=torch.bfloat16,
                                               local_files_only=True).to(device).eval()
        mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
        reciprocal_std = 1.0/torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
        for arm in arm_rows:
            for case in case_rows:
                budget()
                row = report['arms'][arm['name']]['cases'][case['case_id']]
                latent = torch.load(row['final_latent']['path'], map_location='cpu', weights_only=False).to(device)
                normalized = latent.float().to(vae.dtype)/reciprocal_std+mean
                if hasattr(vae, 'clear_cache'):
                    vae.clear_cache()
                decoded = decode_spatial_tiled(vae, normalized, core=128, halo=0)
                if decoded.shape != (1, 3, 77, 480, 832) or not torch.isfinite(decoded).all():
                    raise RuntimeError('Invalid decoded video')
                frames = video_processor.postprocess_video(decoded, output_type='np')[0]
                path = Path(row['final_latent']['path']).with_name('video.mp4')
                if path.exists():
                    raise FileExistsError(path)
                export_to_video(frames, str(path), fps=16)
                row['video'] = record_file(path) | dict(frames=len(frames), height=480, width=832, fps=16)
                row['status'] = 'complete'
                del latent, normalized, decoded, frames
                if hasattr(vae, 'clear_cache'):
                    vae.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
                checkpoint()
            report['arms'][arm['name']]['status'] = 'complete'
        report['actual_totals'] = dict(native_mm_calls=counts['native'], sdpa_calls=counts['sdpa'],
                                      dit_calls=4*len(case_rows)*len(arm_rows), videos=len(case_rows)*len(arm_rows))
        report['status'] = 'complete'
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='CPU-only fixed manifest draft; no torch import')
    parser.add_argument('--inventory', type=Path, default=INVENTORY)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--artifact-dir', type=Path, default=DATA/'research/20261003/E021')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E021/generation_run.json')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--deadline-unix', type=float, help='Required for GPU execution; budget chosen by the caller')
    args = parser.parse_args()
    if args.prepare_only:
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        save_json(args.manifest, prepare_manifest(args.inventory))
        print(json.dumps(dict(status='draft_not_generated', manifest=str(args.manifest), cuda_imported=False)))
        return
    if args.deadline_unix is None:
        parser.error('GPU generation requires --deadline-unix; --prepare-only is CPU-only')
    if args.deadline_unix <= time.time():
        parser.error('deadline has already expired')
    if len(set(args.arms)) != len(args.arms) or len(set(args.cases)) != len(args.cases):
        parser.error('duplicate arm/case selections are not allowed')
    if args.output.exists() or args.artifact_dir.exists():
        raise FileExistsError('Use fresh output and artifact paths; existing runs are preserved')
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    report = dict(experiment='E021', status='running', manifest=manifest,
                  requested_arms=args.arms, requested_cases=args.cases, deadline_unix=args.deadline_unix)
    try:
        execute(args, manifest, report)
    except Exception:
        report['status'] = 'failed_partial_preserved'
        report['error'] = traceback.format_exc()
        save_json(args.output, report)
        raise
    print(json.dumps(dict(status=report['status'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
