#!/usr/bin/env python3
"""E008: one fixed prompt, paired rCM four-step BF16/nativefast videos.

Reuse historical rCM sampling and VAE decoding. No quality metric or held-out
claim. Seed1 historical candidate, sole initialization fallback seed42. Shared
initial FP64 latent and FP32 noises, independently evolved arm trajectories.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import gc
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'third_party/deepcompressor'))
from bench_wan_native_nvfp4 import BASE, RCM, CHECKPOINT, CACHE, DATA, sha256, tensor_sha, metric, save

CORRECTNESS = ROOT/'results/research/E007_wan_native_correctness.json'
PARITY = ROOT/'results/research/E007_fastpack_real_parity.json'
PLAN = ROOT/'research_state/06_experiments/E008_wan_native_paired_video_plan.md'


def prerequisite_check(native_profile):
    """CPU-only fail-closed check before model loading or GPU use."""
    correctness = json.loads(CORRECTNESS.read_text())
    parity = json.loads(PARITY.read_text())
    profile = json.loads(native_profile.read_text())
    if correctness['status'] != 'complete' or parity['status'] != 'complete':
        raise RuntimeError('E007 correctness/fastpacker prerequisites incomplete')
    if any(any(row['byte_mismatches'].values()) for row in parity['cases']):
        raise RuntimeError('fastpacker byte parity failed')
    if profile.get('status') != 'complete' or profile.get('arm') != 'native':
        raise RuntimeError('completed nativefast whole-model profile required before E008')
    expected = correctness['stages']['native_full']['endpoint_sha256']
    sha_rows = [profile['warmup'], profile['profile'], *profile['unprofiled_repeats']]
    if len(profile['unprofiled_repeats']) != 3 or any(r['endpoint_sha256'] != expected for r in sha_rows):
        raise RuntimeError('E007 nativefast whole-model endpoint SHA prerequisite failed')
    if profile['profile']['counts']['native_gemm'] != 300 or profile['profile']['counts']['attention'] != 60:
        raise RuntimeError('E007 nativefast actual native/attention path not verified')
    fast_path = Path(__file__).with_name('wan_nvfp4_fastpack.py')
    if parity['source_sha256'][fast_path.name] != sha256(fast_path):
        raise RuntimeError('fastpacker differs from the source verified by real-input parity')
    # The profile already pins its model/checkpoint/helpers to E007 correctness.
    # Recheck every recorded source/artifact now instead of trusting status alone.
    for source_report in (correctness, profile):
        for name, info in source_report['files'].items():
            if sha256(Path(name)) != info['sha256']:
                raise RuntimeError(f'prerequisite file changed after E007 verification: {name}')
    return {'expected_native_endpoint_sha256': expected,
        'correctness': {'path': str(CORRECTNESS), 'sha256': sha256(CORRECTNESS)},
        'fast_parity': {'path': str(PARITY), 'sha256': sha256(PARITY)},
        'native_profile': {'path': str(native_profile), 'sha256': sha256(native_profile)}}


@torch.inference_mode()
def execute(args, report):
    report['prerequisites'] = prerequisite_check(args.native_profile)
    import diffusers
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers.utils import export_to_video
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from infer_rcm_wan_4step import load_quantized_transformer, decode_spatial_tiled
    import wan_native_nvfp4 as native_lib
    from wan_nvfp4_fastpack import pack_activation_fast, collect_fastpack_checks, validate_quantizer_contract
    if diffusers.__version__ != '0.33.1' or torch.__version__ != '2.11.0+cu128':
        raise RuntimeError('same verified E007 torch2.11.0+cu128 / Diffusers0.33.1 required')
    if args.artifact_dir.exists() and any(args.artifact_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite E008 artifacts: {args.artifact_dir}')
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    def checkpoint():
        report['elapsed_seconds_not_benchmark'] = time.time()-started
        report['peak_gpu_gib_not_benchmark'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        if report['elapsed_seconds_not_benchmark'] > 3600:
            raise TimeoutError('E008 one-hour internal budget exceeded')
        if report['peak_gpu_gib_not_benchmark'] > 60:
            raise MemoryError('E008 60GiB stop line exceeded; do not change frames or tokens')
    def artifact(name, value):
        path = args.artifact_dir/name
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')
        torch.save(value, path)
        return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}

    sources = [Path(__file__), PLAN, ROOT/'scripts/infer_rcm_wan_4step.py',
        Path(native_lib.__file__), Path(__file__).with_name('wan_nvfp4_fastpack.py'),
        BASE/'vae/config.json', BASE/'vae/diffusion_pytorch_model.safetensors']
    cache_paths = [CACHE.with_name(f'0001-{step:05d}-0.pt') for step in range(4)]
    sources += cache_paths
    report['files'] = {str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sources}
    report.update(torch=torch.__version__, diffusers=diffusers.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), artifact_dir=str(args.artifact_dir))
    caches = [torch.load(p, map_location='cpu', weights_only=False) for p in cache_paths]
    embedding = caches[0]['input_kwargs']['encoder_hidden_states'].cuda()
    for step, cache in enumerate(caches):
        if cache['filename'] != '0001' or cache['step'] != step or cache['guidance'] != 0:
            raise RuntimeError('fixed prompt0001 cache identity changed')
        if not torch.equal(cache['input_kwargs']['encoder_hidden_states'], embedding.cpu()):
            raise RuntimeError('prompt embedding differs between historical cache steps')
    if list(embedding.shape) != [1, 512, 4096] or embedding.dtype != torch.bfloat16:
        raise RuntimeError('fixed BF16 cached embedding contract violated')
    report['embedding_sha256'] = tensor_sha(embedding)
    print('Loading BF16 rCM model without text encoder or VAE', flush=True)
    model = WanTransformer3DModel.from_pretrained(RCM, torch_dtype=torch.bfloat16).cuda().eval()
    pipe = WanPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, transformer=model,
                                       text_encoder=None, tokenizer=None, vae=None)
    video_processor = pipe.video_processor
    # Directly reproduce the existing infer_rcm_wan_4step.py schedule.
    angles = torch.tensor([math.atan(80.), 1.5, 1.4, 1., 0.], dtype=torch.float64, device='cuda')
    t_steps = angles.sin()/(angles.cos()+angles.sin())
    ones = torch.ones((1,), device='cuda', dtype=torch.float64)
    report['schedule'] = {'angles': angles.cpu().tolist(), 't_steps': t_steps.cpu().tolist(),
                         'timesteps_bf16': [(t.float()*ones*1000).to(torch.bfloat16).item() for t in t_steps[:-1]]}

    def initial_state(seed):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        initial_noise = pipe.prepare_latents(batch_size=1, num_channels_latents=16,
            height=480, width=832, num_frames=77, dtype=torch.float32, device=torch.device('cuda'), generator=generator)
        return initial_noise.to(torch.float64)*t_steps[0], generator

    initial, generator = initial_state(1)
    if list(initial.shape) != [1, 16, 20, 60, 104]:
        raise RuntimeError('existing prepare_latents changed expected 77-frame shape')
    candidate_input = initial.to(torch.bfloat16).cpu()
    candidate_matches = torch.equal(candidate_input, caches[0]['input_args'][0])
    report['seed1_candidate'] = {'initialization_matches_history': candidate_matches,
        'input_metric_vs_cache': metric(candidate_input, caches[0]['input_args'][0]),
        'candidate_input_sha256': tensor_sha(candidate_input),
        'artifact': artifact('seed1_candidate.pt', {'initial_latent_fp64': initial.cpu(),
            'model_input_bf16': candidate_input, 'matches_cache_step0': candidate_matches})}
    del candidate_input
    if candidate_matches:
        seed, run_kind = 1, 'historical_seed1_candidate_pending_full_bf16_replay'
    else:
        report['seed1_candidate']['failure_reason'] = 'regenerated CUDA seed1 step0 model input differs from saved BF16 cache'
        initial, generator = initial_state(42)
        seed, run_kind = 42, 'new_seed42_paired_run_not_historical_reproduction'
    # Original code consumes a new FP32 noise after every forward, including the
    # last update with zero coefficient; preserve the same generator sequence.
    noises = [torch.randn(initial.shape, dtype=torch.float32, device='cuda', generator=generator)
              for _ in range(4)]
    report.update(selected_seed=seed, run_kind=run_kind,
        initial_latent_sha256=tensor_sha(initial), noise_sha256=[tensor_sha(n) for n in noises])
    report['shared_random_inputs'] = artifact('shared_random_inputs.pt', {
        'seed': seed, 'initial_latent_fp64': initial.cpu(), 'update_noises_fp32': [n.cpu() for n in noises],
        't_steps_fp64': t_steps.cpu(), 'embedding_bf16': embedding.cpu(),
        'generator_state_after_all_draws': generator.get_state().cpu()})
    checkpoint()
    original_sdpa, original_mm = F.scaled_dot_product_attention, F.scaled_mm
    audit = {'sdpa': 0, 'native_mm': 0}
    def audited_sdpa(q, k, v, *pos, **kw):
        if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
            raise RuntimeError('actual DiT SDPA QKV must remain BF16')
        audit['sdpa'] += 1
        return original_sdpa(q, k, v, *pos, **kw)
    def audited_mm(*pos, **kw):
        audit['native_mm'] += 1
        return original_mm(*pos, **kw)
    F.scaled_dot_product_attention, F.scaled_mm = audited_sdpa, audited_mm
    report['arms'] = {}

    def run_arm(name, fast):
        latents = initial.clone()
        arm = {'status': 'running', 'steps': [], 'initial_latent_sha256': tensor_sha(latents)}
        report['arms'][name] = arm
        if arm['initial_latent_sha256'] != report['initial_latent_sha256']:
            raise RuntimeError('paired initial latent changed')
        for step, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            timestep = (t_cur.float()*ones*1000).to(torch.bfloat16)
            model_input = latents.to(torch.bfloat16)
            row = {'step': step, 'timestep': timestep.item(), 'input_sha256': tensor_sha(model_input),
                   'latent_before_sha256': tensor_sha(latents), 'noise_sha256': tensor_sha(noises[step])}
            arm['steps'].append(row)
            historical = name == 'bf16' and candidate_matches
            if historical:
                row['history_input_metric'] = metric(model_input.cpu(), caches[step]['input_args'][0])
                if row['history_input_metric']['err2'] != 0 or not torch.equal(timestep.cpu(), caches[step]['input_kwargs']['timestep']):
                    row['failure_artifact'] = artifact(f'{name}_step{step}_input_mismatch.pt', {
                        'latent_fp64': latents.cpu(), 'model_input_bf16': model_input.cpu(), 'timestep': timestep.cpu()})
                    row['failure'] = 'historical seed1 input/timestep replay failed; no teacher-state substitution'
                    checkpoint()
                    raise RuntimeError(f'historical BF16 replay failed before step {step}')
            before = dict(audit)
            checks_context = collect_fastpack_checks() if fast else nullcontext([])
            with checks_context as checks:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    velocity_bf16 = model(hidden_states=model_input, timestep=timestep,
                                         encoder_hidden_states=embedding, return_dict=False)[0]
            row.update(actual_sdpa_calls=audit['sdpa']-before['sdpa'],
                       actual_native_mm_calls=audit['native_mm']-before['native_mm'],
                       validated_fastpack_flags=len(checks))
            if row['actual_sdpa_calls'] != 60 or row['actual_native_mm_calls'] != (300 if fast else 0) or len(checks) != (300 if fast else 0):
                raise RuntimeError(f'actual full DiT path changed at {name} step{step}: {row}')
            if not torch.isfinite(velocity_bf16).all():
                raise RuntimeError(f'nonfinite {name} velocity at step{step}')
            velocity = velocity_bf16.to(torch.float64)
            # The existing rCM update, with the already shared original noise.
            next_latents = (1-t_next)*(latents-t_cur*velocity)+t_next*noises[step]
            if not torch.isfinite(next_latents).all():
                raise RuntimeError(f'nonfinite {name} next latent at step{step}')
            row.update(output_sha256=tensor_sha(velocity_bf16), latent_after_sha256=tensor_sha(next_latents),
                artifact=artifact(f'{name}_step{step}.pt', {'latent_before_fp64': latents.cpu(),
                    'velocity_bf16': velocity_bf16.cpu(), 'latent_after_fp64': next_latents.cpu(),
                    'timestep_bf16': timestep.cpu(), 'noise_sha256': row['noise_sha256']}))
            if historical:
                row['history_velocity_metric'] = metric(velocity_bf16.cpu(), caches[step]['outputs'][0])
                if row['history_velocity_metric']['err2'] != 0:
                    row['failure'] = 'historical seed1 velocity replay failed; original output and latent preserved'
                    checkpoint()
                    raise RuntimeError(f'historical BF16 velocity differs at step {step}')
            latents = next_latents
            print(json.dumps({'arm': name, 'step': step, 'history_verified': historical,
                'native_calls': row['actual_native_mm_calls'], 'flags': len(checks)}, ensure_ascii=False), flush=True)
            checkpoint()
        arm['final_latent'] = artifact(f'{name}_final_latent.pt', latents.cpu())
        arm['final_latent_sha256'] = tensor_sha(latents)
        arm['status'] = 'denoising_complete'
        checkpoint()

    try:
        run_arm('bf16', fast=False)
        if candidate_matches:
            report['run_kind'] = 'historical_seed1_all_four_bf16_inputs_and_velocities_exact'
        print('Installing the verified nativefast graph for its independent four-step rollout', flush=True)
        load_quantized_transformer(pipe, CHECKPOINT, BASE)
        model = pipe.transformer.eval()
        conversion = native_lib.convert_wan_transformer_to_native(model, CHECKPOINT,
            activation_packer='legacy', chunk_rows=1024)
        if conversion['target_count'] != 300 or conversion['exact_roundtrip_count'] != 300:
            raise RuntimeError('nativefast requires all 300 exact saved-weight roundtrips')
        names = []
        for name, module in model.named_modules():
            if isinstance(module, native_lib.NativeWanLinear):
                validate_quantizer_contract(module.activation_quantizer)
                module.activation_packer = pack_activation_fast
                names.append(name)
        if len(names) != 300:
            raise RuntimeError('exactly 300 nativefast recipes must validate')
        del module
        report['native_conversion'] = {'target_count': 300, 'exact_roundtrip_count': 300,
                                       'fast_recipe_validated_modules': names}
        run_arm('nativefast', fast=True)
    finally:
        F.scaled_dot_product_attention, F.scaled_mm = original_sdpa, original_mm
    # No VAE coexistence with either DiT graph. Shared randomness already saved.
    del model, pipe, initial, noises, embedding, caches, generator
    gc.collect()
    torch.cuda.empty_cache()
    report['after_dit_release_allocated_gib'] = torch.cuda.memory_allocated()/1024**3
    if report['after_dit_release_allocated_gib'] > 1:
        raise RuntimeError('DiT release incomplete; do not load VAE alongside retained model storage')
    checkpoint()
    print('Both four-step denoisers complete; loading one shared BF16 VAE for sequential decoding', flush=True)
    vae = AutoencoderKLWan.from_pretrained(BASE/'vae', torch_dtype=torch.bfloat16).cuda().eval()
    mean = torch.tensor(vae.config.latents_mean, device='cuda', dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    reciprocal_std = 1.0/torch.tensor(vae.config.latents_std, device='cuda', dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    for name in ('bf16', 'nativefast'):
        latent_path = Path(report['arms'][name]['final_latent']['path'])
        latents = torch.load(latent_path, map_location='cpu', weights_only=False).cuda()
        normalized = latents.float().to(vae.dtype)/reciprocal_std+mean
        if hasattr(vae, 'clear_cache'):
            vae.clear_cache()
        decoded = decode_spatial_tiled(vae, normalized, core=128, halo=0)
        if list(decoded.shape) != [1, 3, 77, 480, 832] or not torch.isfinite(decoded).all():
            raise RuntimeError(f'invalid decoded video for {name}: {tuple(decoded.shape)}')
        frames = video_processor.postprocess_video(decoded, output_type='np')[0]
        video_path = args.artifact_dir/f'{name}.mp4'
        if video_path.exists():
            raise FileExistsError(video_path)
        export_to_video(frames, str(video_path), fps=16)
        report['arms'][name]['video'] = {'path': str(video_path), 'sha256': sha256(video_path),
            'frames': len(frames), 'height': 480, 'width': 832, 'fps': 16,
            'decoder': 'identical BF16 VAE; existing decode_spatial_tiled core128 halo0; inference_mode'}
        report['arms'][name]['status'] = 'complete'
        del latents, normalized, decoded, frames
        if hasattr(vae, 'clear_cache'):
            vae.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint()
    report['status'] = 'complete'
    checkpoint()
    print(json.dumps({'status': report['status'], 'seed': seed, 'run_kind': report['run_kind'],
        'videos': {k: v['video']['path'] for k, v in report['arms'].items()}}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-profile', type=Path, default=ROOT/'results/research/E007_profile_native.json')
    parser.add_argument('--artifact-dir', type=Path, default=DATA/'research/20261002/E008/p0001')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E008_wan_native_paired_video.json')
    parser.add_argument('--check-prerequisites-only', action='store_true')
    args = parser.parse_args()
    if args.check_prerequisites_only:
        print(json.dumps(prerequisite_check(args.native_profile), indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite a prior E008 report: {args.output}')
    report = {'experiment': 'E008', 'status': 'partial', 'prompt_id': '0001',
        'prompt': 'A person is roller skating', 'frames': 77, 'height': 480, 'width': 832, 'fps': 16,
        'guidance': 0., 'sigma_max': 80., 'steps': 4,
        'limitations': ['one original calibration prompt; qualitative paired videos only',
            'no VBench/MJVIDEO or perceptual quality/generalization claim',
            'independent four-step rollout with common initial latent/noises; no teacher-state injection',
            'seed1 history requires exact four-step checks; initialization failure only falls back to new seed42']}
    try:
        execute(args, report)
    except Exception:
        report['status'] = 'failed_no_paired_quality_claim'
        report['error'] = traceback.format_exc()
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
