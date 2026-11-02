#!/usr/bin/env python3
"""Closed-loop H3 native W4A4 with 10% BF16 DiT-output token restoration."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/research')]
import run_e121_h3_split_generation as e121
import run_e124_h3_attention_prompts as e124
import run_e126_h3_vbench5_attention as e126
import run_e127_h3_w4a4_attention as e127
from bench_h3_native_nvfp4 import make_h3_resident
from h3_plain_native_nvfp4 import install_plain_h3, PlainNativeH3Linear

REPORT = ROOT / 'results/research/E133'
DATA = Path('/data1/models/svdquant-wjq/research/20261026/E133_h3_timestep_token_restore')
EXPORT = Path('/data1/models/svdquant-wjq/research/20261024/E128_plain_export_rehydrated')
ARMS = ('random10', 'cycle10')


def case_info(case):
    return e127.case_info(case, 'e124' if case == 'p016' else 'e126')


def settings(case):
    return json.loads(e124.MANIFEST.read_text())['generation'] if case == 'p016' else e126.SETTINGS


def select_tokens(n, arm, step, seed, modality, device):
    count = max(1, round(n / 10))
    if arm == 'cycle10':
        # Contiguous partitions cover each token exactly once every 10 steps.
        start = ((step % 10) * n) // 10
        end = (((step % 10) + 1) * n) // 10
        return torch.arange(start, end, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 133000000 + 1000 * step + modality)
    return torch.randperm(n, device=device, generator=generator)[:count]


def restore(reference, quantized, arm, step, seed, modality):
    assert reference.shape == quantized.shape and reference.dtype == quantized.dtype == torch.bfloat16
    n = reference.shape[0]
    chosen = select_tokens(n, arm, step, seed, modality, reference.device)
    corrected = quantized.clone()
    corrected.index_copy_(0, chosen, reference.index_select(0, chosen))
    before = (quantized.float() - reference.float()).square().sum(dtype=torch.float64)
    after = (corrected.float() - reference.float()).square().sum(dtype=torch.float64)
    energy = reference.float().square().sum(dtype=torch.float64)
    return corrected, {'tokens': n, 'replaced': chosen.numel(), 'fraction': chosen.numel()/n,
                       'nmse_before': float(before / energy), 'nmse_after': float(after / energy),
                       'error_energy_removed': float(1 - after / before)}


@torch.inference_mode()
def denoise(case, arm):
    info, cfg = case_info(case), settings(case)
    steps = cfg['steps']
    report_path = REPORT / case / f'{arm}_denoise.json'
    stage = DATA / case / arm
    if report_path.exists() or stage.exists():
        raise FileExistsError('existing E133 result; refusing overwrite')
    stage.mkdir(parents=True)
    pipe = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    make_h3_resident(pipe.dit)
    install_plain_h3(pipe.dit, EXPORT, retain_bf16=True)
    modules = [m for m in pipe.dit.modules() if isinstance(m, PlainNativeH3Linear)]
    assert len(modules) == 200
    original_forward = pipe.dit.forward
    original_step = pipe.step
    original_loader = pipe.load_models_to_device
    final, initial = {}, {}
    report = {'format': 'e133-timestep-token-restore-v1', 'status': 'running',
              'case': case, 'arm': arm, 'seed': info['seed'], 'settings': cfg,
              'quantization': 'E128 native NVFP4 W4A4; 200 target linears',
              'intervention': '10% BF16 output-token replacement per modality per step',
              'steps': [], 'started_unix': time.time()}
    e121.base.save_json(report_path, report)

    def paired_forward(_self, *args, **kwargs):
        step = len(report['steps'])
        assert step < steps
        for module in modules:
            module.execution_mode = 'bf16'
        reference = e121.base.split_output(original_forward(*args, **kwargs))
        for module in modules:
            module.execution_mode = 'native'
        quantized = e121.base.split_output(original_forward(*args, **kwargs))
        output, metrics = [], {}
        for modality, (name, ref, quant) in enumerate(zip(('video', 'audio'), reference,
                                                            quantized, strict=True)):
            modified, row = restore(ref, quant, arm, step, info['seed'], modality)
            assert torch.isfinite(modified).all().item()
            output.append(modified)
            metrics[name] = row
        report['steps'].append({'step': step, 'modalities': metrics})
        e121.base.save_json(report_path, report)
        print(f'{case}/{arm} step {step + 1}/{steps}: video NMSE '
              f"{metrics['video']['nmse_before']:.6f} -> {metrics['video']['nmse_after']:.6f}", flush=True)
        return tuple(output)

    def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
        name = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
        assert name and progress_id == len(report['steps']) - 1
        if progress_id == 0:
            initial[name] = e121.base.tensor_sha256(latents)
        updated = original_step(scheduler, latents, progress_id,
                                noise_pred=noise_pred, **kwargs)
        assert torch.isfinite(updated).all().item()
        if progress_id == steps - 1:
            final[name + '_latents'] = updated.detach().cpu()
        return updated

    class Complete(Exception):
        pass

    def stop_before_decode(_self, names):
        if list(names) == ['video_vae'] and len(report['steps']) == steps:
            assert set(final) == {'video_latents', 'audio_latents'}
            raise Complete()
        return original_loader(names)

    pipe.dit.forward = types.MethodType(paired_forward, pipe.dit)
    pipe.step = types.MethodType(observed_step, pipe)
    pipe.load_models_to_device = types.MethodType(stop_before_decode, pipe)
    try:
        try:
            pipe(prompt=info['prompt'], seed=info['seed'], height=cfg['height'], width=cfg['width'],
                 num_frames=cfg['frames'], num_inference_steps=steps, cfg_scale=cfg['cfg_scale'],
                 flow_shift=cfg['flow_shift'], audio_flow_shift=cfg['audio_flow_shift'],
                 rand_device='cpu', tiled=True)
        except Complete:
            pass
        else:
            raise RuntimeError('did not stop after denoising')
        assert len(report['steps']) == steps and set(initial) == {'video', 'audio'}
        terminal = stage / 'final_latents.pt'
        with terminal.open('xb') as stream:
            torch.save(final, stream)
        report.update(status='denoise_complete', terminal_latents=str(terminal),
                      terminal_sha256=e121.base.sha256(terminal),
                      initial_latent_sha256=initial,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      completed_unix=time.time())
        e121.base.save_json(report_path, report)
        print(f'DENOISE COMPLETE {case}/{arm}', flush=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        e121.base.save_json(report_path, report)
        raise
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def decode(case, arm):
    e121.base.REPORT = REPORT
    e121.base.DATA = DATA
    if case == 'p016':
        e121.base.decode(case, arm)
    else:
        e126.REPORT = REPORT
        e126.DATA = DATA
        e126.decode(case, arm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('p016', 'vbench_000'), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--phase', choices=('denoise', 'decode'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    if args.phase == 'denoise':
        denoise(args.case, args.arm)
    else:
        decode(args.case, args.arm)
