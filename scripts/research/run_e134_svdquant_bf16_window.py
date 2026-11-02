#!/usr/bin/env python3
"""Closed-loop H3 generation with BF16 first and/or last 10% steps."""
import argparse
import gc
import importlib
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
import run_e129_h3_svdquant_attention as e129
from infer_minimax_h3_svdquant_standard import apply_svdquant

REPORT = ROOT / 'results/research/E134'
DATA = Path('/data1/models/svdquant-wjq/research/20261028/E134_svdquant_bf16_step_window')
STATE = ROOT / 'results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt'
ARMS = ('bf16_first10pct', 'bf16_last10pct', 'bf16_both10pct')


def case_info(case):
    return e129.case_info(case, 'e124' if case == 'p016' else 'e126')


def settings(case):
    return json.loads(e124.MANIFEST.read_text())['generation'] if case == 'p016' else e126.SETTINGS


@torch.inference_mode()
def denoise(case, arm):
    info, cfg = case_info(case), settings(case)
    steps = cfg['steps']
    window = steps // 10
    if arm == 'bf16_first10pct':
        bf16_steps = set(range(window))
    elif arm == 'bf16_last10pct':
        bf16_steps = set(range(steps-window, steps))
    else:
        assert arm == 'bf16_both10pct'
        bf16_steps = set(range(window)) | set(range(steps-window, steps))
    report_path = REPORT / case / f'{arm}_denoise.json'
    stage = DATA / case / arm
    if report_path.exists() or stage.exists():
        raise FileExistsError('existing E134 result; refusing overwrite')
    stage.mkdir(parents=True)
    quant = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    quant.load_models_to_device(['dit'])
    quant.dit.eval()
    hooks = apply_svdquant(quant.dit, STATE, tensor_scale_mode='dynamic')
    assert len(hooks) == 200
    reference = e121.base.load_h3_pipeline(full=False, vram_limit_gib=18.0)
    reference.load_models_to_device(['dit'])
    reference.dit.eval()
    assert torch.cuda.mem_get_info()[0] > 5 * 1024**3
    original_forward = quant.dit.forward
    original_step = quant.step
    original_loader = quant.load_models_to_device
    initial, final = {}, {}
    report = {'format': 'e134-svdquant-bf16-window-v1', 'status': 'running',
              'case': case, 'arm': arm, 'seed': info['seed'], 'settings': cfg,
              'svdquant_state_sha256': e121.base.sha256(STATE),
              'bf16_steps_zero_based': sorted(bf16_steps), 'steps': [],
              'started_unix': time.time()}
    e121.base.save_json(report_path, report)

    def selected_forward(_self, *args, **kwargs):
        step = len(report['steps'])
        assert step < steps
        use_bf16 = step in bf16_steps
        start = time.monotonic()
        output = reference.dit(*args, **kwargs) if use_bf16 else original_forward(*args, **kwargs)
        video, audio = e121.base.split_output(output)
        assert torch.isfinite(video).all().item() and torch.isfinite(audio).all().item()
        report['steps'].append({'step': step, 'precision': 'bf16' if use_bf16 else 'svdquant',
                                'dit_seconds': time.monotonic() - start,
                                'video_output_energy': float(video.float().square().mean()),
                                'audio_output_energy': float(audio.float().square().mean())})
        e121.base.save_json(report_path, report)
        print(f'{case}/{arm} step {step+1}/{steps}: '
              f'{"BF16" if use_bf16 else "SVDQuant"} {report["steps"][-1]["dit_seconds"]:.2f}s', flush=True)
        return output

    def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
        name = 'video' if scheduler is quant.scheduler else 'audio' if scheduler is quant.scheduler_audio else None
        assert name and progress_id == len(report['steps']) - 1
        if progress_id == 0:
            initial[name] = e121.base.tensor_sha256(latents)
        updated = original_step(scheduler, latents, progress_id, noise_pred=noise_pred, **kwargs)
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

    quant.dit.forward = types.MethodType(selected_forward, quant.dit)
    quant.step = types.MethodType(observed_step, quant)
    quant.load_models_to_device = types.MethodType(stop_before_decode, quant)
    try:
        try:
            quant(prompt=info['prompt'], seed=info['seed'], height=cfg['height'], width=cfg['width'],
                  num_frames=cfg['frames'], num_inference_steps=steps, cfg_scale=cfg['cfg_scale'],
                  flow_shift=cfg['flow_shift'], audio_flow_shift=cfg['audio_flow_shift'],
                  rand_device='cpu', tiled=True)
        except Complete:
            pass
        else:
            raise RuntimeError('did not stop after denoising')
        assert len(report['steps']) == steps and set(initial) == {'video', 'audio'}
        assert min(h.act.calls for h in hooks) == steps - len(bf16_steps)
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
        del quant, reference
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
    importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION = 'torch'
    torch.set_num_threads(6)
    if args.phase == 'denoise':
        denoise(args.case, args.arm)
    else:
        decode(args.case, args.arm)
