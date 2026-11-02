#!/usr/bin/env python3
"""E136 paired H3 rollouts: INT8 endpoints or BF16 endpoints with Sage3 middle."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time
import types

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/research')]
import run_e121_h3_split_generation as e121
import run_e124_h3_attention_prompts as e124
import run_e126_h3_vbench5_attention as e126
import run_e129_h3_svdquant_attention as e129
from bench_h3_native_nvfp4 import make_h3_resident
from infer_minimax_h3_svdquant_standard import apply_svdquant
from minimax_h3_svdquant_common import target_linears, nvfp4_qdq

REPORT = ROOT / 'results/research/step_end_w4a8'
DATA = ROOT / 'results/local/step_end_w4a8'
STATE = ROOT / 'results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt'
ARMS = ('ends_nvfp4_int8_g128', 'ends_bf16_svd_sage3')
GROUP = 128


@torch.no_grad()
def int8_group128_qdq(x: torch.Tensor, chunk_rows: int = 1024) -> torch.Tensor:
    """Symmetric signed W8A8 oracle: per-row K128 absmax, RNE, BF16 output."""
    if x.dtype != torch.bfloat16 or x.shape[-1] % GROUP:
        raise ValueError('INT8 group-128 expects BF16 and K divisible by 128')
    rows = x.reshape(-1, x.shape[-1])
    result = torch.empty_like(rows)
    for start in range(0, rows.shape[0], chunk_rows):
        chunk = rows[start:start + chunk_rows].float().reshape(-1, rows.shape[-1] // GROUP, GROUP)
        scale = (chunk.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-12)
        reconstructed = chunk.div(scale).round_().clamp_(-127, 127).mul_(scale)
        result[start:start + chunk_rows].copy_(reconstructed.reshape(-1, rows.shape[-1]))
    return result.reshape(x.shape)


class Int8QDQLinear(nn.Module):
    """Numerical INT8 group-128 proxy; not a deployment INT8 GEMM."""

    def __init__(self, old: nn.Linear):
        super().__init__()
        if not isinstance(old, nn.Linear) or old.weight.is_meta:
            raise TypeError('expected materialized original BF16 Linear')
        self.in_features = old.in_features
        self.out_features = old.out_features
        self.register_buffer('weight', nvfp4_qdq(old.weight.detach(), group_size=16))
        self.register_buffer('bias', old.bias.detach() if old.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        outputs = []
        for start in range(0, rows.shape[0], 1024):
            qx = int8_group128_qdq(rows[start:start + 1024])
            outputs.append(F.linear(qx, self.weight, self.bias))
        return torch.cat(outputs, dim=0).reshape(*x.shape[:-1], self.out_features)


@torch.no_grad()
def install_int8_linears(dit):
    resident = make_h3_resident(dit)
    linears = target_linears(dit)
    for index, (name, old) in enumerate(linears, 1):
        parent, attr = name.rsplit('.', 1)
        replacement = Int8QDQLinear(old)
        setattr(dit.get_submodule(parent), attr, replacement)
        old.weight = None
        old.bias = None
        if index % 20 == 0:
            print(f'INT8 group-128 installed {index}/200', flush=True)
    assert sum(isinstance(module, Int8QDQLinear) for module in dit.modules()) == 200
    return {'count': 200, 'resident_allocated_gib': resident['allocated_gib'],
            'weight': 'original BF16 W, NVFP4 E2M1 group16 E4M3 scale and dynamic tensor scale Q/DQ',
            'activation': 'per-token K128 symmetric INT8 Q/DQ',
            'scale': 'FP32 absmax/127', 'codes': 'signed [-127,127]',
            'rounding': 'torch.round ties-to-even', 'gemm': 'BF16 F.linear after Q/DQ'}


def case_info(case):
    return e129.case_info(case, 'e124' if case == 'p016' else 'e126')


def settings(case):
    return json.loads(e124.MANIFEST.read_text())['generation'] if case == 'p016' else e126.SETTINGS


@torch.inference_mode()
def denoise(case, arm):
    info, cfg = case_info(case), settings(case)
    steps = cfg['steps']
    window = steps // 10
    ends = set(range(window)) | set(range(steps - window, steps))
    report_path = REPORT / case / f'{arm}_denoise.json'
    stage = DATA / case / arm
    if report_path.exists() or stage.exists():
        raise FileExistsError(f'existing E136 result: {report_path} or {stage}')
    stage.mkdir(parents=True)
    report = {'format': 'e136-h3-precision-window-v1', 'status': 'loading',
              'case': case, 'arm': arm, 'seed': info['seed'], 'settings': cfg,
              'endpoint_steps_zero_based': sorted(ends), 'steps': [],
              'started_unix': time.time()}
    e121.base.save_json(report_path, report)
    quant = alternate = router = None
    try:
        quant = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
        quant.load_models_to_device(['dit'])
        quant.dit.eval()
        hooks = apply_svdquant(quant.dit, STATE, tensor_scale_mode='dynamic')
        assert len(hooks) == 200
        alternate = e121.base.load_h3_pipeline(full=False, vram_limit_gib=18.0)
        alternate.load_models_to_device(['dit'])
        alternate.dit.eval()
        int8_info = install_int8_linears(alternate.dit) if arm == 'ends_nvfp4_int8_g128' else None
        if arm == 'ends_bf16_svd_sage3':
            e121.check_backend()
        assert torch.cuda.mem_get_info()[0] > 5 * 1024**3
        report.update(status='running', svdquant_state_sha256=e121.base.sha256(STATE),
                      middle_linears='rank-64 SVDQuant W4A4 Q/DQ, dynamic tensor scale',
                      endpoint_linears=int8_info if int8_info else 'BF16 original',
                      middle_attention='corrected Sage3 FP4' if int8_info is None else 'BF16 torch',
                      endpoint_attention='BF16 torch',
                      loaded_allocated_gib=torch.cuda.memory_allocated()/1024**3)
        e121.base.save_json(report_path, report)
        original_forward = quant.dit.forward
        original_step = quant.step
        original_loader = quant.load_models_to_device
        initial, final = {}, {}

        def selected_forward(_self, *args, **kwargs):
            nonlocal router
            step = len(report['steps'])
            assert step < steps
            at_end = step in ends
            start = time.monotonic()
            attention = None
            if at_end:
                assert router is None
                output = alternate.dit(*args, **kwargs)
                precision = 'nvfp4_w_int8_a_g128' if arm == 'ends_nvfp4_int8_g128' else 'bf16'
            elif arm == 'ends_nvfp4_int8_g128':
                output = original_forward(*args, **kwargs)
                precision = 'svdquant'
            else:
                if router is None:
                    router = e121.Router(quant.dit, 'sage3_fixed')
                contract = e121.base.attention_contract(kwargs)
                with router.forward_context(**contract, diagnostics=False) as routed:
                    output = original_forward(*args, **kwargs)
                attention = {key: routed.summary[key] for key in
                             ('main_helper_calls', 'refiner_helper_calls', 'fp4_calls')}
                assert attention == {'main_helper_calls': 50, 'refiner_helper_calls': 2,
                                     'fp4_calls': 50}
                if step == steps - window - 1:
                    router.close()
                    router = None
                precision = 'svdquant_sage3'
            video, audio = e121.base.split_output(output)
            assert torch.isfinite(video).all().item() and torch.isfinite(audio).all().item()
            row = {'step': step, 'precision': precision, 'dit_seconds': time.monotonic() - start,
                   'video_output_energy': float(video.float().square().mean()),
                   'audio_output_energy': float(audio.float().square().mean())}
            if attention is not None:
                row['attention_calls'] = attention
            report['steps'].append(row)
            e121.base.save_json(report_path, report)
            print(f'{case}/{arm} step {step+1}/{steps}: {precision} {row["dit_seconds"]:.2f}s', flush=True)
            return output

        def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
            name = ('video' if scheduler is quant.scheduler else
                    'audio' if scheduler is quant.scheduler_audio else None)
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
            quant(prompt=info['prompt'], seed=info['seed'], height=cfg['height'], width=cfg['width'],
                  num_frames=cfg['frames'], num_inference_steps=steps, cfg_scale=cfg['cfg_scale'],
                  flow_shift=cfg['flow_shift'], audio_flow_shift=cfg['audio_flow_shift'],
                  rand_device='cpu', tiled=True)
        except Complete:
            pass
        else:
            raise RuntimeError('did not stop after denoising')
        assert router is None and len(report['steps']) == steps and set(initial) == {'video', 'audio'}
        assert min(h.act.calls for h in hooks) == steps - len(ends)
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
        if router is not None:
            router.close()
        del quant, alternate
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.phase == 'denoise':
        denoise(args.case, args.arm)
    else:
        decode(args.case, args.arm)
