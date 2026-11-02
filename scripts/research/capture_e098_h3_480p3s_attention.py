#!/usr/bin/env python3
"""Capture real post-norm/RoPE QKV at H3 block 25 for one 480p/3s call."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DS = Path('/home/wjq/workspace/DiffSynth-Studio')
for key, value in {
    'DIFFSYNTH_SKIP_DOWNLOAD': 'True',
    'DIFFSYNTH_ATTENTION_IMPLEMENTATION': 'torch',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'DIFFSYNTH_ROOT': str(DS),
    'MINIMAX_H3_MODEL_ROOT': str(DS / 'models/MiniMax/MiniMax-H3'),
    'MINIMAX_H3_DIT_PATH': str(DS / 'models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors'),
    'SVDQUANT_DATA_ROOT': '/data1/models/svdquant-wjq',
}.items():
    os.environ[key] = value
sys.path[:0] = [str(ROOT / 'scripts'), str(DS), str(ROOT / 'third_party/deepcompressor')]

import torch
from minimax_h3_svdquant_common import load_h3_pipeline, read_prompt
from diffsynth.models import minimax_h3_dit_comfy as comfy


class CapturedAttention(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt-id', type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    row = read_prompt(args.prompt_id, Path('/home/wjq/workspace/178866172854036'))
    started = time.monotonic()
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    pipe.dit.eval()
    active = [False]
    captured = {}

    def enter(_module, _pos, _kw):
        active[0] = True

    hook = pipe.dit.blocks[25].attn.register_forward_pre_hook(enter, with_kwargs=True)
    original = comfy._sdpa_varlen_attention

    def intercept(q, k, v, cu_seqlens, softmax_scale):
        if not active[0]:
            return original(q, k, v, cu_seqlens, softmax_scale)
        bounds = cu_seqlens.tolist()
        if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != torch.bfloat16:
            raise RuntimeError('Unexpected H3 attention dtype')
        if q.shape != k.shape or q.shape != v.shape or q.shape[1:] != (56, 128):
            raise RuntimeError(f'Unexpected H3 attention shape: {q.shape}')
        captured.update(q=q.detach().cpu(), k=k.detach().cpu(), v=v.detach().cpu(),
                        cu_seqlens=cu_seqlens.detach().cpu(), softmax_scale=float(softmax_scale))
        raise CapturedAttention()

    comfy._sdpa_varlen_attention = intercept
    try:
        with torch.inference_mode():
            pipe(prompt=row['prompt'], seed=int(row['seed']), height=480, width=832,
                 num_frames=73, num_inference_steps=1, cfg_scale=1.0,
                 rand_device='cpu', tiled=True)
    except CapturedAttention:
        pass
    finally:
        comfy._sdpa_varlen_attention = original
        hook.remove()
    if set(captured) != {'q', 'k', 'v', 'cu_seqlens', 'softmax_scale'}:
        raise RuntimeError('H3 block-25 attention was not captured')
    bounds = captured['cu_seqlens'].tolist()
    if len(bounds) != 3 or bounds[0] != 0 or bounds[-1] != captured['q'].shape[0]:
        raise RuntimeError(f'Unexpected H3 attention boundaries: {bounds}')
    payload = dict(format='h3-480p3s-real-attention-v1', block=25,
                   prompt_id=args.prompt_id, seed=int(row['seed']),
                   height=480, width=832, frames=73, fps=24,
                   video_tokens=22 * (480 // 32) * (832 // 32),
                   **captured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(dict(output=str(args.output), shape=list(captured['q'].shape),
                          bounds=bounds, video_tokens=payload['video_tokens'],
                          seconds=time.monotonic() - started), indent=2), flush=True)


if __name__ == '__main__':
    main()
