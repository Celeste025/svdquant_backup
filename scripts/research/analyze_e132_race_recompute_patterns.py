#!/usr/bin/env python3
"""Teacher-forced H3 output-error maps: native W4A4 versus 10% token recompute."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/research')]
from minimax_h3_svdquant_common import tree_device
from eval_minimax_h3_paired_step_error import split_output
from bench_h3_native_nvfp4 import make_h3_resident
from h3_plain_native_nvfp4 import install_plain_h3, PlainNativeH3Linear
import run_e121_h3_split_generation as e121

CALLS = Path('/data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases/race/bf16_teacher_calls.pt')
EXPORT = Path('/data1/models/svdquant-wjq/research/20261024/E128_plain_export_rehydrated')
OUT = ROOT / 'results/research/E132'
GRID = (37, 18, 32)


def corr(x, y):
    a, b = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def top_ids(x, fraction=.01):
    k = max(1, round(x.size * fraction))
    return np.argpartition(x.ravel(), -k)[-k:]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fraction', type=float, choices=(0.0, .1), required=True)
    args = p.parse_args()
    name = 'w4a4' if args.fraction == 0 else 'recompute10'
    OUT.mkdir(parents=True, exist_ok=True)
    calls = torch.load(CALLS, map_location='cpu', weights_only=False)
    assert len(calls) == 20
    attention = importlib.import_module('diffsynth.core.attention.attention')
    attention.ATTENTION_IMPLEMENTATION = 'torch'
    pipe = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    make_h3_resident(pipe.dit)
    install_plain_h3(pipe.dit, EXPORT, recompute_fraction=args.fraction,
                     recompute_seed=20261025)
    modules = [m for m in pipe.dit.modules() if isinstance(m, PlainNativeH3Linear)]
    assert len(modules) == 200
    token_maps, step_nmse, signed_adjacent = [], [], []
    prev = None
    for step, call in enumerate(calls):
        video, _ = split_output(pipe.dit(*tree_device(call['args'], 'cuda'),
                                          **tree_device(call['kwargs'], 'cuda')))
        ref = call['video'].to('cuda')
        err = (video.float() - ref.float()).reshape(*GRID, 96)
        energy = err.square().sum(-1)
        token_maps.append(energy.cpu().numpy())
        step_nmse.append(float(energy.sum() / ref.float().square().sum()))
        if prev is not None:
            signed_adjacent.append(float((err * prev).sum() /
                                         (err.square().sum() * prev.square().sum()).sqrt()))
        prev = err
        print(f'{name} step {step + 1}/20 nmse={step_nmse[-1]:.6f}', flush=True)
    assert all(getattr(m, 'recompute_calls', 0) == (0 if args.fraction == 0 else 20)
               for m in modules)
    maps = np.stack(token_maps)
    adjacent_magnitude = [corr(np.log(np.maximum(maps[i], 1e-12)),
                               np.log(np.maximum(maps[i + 1], 1e-12))) for i in range(19)]
    adjacent_hot_jaccard = []
    for i in range(19):
        a, b = set(top_ids(maps[i])), set(top_ids(maps[i + 1]))
        adjacent_hot_jaccard.append(len(a & b) / len(a | b))
    np.savez_compressed(OUT / f'{name}_token_maps.npz', token_error_energy=maps)
    result = {'case': 'race_p016', 'arm': name, 'definition': 'same BF16 teacher-forced DiT inputs',
              'fraction': args.fraction, 'grid_thw': GRID, 'step_nmse': step_nmse,
              'adjacent_log_token_energy_corr': adjacent_magnitude,
              'adjacent_signed_error_cosine': signed_adjacent,
              'adjacent_top1pct_token_jaccard': adjacent_hot_jaccard,
              'mean_adjacent_log_token_energy_corr': float(np.mean(adjacent_magnitude)),
              'mean_adjacent_signed_error_cosine': float(np.mean(signed_adjacent)),
              'mean_adjacent_top1pct_token_jaccard': float(np.mean(adjacent_hot_jaccard))}
    (OUT / f'{name}_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in result if k.startswith('mean_')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
