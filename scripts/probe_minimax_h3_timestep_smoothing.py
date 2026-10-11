#!/usr/bin/env python3
"""Targeted timestep-aware H3 SVDQuant smoothing probe.

This is deliberately smaller than a full 50-block PTQ run: it preserves the
production candidate objective (one W4A4 linear, evaluated at its complete
parent DiT-block endpoint) for selected representative layers.  Raw calls
live outside the repository; one block's replay state lives in host RAM.
"""
from __future__ import annotations

import argparse, csv, json, gc
from pathlib import Path

import torch
import torch.nn as nn

from minimax_h3_svdquant_common import (
    nvfp4_qdq, raw_call_to_block0, run_block, temporary_quantized_linear,
    tree_device,
)
from ptq_minimax_h3_svdquant_standard import _capture_inputs, _smooth_candidates
from minimax_h3_svdquant_common import load_h3_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    return p.parse_args()


def linear_at(dit, name: str):
    obj = dit
    for part in name.split('.'):
        if part == 'blocks':
            continue
        if part.isdigit():
            obj = obj.blocks[int(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, nn.Linear):
        raise TypeError(f'{name} is not Linear')
    return obj


def block_id(name: str) -> int:
    return int(name.split('.')[1])


def load_raw(cache: Path, prompt_ids: list[int]):
    rows = []
    for pid in prompt_ids:
        manifest = json.loads((cache / f'p{pid}' / 'manifest.json').read_text())
        if len(manifest['samples']) != 20:
            raise RuntimeError(f'p{pid}: expected 20 cached calls')
        for r in sorted(manifest['samples'], key=lambda x: int(x['step'])):
            rows.append(({'prompt_id': pid, 'step': int(r['step'])}, cache / f'p{pid}' / r['file']))
    return rows


def stage_block_in_memory(dit, target_block: int, raw_rows):
    """Replay one BF16 predecessor prefix and retain the compact study set in RAM.

    Persisting a ``kwargs`` tree for every call was ~0.5 GiB/call and made the
    former 16-prompt probe I/O-bound.  The fast four-prompt study fits easily
    in host RAM, so a block's two target linears share one in-memory stage.
    """
    samples = []
    metas = []
    for meta, path in raw_rows:
        raw = torch.load(path, map_location='cpu', weights_only=False)
        samples.append(raw_call_to_block0(dit, raw)); metas.append(meta)
    for b in range(target_block):
        print(f'block {target_block}: replay predecessor block {b}/{target_block - 1}', flush=True)
        samples = [(out, kw) for out, (_, kw) in zip(run_block(dit.blocks[b], samples), samples, strict=True)]
    refs = run_block(dit.blocks[target_block], samples)
    return [{'hidden': hidden, 'kwargs': kwargs, 'ref': ref, 'meta': meta}
            for (hidden, kwargs), ref, meta in zip(samples, refs, metas, strict=True)]


def load_stage(staged_rows, prompt_ids: set[int], timestep: int):
    selected = [r for r in staged_rows if r['meta']['prompt_id'] in prompt_ids and r['meta']['step'] == timestep]
    if not selected:
        raise RuntimeError(f'no rows for timestep={timestep}')
    return selected


@torch.inference_mode()
def endpoint_nmse(block, linear, rows, qweight, smooth):
    num = den = 0.0
    with temporary_quantized_linear(linear, qweight, smooth):
        for row in rows:
            out = block(row['hidden'].to('cuda'), **tree_device(row['kwargs'], 'cuda'))
            ref = row['ref'].to('cuda')
            num += float((out.float() - ref.float()).square().sum(dtype=torch.float64))
            den += float(ref.float().square().sum(dtype=torch.float64))
    return num / max(den, 1e-30)


def choose(layer, block, linear, fit_rows, val_rows, num_grids: int):
    pairs = [(r['hidden'], r['kwargs']) for r in fit_rows]
    inputs = _capture_inputs(block, linear, pairs)
    best = None
    w = linear.weight.detach()
    for alpha, smooth in _smooth_candidates(linear, inputs, num_grids):
        err = endpoint_nmse(block, linear, fit_rows, nvfp4_qdq(w * smooth), smooth)
        if best is None or err < best[0]:
            best = (err, alpha, smooth.detach().cpu())
        torch.cuda.empty_cache()
    fit_err, alpha, smooth = best
    val_err = endpoint_nmse(block, linear, val_rows, nvfp4_qdq(w * smooth.to(w)), smooth.to(w))
    return fit_err, alpha, smooth, val_err


def main():
    a = parse_args(); cfg = json.loads(a.config.read_text()); a.output_dir.mkdir(parents=True, exist_ok=True)
    grids = int(cfg.get('smoothing_grid_size', 10))
    if grids != 10:
        raise ValueError('this study fixes the requested 10-candidate smoothing grid')
    all_ids = list(map(int, cfg['prompt_ids'])); fit_ids = set(map(int, cfg['fit_prompt_ids'])); val_ids = set(map(int, cfg['validation_prompt_ids']))
    if fit_ids & val_ids or fit_ids | val_ids != set(all_ids):
        raise ValueError('fit/validation prompts must be a disjoint complete split')
    raw_rows = load_raw(a.cache_dir, all_ids)
    pipe = load_h3_pipeline(full=False, reserve_gib=24.0); pipe.load_models_to_device(['dit']); dit = pipe.dit
    records = []
    completed_layers: set[str] = set()
    existing_csv = a.output_dir / 'timestep_smoothing_probe.csv'
    if existing_csv.exists():
        records = list(csv.DictReader(existing_csv.open()))
        per_layer: dict[str, set[int]] = {}
        for row in records:
            per_layer.setdefault(row['layer'], set()).add(int(row['timestep']))
        completed_layers = {name for name, steps in per_layer.items() if steps == set(range(20))}
        if completed_layers:
            print(f'resuming: preserving completed layers {sorted(completed_layers)}', flush=True)
    def checkpoint_records():
        """Persist completed layers so a later layer cannot discard them."""
        if not records:
            return
        tmp = a.output_dir / 'timestep_smoothing_probe.csv.tmp'
        with tmp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        tmp.replace(a.output_dir / 'timestep_smoothing_probe.csv')
        tmp_json = a.output_dir / 'timestep_smoothing_probe.json.tmp'
        tmp_json.write_text(json.dumps({'config': cfg, 'records': records}, indent=2))
        tmp_json.replace(a.output_dir / 'timestep_smoothing_probe.json')
    layers_by_block = {}
    for name in cfg['probe_layers']:
        if name not in completed_layers:
            layers_by_block.setdefault(block_id(name), []).append(name)
    for bid, names in layers_by_block.items():
        staged_rows = stage_block_in_memory(dit, bid, raw_rows)
        for name in names:
            linear = linear_at(dit, name); block = dit.blocks[bid]
            if linear.weight.is_meta: linear.load_from_disk(torch.bfloat16, 'cuda', assign=True)
            else: linear.to('cuda', torch.bfloat16)
        # A global (all timestep) fit is the baseline. Its smooth vector is
        # reused at every validation timestep, isolating the value of timing.
            fit_all = [r for r in staged_rows if r['meta']['prompt_id'] in fit_ids]
            val_all = [r for r in staged_rows if r['meta']['prompt_id'] in val_ids]
            gfit, galpha, gs, _ = choose(name, block, linear, fit_all, val_all, grids)
            for t in range(20):
                fit, val = load_stage(staged_rows, fit_ids, t), load_stage(staged_rows, val_ids, t)
                f_err, alpha, smooth, v_err = choose(name, block, linear, fit, val, grids)
                g_val = endpoint_nmse(block, linear, val, nvfp4_qdq(linear.weight.detach() * gs.to(linear.weight)), gs.to(linear.weight))
                ratio = (smooth.float().clamp_min(1e-12) / gs.float().clamp_min(1e-12)).log()
                records.append({'layer': name, 'block_id': bid, 'timestep': t, 'local_alpha': alpha,
                                'local_fit_nmse': f_err, 'local_validation_nmse': v_err,
                                'global_alpha': galpha, 'global_fit_nmse': gfit, 'global_validation_nmse': g_val,
                                'validation_relative_gain': (g_val - v_err) / max(g_val, 1e-30),
                                'smooth_log_ratio_median_abs': float(ratio.abs().median()),
                                'smooth_log_ratio_p90_abs': float(torch.quantile(ratio.abs(), .9)),
                                'smooth_cosine_to_global': float(torch.nn.functional.cosine_similarity(smooth.float(), gs.float(), dim=0))})
                print(f'{name} t={t:02d}: alpha={alpha:.1f}, val={v_err:.6g}, global={g_val:.6g}', flush=True)
            checkpoint_records()
        # Keep completed target linears resident.  A later block-prefix replay
        # still executes every earlier block; moving an already materialized
        # DiffSynth VRAMLinear back to CPU makes its next lazy reload an
        # inference tensor without a version counter.  Eight selected linears
        # fit comfortably alongside one block's in-memory replay state.
        torch.cuda.empty_cache()
        del staged_rows
        gc.collect()
    checkpoint_records()

if __name__ == '__main__': main()
