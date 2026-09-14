#!/usr/bin/env python3
"""Causal trajectory-sensitivity audit for rCM-Wan W4A4.

This is an audit, not a new quantizer.  It injects the *existing* W4A4+
low-rank output error of one Linear at a time into an otherwise BF16 model.
All metrics are accumulated online; no activation, Linear output, or latent is
written by this program.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import WanPipeline

from infer_rcm_wan_4step import load_quantized_transformer
from linear_quant_context import UnsupportedExternalTransform, prepare_isolated_linear_input


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/data/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers')
CKPT = Path('/data/models/svdquant-wjq/ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16')
CACHE = Path('/data/models/svdquant-wjq/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches')
OUT = ROOT / 'results/reports/rcm_trajectory_sensitivity_audit'
EPS = 1e-30


def as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f'Expected Tensor or tuple/list starting with Tensor, got {type(value)}')


def replace_first(value: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return tensor
    if isinstance(value, tuple):
        return (tensor, *value[1:])
    if isinstance(value, list):
        return [tensor, *value[1:]]
    raise TypeError(f'Cannot replace first value of {type(value)}')


def sums(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, int]:
    err = got.float() - ref.float()
    return float(err.square().sum()), float(ref.float().square().sum()), ref.numel()


def metric(err2: float, ref2: float, numel: int) -> dict[str, float]:
    return {'nmse': err2 / max(ref2, EPS), 'mse': err2 / max(numel, 1), 'err2': err2, 'ref2': ref2, 'numel': numel}


def combine(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    err2 = sum(float(row[f'{prefix}_err2']) for row in rows)
    ref2 = sum(float(row[f'{prefix}_ref2']) for row in rows)
    numel = sum(int(row[f'{prefix}_numel']) for row in rows)
    return metric(err2, ref2, numel)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def layer_type(name: str) -> str:
    if '.attn1.' in name:
        family = 'self'
    elif '.attn2.' in name:
        family = 'cross'
    elif '.ffn.' in name:
        family = 'ffn'
    else:
        return 'other'
    if '.to_out.' in name or name.endswith('.to_out'):
        return f'{family}-o'
    suffix = name.rsplit('.', 1)[-1]
    if suffix in {'to_q', 'to_k', 'to_v'}: return f'{family}-{suffix[3:]}'
    if suffix in {'to_out', 'out_proj'}: return f'{family}-o'
    if '.ffn.net.0.proj' in name: return 'ffn-up'
    if '.ffn.net.2' in name: return 'ffn-down'
    return f'{family}-{suffix}'


def block_index(name: str) -> int:
    parts = name.split('.')
    return int(parts[1]) if len(parts) > 2 and parts[0] == 'blocks' else -1


def endpoint_name(name: str) -> str:
    if '.attn1.' in name: return name.split('.attn1.')[0] + '.attn1'
    if '.attn2.' in name: return name.split('.attn2.')[0] + '.attn2'
    if '.ffn.' in name: return name.split('.ffn.')[0] + '.ffn'
    raise RuntimeError(f'No block submodule endpoint for {name}')


def clone_tree(value: Any) -> Any:
    if torch.is_tensor(value): return value.detach()
    if isinstance(value, tuple): return tuple(clone_tree(x) for x in value)
    if isinstance(value, list): return [clone_tree(x) for x in value]
    if isinstance(value, dict): return {k: clone_tree(v) for k, v in value.items()}
    return value


def cpu_tree(value: Any) -> Any:
    """Keep phase-1 block invocation snapshots out of GPU memory."""
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, tuple): return tuple(cpu_tree(x) for x in value)
    if isinstance(value, list): return [cpu_tree(x) for x in value]
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    return value


def device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value): return value.to(device=device)
    if isinstance(value, tuple): return tuple(device_tree(x, device) for x in value)
    if isinstance(value, list): return [device_tree(x, device) for x in value]
    if isinstance(value, dict): return {k: device_tree(v, device) for k, v in value.items()}
    return value


@dataclass
class CacheItem:
    prompt_id: str
    step: int
    path: Path
    payload: dict[str, Any]

    @property
    def hidden(self) -> torch.Tensor: return self.payload['input_args'][0]
    @property
    def velocity(self) -> torch.Tensor: return as_tensor(self.payload['outputs'])


def load_cache(cache_dir: Path, seed: int, num_prompts: int, discovery: int) -> tuple[dict[str, list[CacheItem]], list[str], list[str]]:
    by_prompt: dict[str, list[CacheItem]] = defaultdict(list)
    for path in sorted(cache_dir.glob('*.pt')):
        obj = torch.load(path, map_location='cpu', weights_only=False)
        label = str(obj.get('filename', path.stem)).split('-')[0]
        by_prompt[label].append(CacheItem(label, int(obj.get('step', -1)), path, obj))
    valid = {name: sorted(items, key=lambda x: x.step) for name, items in by_prompt.items() if len(items) == 4 and [x.step for x in sorted(items, key=lambda x: x.step)] == [0, 1, 2, 3]}
    if len(valid) < num_prompts:
        raise RuntimeError(f'Only {len(valid)} complete 4-step prompts in {cache_dir}; need {num_prompts}')
    choices = sorted(valid)
    random.Random(seed).shuffle(choices)
    selected = choices[:num_prompts]
    return {name: valid[name] for name in selected}, selected[:discovery], selected[discovery:]


def quantized_lowrank_linears(qtransformer: torch.nn.Module, bftransformer: torch.nn.Module) -> list[str]:
    qmods, bfmods = dict(qtransformer.named_modules()), dict(bftransformer.named_modules())
    names: list[str] = []
    for name, module in qmods.items():
        if not isinstance(module, torch.nn.Linear) or not isinstance(bfmods.get(name), torch.nn.Linear):
            continue
        hooks = list(module._forward_pre_hooks.values()) + list(module._forward_hooks.values())
        has_branch = any('LowRankBranch' in type(getattr(hook, 'branch', None)).__name__ for hook in hooks)
        has_quant = any(getattr(hook, 'processor', None) is not None for hook in hooks)
        if has_branch and has_quant:
            names.append(name)
    names.sort()
    if len(names) != 300:
        raise RuntimeError(f'Expected exactly 300 W4A4+LowRankBranch Linear modules, found {len(names)}')
    if any(block_index(name) < 0 for name in names):
        raise RuntimeError('A target Linear is outside transformer.blocks; block intervention is undefined')
    return names


@contextmanager
def output_override(module: torch.nn.Module, fn: Any) -> Iterator[None]:
    handle = module.register_forward_hook(lambda _m, args, out: fn(args, out))
    try: yield
    finally: handle.remove()


@contextmanager
def output_capture(module: torch.nn.Module, sink: dict[str, torch.Tensor], key: str = 'value') -> Iterator[None]:
    def capture(_m: torch.nn.Module, _args: tuple[Any, ...], out: Any) -> None:
        sink[key] = as_tensor(out).detach()
    handle = module.register_forward_hook(capture)
    try: yield
    finally: handle.remove()


def q_output(qtransformer: torch.nn.Module, name: str, raw_input: torch.Tensor) -> torch.Tensor:
    qinput, _ = prepare_isolated_linear_input(qtransformer, name, raw_input)
    return qtransformer.get_submodule(name)(qinput)


def normalized_output(ref: torch.Tensor, quant: torch.Tensor, target_nmse: float) -> tuple[torch.Tensor, float]:
    error = quant.float() - ref.float()
    local = float(error.square().sum() / ref.float().square().sum().clamp_min(EPS))
    if local <= EPS: return ref, 0.0
    scale = math.sqrt(target_nmse / local)
    # The injected value is BF16, so one analytical scale can miss the target
    # after rounding.  Calibrate in the *actual injected coordinate*.
    candidate = ref
    for _ in range(6):
        candidate = (ref.float() + error * scale).to(ref.dtype)
        observed = float((candidate.float() - ref.float()).square().sum() / ref.float().square().sum().clamp_min(EPS))
        if math.isclose(observed, target_nmse, rel_tol=.002, abs_tol=1e-8):
            break
        if observed <= EPS:
            break
        scale *= math.sqrt(target_nmse / observed)
    return candidate, scale


def rcm_times(device: torch.device) -> torch.Tensor:
    trig = torch.tensor([math.atan(80.0), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    return torch.sin(trig) / (torch.cos(trig) + torch.sin(trig))


def run_transformer(model: torch.nn.Module, item: CacheItem, device: torch.device) -> torch.Tensor:
    kw = item.payload['input_kwargs']
    return as_tensor(model(hidden_states=item.hidden.to(device=device, dtype=torch.bfloat16),
                           timestep=kw['timestep'].to(device=device, dtype=torch.bfloat16),
                           encoder_hidden_states=kw['encoder_hidden_states'].to(device=device, dtype=torch.bfloat16),
                           return_dict=False))


@torch.inference_mode()
def phase1(
    bf: torch.nn.Module, q: torch.nn.Module, caches: dict[str, list[CacheItem]], layers: list[str],
    normalized_nmse: float, out: Path, device: torch.device,
) -> list[dict[str, Any]]:
    """Replay one BF16 block at a time with a single Linear-output override."""
    bfmods = dict(bf.named_modules())
    rows: list[dict[str, Any]] = []
    by_block: dict[int, list[str]] = defaultdict(list)
    for name in layers: by_block[block_index(name)].append(name)
    for prompt_id, items in caches.items():
        for item in items:
            # Cache input/output checks and collect each block's exact BF16 invocation.
            invocations: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
            handles = []
            for index in by_block:
                block = bf.blocks[index]
                def capture_invocation(_m: torch.nn.Module, call_args: tuple[Any, ...], call_kwargs: dict[str, Any], i: int = index) -> None:
                    invocations[i] = (cpu_tree(call_args), cpu_tree(call_kwargs))
                handles.append(block.register_forward_pre_hook(
                    capture_invocation, with_kwargs=True))
            base_velocity = run_transformer(bf, item, device)
            for handle in handles: handle.remove()
            ref_velocity = item.velocity.to(device=device, dtype=base_velocity.dtype)
            e2, r2, n = sums(base_velocity, ref_velocity)
            if e2 / max(r2, EPS) >= 5e-5:
                raise RuntimeError(f'BF16 cache replay mismatch {prompt_id}/step{item.step}: {e2/r2:.3e}')
            for index, names in by_block.items():
                block = bf.blocks[index]
                args, kwargs = device_tree(invocations[index][0], device), device_tree(invocations[index][1], device)
                # A baseline block call captures only the relevant endpoint for this block.
                endpoint_refs: dict[str, torch.Tensor] = {}
                endpoint_handles = []
                for endpoint in sorted({endpoint_name(name) for name in names}):
                    endpoint_handles.append(bfmods[endpoint].register_forward_hook(
                        lambda _m, _a, value, ep=endpoint: endpoint_refs.__setitem__(ep, as_tensor(value).detach())))
                base_block = as_tensor(block(*args, **kwargs)).detach()
                for handle in endpoint_handles: handle.remove()
                for name in names:
                    target, endpoint = bfmods[name], endpoint_name(name)
                    values: dict[str, Any] = {}
                    def injected(call_args: tuple[Any, ...], original: Any) -> Any:
                        ref = as_tensor(original)
                        quant = q_output(q, name, call_args[0])
                        normal, scale = normalized_output(ref, quant, normalized_nmse)
                        values.update(ref=ref.detach(), quant=quant.detach(), normal=normal.detach(), scale=scale)
                        return original
                    # First run only invokes the child hook and gathers its true direction.
                    with output_override(target, injected):
                        _ = block(*args, **kwargs)
                    ref, quant, normal = values['ref'], values['quant'], values['normal']
                    for mode, replacement in (('actual', quant), ('normalized', normal)):
                        affected: dict[str, torch.Tensor] = {}
                        with output_capture(bfmods[endpoint], affected):
                            with output_override(target, lambda _a, original, repl=replacement: replace_first(original, repl)):
                                changed_block = as_tensor(block(*args, **kwargs))
                        le2, lr2, ln = sums(replacement, ref)
                        se2, sr2, sn = sums(affected['value'], endpoint_refs[endpoint])
                        be2, br2, bn = sums(changed_block, base_block)
                        row = {
                            'prompt_id': prompt_id, 'step': item.step, 'layer': name, 'layer_type': layer_type(name),
                            'block': index, 'endpoint': endpoint, 'mode': mode, 'scale': values['scale'],
                        }
                        for prefix, value in (('local', metric(le2, lr2, ln)), ('submodule', metric(se2, sr2, sn)), ('block', metric(be2, br2, bn))):
                            row.update({f'{prefix}_{k}': v for k, v in value.items()})
                        rows.append(row)
                del base_block, endpoint_refs
            del invocations, base_velocity
            torch.cuda.empty_cache()
            print(f'phase1 {prompt_id} step={item.step}: {len(rows)} rows', flush=True)
    write_csv(out / 'per_layer_block_impact.csv', rows)
    (out / 'per_layer_block_impact.json').write_text(json.dumps(rows, indent=2, default=json_default))
    return rows


def aggregate_layer(rows: list[dict[str, Any]], prompt_ids: set[str], mode: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['prompt_id'] in prompt_ids and row['mode'] == mode: grouped[row['layer']].append(row)
    result = {}
    for name, group in grouped.items():
        local, block = combine(group, 'local'), combine(group, 'block')
        result[name] = {'layer': name, 'layer_type': group[0]['layer_type'], 'block': group[0]['block'],
                        'local_nmse': local['nmse'], 'block_nmse': block['nmse'],
                        'block_amplification': block['nmse'] / max(local['nmse'], EPS)}
    return result


def select_layers(rows: list[dict[str, Any]], discovery: list[str], count: int) -> dict[str, Any]:
    summary = aggregate_layer(rows, set(discovery), 'normalized')
    if len(summary) != 300: raise RuntimeError(f'Expected 300 discovery summaries, got {len(summary)}')
    sensitive = sorted(summary.values(), key=lambda x: (-x['block_amplification'], x['layer']))[:count // 2]
    used = {x['layer'] for x in sensitive}
    controls = []
    for source in sensitive:
        candidates = [x for x in summary.values() if x['layer'] not in used and x['layer_type'] == source['layer_type']]
        lower = [x for x in candidates if x['block_amplification'] < source['block_amplification']]
        candidates = lower or candidates
        if not candidates: raise RuntimeError(f'No same-type control for {source["layer"]}')
        chosen = min(candidates, key=lambda x: (abs(math.log(max(x['local_nmse'], EPS)) - math.log(max(source['local_nmse'], EPS))), x['block_amplification'], x['layer']))
        controls.append(chosen); used.add(chosen['layer'])
    records = [{'role': 'sensitive', **x} for x in sensitive] + [{'role': 'control', **x} for x in controls]
    return {'method': 'top normalized block amplification on discovery prompts with same-type local-NMSE matched lower-amplification controls',
            'discovery_prompts': discovery, 'records': records, 'layers': [x['layer'] for x in records]}


def recovered_noises(items: list[CacheItem], times: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
    result = []
    for step in range(3):
        z, velocity, nxt = items[step].hidden.to(device=device, dtype=torch.float64), items[step].velocity.to(device=device, dtype=torch.float64), times[step + 1]
        znext = items[step + 1].hidden.to(device=device, dtype=torch.float64)
        noise = (znext - (1 - nxt) * (z - times[step] * velocity)) / nxt
        rebuilt = (1 - nxt) * (z - times[step] * velocity) + nxt * noise
        e2, r2, _ = sums(rebuilt, znext)
        if e2 / max(r2, EPS) >= 5e-5: raise RuntimeError(f'Noise reconstruction failure step {step}: {e2/r2:.3e}')
        result.append(noise)
    return result


def future_final(bf: torch.nn.Module, items: list[CacheItem], start: int, pert_velocity: torch.Tensor, noises: list[torch.Tensor], times: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    z = items[start].hidden.to(device=device, dtype=torch.float64)
    nxt = times[start + 1]
    noise = noises[start] if start < 3 else torch.zeros_like(z)
    z = (1 - nxt) * (z - times[start] * pert_velocity.to(torch.float64)) + nxt * noise
    next_latent = z.detach()
    for step in range(start + 1, 4):
        item = items[step]
        velocity = run_transformer(bf, item_from_hidden(item, z), device).to(torch.float64)
        nxt = times[step + 1]
        noise = noises[step] if step < 3 else torch.zeros_like(z)
        z = (1 - nxt) * (z - times[step] * velocity) + nxt * noise
    return next_latent, z


def item_from_hidden(item: CacheItem, hidden: torch.Tensor) -> CacheItem:
    # Shallow payload copy: only hidden changes; embeddings/timestep remain cache-derived.
    payload = dict(item.payload); payload['input_args'] = [hidden.to(torch.bfloat16)]
    return CacheItem(item.prompt_id, item.step, item.path, payload)


@torch.inference_mode()
def phase2(bf: torch.nn.Module, q: torch.nn.Module, caches: dict[str, list[CacheItem]], selected: dict[str, Any], normalized_nmse: float, out: Path, device: torch.device, resume: bool = False) -> list[dict[str, Any]]:
    bfmods = dict(bf.named_modules()); roles = {r['layer']: r['role'] for r in selected['records']}
    partial = out / 'trajectory_impact.partial.json'
    rows: list[dict[str, Any]] = json.loads(partial.read_text()) if resume and partial.exists() else []
    done = {(r['prompt_id'], int(r['step']), r['layer'], r['mode']) for r in rows}
    times = rcm_times(device)
    for prompt_id, items in caches.items():
        noises = recovered_noises(items, times, device)
        reference_final = (items[3].hidden.to(device=device, dtype=torch.float64) - times[3] * items[3].velocity.to(device=device, dtype=torch.float64)).detach()
        for item in items:
            reference_velocity = item.velocity.to(device=device, dtype=torch.bfloat16)
            reference_next = items[item.step + 1].hidden.to(device=device, dtype=torch.float64) if item.step < 3 else reference_final
            for name in selected['layers']:
                target = bfmods[name]
                for mode in ('actual', 'normalized'):
                    if (prompt_id, item.step, name, mode) in done:
                        continue
                    values: dict[str, Any] = {}
                    def override(call_args: tuple[Any, ...], original: Any) -> Any:
                        ref = as_tensor(original); quant = q_output(q, name, call_args[0])
                        normal, scale = normalized_output(ref, quant, normalized_nmse)
                        replacement = quant if mode == 'actual' else normal
                        values.update(ref=ref.detach(), replacement=replacement.detach(), scale=scale)
                        return replace_first(original, replacement)
                    with output_override(target, override):
                        pert_velocity = run_transformer(bf, item, device)
                    next_latent, final_latent = future_final(bf, items, item.step, pert_velocity, noises, times, device)
                    le2, lr2, ln = sums(values['replacement'], values['ref'])
                    de2, dr2, dn = sums(pert_velocity, reference_velocity)
                    ne2, nr2, nn = sums(next_latent, reference_next)
                    fe2, fr2, fn = sums(final_latent, reference_final)
                    row = {'prompt_id': prompt_id, 'step': item.step, 'layer': name, 'role': roles[name],
                           'layer_type': layer_type(name), 'block': block_index(name), 'mode': mode, 'scale': values['scale']}
                    for prefix, value in (('local', metric(le2, lr2, ln)), ('denoiser', metric(de2, dr2, dn)), ('next_latent', metric(ne2, nr2, nn)), ('final_latent', metric(fe2, fr2, fn))):
                        row.update({f'{prefix}_{k}': v for k, v in value.items()})
                    rows.append(row)
                    done.add((prompt_id, item.step, name, mode))
                print(f'phase2 {prompt_id} step={item.step} layer={name}: {len(rows)} rows', flush=True)
            # Long all-layer runs retain no tensors, but preserve completed
            # scalar rows at every timestep in case an external job stops.
            write_csv(out / 'trajectory_impact.partial.csv', rows)
            (out / 'trajectory_impact.partial.json').write_text(json.dumps(rows, indent=2, default=json_default))
        del noises; torch.cuda.empty_cache()
    write_csv(out / 'trajectory_impact.csv', rows)
    (out / 'trajectory_impact.json').write_text(json.dumps(rows, indent=2, default=json_default))
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort'); ranks = np.empty(len(values), dtype=float); ranks[order] = np.arange(len(values), dtype=float)
    unique, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1: ranks[inv == i] = ranks[inv == i].mean()
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2: return float('nan')
    x, y = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float('nan')


def analysis(block_rows: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]], selected: dict[str, Any], discovery: list[str], confirmation: list[str], out: Path) -> dict[str, Any]:
    norm = [r for r in trajectory_rows if r['mode'] == 'normalized']
    block_summary = aggregate_layer(block_rows, set(discovery), 'normalized')
    def summary_for(prompts: set[str]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in norm:
            if row['prompt_id'] in prompts: grouped[row['layer']].append(row)
        answer = {}
        for name, group in grouped.items():
            final = combine(group, 'final_latent'); answer[name] = {'final_nmse': final['nmse'], **block_summary.get(name, {})}
        return answer
    confirm_summary = summary_for(set(confirmation))
    names = sorted(confirm_summary)
    local = [confirm_summary[n]['local_nmse'] for n in names]; block = [confirm_summary[n]['block_nmse'] for n in names]; final = [confirm_summary[n]['final_nmse'] for n in names]
    role = {r['layer']: r['role'] for r in selected['records']}
    pairs = []
    sensitive = [r['layer'] for r in selected['records'] if r['role'] == 'sensitive']
    controls = [r['layer'] for r in selected['records'] if r['role'] == 'control']
    for s, c in zip(sensitive, controls, strict=True): pairs.append({'sensitive': s, 'control': c, 'ratio': confirm_summary[s]['final_nmse']/max(confirm_summary[c]['final_nmse'], EPS)})
    timestep_stats = {}
    for step in range(4):
        vals = [r['final_latent_nmse'] for r in norm if r['prompt_id'] in confirmation and r['step'] == step]
        timestep_stats[str(step)] = {'p90_p10': float(np.percentile(vals, 90)/max(np.percentile(vals, 10), EPS)), 'count': len(vals)}
    prompt_stats = {}
    for p in confirmation:
        vals = [r['final_latent_nmse'] for r in norm if r['prompt_id'] == p]
        prompt_stats[p] = {'p90_p10': float(np.percentile(vals, 90)/max(np.percentile(vals, 10), EPS)), 'count': len(vals)}
    pair_median = float(np.median([p['ratio'] for p in pairs])) if pairs else None
    result = {'local_vs_final_spearman': spearman(local, final), 'block_vs_final_spearman': spearman(block, final),
              'block_minus_local_spearman': spearman(block, final)-spearman(local, final),
              'confirmation_final_p90_p10': float(np.percentile(final,90)/max(np.percentile(final,10),EPS)),
              'matched_pair_ratios': pairs, 'matched_pair_median_ratio': pair_median,
              'per_timestep': timestep_stats, 'per_confirmation_prompt': prompt_stats,
              'selected_roles': role}
    result['decision'] = {'enter_trajectory_aware_lowrank': bool(
        result['confirmation_final_p90_p10'] >= 3 and pair_median is not None and pair_median >= 2 and
        result['local_vs_final_spearman'] <= .6 and result['block_minus_local_spearman'] >= .15 and
        sum(x['p90_p10'] >= 3 for x in timestep_stats.values()) >= 3 and sum(x['p90_p10'] >= 3 for x in prompt_stats.values()) >= 3)}
    (out / 'correlations.json').write_text(json.dumps(result, indent=2, default=json_default))
    # Compact plots, all derived from scalar CSV data.
    plt.figure(figsize=(6,5)); plt.scatter(local, block, c=['tab:red' if role[n]=='sensitive' else 'tab:blue' for n in names]); plt.xscale('log'); plt.yscale('log'); plt.xlabel('local NMSE'); plt.ylabel('block NMSE'); plt.tight_layout(); plt.savefig(out/'local_vs_block_scatter.png',dpi=180); plt.close()
    plt.figure(figsize=(6,5)); plt.scatter(local, final, c=['tab:red' if role[n]=='sensitive' else 'tab:blue' for n in names]); plt.xscale('log'); plt.yscale('log'); plt.xlabel('local NMSE'); plt.ylabel('final latent NMSE'); plt.tight_layout(); plt.savefig(out/'local_vs_final_scatter.png',dpi=180); plt.close()
    matrix = np.array([[np.mean([r['final_latent_nmse'] for r in norm if r['layer']==n and r['step']==s]) for s in range(4)] for n in names])
    plt.figure(figsize=(8, max(4, len(names)*.22))); plt.imshow(np.log10(np.maximum(matrix, EPS)), aspect='auto', cmap='magma'); plt.colorbar(label='log10 final latent NMSE'); plt.xticks(range(4), range(4)); plt.yticks(range(len(names)), names, fontsize=6); plt.xlabel('injection timestep'); plt.tight_layout(); plt.savefig(out/'layer_timestep_final_impact_heatmap.png',dpi=180); plt.close()
    depths = sorted({r['block'] for r in norm}); vals = [np.mean([r['final_latent_nmse'] for r in norm if r['block']==d]) for d in depths]
    plt.figure(figsize=(7,4)); plt.plot(depths, vals, 'o-'); plt.yscale('log'); plt.xlabel('block depth'); plt.ylabel('mean final latent NMSE'); plt.tight_layout(); plt.savefig(out/'block_depth_final_impact.png',dpi=180); plt.close()
    if pairs:
        plt.figure(figsize=(10,4)); plt.bar(np.arange(len(pairs))-.2,[p['ratio'] for p in pairs],.4); plt.axhline(1,color='black',lw=1); plt.xticks(range(len(pairs)),[f'{i+1}' for i in range(len(pairs))]); plt.ylabel('sensitive / control final NMSE'); plt.tight_layout(); plt.savefig(out/'matched_pairs.png',dpi=180); plt.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-prompts', type=int, default=8); parser.add_argument('--discovery-prompts', type=int, default=4)
    parser.add_argument('--exact-layers', type=int, default=24); parser.add_argument('--normalized-local-nmse', type=float, default=.01)
    parser.add_argument('--checkpoint', type=Path, default=CKPT); parser.add_argument('--cache-dir', type=Path, default=CACHE); parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--verify-only', action='store_true'); parser.add_argument('--phase', choices=('all','block','trajectory'), default='all')
    parser.add_argument('--resume', action='store_true', help='Resume phase-2 scalar rows from trajectory_impact.partial.json.')
    args = parser.parse_args()
    if args.num_prompts < 2 or not 0 < args.discovery_prompts < args.num_prompts or args.exact_layers < 1:
        raise ValueError('Use at least two prompts, a non-empty discovery/confirmation split, and a positive exact-layer count.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    caches, discovery, confirmation = load_cache(args.cache_dir, 20260901, args.num_prompts, args.discovery_prompts)
    config = vars(args) | {'random_seed': 20260901, 'selected_prompt_ids': list(caches), 'discovery_prompt_ids': discovery, 'confirmation_prompt_ids': confirmation, 'all_tokens': True, 'no_intermediate_tensors_saved': True}
    (args.output_dir/'config.json').write_text(json.dumps(config, indent=2, default=json_default))
    device = torch.device('cuda'); pipe_bf = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device); pipe_bf.text_encoder.to('cpu')
    pipe_q = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device); load_quantized_transformer(pipe_q, args.checkpoint, MODEL); pipe_q.text_encoder.to('cpu')
    layers = quantized_lowrank_linears(pipe_q.transformer, pipe_bf.transformer)
    (args.output_dir/'module_audit.json').write_text(json.dumps({'target_count':len(layers),'targets':layers},indent=2))
    if args.verify_only:
        item = next(iter(caches.values()))[0]; got = run_transformer(pipe_bf.transformer, item, device); e2,r2,_=sums(got,item.velocity.to(device));
        if e2/max(r2,EPS)>=5e-5: raise RuntimeError('BF16 cache verification failed')
        print(json.dumps({'status':'verified','targets':len(layers),'bf16_cache_nmse':e2/max(r2,EPS)},indent=2)); return
    if args.phase in {'all','block'}:
        block_rows = phase1(pipe_bf.transformer, pipe_q.transformer, caches, layers, args.normalized_local_nmse, args.output_dir, device)
        selected = select_layers(block_rows, discovery, args.exact_layers)
        (args.output_dir/'selected_layers.json').write_text(json.dumps(selected,indent=2,default=json_default))
    else:
        block_rows=json.loads((args.output_dir/'per_layer_block_impact.json').read_text()); selected=json.loads((args.output_dir/'selected_layers.json').read_text())
    if args.phase in {'all','trajectory'}:
        trajectory_rows=phase2(pipe_bf.transformer,pipe_q.transformer,caches,selected,args.normalized_local_nmse,args.output_dir,device,args.resume)
        result=analysis(block_rows,trajectory_rows,selected,discovery,confirmation,args.output_dir)
        print(json.dumps(result['decision'],indent=2))
    pipe_bf.to('cpu'); pipe_q.to('cpu'); del pipe_bf,pipe_q; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__': main()
