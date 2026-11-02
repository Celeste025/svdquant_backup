#!/usr/bin/env python3
"""Render one-data-element-per-pixel 3000x3000 NVFP4 Q/DQ error maps."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import FP4_LEVELS, KEY, WEIGHT, dequantize, load_input


ROW_START, ROW_STOP = 0, 3000
COL_START, COL_STOP = 4500, 7500
CASES = {
    'p1_activation': (Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p1_s00.pt'), 22400),
    'p11_activation': (Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p11_s00_calib.pt'), 22464),
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=Path('results/research/E093/figures'))
    p.add_argument('--gpu', type=int, default=7)
    return p.parse_args()


def error_crop(x, levels, nvfp4_quantize, sf_layout):
    if x.shape[0] < ROW_STOP or x.shape[1] < COL_STOP:
        raise ValueError('Input smaller than selected 3000x3000 window')
    amax = float(x.abs().amax().float().item())
    inverse = torch.tensor([2688.0 / amax], device=x.device, dtype=torch.float32)
    q, sf = nvfp4_quantize(x, inverse, sfLayout=sf_layout, backend='cuda')
    crop = np.empty((ROW_STOP - ROW_START, COL_STOP - COL_START), dtype=np.float32)
    for start in range(ROW_START, ROW_STOP, 128):
        stop = min(start + 128, ROW_STOP)
        decoded = dequantize(q[start:stop], sf[start:stop],
                             inverse.reciprocal(), levels)
        delta = decoded[:, COL_START:COL_STOP] - x[start:stop, COL_START:COL_STOP].float()
        crop[start-ROW_START:stop-ROW_START] = delta.cpu().numpy()
    del q, sf
    if not np.isfinite(crop).all():
        raise RuntimeError('Nonfinite error crop')
    return crop, amax


def lut(stops, size=4096):
    anchors = np.asarray(stops, dtype=np.float64)
    xp = np.linspace(0, size - 1, len(anchors))
    positions = np.arange(size)
    return np.stack([np.interp(positions, xp, anchors[:, i])
                     for i in range(3)], axis=-1).round().astype(np.uint8)


SIGNED = lut([(34, 70, 140), (247, 247, 243), (183, 58, 28)])
ABSOLUTE = lut([(250, 250, 246), (242, 183, 75), (153, 32, 27)])


def render(crop, vmax, output_dir, name):
    # Native-size images: precisely one tensor element maps to one PNG pixel.
    signed_index = np.rint((np.clip(crop / vmax, -1, 1) + 1) * 2047.5).astype(np.uint16)
    abs_index = np.rint(np.clip(np.abs(crop) / vmax, 0, 1) * 4095).astype(np.uint16)
    paths = {}
    for label, indices, palette in (('signed', signed_index, SIGNED),
                                    ('absolute', abs_index, ABSOLUTE)):
        path = output_dir / f'{name}_{label}_3000x3000.png'
        Image.fromarray(palette[indices], mode='RGB').save(path, compress_level=6)
        with Image.open(path) as check:
            if check.size != (3000, 3000):
                raise RuntimeError(f'Wrong heatmap dimensions: {path}')
        paths[label] = str(path.resolve())
    return dict(files=paths, display_limit_abs=vmax,
                saturated_elements=int((np.abs(crop) > vmax).sum()),
                saturated_fraction=float((np.abs(crop) > vmax).mean()),
                raw_min=float(crop.min()), raw_max=float(crop.max()))


def main():
    a = args()
    from flashinfer import SfLayout, nvfp4_quantize
    torch.cuda.set_device(a.gpu)
    device = torch.device(f'cuda:{a.gpu}')
    levels = torch.tensor(FP4_LEVELS, dtype=torch.float32, device=device)
    crops = {}
    metadata = dict(format='h3-nvfp4-error-heatmap-3000x3000-v1',
                    timestamp_utc=datetime.now(timezone.utc).isoformat(), gpu=a.gpu,
                    definition='native NVFP4 Q/DQ minus original BF16, elementwise; no pooling, averaging, interpolation, or smoothing',
                    quantization='FlashInfer E2M1, K16 E4M3, current-input per-tensor FP32 global scale; weight scale fixed from BF16 weight',
                    crop=dict(row_start=ROW_START, row_stop_exclusive=ROW_STOP,
                              input_channel_start=COL_START, input_channel_stop_exclusive=COL_STOP,
                              shape=[3000, 3000]),
                    color=dict(signed='blue negative, off-white zero, orange positive; linear symmetric limit',
                               absolute='off-white zero, gold mid, red high; linear limit',
                               limits='common p1/p11 activation 99.5th percentile of deterministic 16x16 stride sample; independent weight limit; values beyond limits saturate, raw pixels are not averaged'),
                    cases={})
    with torch.inference_mode():
        for name, (path, expected_rows) in CASES.items():
            cpu, call = load_input(path)
            if cpu.shape[0] != expected_rows:
                raise ValueError(f'Unexpected {name} shape: {tuple(cpu.shape)}')
            crop, amax = error_crop(cpu.to(device), levels, nvfp4_quantize,
                                    SfLayout.layout_linear)
            crops[name] = crop
            metadata['cases'][name] = dict(source=str(path), cache_call=call,
                                           full_shape=list(cpu.shape), global_amax=amax)
            print(f'{name} captured {crop.shape}; amax={amax}', flush=True)
        with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
            weight = f.get_tensor(KEY)
        if tuple(weight.shape) != (5376, 14336):
            raise ValueError('Unexpected H3 FC2 weight shape')
        crop, amax = error_crop(weight.to(device), levels, nvfp4_quantize,
                                SfLayout.layout_linear)
        crops['weight'] = crop
        metadata['cases']['weight'] = dict(source=str(WEIGHT), key=KEY,
                                          full_shape=list(weight.shape), global_amax=amax)
        print(f'weight captured {crop.shape}; amax={amax}', flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    activation_limit = max(float(np.quantile(np.abs(crops[name][::16, ::16]), 0.995))
                           for name in CASES)
    weight_limit = float(np.quantile(np.abs(crops['weight'][::16, ::16]), 0.995))
    if activation_limit <= 0 or weight_limit <= 0:
        raise RuntimeError('Invalid heatmap color limits')
    for name, crop in crops.items():
        vmax = weight_limit if name == 'weight' else activation_limit
        metadata['cases'][name].update(render(crop, vmax, a.output_dir, name))
        print(f'{name}: color ±{vmax:.6g}; saturated '
              f'{metadata["cases"][name]["saturated_fraction"]:.3%}', flush=True)
    metadata['activation_common_display_limit_abs'] = activation_limit
    metadata['weight_display_limit_abs'] = weight_limit
    path = a.output_dir / 'heatmap_metadata.json'
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'wrote {path}', flush=True)


if __name__ == '__main__':
    main()
