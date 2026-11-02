#!/usr/bin/env python3
"""Render native H3 FC2 W4A4 output-error maps for the E093 inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import KEY, WEIGHT, load_input
from render_h3_nvfp4_error_heatmaps import ABSOLUTE, CASES, SIGNED


ROWS = 3000
PACK_ROWS = 3072  # Native kernel's row alignment; only the first 3000 are reported.
COLS = 3000
BOUNDARIES = {'p1_activation': (658, 1072), 'p11_activation': (738, 1152)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/research/E093/figures'))
    parser.add_argument('--gpu', type=int, default=4)
    return parser.parse_args()


def region_stats(reference, error, start, stop):
    ref = reference[start:stop].float()
    err = error[start:stop].float()
    count = err.numel()
    return dict(rows=stop-start, elements=count,
                output_rms=float(ref.square().mean().sqrt().item()),
                error_rms=float(err.square().mean().sqrt().item()),
                error_mae=float(err.abs().mean().item()),
                output_nmse=float((err.square().sum()/ref.square().sum()).item()),
                signal_energy=float(ref.square().sum().item()),
                error_energy=float(err.square().sum().item()))


def render(crop, vmax, out_dir, name):
    result = {}
    for label, pixels, palette in (
        ('signed', np.rint((np.clip(crop/vmax, -1, 1)+1)*2047.5).astype(np.uint16), SIGNED),
        ('absolute', np.rint(np.clip(np.abs(crop)/vmax, 0, 1)*4095).astype(np.uint16), ABSOLUTE),
    ):
        path = out_dir / f'{name}_output_{label}_3000x3000.png'
        Image.fromarray(palette[pixels], mode='RGB').save(path, compress_level=6)
        with Image.open(path) as im:
            assert im.size == (COLS, ROWS)
        result[label] = str(path.resolve())
    return dict(files=result, display_limit_abs=vmax,
                saturated_fraction=float((np.abs(crop)>vmax).mean()),
                raw_min=float(crop.min()), raw_max=float(crop.max()))


def main():
    a = parse_args()
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    torch.cuda.set_device(a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{a.gpu}')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(format='h3-nvfp4-output-error-heatmap-3000x3000-v1',
                    timestamp_utc=datetime.now(timezone.utc).isoformat(), gpu=a.gpu,
                    layer=KEY.removesuffix('.weight'),
                    definition='native W4A4 NVFP4 FC2 output minus BF16 FC2 output, both BF16; bias-free; elementwise',
                    quantization='FlashInfer E2M1, K16 E4M3; each full input per-tensor current amax; fixed original-weight amax',
                    crop=dict(token_rows=[0, ROWS], output_channels=[0, COLS], shape=[ROWS,COLS]),
                    color=dict(signed='blue negative, off-white zero, orange positive',
                               absolute='off-white zero, gold mid, red high',
                               limits='common p1/p11 linear absolute limit from max 99.5th percentile of 16x16 stride samples'),
                    cases={})
    crops = {}
    with torch.inference_mode():
        with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
            w_cpu = f.get_tensor(KEY)
        assert tuple(w_cpu.shape) == (5376,14336)
        w = w_cpu.to(device)
        del w_cpu
        w_amax = float(w.abs().amax().item())
        w_inverse = torch.tensor([2688.0/w_amax], device=device, dtype=torch.float32)
        wq, wsf = nvfp4_quantize(w, w_inverse, sfLayout=SfLayout.layout_128x4, backend='cuda')
        metadata['weight'] = dict(path=str(WEIGHT), key=KEY, amax=w_amax)
        for name, (path, expected_rows) in CASES.items():
            x_cpu, call = load_input(path)
            assert x_cpu.shape[0] == expected_rows
            full_amax = float(x_cpu.abs().amax().float().item())
            x = x_cpu[:PACK_ROWS].to(device)
            del x_cpu
            x_inverse = torch.tensor([2688.0/full_amax], device=device, dtype=torch.float32)
            xq, xsf = nvfp4_quantize(x, x_inverse, sfLayout=SfLayout.layout_128x4, backend='cuda')
            native = mm_fp4(xq, wq.T, xsf, wsf.T,
                            alpha=(x_inverse*w_inverse).reciprocal(),
                            out_dtype=torch.bfloat16, backend='cutlass')
            reference = x @ w.T
            error = native[:ROWS].float() - reference[:ROWS].float()
            text_end, audio_end = BOUNDARIES[name]
            regions = {'text': (0,text_end), 'audio': (text_end,audio_end),
                       'video_start': (audio_end,ROWS), 'all_crop_rows': (0,ROWS)}
            stats = {kind: region_stats(reference, error, start, stop)
                     for kind,(start,stop) in regions.items()}
            # Statistics cover all 5376 output channels; PNGs show the first 3000.
            crops[name] = error[:, :COLS].cpu().numpy()
            metadata['cases'][name] = dict(source=str(path), cache_call=call,
                                           full_input_shape=[expected_rows,14336],
                                           activation_amax=full_amax,
                                           modality_boundaries=dict(text_end=text_end,audio_end=audio_end),
                                           statistics_all_output_channels=stats,
                                           crop_statistics=region_stats(reference[:ROWS,:COLS],
                                                                       error[:,:COLS],0,ROWS))
            print(f'{name}: {stats}', flush=True)
            del x, xq, xsf, native, reference, error
            torch.cuda.empty_cache()
    vmax = max(float(np.quantile(np.abs(crop[::16,::16]), .995)) for crop in crops.values())
    if not np.isfinite(vmax) or vmax <= 0:
        raise RuntimeError('Invalid common output-error color limit')
    metadata['common_display_limit_abs'] = vmax
    for name,crop in crops.items():
        metadata['cases'][name].update(render(crop,vmax,a.output_dir,name))
    out = a.output_dir/'output_heatmap_metadata.json'
    out.write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
