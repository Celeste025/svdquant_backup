#!/usr/bin/env python3
"""E095: separate outlier-zeroing distortion from native W4A4 error."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import KEY, WEIGHT, load_input


CALLS = {
    'p11': (Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p11_s00_calib.pt'), 738, 1152, 22464),
    'p1': (Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p1_s00.pt'), 658, 1072, 22384),
}
THRESHOLDS = (None, 2512.0, 1024.0, 512.0, 256.0)


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=6)
    p.add_argument('--output', type=Path,
                   default=Path('results/research/E095/h3_fc2_outlier_zeroing_nvfp4.json'))
    return p.parse_args()


def nmse(a, b):
    return float(((a.float()-b.float()).square().sum()/b.float().square().sum()).item())


def main():
    a = args()
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    torch.cuda.set_device(a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{a.gpu}')
    report = dict(format='h3-fc2-outlier-zeroing-nvfp4-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='H3 block-25 FC2, full p1/p11 step-0 cached calls; plain W4A4 only',
                  thresholds_abs_gt=list(THRESHOLDS),gpu=a.gpu,
                  quantization='native FlashInfer NVFP4 E2M1, K16 E4M3, original BF16 weight',
                  definitions=dict(primary='NMSE(native W4A4 output after zeroing, original unmodified BF16 output)',
                                   quantization_only='NMSE(native W4A4 output after zeroing, BF16 output after zeroing)',
                                   removal='NMSE(BF16 output after zeroing, original unmodified BF16 output)'),
                  weight=str(WEIGHT),weight_key=KEY,calls={})
    with torch.inference_mode():
        with safe_open(str(WEIGHT),framework='pt',device='cpu') as f:
            w_cpu = f.get_tensor(KEY)
        assert tuple(w_cpu.shape)==(5376,14336)
        w=w_cpu.to(device)
        del w_cpu
        w_amax=float(w.abs().amax().item())
        wi=torch.tensor([2688.0/w_amax],device=device,dtype=torch.float32)
        wq,wsf=nvfp4_quantize(w,wi,sfLayout=SfLayout.layout_128x4,backend='cuda')
        report['weight_amax']=w_amax
        # E089 calibrated activation amax is the original p11 full-call maximum.
        fixed_amax=2512.0
        fixed_inv=torch.tensor([2688.0/fixed_amax],device=device,dtype=torch.float32)
        report['fixed_activation_amax_from_unmodified_p11']=fixed_amax
        for name,(path,text_end,audio_end,video_end) in CALLS.items():
            cpu,call=load_input(path)
            x=cpu.to(device)
            del cpu
            ref=x@w.T
            rows=[]
            for threshold in THRESHOLDS:
                if threshold is None:
                    mask=None
                    xx=x
                    counts={'text':0,'audio':0,'video':0,'padding':0}
                else:
                    mask=x.abs()>threshold
                    xx=x.masked_fill(mask,0)
                    counts={'text':int(mask[:text_end].sum().item()),
                            'audio':int(mask[text_end:audio_end].sum().item()),
                            'video':int(mask[audio_end:video_end].sum().item()),
                            'padding':int(mask[video_end:].sum().item())}
                amax=float(xx.abs().amax().item())
                inv=torch.tensor([2688.0/amax],device=device,dtype=torch.float32)
                modified=ref if threshold is None else xx@w.T
                original_energy=ref.float().square().sum()
                modified_energy=modified.float().square().sum()
                modified_to_original_energy_ratio=float((modified_energy/original_energy).item())
                measured={}
                for scale_name,xi in (('current_input',inv),('fixed_p11_original',fixed_inv)):
                    aq,asf=nvfp4_quantize(xx,xi,sfLayout=SfLayout.layout_128x4,backend='cuda')
                    y=mm_fp4(aq,wq.T,asf,wsf.T,
                             alpha=(xi*wi).reciprocal(),
                             out_dtype=torch.bfloat16,backend='cutlass')
                    measured[scale_name]=dict(
                        final_original_target_nmse=nmse(y,ref),
                        quantization_only_modified_target_nmse=nmse(y,modified),
                        quantization_only_original_energy_nmse=float(
                            ((y.float()-modified.float()).square().sum()/original_energy).item()),
                        removal_original_target_nmse=nmse(modified,ref))
                    del aq,asf,y
                row=dict(threshold_abs_gt=threshold,removed_elements=sum(counts.values()),
                         removed_fraction=sum(counts.values())/x.numel(),
                         removed_by_modality=counts,activation_amax_after_zeroing=amax,
                         modified_to_original_output_energy_ratio=modified_to_original_energy_ratio,
                         scales=measured)
                rows.append(row)
                print(name,threshold,counts,amax,
                      measured['current_input']['final_original_target_nmse'],
                      measured['current_input']['quantization_only_modified_target_nmse'],
                      measured['current_input']['removal_original_target_nmse'],flush=True)
                if threshold is not None:
                    del mask,xx,modified
            report['calls'][name]=dict(source=str(path),cache_call=call,
                                       shape=list(x.shape),boundaries=dict(text_end=text_end,
                                       audio_end=audio_end,video_end=video_end),rows=rows)
            del x,ref
            torch.cuda.empty_cache()
    # Unsmodified controls reproduce the previously published E089 native path.
    expected={'p11':(0.009607384,0.009607384),
              'p1':(0.009643156,0.010424481)}
    for name,(dynamic,static) in expected.items():
        baseline=report['calls'][name]['rows'][0]['scales']
        assert abs(baseline['current_input']['final_original_target_nmse']-dynamic)<2e-7
        assert abs(baseline['fixed_p11_original']['final_original_target_nmse']-static)<2e-7
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('wrote',a.output,flush=True)


if __name__=='__main__':
    main()
