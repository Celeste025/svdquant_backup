#!/usr/bin/env python3
"""Native NVFP4/FP8 Triton attention versus PyTorch mixed-PV reference."""
from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path

from e113_native_mixed_attention import native_mixed_sage3_attention
from validate_e113_mixed_triton import reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',
                        help='also validate one full 9482-token H3 attention head')
    parser.add_argument('--output',type=Path,
                        help='write validation to this path instead of the historical E113 artifact')
    args=parser.parse_args()
    capture=Path('/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt')
    data=torch.load(capture,map_location='cpu',weights_only=True,mmap=True)
    rows=[]
    with torch.inference_mode():
        for n in ((54,128,256,1024,9482) if args.full else (54,128,256,1024)):
            head=4
            start=9482 if n==54 else 0
            q,k,v=(data[key][start:start+n,head].unsqueeze(0).unsqueeze(0).contiguous().cuda()
                   for key in ('q','k','v'))
            for th in ((.1,) if n==9482 else (.001,.1,.5)):
                ref,route=reference(q,k,v,th,fp8_group=128,
                                    reverse_tiles=True)
                got=native_mixed_sage3_attention(q,k,v,threshold=th)
                diff=got.float()-ref.float()
                nmse=(diff.square().sum()/ref.float().square().sum()).item()
                row={'tokens':n,'head':head,'threshold':th,
                     'fp4_tiles':sum(x[0] for x in route),
                     'nmse_vs_pytorch':nmse,
                     'finite':bool(torch.isfinite(got).all()),
                     'mma_ptx':native_mixed_sage3_attention.last_mma_check}
                rows.append(row)
                print(json.dumps(row),flush=True)
                assert row['finite'] and nmse < 1e-5 and all(row['mma_ptx'].values())
    assert any(0 < row['fp4_tiles'] < row['tokens']//128 for row in rows if row['tokens']>=256)
    filename='native_vs_pytorch_group128_full.json' if args.full else 'native_vs_pytorch_group128.json'
    target=args.output or Path(__file__).resolve().parents[2]/'results/research/E113'/filename
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'format':'e113-native-vs-pytorch-group128-v1',
                                  'fp4_group':16,'fp8_group':128,'results':rows},indent=2)+'\n')


if __name__=='__main__':
    main()
