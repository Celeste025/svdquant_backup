#!/usr/bin/env python3
"""Single labeled E114 native V-route/FP8-pack launch for Nsight Compute."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "third_party/sage3_native_mixed"),
    "/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe",
]
import sage3_mixed_router_cuda
from sageattn3.api import preprocess_qkv


def main():
    capture = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
                   "h3_p1_s0_b25_480x832_73f_qkv.pt")
    payload = torch.load(capture, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:9482].transpose(0, 1).unsqueeze(0)
               .contiguous().cuda() for name in ("q", "k", "v"))
    with torch.inference_mode():
        _, _, vp, _ = preprocess_qkv(q, k.clone(), v, per_block_mean=True)
        for _ in range(8):
            sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
        torch.cuda.synchronize()
        print("E114: profiling one 56-head route_pack_v launch", flush=True)
        torch.cuda.nvtx.range_push("e114_sage3_native_route_pack_v")
        sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
