#!/usr/bin/env python3
"""Capture one real long-sequence H3 attention call and compare E121 cores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

import run_e121_h3_split_generation as exp
from h3_mixed_sage3_api import (_compact_route_all, native_mixed_attention,
                                split_mixed_attention)
from sageattn3 import sageattn3_blackwell
from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                           scale_and_quant_fp4_permute,
                           scale_and_quant_fp4_transpose)
import fp4attn_lse_cuda


class Captured(RuntimeError):
    pass


class CaptureRouter(exp.Router):
    def __init__(self, dit, block: int, heads: int, path: Path):
        self.capture_block, self.capture_heads, self.capture_path = block, heads, path
        super().__init__(dit, "sage3_fixed")

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        if self._active_main.get() == self.capture_block:
            n = self._active_log.get().expected_cu[1]
            tensors = {name: x[:n, :self.capture_heads].transpose(0, 1)
                       .unsqueeze(0).contiguous().cpu()
                       for name, x in (("q", q), ("k", k), ("v", v))}
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            with self.capture_path.open("xb") as handle:
                torch.save(tensors, handle)
            raise Captured()
        return super()._dispatch(q, k, v, cu_seqlens, softmax_scale)


@torch.inference_mode()
def capture(block: int, heads: int, path: Path) -> None:
    cached = (Path("/data1/models/svdquant-wjq/research/20261015/"
                   "E101_h3_r32_fixed1_three_cases/race/bf16_teacher_calls.pt"))
    calls = torch.load(cached, map_location="cpu", weights_only=False)
    pipe = exp.base.load_h3_pipeline(full=False, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    args = exp.base.tree_device(calls[0]["args"], "cuda")
    kwargs = exp.base.tree_device(calls[0]["kwargs"], "cuda")
    router = CaptureRouter(pipe.dit, block, heads, path)
    try:
        try:
            with router.forward_context(**exp.base.attention_contract(kwargs),
                                        diagnostics=False):
                pipe.dit(*args, **kwargs)
        except Captured:
            pass
        else:
            raise RuntimeError("attention capture did not fire")
    finally:
        router.close()
    print(f"Captured block {block}: {path}", flush=True)


def nmse(x, y):
    return float((x.float() - y.float()).square().sum() / y.float().square().sum())


@torch.inference_mode()
def compare(path: Path) -> None:
    data = torch.load(path, map_location="cpu", weights_only=True)
    q, k, v = (data[name].cuda() for name in ("q", "k", "v"))
    n = q.shape[2]
    tiles = triton.cdiv(n, 128)
    route_test = torch.zeros((1, q.shape[1], tiles), dtype=torch.int32,
                             device="cuda")
    route_test[:, :, 0] = 1
    route_test[:, :, tiles - 1] = 1
    ids_test = torch.empty_like(route_test)
    counts_test = torch.empty((1, q.shape[1]), dtype=torch.int32,
                              device="cuda")
    _compact_route_all[(q.shape[1],)](
        route_test, ids_test, counts_test, tiles, triton.next_power_of_2(tiles))
    assert bool(torch.all(counts_test == 2))
    assert bool(torch.all(ids_test[:, :, 0] == 0))
    assert bool(torch.all(ids_test[:, :, 1] == tiles - 1))
    pure = sageattn3_blackwell(q, k.clone(), v, is_causal=False, per_block_mean=True)
    qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
    q4 = scale_and_quant_fp4(qp)
    k4 = scale_and_quant_fp4_permute(kp)
    v4 = scale_and_quant_fp4_transpose(vp)
    packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
              ds, n, None, 128 ** -0.5, False, True, True)
    lse_base, lse = fp4attn_lse_cuda.fwd(*packet)[:2]
    lse_base = lse_base[:, :, :n].contiguous()
    native = native_mixed_attention(q, k, v, threshold=0.10)
    split = split_mixed_attention(q, k, v, threshold=0.10)
    torch.cuda.synchronize()
    report = {"format": "e121-long-attention-diagnosis-v2", "shape": list(q.shape),
              "capture": str(path), "pure_vs_lse_base_nmse": nmse(pure, lse_base),
              "route_tail_boundary_test": "passed",
              "split_vs_pure_nmse": nmse(split, pure),
              "native_vs_pure_nmse": nmse(native, pure),
              "split_vs_native_nmse": nmse(split, native),
              "finite": {name: bool(torch.isfinite(x).all()) for name, x in
                         (("pure", pure), ("lse_base", lse_base),
                          ("native", native), ("split", split), ("lse", lse))},
              "range": {name: [float(x.min()), float(x.max())] for name, x in
                        (("pure", pure), ("lse_base", lse_base),
                         ("native", native), ("split", split), ("lse", lse))}}
    out = exp.REPORT / "race" / "long_attention_diagnosis_v2.json"
    exp.base.save_json(out, report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("capture", "compare"), required=True)
    parser.add_argument("--block", type=int, default=25)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--path", type=Path, default=exp.DATA / "race/block25_qkv_h8.pt")
    args = parser.parse_args()
    exp.check_backend()
    torch.set_num_threads(6)
    if args.phase == "capture":
        capture(args.block, args.heads, args.path)
    else:
        compare(args.path)


if __name__ == "__main__":
    main()
