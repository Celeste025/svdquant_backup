#!/usr/bin/env python3
"""Experimental sparse V correction on top of a spill-free Sage3 FP4 core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


@triton.jit
def pack_selected_delta_v(V, V8, S8, ROUTE, DELTA,
                          H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                          TILES: tl.constexpr):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    if tl.load(ROUTE + (batch * H + head) * TILES + tile) != 0:
        tok = tile * 128 + tl.arange(0, 128)
        dim = tl.arange(0, 128)
        base = (batch * H + head)
        x = tl.load(V + base * K * D + tok[:, None] * D + dim[None, :],
                    mask=dim[None, :] < D, other=0).to(tl.float32)
        x_groups = tl.reshape(x, (8, 16, 128))
        vmax = tl.max(tl.abs(x_groups), axis=1)
        sf = (vmax / 6.0).to(tl.float8e4nv).to(tl.float32)
        sf_broadcast = tl.reshape(tl.broadcast_to(sf[:, None, :], (8, 16, 128)),
                                  (128, 128))
        inv = tl.where(sf_broadcast > 0, 1.0 / sf_broadcast, 0.0)
        norm = tl.abs(x * inv)
        mag = tl.where(norm >= 5.0, 6.0,
              tl.where(norm > 3.5, 4.0,
              tl.where(norm >= 2.5, 3.0,
              tl.where(norm > 1.75, 2.0,
              tl.where(norm >= 1.25, 1.5,
              tl.where(norm > 0.75, 1.0,
              tl.where(norm >= 0.25, 0.5, 0.0)))))))
        v4 = tl.where(x < 0, -mag, mag) * sf_broadcast
        v8_code = tl.load(V8 + base * D * K + dim[:, None] * K + tok[None, :]).to(tl.float32)
        v8_scale = tl.load(S8 + base * D * TILES + dim * TILES + tile)
        v8 = tl.trans(v8_code) * v8_scale[None, :]
        delta = v8 - v4
        tl.store(DELTA + base * K * D + tok[:, None] * D + dim[None, :], delta,
                 mask=dim[None, :] < D)


@triton.jit
def compact_route(ROUTE, IDS, COUNTS, TILES: tl.constexpr):
    head = tl.program_id(0)
    idx = tl.arange(0, 128)
    chosen = tl.load(ROUTE + head * TILES + idx,
                     mask=idx < TILES, other=0).to(tl.int32)
    pos = tl.cumsum(chosen) - 1
    tl.store(IDS + head * TILES + pos, idx, mask=(idx < TILES) & (chosen != 0))
    tl.store(COUNTS + head, tl.sum(chosen))


@triton.jit
def sparse_correct(Q, K, DELTA, IDS, COUNTS, LSE, BASE, OUT,
                   QLEN: tl.constexpr, KLEN: tl.constexpr,
                   D: tl.constexpr, TILES: tl.constexpr,
                   QSH: tl.constexpr, QSQ: tl.constexpr, QSD: tl.constexpr,
                   BM: tl.constexpr):
    qb = tl.program_id(0)
    head = tl.program_id(1)
    row = qb * BM + tl.arange(0, BM)
    dim = tl.arange(0, 128)
    tok = tl.arange(0, 128)
    q = tl.load(Q + head * QSH + row[:, None] * QSQ + dim[None, :] * QSD,
                mask=(row[:, None] < QLEN) & (dim[None, :] < D),
                other=0)
    lse = tl.load(LSE + head * QLEN + row, mask=row < QLEN, other=0)
    acc = tl.full((BM, 128), 0.0, tl.float32)
    count = tl.load(COUNTS + head)
    for i in range(count):
        tile = tl.load(IDS + head * TILES + i)
        key = tile * 128 + tok
        key_matrix = tl.load(K + head * KLEN * D + key[None, :] * D + dim[:, None],
                             mask=dim[:, None] < D, other=0)
        score = tl.dot(q, key_matrix) * 0.08838834764831845
        prob = tl.exp(score - lse[:, None]).to(tl.bfloat16)
        dv = tl.load(DELTA + head * KLEN * D + key[:, None] * D + dim[None, :],
                     mask=dim[None, :] < D, other=0)
        acc += tl.dot(prob, dv)
    base = tl.load(BASE + head * QLEN * D + row[:, None] * D + dim[None, :],
                   mask=(row[:, None] < QLEN) & (dim[None, :] < D), other=0)
    tl.store(OUT + head * QLEN * D + row[:, None] * D + dim[None, :],
             base.to(tl.float32) + acc,
             mask=(row[:, None] < QLEN) & (dim[None, :] < D))


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).square().sum() / b.square().sum())


def bench(fn, warmup=10, batches=5, repeats=10) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(batches):
        events = []
        for _ in range(repeats):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            events.append((a, b))
        torch.cuda.synchronize()
        means.append(statistics.mean(a.elapsed_time(b) for a, b in events))
    return {"mean_ms": statistics.mean(means), "batch_means_ms": means,
            "calls": batches * repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=9482)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--bm", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--force-all-route", action="store_true")
    parser.add_argument("--full-reference", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E119/sparse_correction_gpu0.json")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed")]
    import fp4attn_lse_cuda
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_cuda
    import sage3_mixed_router_sparse_cuda
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:args.tokens].to(f"cuda:{args.gpu}")
               .transpose(0, 1)[:args.heads].unsqueeze(0) for name in ("q", "k", "v"))
    with torch.inference_mode():
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                  ds, args.tokens, None, 128 ** -0.5, False, True, True)
        v8, s8, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(vp, 0.10)
        if args.force_all_route:
            v8, s8, _, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
            route = torch.ones_like(route)
        pure, lse = fp4attn_lse_cuda.fwd(*packet)[:2]
        pure = pure[:, :, :args.tokens].contiguous()
        lse = lse[:, :, :args.tokens].contiguous()
        mixed = fp4attn_mixed_maskfix_cuda.fwd_mixed(
            *packet, v8, s8, route)[0][:, :, :args.tokens].contiguous()
        delta = torch.empty_like(vp)
        ids = torch.empty_like(route, dtype=torch.int32)
        counts = torch.empty((1, args.heads), device=q.device, dtype=torch.int32)
        corrected = torch.empty_like(pure)
        klen = vp.shape[2]
        tiles = klen // 128

        def pack_delta():
            pack_selected_delta_v[(tiles, args.heads, 1)](
                vp, v8, s8, route, delta, args.heads, klen, 128, tiles)

        def compact():
            compact_route[(args.heads,)](route, ids, counts, tiles)

        def correction():
            sparse_correct[(triton.cdiv(args.tokens, args.bm), args.heads)](
                q, kp, delta, ids, counts, lse, pure, corrected,
                args.tokens, klen, 128, tiles,
                q.stride(1), q.stride(2), q.stride(3), args.bm,
                num_warps=4)

        pack_delta()
        compact()
        correction()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(corrected).all())
        if args.tokens <= 128:
            query_rows = torch.arange(args.tokens, device=q.device)
        else:
            query_rows = torch.tensor([0, 1, 13, 127, 128, 500, 1024, 3000,
                                       4096, 7000, 9000, args.tokens - 1], device=q.device)
        query_rows = query_rows[query_rows < args.tokens]
        reference_prob = torch.softmax(
            torch.matmul(q[:, :, query_rows].float(), k.float().transpose(-1, -2))
            * (128 ** -0.5), dim=-1)
        reference = torch.matmul(reference_prob, v.float())
        report = {
            "format": "e119-sparse-v-correction-v1", "capture": str(CAPTURE),
            "gpu": torch.cuda.get_device_name(), "shape": list(q.shape),
            "query_block": args.bm, "selected_tiles": int(route.sum()),
            "force_all_route": args.force_all_route,
            "total_tiles": route.numel(),
            "correction_vs_native_mixed_nmse": nmse(corrected, mixed),
            "correction_vs_pure_nmse": nmse(corrected, pure),
            "native_mixed_vs_pure_nmse": nmse(mixed, pure),
            "reference_query_rows": query_rows.tolist(),
            "pure_vs_bf16_sample_nmse": nmse(pure[:, :, query_rows], reference),
            "native_mixed_vs_bf16_sample_nmse": nmse(mixed[:, :, query_rows], reference),
            "correction_vs_bf16_sample_nmse": nmse(corrected[:, :, query_rows], reference),
            "timing": {
                "pure_lse_core": bench(lambda: fp4attn_lse_cuda.fwd(*packet)),
                "delta_pack": bench(pack_delta),
                "route_compact": bench(compact),
                "sparse_correction": bench(correction),
                "mixed_baseline_core": bench(lambda:
                    fp4attn_mixed_maskfix_cuda.fwd_mixed(*packet, v8, s8, route)),
            },
        }
        if args.full_reference:
            full_reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            report["pure_vs_bf16_full_nmse"] = nmse(pure, full_reference)
            report["native_mixed_vs_bf16_full_nmse"] = nmse(mixed, full_reference)
            report["correction_vs_bf16_full_nmse"] = nmse(corrected, full_reference)
            direction = corrected.float() - pure.float()
            denominator = direction.square().sum()
            alpha_reference = float(((full_reference.float() - pure.float()) *
                                     direction).sum() / denominator)
            alpha_native = float(((mixed.float() - pure.float()) *
                                  direction).sum() / denominator)
            report["oracle_alpha_to_bf16"] = alpha_reference
            report["oracle_alpha_to_native_mixed"] = alpha_native
            report["oracle_alpha_bf16_nmse"] = nmse(
                pure.float() + alpha_reference * direction, full_reference)
            report["oracle_alpha_native_nmse"] = nmse(
                pure.float() + alpha_native * direction, mixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "timing"}
                     | {"timing_ms": {key: value["mean_ms"] for key, value in
                                      report["timing"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
