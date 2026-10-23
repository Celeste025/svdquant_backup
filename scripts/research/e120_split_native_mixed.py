#!/usr/bin/env python3
"""Split selected FP8 PV work from Sage3's spill-free FP4 attention core."""
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
def decode_e2m1(code):
    magnitude_code = code & 7
    magnitude = tl.where(magnitude_code == 0, 0.0,
                tl.where(magnitude_code == 1, 0.5,
                tl.where(magnitude_code == 2, 1.0,
                tl.where(magnitude_code == 3, 1.5,
                tl.where(magnitude_code == 4, 2.0,
                tl.where(magnitude_code == 5, 3.0,
                tl.where(magnitude_code == 6, 4.0, 6.0)))))))
    return tl.where((code & 8) != 0, -magnitude, magnitude)


@triton.jit
def zero_selected_v4(V4, ROUTE, H: tl.constexpr, KP: tl.constexpr,
                     TILES: tl.constexpr):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    if tl.load(ROUTE + head * TILES + tile) != 0:
        dim = tl.arange(0, 128)
        pair = tl.arange(0, 64)
        offset = head * 128 * (KP // 2) + dim[:, None] * (KP // 2) + \
                 tile * 64 + pair[None, :]
        tl.store(V4 + offset, tl.full((128, 64), 0, tl.uint8))


@triton.jit
def dequant_qk4(SRC, SCALE, ROUTE, DST,
               LEN: tl.constexpr, TILES: tl.constexpr,
               IS_KEY: tl.constexpr):
    rb = tl.program_id(0)
    head = tl.program_id(1)
    row = rb * 32 + tl.arange(0, 32)
    dim = tl.arange(0, 128)
    if (not IS_KEY) or (tl.load(ROUTE + head * TILES + rb // 4) != 0):
        if IS_KEY:
            local = row % 32
            physical = (row // 32) * 32 + (local // 8) * 2 + \
                       ((local % 8) // 2) * 8 + local % 2
        else:
            physical = row
        packed = tl.load(SRC + head * LEN * 64 + physical[:, None] * 64 +
                         dim[None, :] // 2)
        code = tl.where((dim[None, :] & 1) == 0,
                        packed & 15, packed >> 4)
        group = dim // 16
        scale_offset = (physical[:, None] // 64) * 512 + \
                       (group[None, :] // 4) * 256 + \
                       (group[None, :] % 4) + \
                       ((physical[:, None] % 64) // 16) * 4 + \
                       (physical[:, None] % 16) * 16
        sf = tl.load(SCALE + head * LEN * 8 + scale_offset).to(tl.float32)
        val = (decode_e2m1(code) * sf).to(tl.float16)
        tl.store(DST + head * LEN * 128 + row[:, None] * 128 +
                 dim[None, :], val)


@triton.jit
def split_selected_fp8_pv(Q4, K4, SFQ, SFK, QDEC, KDEC, DS, V8, S8,
                          IDS, COUNTS, LSE, BASE, OUT,
                          QLEN: tl.constexpr, QP: tl.constexpr,
                          KP: tl.constexpr, TILES: tl.constexpr,
                          BM: tl.constexpr, PREDECODE: tl.constexpr):
    qb = tl.program_id(0)
    head = tl.program_id(1)
    row = qb * BM + tl.arange(0, BM)
    dim = tl.arange(0, 128)
    key_local = tl.arange(0, 128)
    channel_group = dim // 16

    if PREDECODE:
        q = tl.load(QDEC + head * QP * 128 + row[:, None] * 128 +
                    dim[None, :])
    else:
        q_packed = tl.load(Q4 + head * QP * 64 + row[:, None] * 64 +
                           (dim[None, :] // 2))
        q_code = tl.where((dim[None, :] & 1) == 0,
                          q_packed & 15, q_packed >> 4)
        q_sf_offset = (row[:, None] // 64) * 512 + \
                      (channel_group[None, :] // 4) * 256 + \
                      (channel_group[None, :] % 4) + \
                      ((row[:, None] % 64) // 16) * 4 + \
                      (row[:, None] % 16) * 16
        q_scale = tl.load(SFQ + head * QP * 8 + q_sf_offset).to(tl.float32)
        q = (decode_e2m1(q_code) * q_scale).to(tl.float16)
    lse = tl.load(LSE + head * QLEN + row, mask=row < QLEN, other=0)
    acc = tl.full((BM, 128), 0.0, tl.float32)

    count = tl.load(COUNTS + head)
    for i in range(count):
        tile = tl.load(IDS + head * TILES + i)
        key = tile * 128 + key_local
        if PREDECODE:
            k = tl.load(KDEC + head * KP * 128 + key[:, None] * 128 +
                        dim[None, :])
        else:
            k_local = key_local % 32
            physical_key = tile * 128 + (key_local // 32) * 32 + \
                           (k_local // 8) * 2 + ((k_local % 8) // 2) * 8 + \
                           (k_local % 2)
            k_packed = tl.load(K4 + head * KP * 64 + physical_key[:, None] * 64 +
                               (dim[None, :] // 2))
            k_code = tl.where((dim[None, :] & 1) == 0,
                              k_packed & 15, k_packed >> 4)
            k_sf_offset = (physical_key[:, None] // 64) * 512 + \
                          (channel_group[None, :] // 4) * 256 + \
                          (channel_group[None, :] % 4) + \
                          ((physical_key[:, None] % 64) // 16) * 4 + \
                          (physical_key[:, None] % 16) * 16
            k_scale = tl.load(SFK + head * KP * 8 + k_sf_offset).to(tl.float32)
            k = (decode_e2m1(k_code) * k_scale).to(tl.float16)
        scores = tl.dot(q, tl.trans(k))
        ds = tl.load(DS + head * (QP // 128) * KP +
                     (row[:, None] // 128) * KP + key[None, :])
        scores += ds
        tile_max = tl.max(scores, axis=1)
        scaled = tl.exp((scores - tile_max[:, None]) * 0.08838834764831845)
        p8 = tl.minimum(scaled * 448.0, 448.0).to(tl.float8e4nv)
        v8 = tl.load(V8 + head * 128 * KP + dim[None, :] * KP + key[:, None])
        pv = tl.dot(p8, v8)
        v_scale = tl.load(S8 + head * 128 * TILES + dim * TILES + tile)
        row_scale = tl.exp(tile_max * 0.08838834764831845 - lse) / 448.0
        acc += pv * row_scale[:, None] * v_scale[None, :]

    base = tl.load(BASE + head * QLEN * 128 + row[:, None] * 128 + dim[None, :],
                   mask=row[:, None] < QLEN, other=0)
    tl.store(OUT + head * QLEN * 128 + row[:, None] * 128 + dim[None, :],
             base.to(tl.float32) + acc, mask=row[:, None] < QLEN)


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
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--bm", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--warps", type=int, choices=(2, 4, 8, 16), default=4)
    parser.add_argument("--stages", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--predecode", action="store_true")
    parser.add_argument("--force-all-route", action="store_true")
    parser.add_argument("--full-reference", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E120/split_control.json")
    args = parser.parse_args()
    if args.tokens < 1 or args.tokens > 9482 or args.heads < 1 or args.heads > 56:
        raise ValueError("unsupported capture slice")
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed"),
                    str(ROOT / "scripts/research")]
    import fp4attn_lse_cuda
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_cuda
    import sage3_mixed_router_sparse_cuda
    from e119_sparse_v_correction import compact_route
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
        native = fp4attn_mixed_maskfix_cuda.fwd_mixed(
            *packet, v8, s8, route)[0][:, :, :args.tokens].contiguous()
        pure, _ = fp4attn_lse_cuda.fwd(*packet)[:2]
        pure = pure[:, :, :args.tokens].contiguous()
        tiles = vp.shape[2] // 128
        v4zero = v4[0].clone()
        zero_packet = (packet[0], packet[1], v4zero, *packet[3:])
        ids = torch.empty_like(route, dtype=torch.int32)
        counts = torch.empty((1, args.heads), device=q.device, dtype=torch.int32)
        out = torch.empty_like(pure)
        qdec = torch.empty((1, args.heads, qp.shape[2], 128),
                           device=q.device, dtype=torch.float16)
        kdec = torch.empty((1, args.heads, kp.shape[2], 128),
                           device=q.device, dtype=torch.float16)

        def zero_v():
            zero_selected_v4[(tiles, args.heads)](
                v4zero, route, args.heads, vp.shape[2], tiles)

        def compact():
            compact_route[(args.heads,)](route, ids, counts, tiles)

        def decode_q():
            dequant_qk4[(qp.shape[2] // 32, args.heads)](
                q4[0], q4[1], route, qdec, qp.shape[2], tiles, False)

        def decode_k():
            dequant_qk4[(kp.shape[2] // 32, args.heads)](
                k4[0], k4[1], route, kdec, kp.shape[2], tiles, True)

        def split():
            return fp4attn_lse_cuda.fwd(*zero_packet)[:2]

        def correction(base, lse):
            return split_selected_fp8_pv[(triton.cdiv(args.tokens, args.bm), args.heads)](
                q4[0], k4[0], q4[1], k4[1], qdec, kdec, ds, v8, s8,
                ids, counts, lse, base, out,
                args.tokens, qp.shape[2], kp.shape[2], tiles, args.bm,
                args.predecode,
                num_warps=args.warps, num_stages=args.stages)

        zero_v()
        compact()
        if args.predecode:
            decode_q()
            decode_k()
        base_full, lse_full = split()
        base = base_full[:, :, :args.tokens].contiguous()
        lse = lse_full[:, :, :args.tokens].contiguous()
        compiled = correction(base, lse)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all())
        report = {
            "format": "e120-split-native-mixed-v1", "capture": str(CAPTURE),
            "gpu": torch.cuda.get_device_name(), "shape": list(q.shape),
            "selected_tiles": int(route.sum()), "total_tiles": route.numel(),
            "force_all_route": args.force_all_route, "query_block": args.bm,
            "predecode": args.predecode, "warps": args.warps,
            "stages": args.stages,
            "kernel_registers": compiled.n_regs,
            "kernel_spills": compiled.n_spills,
            "split_vs_native_nmse": nmse(out, native),
            "split_vs_pure_nmse": nmse(out, pure),
            "native_vs_pure_nmse": nmse(native, pure),
            "split_max_abs_diff": float((out.float() - native.float()).abs().max()),
            "timing": {
                "zero_v4": bench(zero_v),
                "compact_route": bench(compact),
                **({"decode_q": bench(decode_q), "decode_k": bench(decode_k)}
                   if args.predecode else {}),
                "split_fp4_lse_core": bench(split),
                "sparse_fp8_pv": bench(lambda: correction(base, lse)),
                "native_mixed_core": bench(lambda:
                    fp4attn_mixed_maskfix_cuda.fwd_mixed(
                        *packet, v8, s8, route)),
            },
        }
        if args.full_reference:
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            report["pure_vs_bf16_nmse"] = nmse(pure, ref)
            report["native_vs_bf16_nmse"] = nmse(native, ref)
            report["split_vs_bf16_nmse"] = nmse(out, ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "timing"}
                     | {"timing_ms": {key: value["mean_ms"] for key, value in
                                      report["timing"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
