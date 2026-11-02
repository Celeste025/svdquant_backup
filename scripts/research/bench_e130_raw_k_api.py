#!/usr/bin/env python3
"""A/B fast Sage3 API against direct raw-K delta plus fused K center/pack."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/research")]
from bench_h3_sage_two_scope import measure
from probe_e130_raw_k_delta import CAPTURES, raw_k_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--capture", choices=CAPTURES, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--mutate-input-k", action="store_true",
                        help="Match the official disposable-K API; restore K before each timed call")
    parser.add_argument("--readonly-kpack", action="store_true",
                        help="Use the experimental fused K pack without cloning or modifying K")
    parser.add_argument("--project-api", action="store_true",
                        help="Call sageattn3_blackwell(fast_raw_k_delta=True) instead of the research assembly")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mutate_input_k and args.readonly_kpack:
        raise ValueError("choose only one K input policy")
    if args.project_api and not args.readonly_kpack:
        raise ValueError("project API route requires the read-only K pack")
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURES[args.capture], map_location="cpu", weights_only=True,
                         mmap=True)
    if payload["q"].ndim == 3:
        q, k, v = (payload[name][:args.tokens].to("cuda:0")
                   .transpose(0, 1).unsqueeze(0) for name in ("q", "k", "v"))
    else:
        q, k, v = (payload[name][:, :, :args.tokens].to("cuda:0")
                   for name in ("q", "k", "v"))
    b, h, n, d = q.shape
    p = triton.cdiv(n, 128) * 128
    if (q.shape != k.shape or q.shape != v.shape or d != 128 or
            n != args.tokens or p < 2048 or
            getattr(api.fp4attn_cuda, "delta_s_dtype", None) != "bf16" or
            not hasattr(api.fp4quant_cuda, "scaled_fp4_quant_group_center")):
        raise RuntimeError("requires long H3 BF16 delta/Q-pack candidate build")
    if args.readonly_kpack and not hasattr(
            api.fp4quant_cuda, "scaled_fp4_quant_permute_center_readonly"):
        raise RuntimeError("read-only fused K extension is missing")
    k_original = k.clone()

    def existing():
        return sageattn3_blackwell(q, k, v, is_causal=False, per_block_mean=True,
                                   fast_preprocess=True, fast_qpack=True,
                                   fast_delta_gemm=True)

    def raw_k():
        qm = torch.empty((b, h, p // 128, d), device=q.device, dtype=q.dtype)
        api._e130_group_mean_q_only[(b, h, p // 128)](
            q, qm, n, d, *q.stride(), *qm.stride(), num_warps=4)
        km = k.mean(dim=2, keepdim=True)
        delta_s = raw_k_delta(qm, k, km)
        if p > n and getattr(api.fp4attn_cuda, "delta_tail_mask_in_delta", False):
            delta_s[..., n:] = float("-inf")
        qcode = torch.empty((b, h, p, d // 2), device=q.device, dtype=torch.uint8)
        qscale = torch.empty((b, h, p, d // 16), device=q.device,
                             dtype=torch.float8_e4m3fn)
        api.fp4quant_cuda.scaled_fp4_quant_group_center(q, qm, qcode, qscale)
        # Existing fused K kernel centers its input in place. The official
        # Sage3 API also treats K as disposable; optional clone keeps the
        # experimental fast-preprocess no-mutation contract instead.
        kcopy = k if (args.mutate_input_k or args.readonly_kpack) else k.clone()
        kcode = torch.empty((b, h, p, d // 2), device=k.device, dtype=torch.uint8)
        kscale = torch.empty((b, h, p, d // 16), device=k.device,
                             dtype=torch.float8_e4m3fn)
        kpack = (api.fp4quant_cuda.scaled_fp4_quant_permute_center_readonly
                 if args.readonly_kpack else
                 api.fp4quant_cuda.scaled_fp4_quant_permute_center)
        kpack(kcopy, km, kcode, kscale)
        vcode = torch.empty((b, h, d, p // 2), device=v.device, dtype=torch.uint8)
        vscale = torch.empty((b, h, d, p // 16), device=v.device,
                             dtype=torch.float8_e4m3fn)
        api.fp4quant_cuda.scaled_fp4_quant_trans(v, vcode, vscale, 1)
        out = api.blockscaled_fp4_attn((qcode, qscale), (kcode, kscale),
                                      (vcode, vscale), delta_s, n,
                                      is_causal=False, per_block_mean=True,
                                      is_bf16=True)[0][:, :, :n, :]
        return out.contiguous()

    def project_raw_k():
        return sageattn3_blackwell(
            q, k, v, is_causal=False, per_block_mean=True,
            fast_preprocess=True, fast_qpack=True,
            fast_delta_gemm=True, fast_raw_k_delta=True)

    candidate_call = project_raw_k if args.project_api else raw_k

    with torch.inference_mode():
        pack_checks = {}
        if args.readonly_kpack:
            km_check = k.mean(dim=2, keepdim=True)
            kp_check = torch.empty((b, h, p, d), device=k.device, dtype=k.dtype)
            api._e130_center_pad_k[(b, h, p // 128)](
                k, km_check, kp_check, n, d, *k.stride(),
                km_check.stride(0), km_check.stride(1), km_check.stride(3),
                *kp_check.stride(), num_warps=4)
            ref_code, ref_scale = api.scale_and_quant_fp4_permute(kp_check)
            got_code = torch.empty_like(ref_code)
            got_scale = torch.empty_like(ref_scale)
            api.fp4quant_cuda.scaled_fp4_quant_permute_center_readonly(
                k, km_check, got_code, got_scale)
            pack_checks = {"code_equal": bool(torch.equal(got_code, ref_code)),
                           "scale_equal": bool(torch.equal(got_scale, ref_scale)),
                           "raw_k_unchanged": bool(torch.equal(k, k_original))}
        k.copy_(k_original)
        reference = existing()
        k.copy_(k_original)
        candidate = candidate_call()
        torch.cuda.synchronize()
        diff = candidate.float() - reference.float()
        report = {
            "format": "e130-raw-k-full-api-ab-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(args.sage3_dir.resolve()),
            "capture": str(CAPTURES[args.capture]),
            "shape_hnd": list(q.shape), "gpu": torch.cuda.get_device_name(),
            "input_k_unchanged": bool(torch.equal(k, k_original)),
            "mutate_input_k": args.mutate_input_k,
            "readonly_kpack": args.readonly_kpack,
            "project_api": args.project_api,
            "bitwise_equal": bool(torch.equal(candidate, reference)),
            "nmse": float(diff.square().sum() / reference.float().square().sum()),
            "max_abs_difference": float(diff.abs().max()),
            "k_pack_checks": pack_checks,
            "policy": ("Full online API; raw-K correction plus read-only fused "
                       "K center/pack; K restoration excluded from events"
                       if args.readonly_kpack else
                       "Full online API; raw-K path mutates disposable K; "
                       "K restoration excluded from events" if args.mutate_input_k
                       else "Full online API; raw-K path clones K before in-place "
                            "fused center/pack; no external K restoration timed"),
            "rounds": {"existing": [], "raw_k": []},
        }
        if (report["bitwise_equal"] and
                (args.mutate_input_k or report["input_k_unchanged"]) and
                all(pack_checks.values())):
            for ordering in (("existing", "raw_k"), ("raw_k", "existing")):
                for name in ordering:
                    row = measure(existing if name == "existing" else candidate_call,
                                  prepare=lambda: k.copy_(k_original))
                    report["rounds"][name].append(row)
                    print(name, row["mean_ms"], flush=True)
            report["means_ms"] = {
                name: statistics.mean(row["mean_ms"] for row in rows)
                for name, rows in report["rounds"].items()}
            report["speedup"] = (report["means_ms"]["existing"] /
                                  report["means_ms"]["raw_k"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report.get(key) for key in
                          ("bitwise_equal", "input_k_unchanged", "nmse",
                           "means_ms", "speedup")}, indent=2), flush=True)
        if (not report["bitwise_equal"] or not all(pack_checks.values()) or
                (not args.mutate_input_k and not report["input_k_unchanged"])):
            raise RuntimeError("raw-K API correctness gate failed; timing skipped")


if __name__ == "__main__":
    main()
