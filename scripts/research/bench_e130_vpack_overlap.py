#!/usr/bin/env python3
"""Measure exact Sage3 V-pack overlap on captured H3 Q/K/V."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

from bench_h3_sage_two_scope import measure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    if payload["q"].ndim == 3:
        q, k, v = (payload[name][:args.tokens].to("cuda:0").transpose(0, 1).unsqueeze(0)
                   for name in ("q", "k", "v"))
    else:
        q, k, v = (payload[name][:, :, :args.tokens].to("cuda:0")
                   for name in ("q", "k", "v"))
    assert q.shape == k.shape == v.shape and q.shape[2] == args.tokens
    v_stream = torch.cuda.Stream(device=q.device)
    flags = dict(fast_preprocess=True, fast_qpack=True,
                 fast_delta_gemm=True, fast_raw_k_delta=True)

    def serial():
        return sageattn3_blackwell(q, k, v, **flags)

    def overlap():
        return sageattn3_blackwell(q, k, v, **flags, v_pack_stream=v_stream)

    def serial_preprocess():
        return api._e130_fast_preprocess_qkv(
            q, k, v, fast_delta_gemm=True, fast_qpack=True,
            fast_raw_k_delta=True)

    def overlap_preprocess():
        return api._e130_fast_preprocess_qkv(
            q, k, v, fast_delta_gemm=True, fast_qpack=True,
            fast_raw_k_delta=True, v_pack_stream=v_stream)

    with torch.inference_mode():
        a = serial()
        b = overlap()
        torch.cuda.synchronize()
        equal = bool(torch.equal(a, b))
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        delta = (a.float() - b.float())
        row = dict(format="e130-sage3-vpack-overlap-v1",
                   timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   source=str(args.sage3_dir.resolve()),
                   capture=str(args.capture.resolve()),
                   tokens=args.tokens, shape=list(q.shape),
                   gpu=torch.cuda.get_device_name(0),
                   bitwise_equal=equal, finite=finite,
                   preprocess_only=args.preprocess_only,
                   nmse=float(delta.square().sum() / a.float().square().sum()),
                   max_abs=float(delta.abs().max()),
                   rounds={"serial": [], "overlap": []})
        if equal and finite:
            timed = ({"serial": serial_preprocess, "overlap": overlap_preprocess}
                     if args.preprocess_only else {"serial": serial, "overlap": overlap})
            for order in (("serial", "overlap"), ("overlap", "serial")):
                for name in order:
                    timing = measure(timed[name])
                    row["rounds"][name].append(timing)
                    print(name, timing["mean_ms"], flush=True)
            row["means_ms"] = {name: statistics.mean(
                timing["mean_ms"] for timing in rounds)
                for name, rounds in row["rounds"].items()}
            row["speedup"] = row["means_ms"]["serial"] / row["means_ms"]["overlap"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: row.get(key) for key in
                          ("bitwise_equal", "finite", "means_ms", "speedup")},
                         indent=2), flush=True)
        if not (equal and finite):
            raise RuntimeError("overlapped V packing changed Sage3 output")


if __name__ == "__main__":
    main()
