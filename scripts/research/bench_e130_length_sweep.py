#!/usr/bin/env python3
"""H3 Sage2++/Sage3 online and core speed versus real-QKV token length."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SAGE2 = Path("/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014")
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt")
LENGTHS = (1024, 2048, 4096, 8192, 16384, 22677)
sys.path[:0] = [str(ROOT / "scripts/research"), str(SAGE2)]
from bench_h3_sage_real_gpu0 import metrics
from e130_sage3_preprocess import attention


def time_cuda(fn, prepare=lambda: None) -> dict:
    for _ in range(10):
        prepare()
        fn()
    torch.cuda.synchronize()
    batches = []
    for _ in range(6):
        events = []
        for _ in range(12):
            prepare()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        batches.append(statistics.mean(a.elapsed_time(b) for a, b in events))
    return {"mean_ms": statistics.mean(batches),
            "cv_batch_percent": 100 * statistics.stdev(batches) / statistics.mean(batches),
            "batch_mean_ms": batches, "calls": 72}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=56)
    args = parser.parse_args()
    if any(n < 1 or n > 22677 for n in args.lengths):
        raise ValueError("lengths must fit within the captured H3 sequence")
    if args.batch_size < 1 or not 1 <= args.heads <= 56:
        raise ValueError("requested batch/head shape is outside the H3 capture")
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattention import sageattn
    from sageattention import _qattn_sm89
    from sageattention.quant import per_channel_fp8, per_warp_int8
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api
    if "fast_delta_gemm" not in inspect.signature(sageattn3_blackwell).parameters:
        raise RuntimeError("benchmark requires explicit tiled delta API support")

    torch.cuda.set_device(0)
    source = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    if source["q"].shape != (1, 56, 22677, 128):
        raise RuntimeError("unexpected H3 capture shape")
    report = {"format": "e130-h3-sage-length-sweep-v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "capture": str(CAPTURE), "sage3_dir": str(args.sage3_dir.resolve()),
              "gpu": torch.cuda.get_device_name(0),
              "input_policy": "contiguous prefix of real H3 HND QKV; requested heads selected and repeated across batch",
              "timing_policy": "CUDA events; 10 warmups; 6 batches x 12 calls; K restore excluded",
              "lengths": []}
    for n in args.lengths:
        q, k, v = [source[name][:, :args.heads, :n, :].to("cuda:0")
                   .expand(args.batch_size, -1, -1, -1).contiguous()
                   for name in ("q", "k", "v")]
        k_work = torch.empty_like(k)
        out2 = torch.empty_like(q)
        scale = 128 ** -0.5

        def bf16():
            return F.scaled_dot_product_attention(q, k, v, scale=scale)

        def sage2_online():
            return sageattn(q, k, v, tensor_layout="HND", is_causal=False,
                            sm_scale=scale)

        def sage3_online():
            return sageattn3_blackwell(q, k_work, v,
                                       is_causal=False, per_block_mean=True)

        def sage3_fused():
            correction_dtype = (torch.bfloat16 if getattr(api.fp4attn_cuda,
                                                            "delta_s_dtype", "fp32") == "bf16"
                                else torch.float32)
            return attention(q, k_work, v, api, exact_delta=True,
                             delta_dtype=correction_dtype, fuse_k_center_pad=True,
                             contiguous_output=True)

        def sage3_api_fast():
            return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                       per_block_mean=True, fast_preprocess=True)

        def sage3_api_delta_tiled():
            return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                       per_block_mean=True, fast_preprocess=True,
                                       fast_delta_gemm=True)

        with torch.inference_mode():
            ref = bf16()
            sage2_out = sage2_online()
            k_work.copy_(k)
            sage3_out = sage3_online()
            k_work.copy_(k)
            fused_out = sage3_fused()
            if not torch.equal(fused_out, sage3_out):
                raise RuntimeError(f"fused Sage3 differs from public API at n={n}")
            k_work.copy_(k)
            api_fast_out = sage3_api_fast()
            if not torch.equal(api_fast_out, sage3_out):
                raise RuntimeError(f"opt-in API differs from public API at n={n}")
            k_work.copy_(k)
            delta_tiled_out = sage3_api_delta_tiled()
            if not torch.equal(delta_tiled_out, sage3_out):
                raise RuntimeError(f"tiled delta API differs from public API at n={n}")
            quality = {"sage2pp_vs_bf16": metrics(sage2_out, ref),
                       "sage3_vs_bf16": metrics(sage3_out, ref),
                       "sage3_fused_equal_public": True,
                       "sage3_api_fast_equal_public": True,
                       "sage3_api_delta_tiled_equal_public": True}

            km = k.mean(dim=2, keepdim=True)
            qi, qs, ki, ks = per_warp_int8(q, k, km, tensor_layout="HND",
                                          BLKQ=128, WARPQ=32, BLKK=64)
            v16, vs16, _ = per_channel_fp8(v, tensor_layout="HND",
                                            scale_max=2.25, smooth_v=False)

            def sage2_core():
                _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                    qi, ki, v16, out2, qs, ks, vs16, 1, 0, 2, scale, 0)
                return out2

            k_work.copy_(k)
            q3, k3, v3, delta_s = api.preprocess_qkv(q, k_work, v, True)
            qpack = api.scale_and_quant_fp4(q3)
            kpack = api.scale_and_quant_fp4_permute(k3)
            vpack = api.scale_and_quant_fp4_transpose(v3)

            def sage3_core():
                return api.blockscaled_fp4_attn(qpack, kpack, vpack, delta_s, n,
                                                is_causal=False, per_block_mean=True,
                                                is_bf16=True)[0]

            if metrics(sage2_core(), sage2_out)["nmse"] > 1e-8:
                raise RuntimeError(f"Sage2++ core mismatch at n={n}")
            if metrics(sage3_core()[:, :, :n], sage3_out)["nmse"] > 1e-8:
                raise RuntimeError(f"Sage3 core mismatch at n={n}")
            arms = {
                "bf16_online": (bf16, lambda: None),
                "sage2pp_online": (sage2_online, lambda: None),
                "sage3_online": (sage3_online, lambda: k_work.copy_(k)),
                "sage3_fused_online": (sage3_fused, lambda: k_work.copy_(k)),
                "sage3_api_fast_online": (sage3_api_fast, lambda: k_work.copy_(k)),
                "sage3_api_delta_tiled_online": (sage3_api_delta_tiled, lambda: k_work.copy_(k)),
                "sage2pp_core": (sage2_core, lambda: None),
                "sage3_core": (sage3_core, lambda: None),
            }
            timings = {name: time_cuda(*arm) for name, arm in arms.items()}
        ops = 4 * args.batch_size * args.heads * n * n * 128
        row = {"tokens": n, "shape_hnd": list(q.shape), "ops": ops,
               "quality": quality, "timings": timings,
               "tops": {name: ops / 1e9 / t["mean_ms"] for name, t in timings.items()},
               "sage3_over_sage2pp_online": timings["sage2pp_online"]["mean_ms"]
                                               / timings["sage3_online"]["mean_ms"],
               "sage3_over_sage2pp_core": timings["sage2pp_core"]["mean_ms"]
                                             / timings["sage3_core"]["mean_ms"]}
        report["lengths"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"tokens": n,
                          "ms": {name: round(t["mean_ms"], 4) for name, t in timings.items()},
                          "online_ratio": row["sage3_over_sage2pp_online"],
                          "core_ratio": row["sage3_over_sage2pp_core"]}), flush=True)


if __name__ == "__main__":
    main()
