#!/usr/bin/env python3
"""E121: paired H3 rollout with BF16, corrected Sage3 and split P8/V8 attention.

The original H3 QKV projections, attention output projections, refiner, and
independent padding segment remain unchanged.  All three arms use the same
BF16 model and prompt/seed/settings from E113, with separate E121 outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research")]

# Import the corrected Sage3 package before the older E113 helper can put its
# historical (unfixed) package first on sys.path.
from h3_mixed_sage3_api import split_mixed_attention
import sageattn3
import sageattn3.api as sage_api
from sageattn3 import sageattn3_blackwell
from h3_native_fp4_attention import H3AttentionRouter
import run_e113_h3_mixed_pv_video as base

ARMS = ("bf16", "sage3_fixed", "split_mixed")
MEASURE_ARMS = (*ARMS, "bf16_warm")
REPORT = ROOT / "results/research/E121"
DATA = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation")
PLAN = REPORT / "experiment_plan.md"
CORRECTED = (ROOT / "third_party/sage3_h3_maskfix/sageattn3").resolve()


def check_backend() -> None:
    actual = Path(sageattn3.__file__).resolve()
    base.require(CORRECTED in actual.parents,
                 f"corrected Sage3 was not loaded: {actual}")


class Router(H3AttentionRouter):
    def __init__(self, dit, arm: str):
        base.require(arm in MEASURE_ARMS, f"invalid arm: {arm}")
        self.arm = arm
        is_bf16 = arm in ("bf16", "bf16_warm")
        super().__init__(dit, "bf16" if is_bf16 else "block_mean",
                         _official=None if is_bf16 else sage_api)
        self.manifest.update(e121_arm=arm,
                             corrected_sage3=str(Path(sageattn3.__file__).resolve()),
                             split_threshold=0.10 if arm == "split_mixed" else None)

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        if self.arm in ("bf16", "bf16_warm"):
            return super()._dispatch(q, k, v, cu_seqlens, softmax_scale)
        log = self._active_log.get()
        block = self._active_main.get()
        base.require(log is not None and softmax_scale == 128 ** -0.5,
                     "attention called outside route or scale changed")
        expected = log.expected_refiner_cu if block is None else log.expected_cu
        base.require(q.shape == k.shape == v.shape == (expected[-1], 56, 128),
                     "QKV shape changed")
        base.require(all(x.dtype == torch.bfloat16 and x.is_cuda for x in (q, k, v)),
                     "QKV dtype/device changed")
        log.remember_cu(cu_seqlens, expected, "refiner" if block is None else "main")
        if block is None:
            log.refiner_helpers += 1
            log.original_bf16_segments += 1
            return self.original_helper(q, k, v, cu_seqlens, softmax_scale)
        n, total = expected[1:]
        Q, K, V = (x[:n].transpose(0, 1).unsqueeze(0).contiguous()
                   for x in (q, k, v))
        if self.arm == "sage3_fixed":
            # Official API mutates K in preprocess; keep the H3 input intact.
            result = sageattn3_blackwell(Q, K.clone(), V, is_causal=False,
                                         per_block_mean=True)
        else:
            result = split_mixed_attention(Q, K, V, threshold=0.10)
        base.require(result.shape == Q.shape and result.dtype == torch.bfloat16,
                     "attention output shape/dtype changed")
        out = torch.empty_like(q)
        out[:n] = result[0].transpose(0, 1)
        if total > n:
            out[n:] = self.original_helper(q[n:], k[n:], v[n:],
                                           log.padding_cpu_cu, softmax_scale)
        log.main_rows.append({"block": block, "valid_length": n,
                              "padding_length": total - n, "arm": self.arm})
        log.fp4_calls += 1
        log.original_bf16_segments += int(total > n)
        return out


@torch.inference_mode()
def preflight(case: str) -> None:
    info = base.case_info(case)
    capture = (Path("/data1/models/svdquant-wjq/research/20261015/"
                    "E101_h3_r32_fixed1_three_cases") / case / "bf16_teacher_calls.pt")
    base.require(capture.is_file(), f"missing cached BF16 call: {capture}")
    path = REPORT / case / "attention_preflight_fixed.json"
    base.require(not path.exists(), f"preflight already exists: {path}")
    calls = torch.load(capture, map_location="cpu", weights_only=False)
    base.require(len(calls) == 20, "cached trajectory must have 20 steps")
    pipe = base.load_h3_pipeline(full=False, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    linears = base.checked_linears(pipe.dit)
    args = base.tree_device(calls[0]["args"], "cuda")
    kwargs = base.tree_device(calls[0]["kwargs"], "cuda")
    contract = base.attention_contract(kwargs)
    report = {"format": "e121-attention-preflight-v1", "status": "running",
              "case": case, "prompt_id": info["prompt_id"], "seed": info["seed"],
              "capture": str(capture), "contract": contract, "linears": linears,
              "backend": str(Path(sageattn3.__file__).resolve()), "arms": {}}
    outputs = {}
    for arm in ARMS:
        router = Router(pipe.dit, arm)
        try:
            torch.cuda.synchronize()
            started = time.monotonic()
            with router.forward_context(**contract, diagnostics=False) as routed:
                prediction = base.split_output(pipe.dit(*args, **kwargs))
                torch.cuda.synchronize()
            base.require(all(torch.isfinite(x).all().item() for x in prediction),
                         f"nonfinite DiT output: {arm}")
            outputs[arm] = tuple(x.float() for x in prediction)
            report["arms"][arm] = {"seconds": time.monotonic() - started,
                                   "attention": {k: v for k, v in routed.summary.items()
                                                 if k != "main_rows"}}
            print(f"PREFLIGHT {case}/{arm}: {report['arms'][arm]['seconds']:.2f} s",
                  flush=True)
        finally:
            router.close()
    for arm in ARMS[1:]:
        report["arms"][arm]["vs_bf16_dit_nmse"] = {
            name: float((x - y).square().sum() / y.square().sum())
            for name, x, y in zip(("video", "audio"), outputs[arm],
                                  outputs["bf16"], strict=True)}
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["status"] = "complete"
    base.save_json(path, report)
    print(json.dumps({"status": "complete", "report": str(path),
                      "peak_allocated_gib": report["peak_allocated_gib"]}), flush=True)


def run_phase(phase: str, case: str, arm: str | None) -> None:
    base.REPORT, base.DATA, base.PLAN, base.Router = REPORT, DATA, PLAN, Router
    started = time.monotonic()
    if phase == "preflight":
        preflight(case)
        return
    base.require(arm in MEASURE_ARMS, "arm required for denoise/decode")
    if phase == "denoise":
        base.denoise(case, arm)
        path = REPORT / case / f"{arm}_denoise.json"
        row = json.loads(path.read_text())
        row["format"] = "e121-h3-attention-denoise-v1"
        row["denoise_wall_seconds"] = time.monotonic() - started
        row["sum_dit_seconds"] = sum(x["dit_seconds"] for x in row["steps"])
        base.save_json(path, row)
        print(f"DENOISE WALL {case}/{arm}: {row['denoise_wall_seconds']:.2f} s",
              flush=True)
    elif phase == "decode":
        base.decode(case, arm)
        path = REPORT / case / f"{arm}_video.json"
        row = json.loads(path.read_text())
        row["format"] = "e121-h3-attention-video-v1"
        row["decode_wall_seconds"] = time.monotonic() - started
        base.save_json(path, row)
        print(f"DECODE WALL {case}/{arm}: {row['decode_wall_seconds']:.2f} s",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    parser.add_argument("--case", choices=base.CASES, required=True)
    parser.add_argument("--arm", choices=MEASURE_ARMS)
    args = parser.parse_args()
    check_backend()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_phase(args.phase, args.case, args.arm)


if __name__ == "__main__":
    main()
