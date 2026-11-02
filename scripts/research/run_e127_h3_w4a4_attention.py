#!/usr/bin/env python3
"""E127: plain fake-NVFP4 W4A4 H3 linears, with BF16 or fixed Sage3 attention.

Reuses the E124/E126 generation and decode contracts without altering their
artifacts.  This is the historical H3 Q/DQ linear path, not a native FP4 GEMM
latency benchmark.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research")]

from infer_minimax_h3_svdquant_standard import apply_plain
import run_e121_h3_split_generation as e121
import run_e124_h3_attention_prompts as e124
import run_e126_h3_vbench5_attention as e126
from minimax_h3_svdquant_common import H3_DIT_PATH, PROMPT_FILE, read_prompt

REPORT = ROOT / "results/research/E127"
DATA = Path("/data1/models/svdquant-wjq/research/20261024/E127_h3_w4a4_attention")
ARMS = ("w4a4_bf16_attention", "w4a4_sage3_fixed")
HOOKS = []
EXECUTION = "fake"
NATIVE_CALLS = 0
RECOMPUTE_MODULES = []
EXPORT = Path("/data1/models/svdquant-wjq/research/20261024/E128_plain_export_rehydrated")


class QuantRouter(e121.Router):
    def __init__(self, dit, arm: str):
        e121.base.require(arm in ARMS, f"unknown E127 arm: {arm}")
        super().__init__(dit, "bf16" if arm == ARMS[0] else "sage3_fixed")
        self.manifest["e127_arm"] = arm
        self.manifest["linear_quantization"] = ("plain W4A4 NVFP4 group-16 Q/DQ, dynamic tensor scale"
                                                if EXECUTION == "fake" else
                                                "packed NVFP4 W4A4 scaled_mm, fast dynamic activation pack")


def install_quant_linears(dit) -> dict:
    global HOOKS, NATIVE_CALLS, RECOMPUTE_MODULES
    if EXECUTION in ("native", "native_recompute"):
        from bench_h3_native_nvfp4 import make_h3_resident
        from h3_plain_native_nvfp4 import install_plain_h3, PlainNativeH3Linear
        from h3_nvfp4_zero_sf_compat import collect_fastpack_checks
        model_path = H3_DIT_PATH
        manifest = json.loads((EXPORT / "manifest.json").read_text())
        source = manifest["source_checkpoint"]
        e121.base.require(model_path.resolve() == Path(source["file"]).resolve() and
                          model_path.stat().st_size == source["bytes"] and
                          model_path.stat().st_mtime_ns == source["mtime_ns"],
                          "Native export source checkpoint changed")
        resident = make_h3_resident(dit)
        installation = install_plain_h3(dit, EXPORT,
                                        recompute_fraction=0.10 if EXECUTION == "native_recompute" else 0.0,
                                        recompute_seed=20261025)
        if EXECUTION == "native_recompute":
            RECOMPUTE_MODULES = [m for m in dit.modules() if isinstance(m, PlainNativeH3Linear)]
        e121.base.require(sum(isinstance(m, PlainNativeH3Linear) for m in dit.modules()) == 200,
                          "Not all 200 native linear modules installed")
        original_forward = dit.forward

        def checked_forward(_self, *args, **kwargs):
            global NATIVE_CALLS
            with collect_fastpack_checks() as checks:
                output = original_forward(*args, **kwargs)
            e121.base.require(checks.summary["checked_calls"] == 200 and
                              checks.summary["invalid_calls"] == 0,
                              "Native activation packing contract failed")
            NATIVE_CALLS += 1
            return output

        dit.forward = types.MethodType(checked_forward, dit)
        return {"count": 200, "format": "native NVFP4 E2M1 scaled_mm",
                "group_size": 16, "tensor_scale": "dynamic", "smoothing": False,
                "low_rank": False, "export_dir": str(EXPORT),
                "export_manifest_sha256": e121.base.sha256(EXPORT / "manifest.json"),
                "resident_allocated_gib": resident["allocated_gib"],
                "exact_weight_roundtrip_count": installation["exact_roundtrip_count"]}
    e121.base.require(not HOOKS, "W4A4 hooks already installed in this worker")
    HOOKS = apply_plain(dit)
    e121.base.require(len(HOOKS) == 200, "Expected 200 W4A4 linear hooks")
    return {"count": 200, "format": "NVFP4 E2M1 weight/activation Q/DQ",
            "group_size": 16, "tensor_scale": "dynamic", "smoothing": False,
            "low_rank": False, "hook_calls_before_forward": [h.act.calls for h in HOOKS]}


def check_hooks() -> dict:
    if EXECUTION in ("native", "native_recompute"):
        e121.base.require(NATIVE_CALLS > 0, "Native FP4 DiT was never executed")
        if EXECUTION == "native_recompute":
            calls = [m.recompute_calls for m in RECOMPUTE_MODULES]
            e121.base.require(len(calls) == 200 and min(calls) == NATIVE_CALLS,
                              "Not all 200 linears recomputed tokens every DiT step")
            return {"native_dit_calls": NATIVE_CALLS, "recompute_fraction": 0.10,
                    "recompute_modules_observed": len(calls),
                    "recompute_calls_per_linear": [min(calls), max(calls)],
                    "recomputed_rows_total": sum(m.recompute_rows for m in RECOMPUTE_MODULES)}
        return {"native_dit_calls": NATIVE_CALLS, "validated_scaled_mm_calls_per_dit": 200}
    calls = [h.act.calls for h in HOOKS]
    e121.base.require(len(calls) == 200 and min(calls) > 0,
                      "At least one target W4A4 linear was not executed")
    return {"count": len(calls), "min_calls": min(calls), "max_calls": max(calls)}


def case_info(case: str, suite: str) -> dict:
    if suite == "e124":
        prompt_id = int(case.removeprefix("p"))
        e121.base.require(case == f"p{prompt_id:03d}", "Unexpected E124 case name")
        manifest = json.loads(e124.MANIFEST.read_text())
        allowed = {int(x) for ids in manifest["groups"].values() for x in ids}
        e121.base.require(prompt_id in allowed and
                          e121.base.sha256(PROMPT_FILE) == manifest["prompt_source_sha256"],
                          "E124 prompt manifest mismatch")
        row = read_prompt(prompt_id)
        return {"prompt_id": prompt_id, "seed": int(row["seed"]), "prompt": row["prompt"]}
    return e126.case_info(case)


def configure(suite: str) -> None:
    e121.check_backend()
    e121.base.checked_linears = install_quant_linears
    e121.base.Router = QuantRouter
    e121.Router = QuantRouter
    e121.base.REPORT = REPORT
    e121.base.DATA = DATA
    e121.base.case_info = lambda case: case_info(case, suite)
    if suite == "e124":
        e121.base.PLAN = e124.MANIFEST
    else:
        e126.REPORT = REPORT
        e126.DATA = DATA
    importlib.import_module("diffsynth.core.attention.attention").ATTENTION_IMPLEMENTATION = "torch"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class FirstStepComplete(Exception):
    pass


@torch.inference_mode()
def preflight(suite: str, case: str, arm: str) -> None:
    info = case_info(case, suite)
    settings = (json.loads(e124.MANIFEST.read_text())["generation"] if suite == "e124"
                else e126.SETTINGS)
    path = REPORT / case / f"{arm}_preflight.json"
    e121.base.require(not path.exists(), f"Preflight already exists: {path}")
    pipe = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    linears = install_quant_linears(pipe.dit)
    router = QuantRouter(pipe.dit, arm)
    original_forward = pipe.dit.forward
    result = {"format": "e127-h3-w4a4-preflight-v1", "suite": suite,
              "case": case, "arm": arm, "seed": info["seed"],
              "settings": settings, "linears": linears}

    def observed_forward(_self, *args, **kwargs):
        contract = e121.base.attention_contract(kwargs)
        torch.cuda.synchronize()
        started = time.monotonic()
        with router.forward_context(**contract, diagnostics=False) as routed:
            output = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
        video, audio = e121.base.split_output(output)
        e121.base.require(torch.isfinite(video).all().item() and
                          torch.isfinite(audio).all().item(), "Nonfinite W4A4 DiT output")
        result.update(dit_seconds=time.monotonic() - started,
                      main_tokens=routed.summary["expected_cu"][1],
                      attention_main_calls=routed.summary["main_helper_calls"],
                      attention_refiner_calls=routed.summary["refiner_helper_calls"],
                      video_output_energy=float(video.float().square().mean()),
                      audio_output_energy=float(audio.float().square().mean()),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      free_gib=torch.cuda.mem_get_info()[0] / 1024**3,
                      hooks=check_hooks(), status="complete")
        raise FirstStepComplete()

    pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
    try:
        try:
            pipe(prompt=info["prompt"], seed=info["seed"],
                 height=settings["height"], width=settings["width"],
                 num_frames=settings["frames"], num_inference_steps=settings["steps"],
                 cfg_scale=settings["cfg_scale"], flow_shift=settings["flow_shift"],
                 audio_flow_shift=settings["audio_flow_shift"], rand_device="cpu", tiled=True)
        except FirstStepComplete:
            pass
        else:
            raise RuntimeError("Preflight did not stop at the first DiT call")
        e121.base.require(result["attention_main_calls"] == 50 and
                          result["attention_refiner_calls"] == 2,
                          "Unexpected attention call count")
        e121.base.save_json(path, result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        router.close()


def main() -> None:
    global REPORT, DATA, EXECUTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("e124", "e126"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    parser.add_argument("--execution", choices=("fake", "native", "native_recompute"), default="fake")
    args = parser.parse_args()
    EXECUTION = args.execution
    if EXECUTION == "native":
        REPORT = ROOT / "results/research/E128"
        DATA = Path("/data1/models/svdquant-wjq/research/20261024/E128_h3_native_w4a4_attention")
    elif EXECUTION == "native_recompute":
        REPORT = ROOT / "results/research/E131"
        DATA = Path("/data1/models/svdquant-wjq/research/20261025/E131_h3_w4a4_token_recompute")
    configure(args.suite)
    info = case_info(args.case, args.suite)
    e121.base.require(bool(info["prompt"]), "Empty prompt")
    if args.phase == "preflight":
        preflight(args.suite, args.case, args.arm)
        return
    report = REPORT / args.case / f"{args.arm}_{'denoise' if args.phase == 'denoise' else 'video'}.json"
    if report.exists():
        row = json.loads(report.read_text())
        complete = ((args.phase == "denoise" and row.get("status") == "denoise_complete")
                    or (args.phase == "decode" and row.get("frames") == 124))
        if complete:
            print(f"SKIP complete {args.case}/{args.arm}/{args.phase}", flush=True)
            return
        raise RuntimeError(f"Existing incomplete report requires inspection: {report}")
    if args.suite == "e124":
        if args.phase == "denoise":
            e121.base.denoise(args.case, args.arm)
            print(json.dumps({"hooks": check_hooks()}), flush=True)
        else:
            e121.base.decode(args.case, args.arm)
    elif args.phase == "denoise":
        e126.denoise(args.case, args.arm)
        print(json.dumps({"hooks": check_hooks()}), flush=True)
    else:
        e126.decode(args.case, args.arm)


if __name__ == "__main__":
    main()
