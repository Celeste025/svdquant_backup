#!/usr/bin/env python3
"""Matched rank-64 SVDQuant/BF16-attention arm for E124 and E126 cases."""
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

from infer_minimax_h3_svdquant_standard import apply_svdquant
import run_e121_h3_split_generation as e121
import run_e124_h3_attention_prompts as e124
import run_e126_h3_vbench5_attention as e126
from minimax_h3_svdquant_common import PROMPT_FILE, read_prompt

REPORT = ROOT / "results/research/E129"
DATA = Path("/data1/models/svdquant-wjq/research/20261025/E129_h3_svdquant_attention")
STATE = ROOT / "results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
ARM = "svdquant_bf16_attention"
HOOKS = []


class BF16Router(e121.Router):
    def __init__(self, dit, arm: str):
        e121.base.require(arm == ARM, f"unexpected E129 arm: {arm}")
        super().__init__(dit, "bf16")
        self.manifest.update(e129_arm=arm, main_attention="BF16 SDPA")


def install_linears(dit) -> dict:
    global HOOKS
    e121.base.require(not HOOKS, "SVDQuant hooks already installed")
    e121.base.require(STATE.is_file(), f"missing SVDQuant checkpoint: {STATE}")
    HOOKS = apply_svdquant(dit, STATE, tensor_scale_mode="dynamic")
    e121.base.require(len(HOOKS) == 200, "expected 200 SVDQuant target linears")
    return {"count": len(HOOKS), "format": "NVFP4 W4A4 BF16-GEMM Q/DQ",
            "rank": 64, "group_size": 16, "tensor_scale": "dynamic",
            "state": str(STATE.resolve()), "state_sha256": e121.base.sha256(STATE)}


def check_hooks() -> dict:
    calls = [hook.act.calls for hook in HOOKS]
    e121.base.require(len(calls) == 200 and min(calls) > 0,
                      "at least one SVDQuant target linear was not executed")
    return {"count": len(calls), "min_calls": min(calls), "max_calls": max(calls)}


def check_initial_latents(suite: str, case: str) -> dict:
    generated = json.loads((REPORT / case / f"{ARM}_denoise.json").read_text())
    reference_root = e124.REPORT if suite == "e124" else e126.MANIFEST.parent
    reference = json.loads((reference_root / case / "bf16_denoise.json").read_text())
    got = generated["initial_latent_sha256"]
    want = reference["initial_latent_sha256"]
    e121.base.require(got == want and set(got) == {"video", "audio"},
                      f"initial video/audio latents differ from BF16: {case}")
    result = {"case": case, "suite": suite, "arm": ARM,
              "reference": str((reference_root / case / "bf16_denoise.json").resolve()),
              "initial_latent_sha256": got, "matched": True}
    e121.base.save_json(REPORT / case / "initial_pairing.json", result)
    return result


def case_info(case: str, suite: str) -> dict:
    if suite == "e124":
        prompt_id = int(case.removeprefix("p"))
        e121.base.require(case == f"p{prompt_id:03d}", "unexpected E124 case")
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
    e121.base.checked_linears = install_linears
    e121.base.Router = BF16Router
    e121.Router = BF16Router
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
def preflight(suite: str, case: str) -> None:
    info = case_info(case, suite)
    settings = (json.loads(e124.MANIFEST.read_text())["generation"] if suite == "e124"
                else e126.SETTINGS)
    path = REPORT / case / "preflight.json"
    e121.base.require(not path.exists(), f"preflight already exists: {path}")
    pipe = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    linears = install_linears(pipe.dit)
    router = BF16Router(pipe.dit, ARM)
    original_forward = pipe.dit.forward
    result = {"format": "e129-svdquant-preflight-v1", "suite": suite,
              "case": case, "seed": info["seed"], "settings": settings,
              "linears": linears}

    def observed_forward(_self, *args, **kwargs):
        contract = e121.base.attention_contract(kwargs)
        torch.cuda.synchronize()
        started = time.monotonic()
        with router.forward_context(**contract, diagnostics=False) as routed:
            output = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
        video, audio = e121.base.split_output(output)
        e121.base.require(torch.isfinite(video).all().item() and
                          torch.isfinite(audio).all().item(), "nonfinite SVDQuant output")
        result.update(dit_seconds=time.monotonic() - started,
                      attention_main_calls=routed.summary["main_helper_calls"],
                      attention_refiner_calls=routed.summary["refiner_helper_calls"],
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
            raise RuntimeError("preflight did not stop at first DiT call")
        e121.base.require(result["attention_main_calls"] == 50 and
                          result["attention_refiner_calls"] == 2,
                          "unexpected attention call count")
        e121.base.save_json(path, result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        router.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("e124", "e126"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    args = parser.parse_args()
    configure(args.suite)
    info = case_info(args.case, args.suite)
    e121.base.require(bool(info["prompt"]), "empty prompt")
    if args.phase == "preflight":
        preflight(args.suite, args.case)
        return
    suffix = "denoise" if args.phase == "denoise" else "video"
    report = REPORT / args.case / f"{ARM}_{suffix}.json"
    if report.exists():
        row = json.loads(report.read_text())
        complete = ((args.phase == "denoise" and row.get("status") == "denoise_complete")
                    or (args.phase == "decode" and row.get("frames") == 124))
        if complete:
            print(f"SKIP complete {args.case}/{ARM}/{args.phase}", flush=True)
            return
        raise RuntimeError(f"existing incomplete report requires inspection: {report}")
    if args.suite == "e124":
        if args.phase == "denoise":
            e121.base.denoise(args.case, ARM)
            print(json.dumps({"hooks": check_hooks(), "pairing":
                              check_initial_latents(args.suite, args.case)}), flush=True)
        else:
            e121.base.decode(args.case, ARM)
    elif args.phase == "denoise":
        e126.denoise(args.case, ARM)
        print(json.dumps({"hooks": check_hooks(), "pairing":
                          check_initial_latents(args.suite, args.case)}), flush=True)
    else:
        e126.decode(args.case, ARM)


if __name__ == "__main__":
    main()
