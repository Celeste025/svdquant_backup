#!/usr/bin/env python3
"""Profile resident MiniMax-H3 DiT calls with true BF16 and Sage3 attention.

Uses two teacher-forced cached H3 calls. Model loading is outside every timed
region. CUDA events delimit complete attention helpers and linear modules;
separate events delimit only AutoWrappedLinear GEMMs. No per-module synchronize
is used. A direct, uninstrumented forward is also timed for each arm/step.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
import time

import torch

import e122_h3_helpers as exp
from diffsynth.core.vram.layers import AutoWrappedLinear

ARMS = ("bf16_sdpa", "sage2_default", "sage2pp", "sage3_fixed",
        "sage3_optimized", "sage3_bf16_fused", "sage3_api_fast",
        "sage3_api_delta_tiled", "sage3_api_qpack_fused",
        "sage3_api_qpack_delta_tiled", "sage3_api_raw_k_delta",
        "split_mixed")
DATA = Path("/data1/models/svdquant-wjq/research/20261015/"
            "E101_h3_r32_fixed1_three_cases/race/bf16_teacher_calls.pt")
OUT = exp.ROOT / "results/research/E122/h3_forward_profile_refactored.json"


def event_pair():
    return (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))


def category(name: str) -> str:
    if ".attn.qkv_proj" in name:
        return "qkv_proj"
    if ".attn.out_proj" in name:
        return "attn_out_proj"
    if ".mlp.fc1" in name:
        return "mlp_fc1"
    if ".mlp.fc2" in name:
        return "mlp_fc2"
    if ".adaln_proj.linear" in name:
        return "adaln"
    return "other_linear"


class ProfileRouter(exp.Router):
    def __init__(self, dit, arm: str, recorder):
        self.recorder = recorder
        super().__init__(dit, arm)

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        if self.recorder is None:
            return super()._dispatch(q, k, v, cu_seqlens, softmax_scale)
        block = self._active_main.get()
        start, end = event_pair()
        start.record()
        out = super()._dispatch(q, k, v, cu_seqlens, softmax_scale)
        end.record()
        self.recorder.attention.append(("refiner" if block is None else "main",
                                        block, start, end))
        return out


class Recorder:
    def __init__(self, dit):
        self.dit = dit
        self.attention = []
        self.linear = []
        self.gemm = []
        self.handles = []
        self.active = {}
        self.name_by_id = {id(m): name for name, m in dit.named_modules()
                           if isinstance(m, torch.nn.Linear)}
        self.original_gemm = AutoWrappedLinear.linear_forward

    def __enter__(self):
        self.modules = []
        for name, module in self.dit.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            self.modules.append(module)
            self.handles.append(module.register_forward_pre_hook(self._pre))
            self.handles.append(module.register_forward_hook(self._post))

        original = self.original_gemm
        recorder = self

        def measured_gemm(module, x, weight, bias):
            start, end = event_pair()
            start.record()
            out = original(module, x, weight, bias)
            end.record()
            recorder.gemm.append((recorder.name_by_id[id(module)], start, end))
            return out

        AutoWrappedLinear.linear_forward = measured_gemm
        return self

    def __exit__(self, exc_type, exc, tb):
        AutoWrappedLinear.linear_forward = self.original_gemm
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        assert not self.active, "linear event stack was not emptied"

    def _pre(self, module, inputs):
        assert id(module) not in self.active, "unexpected recursive linear"
        start, end = event_pair()
        start.record()
        self.active[id(module)] = (start, end)

    def _post(self, module, inputs, output):
        start, end = self.active.pop(id(module))
        end.record()
        self.linear.append((self.name_by_id[id(module)], start, end))

    @staticmethod
    def _summarize(samples, key):
        totals = defaultdict(float)
        counts = defaultdict(int)
        for name, *rest in samples:
            start, end = rest[-2:]
            group = key(name)
            totals[group] += start.elapsed_time(end)
            counts[group] += 1
        return {group: {"ms": totals[group], "calls": counts[group]}
                for group in sorted(totals)}

    def finish(self, total_event_ms, total_wall_s):
        attention = self._summarize(self.attention, lambda name: name)
        linear = self._summarize(self.linear, category)
        gemm = self._summarize(self.gemm, category)
        helper_ms = sum(x["ms"] for x in attention.values())
        linear_ms = sum(x["ms"] for x in linear.values())
        gemm_ms = sum(x["ms"] for x in gemm.values())
        return {"cuda_event_total_ms": total_event_ms,
                "synchronized_wall_seconds": total_wall_s,
                "attention_helper": attention,
                "linear_module_including_weight_load": linear,
                "linear_gemm_only": gemm,
                "attention_helper_ms": helper_ms,
                "all_linear_module_ms": linear_ms,
                "linear_gemm_ms": gemm_ms,
                "linear_non_gemm_estimate_ms": linear_ms - gemm_ms,
                "other_exclusive_estimate_ms": total_event_ms - helper_ms - linear_ms,
                "linear_module_count": len(self.name_by_id)}


def backend_for(arm: str) -> str:
    return "sage_attention" if arm == "sage2_default" else "torch"


def router_arm(arm: str) -> str:
    return "bf16" if arm in ("bf16_sdpa", "sage2_default") else arm


@torch.inference_mode()
def run_call(dit, args, kwargs, arm: str, *, instrument: bool):
    backend = importlib.import_module("diffsynth.core.attention.attention")
    backend.ATTENTION_IMPLEMENTATION = backend_for(arm)
    recorder = Recorder(dit) if instrument else None
    router = ProfileRouter(dit, router_arm(arm), recorder)
    contract = exp.attention_contract(kwargs)
    try:
        if recorder is not None:
            recorder.__enter__()
        start_event, end_event = event_pair()
        torch.cuda.synchronize()
        started = time.perf_counter()
        start_event.record()
        with router.forward_context(**contract, diagnostics=False) as routed:
            output = exp.split_output(dit(*args, **kwargs))
            end_event.record()
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
        event_ms = start_event.elapsed_time(end_event)
        assert all(bool(torch.isfinite(x).all()) for x in output)
        result = {"wall_seconds": wall, "cuda_event_ms": event_ms,
                  "main_helpers": routed.summary["main_helper_calls"],
                  "refiner_helpers": routed.summary["refiner_helper_calls"],
                  "attention_implementation_for_unmodified_helpers":
                      backend.ATTENTION_IMPLEMENTATION}
        if recorder is not None:
            result["breakdown"] = recorder.finish(event_ms, wall)
        return result, tuple(x.float().cpu() for x in output)
    finally:
        if recorder is not None:
            recorder.__exit__(None, None, None)
        router.close()


def nmse(x, y):
    return float((x - y).square().sum() / y.square().sum())


def linear_residency(dit):
    modules = [m for m in dit.modules() if isinstance(m, torch.nn.Linear)]
    gpu = sum(m.weight.device.type == "cuda" for m in modules)
    return {"total": len(modules), "gpu_weights": gpu,
            "all_gpu": gpu == len(modules)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=(0, 10))
    parser.add_argument("--arms", nargs="+", choices=ARMS,
                        default=("bf16_sdpa", "sage2_default", "sage3_fixed", "split_mixed"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resident", action="store_true",
                        help="keep DiT BF16 weights on GPU; no VAE/text model")
    parser.add_argument("--save-output-dir", type=Path,
                        help="optionally save FP32 video/audio outputs for cross-build comparisons")
    opts = parser.parse_args()
    exp.check_backend()
    if "sage2pp" in opts.arms:
        exp.check_sage2pp_backend()
    assert opts.arms[0] == "bf16_sdpa" and len(set(opts.arms)) == len(opts.arms)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert DATA.is_file() and all(0 <= x < 20 for x in opts.steps)
    calls = torch.load(DATA, map_location="cpu", weights_only=False)
    assert len(calls) == 20
    pipe = exp.load_h3_pipeline(full=False, dit_disk=not opts.resident,
                                vram_limit_gib=65.0 if opts.resident else 18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    linears = exp.checked_linears(pipe.dit)
    report = {"format": "e122-h3-attention-linear-profile-v1",
              "gpu": torch.cuda.get_device_name(), "model_load_excluded": True,
              "cached_calls": str(DATA), "steps": list(opts.steps),
              "linears": linears, "resident_requested": opts.resident,
              "arms": {}}
    original_backend = importlib.import_module(
        "diffsynth.core.attention.attention").ATTENTION_IMPLEMENTATION
    try:
        for step in opts.steps:
            args = exp.tree_device(calls[step]["args"], "cuda")
            kwargs = exp.tree_device(calls[step]["kwargs"], "cuda")
            reference = None
            sage3_reference = None
            for arm in opts.arms:
                # Prime lazy kernel compilation and weight cache outside timing.
                run_call(pipe.dit, args, kwargs, arm, instrument=False)
                if opts.resident:
                    residency = linear_residency(pipe.dit)
                    if not residency["all_gpu"]:
                        raise RuntimeError(f"DiT weights not resident: {residency}")
                    report["linear_residency_after_warmup"] = residency
                direct, output = run_call(pipe.dit, args, kwargs, arm,
                                          instrument=False)
                if opts.save_output_dir is not None:
                    opts.save_output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"step": step, "arm": arm, "video": output[0],
                                "audio": output[1]},
                               opts.save_output_dir / f"step{step:02d}_{arm}.pt")
                profiled, profiled_output = run_call(
                    pipe.dit, args, kwargs, arm, instrument=True)
                assert all(torch.equal(a, b) for a, b in
                           zip(output, profiled_output, strict=True))
                if arm == "bf16_sdpa":
                    reference = output
                if arm == "sage3_fixed":
                    sage3_reference = output
                row = {"step": step, "direct": direct, "profiled": profiled,
                       "vs_true_bf16_dit_nmse": None if reference is None else {
                           name: nmse(x, y) for name, x, y in
                           zip(("video", "audio"), output, reference,
                               strict=True)},
                       "vs_sage3_fixed_nmse": None if sage3_reference is None else {
                           name: nmse(x, y) for name, x, y in
                           zip(("video", "audio"), output, sage3_reference,
                               strict=True)},
                       "bitwise_equal_to_sage3_fixed": None if sage3_reference is None else {
                           name: bool(torch.equal(x, y)) for name, x, y in
                           zip(("video", "audio"), output, sage3_reference,
                               strict=True)}}
                report["arms"].setdefault(arm, []).append(row)
                print(f"PROFILE step={step} arm={arm} direct={direct['wall_seconds']:.3f}s "
                      f"profiled={profiled['wall_seconds']:.3f}s "
                      f"attn={profiled['breakdown']['attention_helper_ms']:.1f}ms "
                      f"linear={profiled['breakdown']['all_linear_module_ms']:.1f}ms",
                      flush=True)
            del args, kwargs
        report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    finally:
        importlib.import_module(
            "diffsynth.core.attention.attention").ATTENTION_IMPLEMENTATION = original_backend
    exp.save_json(opts.output, report)
    print(json.dumps({"status": "complete", "output": str(opts.output)}),
          flush=True)


if __name__ == "__main__":
    main()
