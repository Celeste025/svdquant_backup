#!/usr/bin/env python3
"""Persistent MiniMax-H3 generator for the VBench-2.0 prompt subset."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = Path(os.environ.get("DIFFSYNTH_ROOT", ROOT / "third_party/DiffSynth-Studio"))
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

import torch

from minimax_h3_vbench51_worker import (
    SETTINGS,
    apply_plain_w4a4,
    apply_svdquant,
    atomic_json,
    complete,
    generate,
    validate,
)
from minimax_h3_sageattn2 import install_sageattn2
from minimax_h3_sageattn3 import install_sageattn3
from minimax_h3_svdquant_common import load_h3_pipeline

STATE = Path("/home/admin/workspace/aop_lab/app_data/artifacts/variants/nvfp4-g10-r64/quant_state.pt")


class VramPoller:
    """Track device-wide VRAM via nvidia-smi while generation runs."""

    def __init__(self, gpu: int, interval: float = 5.0):
        self.gpu = gpu
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                self.peak_mib = max(self.peak_mib, int(out))
            except (ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "VramPoller":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("bf16", "w4a4", "svdquant"), required=True)
    parser.add_argument("--attention", choices=("sdpa", "sageattn2", "sageattn3"), default="sdpa")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--gpu-physical", type=int, default=0,
                        help="physical GPU id for nvidia-smi VRAM polling")
    parser.add_argument("--reserve-gib", type=float, default=35.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip", type=int, default=0, help="skip the first M cases")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases after skip (0=all)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("settings") != SETTINGS:
        raise RuntimeError("manifest settings differ from the H3 vbench51 contract")
    expected = manifest["sampling"]["cases"]
    if not 1 <= expected <= 110:
        raise RuntimeError(f"unexpected case count: {expected}")
    cases = manifest["cases"][:1] if args.smoke else manifest["cases"]
    if len(manifest["cases"]) != expected:
        raise RuntimeError("manifest case list does not match sampling metadata")
    if args.skip:
        cases = cases[args.skip:]
    if args.limit:
        cases = cases[:args.limit]

    pipe = load_h3_pipeline(full=True, reserve_gib=args.reserve_gib)
    runtimes = []
    if args.variant != "bf16":
        pipe.load_models_to_device(["dit"])
        runtimes = apply_plain_w4a4(pipe.dit) if args.variant == "w4a4" else apply_svdquant(pipe.dit, args.state)
    sage_runtime = install_sageattn3(pipe.dit) if args.attention == "sageattn3" else None
    sage2_runtime = install_sageattn2() if args.attention == "sageattn2" else None
    validation = validate(pipe.dit, args.variant, runtimes)
    validation["attention_backend"] = args.attention
    validation["sageattn3_modules"] = len(sage_runtime.patched_modules) if sage_runtime is not None else 0
    validation["sageattn2"] = sage2_runtime

    worker_status = args.output / "worker_status" / f"{args.variant}.json"
    atomic_json(worker_status, {"state": "running", "variant": args.variant, "attention_backend": args.attention,
                                "quant_state": str(args.state) if args.variant == "svdquant" else None,
                                "validation": validation, "pid": os.getpid(), "started_at": time.time()})
    generated = skipped = 0
    failures = []
    with VramPoller(args.gpu_physical) as poller:
        for case in cases:
            case_dir = args.output / "cases" / case["case_id"]
            video, status = case_dir / f"{args.variant}.mp4", case_dir / f"{args.variant}.json"
            if complete(video, status):
                skipped += 1
                continue
            started = time.time()
            try:
                generate(pipe, case, video)
                if args.variant != "bf16" and any(runtime.act.calls == 0 for runtime in runtimes):
                    raise RuntimeError("a quantized activation hook was not called")
                atomic_json(status, {"state": "complete", "variant": args.variant,
                                     "attention_backend": args.attention,
                                     "case_id": case["case_id"], "prompt": case["prompt"],
                                     "dimensions": case["dimensions"], **SETTINGS,
                                     "quant_state": str(args.state) if args.variant == "svdquant" else None,
                                     "video": str(video), "seconds": time.time() - started,
                                     "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                                     "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                                     "peak_vram_device_mib": poller.peak_mib})
                generated += 1
            except Exception as exc:
                failures.append({"case_id": case["case_id"], "error": repr(exc)})
                atomic_json(status, {"state": "failed", "variant": args.variant,
                                     "attention_backend": args.attention,
                                     "case_id": case["case_id"], "error": repr(exc)})
                torch.cuda.empty_cache()
    atomic_json(worker_status, {"state": "complete" if not failures else "completed_with_failures",
                                "variant": args.variant, "attention_backend": args.attention,
                                "quant_state": str(args.state) if args.variant == "svdquant" else None,
                                "validation": validation, "generated": generated, "skipped": skipped,
                                "failures": failures,
                                "peak_vram_device_mib": poller.peak_mib,
                                "finished_at": time.time()})
    if failures:
        raise SystemExit(f"{args.variant}: {len(failures)} case(s) failed")


if __name__ == "__main__":
    main()
