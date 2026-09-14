#!/usr/bin/env python3
"""Build the reproducible VBench-251 manifest and run its 51-case pilot."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VBENCH = Path("/home/wjq/workspace/ViDiT-Q/eval/video/Vbench/vbench/VBench_full_info.json")
OUT = ROOT / "results/samples/rcm_vbench251_seed0_480p77f_4step"
DIMS = ("aesthetic_quality", "scene", "imaging_quality", "overall_consistency", "background_consistency", "subject_consistency", "dynamic_degree", "motion_smoothness")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def build_manifest(path: Path) -> dict:
    raw = json.loads(VBENCH.read_text())
    prompts: OrderedDict[str, dict] = OrderedDict()
    for item in raw:
        dims = [d for d in item["dimension"] if d in DIMS]
        if not dims:
            continue
        rec = prompts.setdefault(item["prompt_en"], {"prompt": item["prompt_en"], "dimensions": []})
        rec["dimensions"] = sorted(set(rec["dimensions"]) | set(dims), key=DIMS.index)
    all_cases = [{"case_id": f"vbench_{i:03d}", **record} for i, record in enumerate(prompts.values())]
    if len(all_cases) != 251:
        raise RuntimeError(f"expected 251 unique VBench prompts, got {len(all_cases)}")
    quotas = {d: round(sum(d in c["dimensions"] for c in all_cases) / 5) for d in DIMS}
    selected, remaining = [], set(range(len(all_cases)))
    covered = {d: 0 for d in DIMS}
    # Greedy weighted set cover: each selected prompt preferentially fills the
    # largest remaining category deficit; index provides stable tie-breaking.
    while len(selected) < 51:
        def score(i: int) -> tuple[float, int]:
            gain = sum(max(quotas[d] - covered[d], 0) / max(quotas[d], 1) for d in all_cases[i]["dimensions"])
            return gain, -i
        index = max(remaining, key=score)
        selected.append(all_cases[index]); remaining.remove(index)
        for d in all_cases[index]["dimensions"]:
            covered[d] += 1
    payload = {"schema_version": 1, "source": str(VBENCH), "created_at": time.time(),
        "sampling": {"method": "deterministic stratified greedy coverage", "pilot_size": 51, "fraction": 0.2,
                     "dimension_quotas": quotas, "pilot_dimension_counts": covered},
        "settings": {"seed": 0, "height": 480, "width": 832, "frames": 77, "steps": 4, "sigma_max": 80.0, "guidance": 0.0},
        "variants": {"bf16": {"format": "bf16"},
          "nvfp4": {"format": "real-NVFP4 dynamic W4A4 QDQ", "smoothing": False, "low_rank": False, "gate": False},
          "nvfp4_svdquant": {"format": "existing real-NVFP4 SVDQuant", "checkpoint": "/data1/models/svdquant-wjq/ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16"}},
        "all_cases": all_cases, "pilot_cases": selected}
    atomic_json(path, payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--gpu", default="7")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true", help="run all 251 after the pilot has been accepted")
    args = p.parse_args()
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else build_manifest(manifest_path)
    if manifest["settings"] != {"seed": 0, "height": 480, "width": 832, "frames": 77, "steps": 4, "sigma_max": 80.0, "guidance": 0.0}:
        raise RuntimeError("existing manifest settings differ from this experiment contract")
    if args.full:
        manifest["pilot_cases"] = manifest["all_cases"]
        atomic_json(manifest_path, manifest)
    ptq_bin = "/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "DEEPCOMPRESSOR_WAN_GATED": "0",
           "PYTHONPATH": str(ROOT / "third_party/deepcompressor"),
           "TOKENIZERS_PARALLELISM": "false", "PATH": ptq_bin + ":" + os.environ["PATH"]}
    for variant in ("bf16", "nvfp4", "nvfp4_svdquant"):
        cmd = [sys.executable, str(ROOT / "scripts/rcm_vbench251_worker.py"), "--manifest", str(manifest_path),
               "--output", str(args.output), "--variant", variant]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
