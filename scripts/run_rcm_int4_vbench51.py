#!/usr/bin/env python3
"""Generate the deterministic rCM VBench-51 subset with the INT4 SVDQuant checkpoint.

The matching BF16 videos are copied from the prior rCM VBench run, never regenerated.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VBENCH = Path("/home/wjq/workspace/ViDiT-Q/eval/video/Vbench/vbench/VBench_full_info.json")
SOURCE = ROOT / "results/samples/rcm_vbench251_seed0_480p77f_4step"
OUT = ROOT / "results/samples/rcm_int4_svdquant_vbench51_seed0_480p77f_4step"
DIMS = ("aesthetic_quality", "scene", "imaging_quality", "overall_consistency", "background_consistency",
        "subject_consistency", "dynamic_degree", "motion_smoothness")
SETTINGS = {"seed": 0, "height": 480, "width": 832, "frames": 77, "steps": 4,
            "sigma_max": 80.0, "guidance": 0.0, "fps": 16}
INT4_CKPT = Path("/data1/models/svdquant-wjq/ckpts/rcm-wan2.1-1.3b-int4-s16-g10")
PLAIN_INT4_VARIANT = {"format": "dynamic INT4 W4A4 QDQ", "smoothing": False, "low_rank": False,
                      "weight_group_size": 64, "activation_group_size": 64}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def build_manifest(path: Path) -> dict:
    prompts: OrderedDict[str, dict] = OrderedDict()
    for item in json.loads(VBENCH.read_text()):
        dims = [dim for dim in item["dimension"] if dim in DIMS]
        if dims:
            record = prompts.setdefault(item["prompt_en"], {"prompt": item["prompt_en"], "dimensions": []})
            record["dimensions"] = sorted(set(record["dimensions"]) | set(dims), key=DIMS.index)
    all_cases = [{"case_id": f"vbench_{i:03d}", **record} for i, record in enumerate(prompts.values())]
    if len(all_cases) != 251:
        raise RuntimeError(f"expected 251 unique VBench prompts, got {len(all_cases)}")
    quotas = {dim: round(sum(dim in case["dimensions"] for case in all_cases) / 5) for dim in DIMS}
    cases, remaining, covered = [], set(range(len(all_cases))), {dim: 0 for dim in DIMS}
    while len(cases) < 51:
        index = max(remaining, key=lambda i: (
            sum(max(quotas[dim] - covered[dim], 0) / max(quotas[dim], 1) for dim in all_cases[i]["dimensions"]), -i))
        cases.append(all_cases[index]); remaining.remove(index)
        for dim in all_cases[index]["dimensions"]:
            covered[dim] += 1
    manifest = {"schema_version": 1, "created_at": time.time(), "source": str(VBENCH), "settings": SETTINGS,
                "sampling": {"method": "deterministic stratified greedy coverage", "source_population": 251,
                             "cases": 51, "dimension_quotas": quotas, "dimension_counts": covered},
                "variants": {"bf16": {"source": str(SOURCE), "action": "copy-existing"},
                             "int4_plain": PLAIN_INT4_VARIANT,
                             "int4_svdquant": {"checkpoint": str(INT4_CKPT), "format": "INT4 W4A4 SVDQuant",
                                                "rank": 32, "smooth_grids": 10, "lowrank_iters": 100}},
                "cases": cases}
    atomic_json(path, manifest)
    return manifest


def copy_bf16(manifest: dict, output: Path, smoke: bool) -> int:
    copied = 0
    cases = manifest["cases"][:1] if smoke else manifest["cases"]
    missing = []
    for case in cases:
        case_dir = output / "cases" / case["case_id"]
        source_dir = SOURCE / "cases" / case["case_id"]
        for suffix in (".mp4", ".json"):
            source = source_dir / f"bf16{suffix}"
            target = case_dir / source.name
            if not source.is_file():
                missing.append(str(source))
            elif not target.exists():
                case_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
    if missing:
        raise RuntimeError("missing required existing BF16 artifacts:\n" + "\n".join(missing))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--variant", choices=("int4_plain", "int4_svdquant"), default="int4_svdquant")
    parser.add_argument("--smoke", action="store_true", help="copy one BF16 case and generate one INT4 case")
    args = parser.parse_args()
    if not (INT4_CKPT / "model.pt").is_file():
        raise RuntimeError(f"missing INT4 checkpoint: {INT4_CKPT}")
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else build_manifest(manifest_path)
    if manifest.get("settings") != SETTINGS or len(manifest.get("cases", [])) != 51:
        raise RuntimeError("existing manifest does not match the rCM VBench-51 contract")
    # Upgrade a manifest created before the paired plain-INT4 baseline existed.
    if manifest.setdefault("variants", {}).get("int4_plain") != PLAIN_INT4_VARIANT:
        manifest["variants"]["int4_plain"] = PLAIN_INT4_VARIANT
        atomic_json(manifest_path, manifest)
    copied = copy_bf16(manifest, args.output, args.smoke)
    atomic_json(args.output / "copy_status.json", {"state": "complete", "copied_artifacts": copied,
                                                     "source": str(SOURCE), "smoke": args.smoke, "time": time.time()})
    ptq_bin = "/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "DEEPCOMPRESSOR_WAN_GATED": "0",
           "PYTHONPATH": str(ROOT / "third_party/deepcompressor"), "TOKENIZERS_PARALLELISM": "false",
           "PATH": ptq_bin + ":" + os.environ["PATH"]}
    command = [sys.executable, str(ROOT / "scripts/rcm_vbench251_worker.py"), "--manifest", str(manifest_path),
               "--output", str(args.output), "--variant", args.variant]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
