#!/usr/bin/env python3
"""Build the deterministic rCM VBench-51 subset and dispatch MiniMax-H3 workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VBENCH_ROOT = Path(os.environ.get("VBENCH_ROOT", ROOT / "third_party/ViDiT-Q/eval/video/Vbench/vbench"))
VBENCH = VBENCH_ROOT / "VBench_full_info.json"
OUT = ROOT / "results/samples/minimax_h3_vbench51_seed0_calibshape"
DIMS = ("aesthetic_quality", "scene", "imaging_quality", "overall_consistency", "background_consistency",
        "subject_consistency", "dynamic_degree", "motion_smoothness")
SETTINGS = {"seed": 0, "height": 576, "width": 1024, "frames": 124, "steps": 20,
            "cfg_scale": 1.0, "rand_device": "cpu", "tiled": True, "fps": 24}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def build_manifest(path: Path, state: Path = ROOT / "results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt") -> dict:
    import torch
    state_doc = torch.load(state, map_location="cpu", weights_only=False)
    recipe = state_doc.get("config", {})
    if state_doc.get("format") != "minimax-h3-svdquant-standard-v1" or recipe.get("rank") not in (32, 64):
        raise RuntimeError(f"invalid H3 SVDQuant state: {state}")
    prompts: OrderedDict[str, dict] = OrderedDict()
    for item in json.loads(VBENCH.read_text()):
        dims = [d for d in item["dimension"] if d in DIMS]
        if not dims:
            continue
        record = prompts.setdefault(item["prompt_en"], {"prompt": item["prompt_en"], "dimensions": []})
        record["dimensions"] = sorted(set(record["dimensions"]) | set(dims), key=DIMS.index)
    all_cases = [{"case_id": f"vbench_{i:03d}", **record} for i, record in enumerate(prompts.values())]
    if len(all_cases) != 251:
        raise RuntimeError(f"expected 251 unique prompts, got {len(all_cases)}")
    quotas = {d: round(sum(d in case["dimensions"] for case in all_cases) / 5) for d in DIMS}
    selected, remaining, covered = [], set(range(len(all_cases))), {d: 0 for d in DIMS}
    while len(selected) < 51:
        index = max(remaining, key=lambda i: (
            sum(max(quotas[d] - covered[d], 0) / max(quotas[d], 1) for d in all_cases[i]["dimensions"]), -i))
        selected.append(all_cases[index]); remaining.remove(index)
        for dim in all_cases[index]["dimensions"]:
            covered[dim] += 1
    manifest = {"schema_version": 1, "source": str(VBENCH), "created_at": time.time(), "settings": SETTINGS,
                "sampling": {"method": "deterministic stratified greedy coverage", "source_population": 251,
                             "cases": 51, "dimension_quotas": quotas, "dimension_counts": covered},
                "variants": {"bf16": {"format": "bf16"},
                             "w4a4": {"format": "real-NVFP4 dynamic W4A4", "smoothing": False, "low_rank": False,
                                        "group_size": 16, "activation_element_size": 128},
                             "svdquant": {"format": "real-NVFP4 SVDQuant", "rank": recipe["rank"], "grid": recipe["num_grids"],
                                          "max_lowrank_iters": recipe["max_lowrank_iters"], "state": str(state)}},
                "cases": selected}
    atomic_json(path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--state", type=Path, default=ROOT / "results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt")
    parser.add_argument("--source", type=Path, help="copy paired BF16/W4A4 artifacts from this completed VBench directory")
    parser.add_argument("--variants", nargs="+", choices=("bf16", "w4a4", "svdquant"), default=("bf16", "w4a4", "svdquant"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else build_manifest(manifest_path, args.state)
    if manifest.get("settings") != SETTINGS or len(manifest.get("cases", [])) != 51:
        raise RuntimeError("existing manifest does not match this experiment contract")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "DIFFSYNTH_SKIP_DOWNLOAD": "True",
           "PYTHONPATH": f"{ROOT / 'scripts'}:{os.environ.get('DIFFSYNTH_ROOT', str(ROOT / 'third_party/DiffSynth-Studio'))}",
           "TOKENIZERS_PARALLELISM": "false"}
    if manifest["variants"]["svdquant"].get("state") != str(args.state):
        raise RuntimeError("existing manifest refers to a different SVDQuant state")
    if args.source:
        for case in manifest["cases"]:
            for variant in ("bf16", "w4a4"):
                for suffix in (".mp4", ".json"):
                    source = args.source / "cases" / case["case_id"] / f"{variant}{suffix}"
                    target = args.output / "cases" / case["case_id"] / source.name
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    if target.exists():
                        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                            raise RuntimeError(f"copied baseline differs from source: {target}")
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(source.read_bytes())
    for variant in args.variants:
        command = [sys.executable, str(ROOT / "scripts/minimax_h3_vbench51_worker.py"), "--manifest", str(manifest_path),
                   "--output", str(args.output), "--variant", variant]
        if variant == "svdquant":
            command += ["--state", str(args.state)]
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
