#!/usr/bin/env python3
"""Generate one new real-NVFP4 rCM SVDQuant checkpoint on the fixed VBench-51 set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/samples/rcm_int4_svdquant_vbench51_seed0_480p77f_4step"
SETTINGS = {"seed": 0, "height": 480, "width": 832, "frames": 77, "steps": 4,
            "sigma_max": 80.0, "guidance": 0.0, "fps": 16}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(output: Path, checkpoint: Path, rank: int, grid: int) -> dict:
    source = json.loads((SOURCE / "manifest.json").read_text())
    cases = source.get("cases")
    if source.get("settings") != SETTINGS or not isinstance(cases, list) or len(cases) != 51:
        raise RuntimeError("source is not the expected fixed rCM VBench-51 manifest")
    doc = {"schema_version": 1, "created_at": time.time(), "source_manifest": str(SOURCE / "manifest.json"),
           "settings": SETTINGS, "sampling": source.get("sampling"),
           "variants": {"bf16": {"source": str(SOURCE), "action": "copy-existing-sha256-verified"},
                        "nvfp4_svdquant": {"checkpoint": str(checkpoint), "format": "real NVFP4 E2M1 W4A4 SVDQuant",
                                             "weight_group_size": 16, "activation_group_size": 16,
                                             "rank": rank, "smooth_grids": grid, "lowrank_iters": 100}},
           "cases": cases}
    atomic_json(output / "manifest.json", doc)
    return doc


def copy_bf16(doc: dict, output: Path, smoke: bool) -> None:
    records = []
    for case in doc["cases"][:1] if smoke else doc["cases"]:
        source = SOURCE / "cases" / case["case_id"]
        target = output / "cases" / case["case_id"]
        for suffix in (".mp4", ".json"):
            src, dst = source / f"bf16{suffix}", target / f"bf16{suffix}"
            if not src.is_file():
                raise RuntimeError(f"missing required BF16 artifact: {src}")
            source_digest = digest(src)
            if dst.exists() and digest(dst) != source_digest:
                raise RuntimeError(f"existing BF16 artifact differs from source: {dst}")
            if not dst.exists():
                target.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            records.append({"file": str(dst.relative_to(output)), "sha256": source_digest})
    atomic_json(output / "copy_status.json", {"state": "complete", "source": str(SOURCE), "smoke": smoke,
                                               "artifacts": records, "time": time.time()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rank", type=int, choices=(32, 64), required=True)
    p.add_argument("--grid", type=int, required=True)
    p.add_argument("--gpu", default="4")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not (args.checkpoint / "manifest.json").is_file() or not (args.checkpoint / "model.pt").is_file():
        raise RuntimeError(f"checkpoint is incomplete: {args.checkpoint}")
    saved = json.loads((args.checkpoint / "manifest.json").read_text())
    spec = saved.get("svdquant", {})
    if saved.get("state") != "complete" or spec.get("rank") != args.rank or spec.get("smooth_grids") != args.grid:
        raise RuntimeError(f"checkpoint manifest does not match rank={args.rank}, grid={args.grid}")
    manifest_path = args.output / "manifest.json"
    doc = json.loads(manifest_path.read_text()) if manifest_path.is_file() else build_manifest(args.output, args.checkpoint, args.rank, args.grid)
    if doc.get("settings") != SETTINGS or len(doc.get("cases", [])) != 51 or doc["variants"]["nvfp4_svdquant"].get("checkpoint") != str(args.checkpoint):
        raise RuntimeError("existing output manifest does not match requested contract")
    copy_bf16(doc, args.output, args.smoke)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "DEEPCOMPRESSOR_WAN_GATED": "0",
           "PYTHONPATH": str(ROOT / "third_party/deepcompressor"), "TOKENIZERS_PARALLELISM": "false",
           "PATH": "/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin:" + os.environ["PATH"]}
    cmd = ["/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python", str(ROOT / "scripts/rcm_vbench251_worker.py"),
           "--manifest", str(manifest_path), "--output", str(args.output), "--variant", "nvfp4_svdquant",
           "--svdquant-checkpoint", str(args.checkpoint)]
    if args.smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
