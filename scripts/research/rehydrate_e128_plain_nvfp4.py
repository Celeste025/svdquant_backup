#!/usr/bin/env python3
"""Recreate E014's deleted plain-H3 packets, checking original tensor hashes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research")]

import torch
from safetensors import safe_open

from h3_native_nvfp4 import file_sha256, tensor_sha256
from h3_plain_native_nvfp4 import FORMAT, RECIPE
from h3_nvfp4_zero_sf_compat import pack_activation_fast

SOURCE = Path("/data1/models/svdquant-wjq/research/20261002/E014/plain_export/manifest.json")
DEST = Path("/data1/models/svdquant-wjq/research/20261024/E128_plain_export_rehydrated")
PROGRESS = ROOT / "results/research/E128/rehydration_progress.json"


def publish_progress(value: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temp = PROGRESS.with_suffix(".partial.json")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(PROGRESS)


@torch.inference_mode()
def main() -> None:
    original = json.loads(SOURCE.read_text())
    assert original["status"] == "complete" and original["format"] == FORMAT
    assert original["recipe"] == RECIPE and len(original["layers"]) == 200
    checkpoint = Path(original["source_checkpoint"]["file"])
    source = original["source_checkpoint"]
    assert checkpoint.stat().st_size == source["bytes"]
    assert checkpoint.stat().st_mtime_ns == source["mtime_ns"]
    if DEST.exists():
        raise FileExistsError(f"Refusing existing export directory: {DEST}")
    DEST.mkdir(parents=True)
    rebuilt = copy.deepcopy(original)
    rebuilt["status"] = "running"
    rebuilt["rehydrated_from_manifest"] = str(SOURCE)
    rebuilt["rehydrated_from_manifest_sha256"] = file_sha256(SOURCE)
    rebuilt["rehydration_method"] = "fast H3 native encoder; all original source/packet tensor hashes required"
    rebuilt["export_dir"] = str(DEST)
    rebuilt["layers"] = []
    progress = {"format": "e128-plain-export-rehydration-v1", "status": "running",
                "source": str(SOURCE), "destination": str(DEST), "completed": 0,
                "started_unix": time.time()}
    publish_progress(progress)
    torch.set_num_threads(6)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for index, original_row in enumerate(original["layers"]):
            row = copy.deepcopy(original_row)
            weight = handle.get_tensor(row["key"])
            if tensor_sha256(weight) != row["source_weight_sha256"]:
                raise ValueError(f"Source BF16 weight changed: {row['name']}")
            packet = pack_activation_fast(weight.cuda())
            stats = row["quant_stats"]
            actual = {"packed_sha256": tensor_sha256(packet.packed),
                      "global_sha256": tensor_sha256(packet.global_scale),
                      "swizzled_sf_sha256": tensor_sha256(packet.swizzled_scales)}
            if any(actual[key] != stats[key] for key in actual):
                raise ValueError(f"Repacked NVFP4 tensor differs from E014: {row['name']}: {actual}")
            bias = handle.get_tensor(row["bias_key"]) if row["bias_key"] else None
            if bias is not None and tensor_sha256(bias) != row["bias_sha256"]:
                raise ValueError(f"Source bias changed: {row['name']}")
            payload = {"format": FORMAT, "recipe": RECIPE, "name": row["name"],
                       "shape": row["shape"], "source_weight_sha256": row["source_weight_sha256"],
                       "tensors": {"weight_packed": packet.packed.cpu().contiguous(),
                                   "weight_scales_swizzled": packet.swizzled_scales.cpu().contiguous(),
                                   "weight_global": packet.global_scale.cpu().contiguous(),
                                   "bias": bias}}
            path = DEST / row["file"]
            temp = path.with_suffix(".partial.pt")
            torch.save(payload, temp)
            temp.replace(path)
            row["historical_file_sha256"] = row["file_sha256"]
            row["file_sha256"] = file_sha256(path)
            row["file_bytes"] = path.stat().st_size
            rebuilt["layers"].append(row)
            progress.update(completed=index + 1, last_layer=row["name"])
            publish_progress(progress)
            print(f"Rehydrated {index + 1}/200 {row['name']}", flush=True)
            del weight, packet, payload, bias
    rebuilt["status"] = "complete"
    rebuilt["target_count"] = rebuilt["exact_roundtrip_count"] = 200
    manifest = DEST / "manifest.json"
    manifest.write_text(json.dumps(rebuilt, indent=2) + "\n")
    progress.update(status="complete", manifest=str(manifest),
                    manifest_sha256=file_sha256(manifest), completed_unix=time.time())
    publish_progress(progress)
    print(json.dumps({"status": "complete", "manifest": str(manifest),
                      "sha256": file_sha256(manifest)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
