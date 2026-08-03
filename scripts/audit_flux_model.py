#!/usr/bin/env python3
"""Audit cached BF16 and Nunchaku INT4 FLUX.1-schnell checkpoints."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open


HF_CACHE = Path("/data/home/jinqiwen/.cache/huggingface/hub")
BF16_REPO = HF_CACHE / "models--frankjoshua--FLUX.1-schnell" / "snapshots"
INT4_REPO = HF_CACHE / "models--mit-han-lab--svdq-int4-flux.1-schnell" / "snapshots"


def only_snapshot(root: Path) -> Path:
    snapshots = [path for path in root.iterdir() if path.is_dir()]
    if len(snapshots) != 1:
        raise RuntimeError(f"expected one snapshot under {root}, got {snapshots}")
    return snapshots[0]


def tensor_inventory(paths: list[Path]) -> tuple[list[dict], Counter]:
    tensors = []
    shapes = Counter()
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_slice = handle.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                dtype = str(tensor_slice.get_dtype())
                numel = 1
                for dimension in shape:
                    numel *= dimension
                tensors.append(
                    {
                        "key": key,
                        "shape": shape,
                        "dtype": dtype,
                        "numel": numel,
                        "file": path.name,
                    }
                )
                if len(shape) == 2:
                    shapes[shape] += 1
    return tensors, shapes


def main() -> None:
    bf16_snapshot = only_snapshot(BF16_REPO)
    int4_snapshot = only_snapshot(INT4_REPO)
    bf16_files = sorted((bf16_snapshot / "transformer").glob("*.safetensors"))
    bf16_files = [path for path in bf16_files if "index" not in path.name]
    int4_files = sorted(int4_snapshot.glob("*.safetensors"))
    bf16, bf16_shapes = tensor_inventory(bf16_files)
    int4, int4_shapes = tensor_inventory(int4_files)
    config = json.loads((bf16_snapshot / "transformer" / "config.json").read_text())
    report = {
        "bf16_snapshot": str(bf16_snapshot),
        "int4_snapshot": str(int4_snapshot),
        "config": config,
        "bf16_transformer_num_tensors": len(bf16),
        "bf16_transformer_parameters": sum(item["numel"] for item in bf16),
        "bf16_2d_weight_shapes": [
            {"shape": shape, "count": count}
            for shape, count in bf16_shapes.most_common()
        ],
        "int4_num_tensors": len(int4),
        "int4_dtype_counts": dict(Counter(item["dtype"] for item in int4)),
        "int4_example_tensors": int4[:50],
        "int4_2d_shapes": [
            {"shape": shape, "count": count}
            for shape, count in int4_shapes.most_common(30)
        ],
    }
    output = Path("outputs/flux_official_reproduction/model_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
