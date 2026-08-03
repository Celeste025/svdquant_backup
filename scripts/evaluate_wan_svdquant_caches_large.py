#!/usr/bin/env python3
"""Evaluate calibrated Wan SVDQuant caches on the same large memmap dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel

from calibrate_wan_svdquant import module_groups
from calibrate_wan_svdquant_large import open_bfloat16_memmap, output_nmse


DEFAULT_MODEL = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--calib-dir", type=Path, default=Path("outputs/wan_svdquant_calib_large")
    )
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--group-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.cache) != len(args.label):
        raise ValueError("--cache and --label counts must match")
    manifest = json.loads((args.calib_dir / "manifest.json").read_text())
    records = {record["group_index"]: record for record in manifest["groups"]}
    caches = {
        label: torch.load(path, map_location="cpu", weights_only=False)["state"]
        for label, path in zip(args.label, args.cache, strict=True)
    }
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    modules = dict(transformer.named_modules())
    device = torch.device("cuda")
    reports = []
    for group_index, names in enumerate(module_groups()):
        if group_index % args.num_shards != args.shard_index:
            continue
        record = records[group_index]
        inputs = open_bfloat16_memmap(args.calib_dir, record).to(
            device=device, dtype=torch.bfloat16
        )
        weight = torch.cat(
            [modules[name].weight.detach().to(device=device, dtype=torch.float32) for name in names]
        )
        result = {
            "global_group_index": group_index,
            "names": names,
            "num_input_tokens": inputs.shape[0],
        }
        for label, state in caches.items():
            entries = [state[name] for name in names]
            smooth = entries[0]["smooth"].to(device=device, dtype=torch.float32)
            qweight = torch.cat(
                [entry["qweight"].to(device=device, dtype=torch.float32) for entry in entries]
            )
            up = torch.cat(
                [entry["up"].to(device=device, dtype=torch.float32) for entry in entries]
            )
            down = entries[0]["down"].to(device=device, dtype=torch.float32)
            result[f"{label}_nmse"] = output_nmse(
                inputs,
                weight,
                smooth,
                qweight,
                up,
                down,
                batch_size=args.eval_batch_size,
                group_size=args.group_size,
                unsigned_activation=bool(entries[0]["unsigned_activation"].item()),
                input_shift=float(entries[0]["input_shift"].item()),
                device=device,
            )
        reports.append(result)
        print(
            f"[{args.shard_index}] group {group_index:03d} "
            + " ".join(f"{label}={result[f'{label}_nmse']:.6g}" for label in caches),
            flush=True,
        )
        del inputs, weight
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"eval_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    output.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
