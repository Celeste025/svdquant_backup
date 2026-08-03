#!/usr/bin/env python3
"""Split the official QDiff prompt set into deterministic GPU shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=128)
    args = parser.parse_args()

    prompts = yaml.safe_load(args.input.read_text())
    selected = list(prompts.items())[: args.num_samples]
    assert len(selected) == args.num_samples
    assert args.num_samples % args.num_shards == 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_size = args.num_samples // args.num_shards
    for shard in range(args.num_shards):
        begin = shard * shard_size
        shard_prompts = dict(selected[begin : begin + shard_size])
        path = args.output_dir / f"qdiff_shard{shard}.yaml"
        path.write_text(yaml.safe_dump(shard_prompts, sort_keys=False))
        print(f"{path}: {len(shard_prompts)} prompts")


if __name__ == "__main__":
    main()
