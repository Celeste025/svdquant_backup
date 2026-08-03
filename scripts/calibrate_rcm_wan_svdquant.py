#!/usr/bin/env python3
"""Calibrate native rCM-Wan with the original SVDQuant search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from imaginaire.lazy_config import LazyCall as L, instantiate
from rcm.networks.wan2pt1 import WanModel
from rcm.utils.model_utils import init_weights_on_device, load_state_dict

from calibrate_wan_svdquant_large import calibrate_group, open_bfloat16_memmap
from collect_rcm_wan_svdquant_calib import module_groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-path", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=7)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--max-iters", type=int, default=100)
    args = ap.parse_args()
    manifest = json.loads((args.calib_dir / "manifest.json").read_text())
    records = {x["group_index"]: x for x in manifest["groups"]}
    cfg = L(WanModel)(dim=1536, eps=1e-6, ffn_dim=8960, freq_dim=256,
        in_dim=16, model_type="t2v", num_heads=12, num_layers=30,
        out_dim=16, text_len=512)
    with init_weights_on_device():
        net = instantiate(cfg).eval()
    raw = load_state_dict(str(args.dit_path))
    net.load_state_dict({k.removeprefix("net."): v for k, v in raw.items()},
                        strict=False, assign=True)
    del raw
    modules = dict(net.named_modules())
    state, reports = {}, []
    selected = [(i, n) for i, n in enumerate(module_groups())
                if i % args.num_shards == args.shard_index]
    for j, (i, names) in enumerate(selected, 1):
        print(f"[shard {args.shard_index} {j}/{len(selected)}] {i}: {'+'.join(names)}",
              flush=True)
        inputs = open_bfloat16_memmap(args.calib_dir, records[i]).cuda()
        weights = [modules[n].weight.detach().cpu() for n in names]
        part, report = calibrate_group(
            names, weights, inputs, batch_size=args.eval_batch_size,
            group_size=args.group_size, rank=args.rank, max_iters=args.max_iters,
            search_niter=1, final_niter=4, quantize_activation=True,
            weight_bits=4, device=torch.device("cuda"))
        state.update(part)
        report["global_group_index"] = i
        reports.append(report)
        del inputs, weights, part
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    torch.save({"format": "rcm-wan-svdquant-calibrated-v1",
                "group_size": args.group_size, "rank": args.rank,
                "weight_bits": 4, "activation_mode": "w4a4", "state": state},
               args.output_dir / f"{stem}.pt")
    (args.output_dir / f"{stem}.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
