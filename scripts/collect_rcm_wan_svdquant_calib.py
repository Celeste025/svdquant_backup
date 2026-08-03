#!/usr/bin/env python3
"""Collect native rCM-Wan linear inputs for SVDQuant calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import repeat

from imaginaire.lazy_config import LazyCall as L, instantiate
from rcm.networks.wan2pt1 import WanModel
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding


def module_groups() -> list[list[str]]:
    groups = []
    for block in range(30):
        # WanModel enables selective activation checkpointing at construction;
        # named_modules therefore exposes each block below this wrapper.
        p = f"blocks.{block}._checkpoint_wrapped_module"
        groups.extend([
            [f"{p}.self_attn.q", f"{p}.self_attn.k", f"{p}.self_attn.v"],
            [f"{p}.self_attn.o"],
            [f"{p}.cross_attn.q"],
            [f"{p}.cross_attn.k", f"{p}.cross_attn.v"],
            [f"{p}.cross_attn.o"],
            [f"{p}.ffn.0"],
            [f"{p}.ffn.2"],
        ])
    return groups


class Sampler:
    def __init__(self, root: Path, index: int, names: list[str], calls: int, tokens: int):
        self.path = root / f"group_{index:03d}.bf16.mmap"
        self.index, self.names, self.calls, self.tokens = index, names, calls, tokens
        self.array = None
        self.offset = self.dim = self.rows = 0

    def __call__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        count = min(self.tokens, x.shape[0])
        if self.array is None:
            self.rows, self.dim = count, x.shape[-1]
            self.array = np.memmap(
                self.path, dtype=np.uint16, mode="w+",
                shape=(self.calls * count, self.dim),
            )
        pos = torch.linspace(0, x.shape[0] - 1, count, device=x.device).long()
        raw = x.index_select(0, pos).to("cpu", torch.bfloat16).contiguous()
        self.array[self.offset:self.offset + count] = raw.view(torch.uint16).numpy()
        self.offset += count

    def finish(self) -> dict:
        expected = self.calls * self.rows
        if self.array is None or self.offset != expected:
            raise RuntimeError(f"group {self.index}: rows={self.offset}, expected={expected}")
        self.array.flush()
        return {
            "group_index": self.index, "names": self.names, "path": self.path.name,
            "shape": [expected, self.dim], "rows_per_call": self.rows,
            "selected_calls": self.calls, "storage_dtype": "bfloat16_raw_uint16",
            "bytes": self.path.stat().st_size,
        }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-path", type=Path, required=True)
    ap.add_argument("--t5-path", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tokens-per-call", type=int, default=2048)
    ap.add_argument("--base-seed", type=int, default=44)
    args = ap.parse_args()
    items = json.loads(args.prompts.read_text())[:8]
    prompts = [x.get("prompt_en", x.get("prompt", x)) if isinstance(x, dict) else x for x in items]
    if len(prompts) != 8:
        raise RuntimeError("exactly 8 prompts are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # T5 first, then release it before loading DiT: required on a 24 GB card.
    embeddings = get_umt5_embedding(
        checkpoint_path=str(args.t5_path), prompts=prompts
    ).to(torch.bfloat16).cpu()
    clear_umt5_memory()

    cfg = L(WanModel)(dim=1536, eps=1e-6, ffn_dim=8960, freq_dim=256,
        in_dim=16, model_type="t2v", num_heads=12, num_layers=30,
        out_dim=16, text_len=512)
    with init_weights_on_device():
        net = instantiate(cfg).eval()
    state = load_state_dict(str(args.dit_path))
    state = {k.removeprefix("net."): v for k, v in state.items()}
    net.load_state_dict(state, strict=False, assign=True)
    del state
    net.to(device="cuda", dtype=torch.bfloat16)

    groups = module_groups()
    modules = dict(net.named_modules())
    samplers, handles = [], []
    for i, names in enumerate(groups):
        sampler = Sampler(args.output_dir, i, names, 8 * 4, args.tokens_per_call)
        samplers.append(sampler)
        handles.append(modules[names[0]].register_forward_pre_hook(sampler))

    t = torch.tensor([math.atan(80), 1.5, 1.4, 1.0, 0.0],
                     dtype=torch.float64, device="cuda")
    t = torch.sin(t) / (torch.cos(t) + torch.sin(t))
    runs = []
    for pi, (prompt, emb) in enumerate(zip(prompts, embeddings, strict=True)):
        seed = args.base_seed + pi
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(1, 16, 21, 60, 104, generator=gen,
                        device="cuda", dtype=torch.float32).double() * t[0]
        condition = {"crossattn_emb": repeat(emb[None].cuda(), "b l d -> b l d")}
        ones = torch.ones(1, 1, device="cuda", dtype=torch.float64)
        print(f"[{pi + 1}/8 seed={seed}] {prompt}", flush=True)
        for cur, nxt in zip(t[:-1], t[1:]):
            pred = net(x_B_C_T_H_W=x.to(torch.bfloat16),
                       timesteps_B_T=(cur.float() * ones * 1000).to(torch.bfloat16),
                       **condition).double()
            x = (1 - nxt) * (x - cur * pred) + nxt * torch.randn(
                x.shape, generator=gen, device="cuda", dtype=torch.float32)
        runs.append({"index": pi, "seed": seed, "prompt": prompt})

    for h in handles:
        h.remove()
    records = [s.finish() for s in samplers]
    manifest = {
        "format": "rcm-wan-svdquant-memmap-v1", "model": str(args.dit_path),
        "prompts": runs, "num_inference_steps": 4,
        "selected_step_indices": [0, 1, 2, 3],
        "tokens_per_spatiotemporal_call": args.tokens_per_call,
        "num_groups": len(records), "groups": records,
        "total_bytes": sum(x["bytes"] for x in records),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {manifest['total_bytes'] / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
