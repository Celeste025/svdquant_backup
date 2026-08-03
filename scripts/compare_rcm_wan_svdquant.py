#!/usr/bin/env python3
"""Paired BF16/fake-W4A4 rCM-Wan inference with error diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from imaginaire.lazy_config import LazyCall as L, instantiate
from imaginaire.utils.io import save_image_or_video

from rcm.networks.wan2pt1 import WanModel
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding


def q_s4(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    y = x.float().reshape(*shape[:-1], shape[-1] // group, group)
    s = y.abs().amax(-1, keepdim=True).clamp_min_(1e-6).div_(7)
    return y.div(s).round_().clamp_(-7, 7).mul_(s).reshape(shape).to(x.dtype)


def q_u4(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    y = x.float().reshape(*shape[:-1], shape[-1] // group, group)
    s = y.amax(-1, keepdim=True).clamp_min_(1e-6).div_(15)
    return y.div(s).round_().clamp_(0, 15).mul_(s).reshape(shape).to(x.dtype)


class FakeLinear(nn.Module):
    def __init__(self, linear: nn.Linear, state: dict, group: int):
        super().__init__()
        self.in_features, self.out_features, self.group = (
            linear.in_features, linear.out_features, group)
        for key in ("smooth", "qweight", "down", "up"):
            self.register_buffer(key, state[key].to("cpu", linear.weight.dtype))
        shift = state.get("input_shift", torch.tensor(0.0))
        unsigned = state.get("unsigned_activation", torch.tensor(False))
        self.register_buffer("input_shift", shift.to("cpu", linear.weight.dtype))
        self.register_buffer("unsigned_activation", unsigned.to("cpu", torch.bool))
        bias = (torch.zeros(linear.out_features, dtype=torch.float64)
                if linear.bias is None else linear.bias.detach().cpu().double())
        if float(shift):
            bias -= linear.weight.detach().cpu().double().sum(1) * float(shift)
        self.register_buffer("bias", bias.to(linear.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = (x + self.input_shift) / self.smooth
        xq = q_u4(xs, self.group) if bool(self.unsigned_activation) else q_s4(xs, self.group)
        return F.linear(xq, self.qweight, self.bias) + F.linear(F.linear(xs, self.down), self.up)


def set_submodule(root: nn.Module, name: str, value: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent), child, value)


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).square().sum()
                 / b.double().square().sum().clamp_min(1e-20) * 100)


def build_model(path: Path) -> WanModel:
    cfg = L(WanModel)(dim=1536, eps=1e-6, ffn_dim=8960, freq_dim=256,
        in_dim=16, model_type="t2v", num_heads=12, num_layers=30,
        out_dim=16, text_len=512)
    with init_weights_on_device():
        net = instantiate(cfg).eval()
    raw = load_state_dict(str(path))
    net.load_state_dict({k.removeprefix("net."): v for k, v in raw.items()},
                        strict=False, assign=True)
    return net


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-path", type=Path, required=True)
    ap.add_argument("--vae-path", type=Path, required=True)
    ap.add_argument("--t5-path", type=Path, required=True)
    ap.add_argument("--calibrated-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt", default="An astronaut feeding ducks on a sunny afternoon, reflection from the water.")
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emb = get_umt5_embedding(checkpoint_path=str(args.t5_path),
                             prompts=args.prompt).to(torch.bfloat16).cpu()
    clear_umt5_memory()
    net = build_model(args.dit_path).to("cuda", torch.bfloat16)
    condition = {"crossattn_emb": emb.cuda()}
    t = torch.tensor([math.atan(80), 1.5, 1.4, 1.0, 0.0],
                     dtype=torch.float64, device="cuda")
    t = torch.sin(t) / (torch.cos(t) + torch.sin(t))
    ones = torch.ones(1, 1, dtype=torch.float64, device="cuda")

    mode = {"name": "bf16", "first": False}
    bf_blocks: dict[int, torch.Tensor] = {}
    block_nmse: dict[int, float] = {}
    bf_sublayers: dict[str, torch.Tensor] = {}
    sublayer_nmse: dict[str, float] = {}
    handles = []
    for i, block in enumerate(net.blocks):
        def hook(module, inputs, output, index=i):
            if not mode["first"]:
                return
            if mode["name"] == "bf16":
                bf_blocks[index] = output.detach().cpu()
            else:
                block_nmse[index] = nmse(output.detach().cpu(), bf_blocks[index])
        handles.append(block.register_forward_hook(hook))
    diagnostic_names = []
    for bi in (21, 22, 23):
        p = f"blocks.{bi}._checkpoint_wrapped_module"
        diagnostic_names += [
            f"{p}.self_attn.q", f"{p}.self_attn.k", f"{p}.self_attn.v",
            f"{p}.self_attn.o", f"{p}.cross_attn.q", f"{p}.cross_attn.k",
            f"{p}.cross_attn.v", f"{p}.cross_attn.o", f"{p}.ffn.0",
            f"{p}.ffn.2",
        ]
    modules = dict(net.named_modules())
    linear_handles = []
    for name in diagnostic_names:
        def save_hook(module, inputs, output, key=name):
            if mode["first"]:
                bf_sublayers[key] = output.detach().cpu()
        linear_handles.append(modules[name].register_forward_hook(save_hook))

    def sample(label: str):
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        x = torch.randn(1, 16, 21, 60, 104, generator=gen,
                        device="cuda", dtype=torch.float32).double() * t[0]
        preds, latents = [], []
        mode["name"] = label
        for si, (cur, nxt) in enumerate(zip(t[:-1], t[1:])):
            mode["first"] = si == 0
            pred = net(x_B_C_T_H_W=x.to(torch.bfloat16),
                       timesteps_B_T=(cur.float() * ones * 1000).to(torch.bfloat16),
                       **condition).double()
            mode["first"] = False
            preds.append(pred.cpu())
            x = (1 - nxt) * (x - cur * pred) + nxt * torch.randn(
                x.shape, generator=gen, device="cuda", dtype=torch.float32)
            latents.append(x.cpu())
        return x.float().cpu(), preds, latents

    bf_latent, bf_preds, bf_latents = sample("bf16")
    for h in linear_handles:
        h.remove()
    shards = sorted(args.calibrated_dir.glob("shard_*_of_*.pt"))
    states = {}
    group = 64
    for path in shards:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        group = obj["group_size"]
        states.update(obj["state"])
    if len(states) != 300:
        raise RuntimeError(f"expected 300 calibrated linears, found {len(states)}")
    modules = dict(net.named_modules())
    for name, state in states.items():
        set_submodule(net, name, FakeLinear(modules[name], state, group))
    modules = dict(net.named_modules())
    linear_handles = []
    for name in diagnostic_names:
        def compare_hook(module, inputs, output, key=name):
            if mode["first"]:
                sublayer_nmse[key] = nmse(output.detach().cpu(), bf_sublayers[key])
        linear_handles.append(modules[name].register_forward_hook(compare_hook))
    net.to("cuda", torch.bfloat16)
    q_latent, q_preds, q_latents = sample("w4a4")
    for h in linear_handles:
        h.remove()
    for h in handles:
        h.remove()

    report = {
        "prompt": args.prompt, "seed": args.seed, "steps": 4,
        "group_size": group, "rank": 32,
        "model_output_nmse_percent": [nmse(q, b) for q, b in zip(q_preds, bf_preds)],
        "post_step_latent_nmse_percent": [nmse(q, b) for q, b in zip(q_latents, bf_latents)],
        "first_step_block_nmse_percent": [block_nmse[i] for i in range(30)],
        "first_step_sublayer_nmse_percent": sublayer_nmse,
    }
    (args.output_dir / "paired_metrics.json").write_text(json.dumps(report, indent=2))
    torch.save({"bf16": bf_latent, "w4a4": q_latent},
               args.output_dir / "final_latents.pt")

    net.cpu()
    del net, bf_preds, q_preds, bf_latents, q_latents, bf_blocks, bf_sublayers
    torch.cuda.empty_cache()
    vae = Wan2pt1VAEInterface(vae_pth=str(args.vae_path))
    for label, latent in (("bf16", bf_latent), ("w4a4", q_latent)):
        video = vae.decode(latent.cuda())
        video = (1 + video.float().cpu().clamp(-1, 1)) / 2
        save_image_or_video(rearrange(video, "b c t h w -> c t h (b w)"),
                            str(args.output_dir / f"{label}_seed44.mp4"), fps=16)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
