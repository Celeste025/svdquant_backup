#!/usr/bin/env python3
"""Compare W_s against W4 residual + rank-32 reconstruction at selected layers."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import torch


ROOT = Path("outputs/quant_error_diagnosis")
WAN_MODEL = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
)
WAN_CACHE = Path("outputs/wan_svdquant_calib_large/svdquant_large_calibrated.pt")
FLUX_CACHE = Path(
    "/data/home/jinqiwen/.cache/huggingface/hub/"
    "models--mit-han-lab--svdq-int4-flux.1-dev/snapshots"
)


def tensor_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict:
    reference = reference.float()
    estimate = estimate.float()
    error = estimate - reference
    power = reference.square().mean().clamp_min(1e-20)
    return {
        "mse": float(error.square().mean()),
        "nmse": float(error.square().mean() / power),
        "nrmse": float((error.square().mean() / power).sqrt()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.flatten(), estimate.flatten(), dim=0
            )
        ),
        "max_abs_error": float(error.abs().max()),
    }


def wan_specs():
    result = {}
    for block in (0, 15, 29):
        p = f"blocks.{block}"
        for kind, suffix in {
            "attention_q": "attn1.to_q",
            "attention_out": "attn1.to_out.0",
            "ffn_up": "ffn.net.0.proj",
            "ffn_down": "ffn.net.2",
        }.items():
            result[f"{kind}_b{block}"] = f"{p}.{suffix}"
    return result


def analyze_wan() -> list[dict]:
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        str(WAN_MODEL), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    modules = dict(transformer.named_modules())
    payload = torch.load(WAN_CACHE, map_location="cpu", weights_only=False)
    reports = []
    for key, name in wan_specs().items():
        state = payload["state"][name]
        weight = modules[name].weight.detach().float()
        smooth = state["smooth"].float()
        ws = weight * smooth.unsqueeze(0)
        lowrank = state["up"].float() @ state["down"].float()
        w4_only = state["qweight"].float()
        reconstructed = w4_only + lowrank
        report = {
            "model": "wan",
            "key": key,
            "module": name,
            "shape": list(weight.shape),
            "w4_residual_only_vs_ws": tensor_metrics(ws, w4_only),
            "w4_plus_rank32_vs_ws": tensor_metrics(ws, reconstructed),
            "lowrank_energy_fraction": float(
                lowrank.square().sum() / ws.square().sum().clamp_min(1e-20)
            ),
        }
        reports.append(report)
        print(key, report["w4_plus_rank32_vs_ws"]["nmse"], flush=True)
    del transformer
    return reports


def unpack_weight(packed: torch.Tensor, n: int, k: int):
    sys.path.insert(0, str(Path("third_party/deepcompressor").resolve()))
    from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    words = packed.contiguous().view(torch.int32)
    shape = (
        n // packer.mem_n,
        k // packer.mem_k,
        packer.num_k_packs,
        packer.num_n_packs,
        packer.num_n_lanes,
        packer.num_k_lanes,
        packer.n_pack_size,
        packer.k_pack_size,
        packer.reg_n,
    )
    words = words.view(*shape)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    values = ((words.unsqueeze(-1) >> shifts) & 0xF)
    values = torch.where(values >= 8, values - 16, values)
    values = values.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    return values.view(n, k).float()


def unpack_scale(packed: torch.Tensor, n: int, groups: int):
    sys.path.insert(0, str(Path("third_party/deepcompressor").resolve()))
    from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker

    p = NunchakuWeightPacker(bits=4)
    s_pack = min(max(p.warp_n // p.num_lanes, 2), 8)
    s_lanes = min(p.num_lanes, p.warp_n // s_pack)
    s_packs = p.warp_n // (s_pack * s_lanes)
    warp_s = s_packs * s_lanes * s_pack
    value = packed.contiguous().view(
        n // warp_s, groups, s_packs, s_lanes // 4, 4, s_pack // 2, 2
    )
    value = value.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    return value.view(n, groups).float()


def flux_specs():
    result = {}
    for block in (0, 9, 18):
        p = f"transformer_blocks.{block}"
        result.update(
            {
                f"attention_qkv_b{block}": (
                    f"{p}.qkv_proj",
                    (f"{p}.attn.to_q", f"{p}.attn.to_k", f"{p}.attn.to_v"),
                ),
                f"attention_out_b{block}": (
                    f"{p}.out_proj",
                    (f"{p}.attn.to_out.0",),
                ),
                f"ffn_up_b{block}": (f"{p}.mlp_fc1", (f"{p}.ff.net.0.proj",)),
                f"ffn_down_b{block}": (f"{p}.mlp_fc2", (f"{p}.ff.net.2",)),
            }
        )
    return result


def analyze_flux() -> list[dict]:
    from diffusers import FluxTransformer2DModel
    from safetensors import safe_open

    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    modules = dict(transformer.named_modules())
    paths = list(FLUX_CACHE.glob("*/transformer_blocks.safetensors"))
    reports = []
    with safe_open(paths[0], framework="pt", device="cpu") as handle:
        for key, (quant_name, module_names) in flux_specs().items():
            weight = torch.cat(
                [modules[name].weight.detach().float() for name in module_names], dim=0
            )
            n, k = weight.shape
            smooth_packed = handle.get_tensor(f"{quant_name}.smooth_orig")
            smooth = unpack_scale(smooth_packed, k, 1).flatten().float()
            ws = weight * smooth.unsqueeze(0)
            qint = unpack_weight(handle.get_tensor(f"{quant_name}.qweight"), n, k)
            scales = unpack_scale(
                handle.get_tensor(f"{quant_name}.wscales"), n, k // 64
            )
            w4 = (
                qint.reshape(n, k // 64, 64)
                * scales.unsqueeze(-1)
            ).reshape(n, k)
            sys.path.insert(0, str(Path("third_party/deepcompressor").resolve()))
            from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker

            packer = NunchakuWeightPacker(bits=4)
            down_packed = handle.get_tensor(f"{quant_name}.lora_down")
            up_packed = handle.get_tensor(f"{quant_name}.lora_up")
            down = packer.unpack_lowrank_weight(down_packed, down=True)[:, :k].float()
            up = packer.unpack_lowrank_weight(up_packed, down=False)[:n, :].float()
            # Unpacked down is [R,K], while unpacked up is [N,R].
            # Nunchaku stores down/smooth so that the BF16 branch consumes the
            # original activation. Convert it back to the smoothed-weight domain.
            lowrank = up @ (down * smooth.unsqueeze(0))
            reconstructed = w4 + lowrank
            report = {
                "model": "flux",
                "key": key,
                "module": quant_name,
                "shape": [n, k],
                "w4_residual_only_vs_ws": tensor_metrics(ws, w4),
                "w4_plus_rank32_vs_ws": tensor_metrics(ws, reconstructed),
                "lowrank_energy_fraction": float(
                    lowrank.square().sum() / ws.square().sum().clamp_min(1e-20)
                ),
            }
            reports.append(report)
            print(key, report["w4_plus_rank32_vs_ws"]["nmse"], flush=True)
    del transformer
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan", "flux", "both"), default="both")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.model in ("wan", "both"):
        wan = analyze_wan()
        (ROOT / "wan_weight_reconstruction.json").write_text(
            json.dumps(wan, indent=2), encoding="utf-8"
        )
    if args.model in ("flux", "both"):
        flux = analyze_flux()
        (ROOT / "flux_weight_reconstruction.json").write_text(
            json.dumps(flux, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
