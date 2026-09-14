#!/usr/bin/env python3
"""Small Wan gated-calib ablations on one free GPU.

1) Assert wrap_wan_gated(attn1/ffn) == y * gate, and that missing temb raises.
2) On one gated layer (default block14 attn1.o_proj): compare
   - calib A: minimize ungated OutputsError, then evaluate gated error
   - calib B: minimize gated OutputsError directly
   under the same INT4 weight fake-quant + scale-grid search.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "deepcompressor"))

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

from deepcompressor.app.diffusion.quant.utils import wrap_wan_gated

WAN_PATH = os.environ.get("WAN_MODEL_PATH", "/ssd/2/yuzhibo.yzh_data/Wan2.1-T2V-1.3B-Diffusers")
OUT_DIR = Path(os.environ.get("GATE_ABLATION_OUT", str(DATA_ROOT / "compare" / "wan_gate_calib_ablation")))
BLOCK_ID = int(os.environ.get("GATE_ABLATION_BLOCK", "14"))
NUM_STEPS = int(os.environ.get("GATE_ABLATION_STEPS", "4"))
BITS = 4
PROMPT = "an airplane soaring through a clear blue sky"
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def nmse(pred: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (pred.float() - ref.float()).pow(2).sum()
    denom = ref.float().pow(2).sum().clamp_min(1e-12)
    return float((diff / denom).item())


def mse(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float((pred.float() - ref.float()).pow(2).mean().item())


@torch.inference_mode()
def fake_quant_linear_weight(weight: torch.Tensor, bits: int, scale_mult: float) -> torch.Tensor:
    """Per-out-channel symmetric fake quant; scale_mult stretches AbsMax range."""
    qmax = (1 << (bits - 1)) - 1
    # weight: [out, in]
    amax = weight.detach().float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) * float(scale_mult)
    scale = amax / qmax
    q = (weight.float() / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).to(dtype=weight.dtype)


@torch.inference_mode()
def forward_o_proj(module: nn.Linear, xs: list[torch.Tensor], weight: torch.Tensor) -> list[torch.Tensor]:
    bias = module.bias
    outs = []
    for x in xs:
        y = torch.nn.functional.linear(x, weight, bias)
        outs.append(y)
    return outs


def part1_wrap_assert(block, device: torch.device) -> dict:
    print("=== Part1: wrap_wan_gated vs hand gate ===", flush=True)
    B, T, C = 2, 32, block.attn1.to_q.in_features
    hs = torch.randn(B, T, C, device=device, dtype=torch.bfloat16)
    temb = torch.randn(B, 6, C, device=device, dtype=torch.float32)

    gate_msa = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[2]
    c_gate = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[5]

    y_attn = block.attn1(hidden_states=hs)
    wrapped = wrap_wan_gated(block.attn1, block, "msa")
    # temb must be present
    try:
        wrapped(hidden_states=hs)
        raise AssertionError("expected KeyError without temb")
    except KeyError as e:
        assert "temb" in str(e)
        print("  missing-temb KeyError: OK", flush=True)

    y_wrap = wrapped(hidden_states=hs, temb=temb)
    assert "temb" not in {}, "sanity"
    # Confirm kwargs contract: wrapper consumes temb (caller must pass it)
    ref = y_attn * gate_msa.type_as(y_attn)
    err_attn = (y_wrap - ref).float().abs().max().item()

    y_ffn = block.ffn(hs)
    y_wrap_f = wrap_wan_gated(block.ffn, block, "ffn")(hs, temb=temb)
    ref_f = y_ffn * c_gate.type_as(y_ffn)
    err_ffn = (y_wrap_f - ref_f).float().abs().max().item()

    # also o_proj path
    x_o = torch.randn(B, T, block.attn1.to_out[0].in_features, device=device, dtype=torch.bfloat16)
    y_o = block.attn1.to_out[0](x_o)
    y_o_w = wrap_wan_gated(block.attn1.to_out[0], block, "msa")(x_o, temb=temb)
    err_o = (y_o_w - y_o * gate_msa.type_as(y_o)).float().abs().max().item()

    print(f"  attn1  max_abs_err={err_attn:.3e}", flush=True)
    print(f"  ffn    max_abs_err={err_ffn:.3e}", flush=True)
    print(f"  o_proj max_abs_err={err_o:.3e}", flush=True)
    ok = max(err_attn, err_ffn, err_o) < 1e-2
    print(f"  Part1: {'OK' if ok else 'FAIL'}", flush=True)
    return {
        "err_attn": err_attn,
        "err_ffn": err_ffn,
        "err_o_proj": err_o,
        "temb_required": True,
        "ok": ok,
    }


@torch.inference_mode()
def capture_block_o_proj_inputs(pipe, block_id: int, device: torch.device):
    """Short denoising run; capture attn1.to_out[0] inputs and matching temb for one block."""
    block = pipe.transformer.blocks[block_id]
    o_proj = block.attn1.to_out[0]
    xs: list[torch.Tensor] = []
    tembs: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []

    # stash latest temb from block.forward
    latest = {"temb": None}

    orig_block = block.forward
    orig_o = o_proj.forward

    def block_forward(hidden_states, encoder_hidden_states, temb, rotary_emb):
        latest["temb"] = temb.detach()
        return orig_block(hidden_states, encoder_hidden_states, temb, rotary_emb)

    def o_forward(x):
        temb = latest["temb"]
        assert temb is not None
        gate = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[2]
        xs.append(x.detach().float().cpu())
        tembs.append(temb.detach().float().cpu())
        gates.append(gate.detach().float().cpu())
        return orig_o(x)

    block.forward = block_forward
    o_proj.forward = o_forward

    gen = torch.Generator(device=device).manual_seed(42)
    _ = pipe(
        PROMPT,
        negative_prompt=NEGATIVE,
        height=480,
        width=832,
        num_frames=17,
        num_inference_steps=NUM_STEPS,
        guidance_scale=6.0,
        generator=gen,
    )

    block.forward = orig_block
    o_proj.forward = orig_o
    print(f"  captured {len(xs)} o_proj forwards (CFG doubles steps)", flush=True)
    return xs, tembs, gates


@torch.inference_mode()
def part2_calib_compare(block, xs, tembs, gates, device: torch.device) -> dict:
    print(f"=== Part2: ungated vs gated calib on block{BLOCK_ID} attn1.o_proj ===", flush=True)
    o_proj = block.attn1.to_out[0]
    w_fp = o_proj.weight.detach()

    xs_d = [x.to(device=device, dtype=torch.bfloat16) for x in xs]
    gates_d = [g.to(device=device, dtype=torch.bfloat16) for g in gates]

    y_fp = forward_o_proj(o_proj, xs_d, w_fp)
    y_fp_g = [y * g for y, g in zip(y_fp, gates_d)]

    # Mean |gate| per out-channel across captures → split high/low gate channels.
    g_mean = torch.stack([g.float().abs().mean(dim=tuple(range(g.ndim - 1))) for g in gates_d], dim=0).mean(0)
    # g_mean: [C]
    thr = float(g_mean.median().item())
    high_mask = g_mean >= thr  # [C]
    low_mask = ~high_mask
    print(
        f"  channels: high|g|={int(high_mask.sum())} low|g|={int(low_mask.sum())} median|g|={thr:.4f}",
        flush=True,
    )

    def fake_quant_grouped(weight: torch.Tensor, alpha_high: float, alpha_low: float) -> torch.Tensor:
        """Per-out-channel INT4 AbsMax, with different range multipliers for high/low-|gate| rows."""
        qmax = (1 << (BITS - 1)) - 1
        w = weight.detach().float()
        amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)  # [out,1]
        mult = torch.ones_like(amax)
        mult[high_mask] = float(alpha_high)
        mult[low_mask] = float(alpha_low)
        scale = (amax * mult) / qmax
        q = (w / scale).round().clamp(-qmax - 1, qmax)
        return (q * scale).to(dtype=weight.dtype)

    alphas = [0.6, 0.8, 1.0, 1.25, 1.6]
    rows = []
    for ah in alphas:
        for al in alphas:
            w_q = fake_quant_grouped(w_fp, ah, al)
            y_q = forward_o_proj(o_proj, xs_d, w_q)
            y_q_g = [y * g for y, g in zip(y_q, gates_d)]
            ung = nmse(torch.cat([t.reshape(-1) for t in y_q]), torch.cat([t.reshape(-1) for t in y_fp]))
            gat = nmse(torch.cat([t.reshape(-1) for t in y_q_g]), torch.cat([t.reshape(-1) for t in y_fp_g]))
            rows.append(
                {
                    "alpha_high": ah,
                    "alpha_low": al,
                    "ungated_nmse": ung,
                    "gated_nmse": gat,
                }
            )

    best_ungated = min(rows, key=lambda r: r["ungated_nmse"])
    best_gated = min(rows, key=lambda r: r["gated_nmse"])
    baseline = next(r for r in rows if r["alpha_high"] == 1.0 and r["alpha_low"] == 1.0)

    print("  top-5 by ungated objective:", flush=True)
    for r in sorted(rows, key=lambda x: x["ungated_nmse"])[:5]:
        print(
            f"    α_hi={r['alpha_high']:.2f} α_lo={r['alpha_low']:.2f}: "
            f"ungated={r['ungated_nmse']:.6e} gated={r['gated_nmse']:.6e}",
            flush=True,
        )
    print("  top-5 by gated objective:", flush=True)
    for r in sorted(rows, key=lambda x: x["gated_nmse"])[:5]:
        print(
            f"    α_hi={r['alpha_high']:.2f} α_lo={r['alpha_low']:.2f}: "
            f"ungated={r['ungated_nmse']:.6e} gated={r['gated_nmse']:.6e}",
            flush=True,
        )

    a = best_ungated["gated_nmse"]
    b = best_gated["gated_nmse"]
    rel = (a - b) / max(a, 1e-12)
    summary = {
        "block": BLOCK_ID,
        "module": "attn1.to_out.0",
        "bits": BITS,
        "n_forwards": len(xs),
        "gate_median": thr,
        "n_high_gate_channels": int(high_mask.sum()),
        "n_low_gate_channels": int(low_mask.sum()),
        "baseline_alpha1": baseline,
        "calib_without_gate": {
            "chosen_alpha_high": best_ungated["alpha_high"],
            "chosen_alpha_low": best_ungated["alpha_low"],
            "objective_ungated_nmse": best_ungated["ungated_nmse"],
            "eval_gated_nmse": best_ungated["gated_nmse"],
        },
        "calib_with_gate": {
            "chosen_alpha_high": best_gated["alpha_high"],
            "chosen_alpha_low": best_gated["alpha_low"],
            "objective_gated_nmse": best_gated["gated_nmse"],
            "eval_gated_nmse": best_gated["gated_nmse"],
            "eval_ungated_nmse": best_gated["ungated_nmse"],
        },
        "gated_nmse_reduction_vs_ungated_calib": rel,
        "grid": rows,
    }
    print(
        f"\n  Under GATED metric:\n"
        f"    calib without gate → gated_nmse={a:.6e} "
        f"(α_hi={best_ungated['alpha_high']}, α_lo={best_ungated['alpha_low']})\n"
        f"    calib with gate    → gated_nmse={b:.6e} "
        f"(α_hi={best_gated['alpha_high']}, α_lo={best_gated['alpha_low']})\n"
        f"    relative reduction={(rel * 100):.2f}%",
        flush=True,
    )
    same = (
        best_ungated["alpha_high"] == best_gated["alpha_high"]
        and best_ungated["alpha_low"] == best_gated["alpha_low"]
    )
    print(f"  chosen params identical? {same}", flush=True)
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} out={OUT_DIR} block={BLOCK_ID}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe = pipe.to(device)
    block = pipe.transformer.blocks[BLOCK_ID]
    block.eval()

    p1 = part1_wrap_assert(block, device)

    print("=== capturing real o_proj inputs ===", flush=True)
    xs, tembs, gates = capture_block_o_proj_inputs(pipe, BLOCK_ID, device)
    # Confirm temb present in captured eval kwargs analogue
    assert all(t is not None and t.numel() > 0 for t in tembs)
    print(f"  temb shapes: {[tuple(t.shape) for t in tembs[:2]]} ...", flush=True)
    print(f"  gate |mean|={float(torch.stack([g.abs().mean() for g in gates]).mean()):.4f}", flush=True)

    p2 = part2_calib_compare(block, xs, tembs, gates, device)

    out = {"part1_wrap": p1, "part2_calib": p2, "prompt": PROMPT, "steps": NUM_STEPS}
    out_path = OUT_DIR / f"block{BLOCK_ID:02d}_o_proj_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0 if p1["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
