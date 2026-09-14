#!/usr/bin/env python3
"""Real PTQ calibrator ablation: ungated vs gated OutputsError on one Wan layer.

Uses the same APIs as PTQ:
  - smooth_linear_modules (SmoothQuant scale search)
  - DiffusionWeightQuantizer.calibrate_dynamic_range (weight range search)

Default target: block14 attn1.o_proj (Linear → wrap with gate_msa; no RoPE needed).
"""
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402 — import torch before deepcompressor/csrc

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "third_party" / "deepcompressor" / "examples" / "diffusion"
sys.path.insert(0, str(REPO / "third_party" / "deepcompressor"))

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline  # noqa: E402

from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct  # noqa: E402
from deepcompressor.app.diffusion.quant.quantizer import DiffusionWeightQuantizer  # noqa: E402
from deepcompressor.app.diffusion.quant.utils import wrap_wan_gated  # noqa: E402
from deepcompressor.calib.smooth import smooth_linear_modules  # noqa: E402
from deepcompressor.data.cache import TensorCache, TensorsCache  # noqa: E402
from deepcompressor.data.utils.reshape import LinearReshapeFn  # noqa: E402
from deepcompressor.quantizer import Quantizer  # noqa: E402

WAN_PATH = os.environ.get("WAN_MODEL_PATH", "/ssd/2/yuzhibo.yzh_data/Wan2.1-T2V-1.3B-Diffusers")
OUT_DIR = Path(os.environ.get("GATE_REAL_ABLATION_OUT", str(DATA_ROOT / "compare" / "wan_gate_real_calib_ablation")))
BLOCK_ID = int(os.environ.get("GATE_ABLATION_BLOCK", "14"))
NUM_STEPS = int(os.environ.get("GATE_ABLATION_STEPS", "4"))
PROMPT = "an airplane soaring through a clear blue sky"
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(((a - b).pow(2).sum() / b.pow(2).sum().clamp_min(1e-12)).item())


def load_quant_config():
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    old_argv, old_cwd = sys.argv, os.getcwd()
    try:
        os.chdir(EXAMPLES)
        sys.argv = [
            "ptq",
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/wan_s16.yaml",
            "--skip-eval",
            "--skip-gen",
        ]
        cfg, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args()
        return cfg.quant
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def make_act_inputs(xs: list[torch.Tensor], device: torch.device) -> TensorsCache:
    x = torch.cat([t.detach().to(dtype=torch.bfloat16).cpu() for t in xs], dim=0)
    n = int(x.shape[0])
    return TensorsCache(
        OrderedDict(
            {
                0: TensorCache(
                    [x],
                    channels_dim=-1,
                    reshape=LinearReshapeFn(),
                    num_cached=n,
                    num_total=n,
                    num_samples=n,
                    orig_device=device,
                )
            }
        )
    )


def make_eval_inputs(xs: list[torch.Tensor], tembs: list[torch.Tensor], device: torch.device) -> TensorsCache:
    x = torch.cat([t.detach().to(dtype=torch.bfloat16).cpu() for t in xs], dim=0)
    # temb stays float32 like Wan (scale_shift_table + temb.float())
    temb = torch.cat([t.detach().float().cpu() for t in tembs], dim=0)
    assert x.shape[0] == temb.shape[0], f"act batch {x.shape[0]} != temb {temb.shape[0]}"
    n = int(x.shape[0])

    def tc(data: torch.Tensor, channels_dim: int) -> TensorCache:
        return TensorCache(
            [data],
            channels_dim=channels_dim,
            reshape=LinearReshapeFn(),
            num_cached=n,
            num_total=n,
            num_samples=n,
            orig_device=device,
        )

    return TensorsCache(OrderedDict({0: tc(x, -1), "temb": tc(temb, 1)}))


def tensor_report(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().reshape(-1).cpu(), b.float().reshape(-1).cpu()
    diff = (a - b).abs()
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return {
        "name": name,
        "shape": list(a.shape),
        "equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine": cos,
        "rel_l2": float(diff.pow(2).sum().sqrt() / a.pow(2).sum().sqrt().clamp_min(1e-12)),
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
        "a_std": float(a.std()),
        "b_std": float(b.std()),
    }


def summarize_state(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            t = v.detach().float().cpu()
            out[k] = {
                "shape": list(t.shape),
                "mean": float(t.mean()),
                "std": float(t.std()) if t.numel() > 1 else 0.0,
                "min": float(t.min()),
                "max": float(t.max()),
            }
        else:
            out[k] = v
    return out


@torch.inference_mode()
def capture(pipe, block_id: int, device: torch.device):
    block = pipe.transformer.blocks[block_id]
    o_proj = block.attn1.to_out[0]
    latest_temb = {"t": None}
    xs, tembs, gates = [], [], []

    orig_block, orig_o = block.forward, o_proj.forward

    def block_fwd(hidden_states, encoder_hidden_states, temb, rotary_emb):
        latest_temb["t"] = temb.detach()
        return orig_block(hidden_states, encoder_hidden_states, temb, rotary_emb)

    def o_fwd(x):
        temb = latest_temb["t"]
        gate = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[2]
        xs.append(x.detach().float().cpu())
        tembs.append(temb.float().cpu())
        gates.append(gate.float().cpu())
        return orig_o(x)

    block.forward = block_fwd
    o_proj.forward = o_fwd
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
    print(f"captured {len(xs)} o_proj forwards", flush=True)
    return xs, tembs, gates


@torch.inference_mode()
def eval_gated_nmse(o_proj, w_fp, w_q, xs, gates, device) -> float:
    errs = []
    bias = o_proj.bias
    for x, g in zip(xs, gates):
        x = x.to(device=device, dtype=torch.bfloat16)
        g = g.to(device=device, dtype=torch.bfloat16)
        y_fp = torch.nn.functional.linear(x, w_fp.to(device=device, dtype=torch.bfloat16), bias)
        y_q = torch.nn.functional.linear(x, w_q.to(device=device, dtype=torch.bfloat16), bias)
        errs.append(nmse(y_q * g, y_fp * g))
    return float(sum(errs) / len(errs))


@torch.inference_mode()
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} block={BLOCK_ID}", flush=True)

    print("loading quant config...", flush=True)
    qcfg = load_quant_config()
    print(
        f"smooth.num_grids={qcfg.smooth.proj.num_grids} "
        f"wgts.calib_range.enabled={getattr(qcfg.wgts, 'enabled_calib_range', None)}",
        flush=True,
    )

    print("loading Wan...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe = pipe.to(device)

    block = pipe.transformer.blocks[BLOCK_ID]
    o_proj = block.attn1.to_out[0]
    model_struct = DiffusionModelStruct.construct(pipe.transformer)
    attn_struct = None
    for tb in model_struct.iter_transformer_block_structs():
        if tb.module is block:
            for a in tb.iter_attention_structs():
                if a.is_self_attn():
                    attn_struct = a
                    break
    assert attn_struct is not None

    xs, tembs, gates = capture(pipe, BLOCK_ID, device)
    act_inputs = make_act_inputs(xs, device)
    eval_inputs = make_eval_inputs(xs, tembs, device)
    w_backup = o_proj.weight.detach().cpu().clone()
    results = {"block": BLOCK_ID, "n_forwards": len(xs), "target": "attn1.o_proj"}

    # ---------- Smooth scale search ----------
    print("=== Smooth o_proj: ungated vs gated OutputsError ===", flush=True)

    def run_smooth(gated: bool) -> torch.Tensor:
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        eval_mod = wrap_wan_gated(o_proj, block, "msa") if gated else o_proj
        # prevs=None: isolate o_proj objective (no fuse into v_proj)
        # inputs: act only; eval_inputs: act + temb when gated
        scale = smooth_linear_modules(
            None,
            o_proj,
            scale=None,
            config=qcfg.smooth.proj,
            weight_quantizer=Quantizer(qcfg.wgts, key=attn_struct.out_proj_key, low_rank=qcfg.wgts.low_rank),
            input_quantizer=Quantizer(qcfg.ipts, channels_dim=-1, key=attn_struct.out_proj_key),
            inputs=act_inputs,
            eval_inputs=eval_inputs if gated else act_inputs,
            eval_module=eval_mod,
            eval_kwargs={},
            develop_dtype=qcfg.develop_dtype,
        )
        out = scale.detach().float().cpu().clone()
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        return out

    scale_u = run_smooth(False)
    scale_g = run_smooth(True)
    smooth_rep = tensor_report("smooth_o_proj_scale", scale_u, scale_g)
    results["smooth"] = smooth_rep
    print(json.dumps(smooth_rep, indent=2), flush=True)

    # ---------- Weight dynamic-range search ----------
    print("=== Weight calib_range o_proj: ungated vs gated ===", flush=True)

    def run_range(gated: bool) -> dict:
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        quantizer = DiffusionWeightQuantizer(qcfg.wgts, develop_dtype=qcfg.develop_dtype, key=attn_struct.out_proj_key)
        eval_mod = wrap_wan_gated(o_proj, block, "msa") if gated else o_proj
        quantizer.calibrate_dynamic_range(
            module=o_proj,
            inputs=act_inputs,
            eval_inputs=eval_inputs if gated else act_inputs,
            eval_module=eval_mod,
            eval_kwargs={},
        )
        sd = quantizer.state_dict()
        summary = summarize_state(sd)

        # Re-quantize weight under this calibrator for gated NMSE
        try:
            qw = quantizer.quantize(o_proj.weight.detach())
            if isinstance(qw, tuple):
                qw = qw[0]
            if hasattr(qw, "data"):
                qw = qw.data
            gated_err = eval_gated_nmse(o_proj, w_backup, qw, xs, gates, device)
        except Exception as e:
            qw = None
            gated_err = None
            summary["quantize_error"] = repr(e)

        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        return {"state_summary": summary, "gated_nmse": gated_err}

    rng_u = run_range(False)
    rng_g = run_range(True)
    results["weight_range"] = {"ungated_calib": rng_u, "gated_calib": rng_g}

    # Compare dynamic_range tensors if present in both
    for key in ("dynamic_range", "scale", "zero"):
        pass
    # Compare from summaries: if both have same keys, report whether summaries match
    su, sg = rng_u["state_summary"], rng_g["state_summary"]
    common = sorted(set(su) & set(sg))
    range_diff = {}
    for k in common:
        if isinstance(su[k], dict) and isinstance(sg[k], dict) and "mean" in su[k]:
            range_diff[k] = {
                "mean_delta": abs(su[k]["mean"] - sg[k]["mean"]),
                "max_delta": abs(su[k]["max"] - sg[k]["max"]),
                "min_delta": abs(su[k]["min"] - sg[k]["min"]),
            }
    results["weight_range_summary_diff"] = range_diff
    print("ungated gated_nmse=", rng_u["gated_nmse"], flush=True)
    print("gated   gated_nmse=", rng_g["gated_nmse"], flush=True)
    print("range summary deltas:", json.dumps(range_diff, indent=2), flush=True)

    # Cross-eval: apply ungated smooth scale under gated metric vs gated scale
    # (optional quick check using AbsMax-style? skip — scales already compared)

    out_path = OUT_DIR / f"block{BLOCK_ID:02d}_o_proj_real_calib.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)

    # Verdict line
    if smooth_rep["equal"]:
        print("VERDICT smooth: IDENTICAL scale (gated objective did not change Smooth search)", flush=True)
    else:
        print(
            f"VERDICT smooth: DIFFERENT scale (max_abs_diff={smooth_rep['max_abs_diff']:.3e}, "
            f"cosine={smooth_rep['cosine']:.6f})",
            flush=True,
        )
    if rng_u["gated_nmse"] is not None and rng_g["gated_nmse"] is not None:
        delta = rng_u["gated_nmse"] - rng_g["gated_nmse"]
        print(
            f"VERDICT range: gated_nmse(ungated_calib)={rng_u['gated_nmse']:.6e} "
            f"gated_nmse(gated_calib)={rng_g['gated_nmse']:.6e} delta={delta:.3e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
