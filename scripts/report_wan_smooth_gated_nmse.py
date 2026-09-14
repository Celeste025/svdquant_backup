#!/usr/bin/env python3
"""Gated NMSE after Smooth: compare scales found by ungated vs gated calib.

Both scales are evaluated under the SAME metric: NMSE(g⊙y_q, g⊙y_fp)
with SmoothQuant-style W/X fake-quant (same as PTQ Smooth search).
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

import torch  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "third_party" / "deepcompressor" / "examples" / "diffusion"
sys.path.insert(0, str(REPO / "third_party" / "deepcompressor"))

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline  # noqa: E402

from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct  # noqa: E402
from deepcompressor.app.diffusion.quant.utils import wrap_wan_gated  # noqa: E402
from deepcompressor.calib.smooth import SmoothLinearCalibrator, get_smooth_scale  # noqa: E402
from deepcompressor.data.cache import TensorCache, TensorsCache  # noqa: E402
from deepcompressor.data.utils.reshape import LinearReshapeFn  # noqa: E402
from deepcompressor.quantizer import Quantizer  # noqa: E402

WAN_PATH = os.environ.get("WAN_MODEL_PATH", "/ssd/2/yuzhibo.yzh_data/Wan2.1-T2V-1.3B-Diffusers")
OUT_DIR = Path(os.environ.get("GATE_REAL_ABLATION_OUT", str(DATA_ROOT / "compare" / "wan_gate_real_calib_ablation")))
NUM_STEPS = int(os.environ.get("GATE_ABLATION_STEPS", "4"))
BLOCKS = [int(x) for x in os.environ.get("GATE_ABLATION_BLOCKS", "7,14,21").split(",")]
PROMPT = "an airplane soaring through a clear blue sky"
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def nmse(pred, ref) -> float:
    pred, ref = pred.float(), ref.float()
    return float(((pred - ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-12)).item())


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


def make_act_inputs(xs, device):
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


def make_eval_inputs(xs, tembs, device):
    x = torch.cat([t.detach().to(dtype=torch.bfloat16).cpu() for t in xs], dim=0)
    temb = torch.cat([t.detach().float().cpu() for t in tembs], dim=0)
    n = int(x.shape[0])

    def tc(data, channels_dim):
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


@torch.inference_mode()
def capture(pipe, block_id, device):
    block = pipe.transformer.blocks[block_id]
    o_proj = block.attn1.to_out[0]
    latest = {"t": None}
    xs, tembs, gates = [], [], []
    orig_block, orig_o = block.forward, o_proj.forward

    def block_fwd(hidden_states, encoder_hidden_states, temb, rotary_emb):
        latest["t"] = temb.detach()
        return orig_block(hidden_states, encoder_hidden_states, temb, rotary_emb)

    def o_fwd(x):
        temb = latest["t"]
        gate = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[2]
        xs.append(x.detach().to(dtype=torch.bfloat16).cpu())
        tembs.append(temb.float().cpu())
        gates.append(gate.detach().float().cpu())
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
    return xs, tembs, gates


def match_alpha_beta(calibrator, best_scale: torch.Tensor) -> tuple[float, float]:
    best = best_scale.detach().float().cpu().reshape(-1)
    best_pair, best_diff = (None, None), 1e9
    for a_span, b_span in calibrator.span_pairs:
        a_span = a_span.detach().float().cpu().reshape(-1)
        b_span = b_span.detach().float().cpu().reshape(-1)
        for alpha, beta in calibrator.alpha_beta_pairs:
            if alpha == 0 and beta == 0:
                cand = torch.ones_like(a_span)
            else:
                cand = get_smooth_scale(alpha_base=a_span, beta_base=b_span, alpha=alpha, beta=beta).reshape(-1)
            diff = (cand - best).abs().max().item()
            if diff < best_diff:
                best_diff = diff
                best_pair = (float(alpha), float(beta))
    return best_pair


@torch.inference_mode()
def smooth_fake_quant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    w_quantizer: Quantizer,
    x_quantizer: Quantizer,
    develop_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply SmoothQuant W/X fake-quant as in SmoothLinearCalibrator._process_*_in_xw."""
    dtype = weight.dtype
    s = scale.to(device=weight.device, dtype=develop_dtype).view(1, -1)  # in-channels
    # weight: [out, in]
    w = weight.to(dtype=develop_dtype).clone()
    w = w_quantizer.quantize(w.mul_(s), kernel=None, default_dtype=dtype, develop_dtype=develop_dtype).data
    w = w.div_(s).to(dtype=dtype)

    xs = x.to(dtype=develop_dtype)
    # x channels last
    s_x = scale.to(device=x.device, dtype=develop_dtype)
    while s_x.ndim < xs.ndim:
        s_x = s_x.view(*([1] * (xs.ndim - 1)), -1)
    xs = xs.div(s_x)
    xs = x_quantizer.quantize(xs, channels_dim=-1, default_dtype=dtype, develop_dtype=develop_dtype).data
    xs = xs.mul(s_x).to(dtype=dtype)
    return torch.nn.functional.linear(xs, w, bias)


@torch.inference_mode()
def eval_gated_nmse(xs, gates, weight, bias, scale, w_q, x_q, develop_dtype, device) -> float:
    errs = []
    for x, g in zip(xs, gates):
        x = x.to(device=device, dtype=torch.bfloat16)
        g = g.to(device=device, dtype=torch.bfloat16)
        w = weight.to(device=device, dtype=torch.bfloat16)
        b = bias.to(device=device) if bias is not None else None
        y_fp = torch.nn.functional.linear(x, w, b)
        y_q = smooth_fake_quant_linear(x, w, b, scale.to(device=device), w_q, x_q, develop_dtype)
        errs.append(nmse(y_q * g, y_fp * g))
    return float(sum(errs) / len(errs))


@torch.inference_mode()
def eval_ungated_nmse(xs, weight, bias, scale, w_q, x_q, develop_dtype, device) -> float:
    errs = []
    for x in xs:
        x = x.to(device=device, dtype=torch.bfloat16)
        w = weight.to(device=device, dtype=torch.bfloat16)
        b = bias.to(device=device) if bias is not None else None
        y_fp = torch.nn.functional.linear(x, w, b)
        y_q = smooth_fake_quant_linear(x, w, b, scale.to(device=device), w_q, x_q, develop_dtype)
        errs.append(nmse(y_q, y_fp))
    return float(sum(errs) / len(errs))


@torch.inference_mode()
def run_one_block(pipe, qcfg, block_id, device):
    block = pipe.transformer.blocks[block_id]
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

    xs, tembs, gates = capture(pipe, block_id, device)
    act_inputs = make_act_inputs(xs, device)
    eval_inputs = make_eval_inputs(xs, tembs, device)
    w_backup = o_proj.weight.detach().cpu().clone()

    w_q = Quantizer(qcfg.wgts, key=attn_struct.out_proj_key, low_rank=qcfg.wgts.low_rank)
    x_q = Quantizer(qcfg.ipts, channels_dim=-1, key=attn_struct.out_proj_key)

    def find_scale(gated: bool):
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        eval_mod = wrap_wan_gated(o_proj, block, "msa") if gated else o_proj
        calib = SmoothLinearCalibrator(
            config=qcfg.smooth.proj,
            weight_quantizer=Quantizer(qcfg.wgts, key=attn_struct.out_proj_key, low_rank=qcfg.wgts.low_rank),
            input_quantizer=Quantizer(qcfg.ipts, channels_dim=-1, key=attn_struct.out_proj_key),
            develop_dtype=qcfg.develop_dtype,
        )
        scale = calib.calibrate(
            x_wgts=[o_proj.weight],
            x_acts=act_inputs,
            x_mods=[o_proj],
            eval_inputs=eval_inputs if gated else act_inputs,
            eval_module=eval_mod,
            eval_kwargs={},
        )
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        alpha, beta = match_alpha_beta(calib, scale)
        return scale.detach().float().cpu(), alpha, beta

    scale_u, a_u, b_u = find_scale(False)
    scale_g, a_g, b_g = find_scale(True)

    # Evaluate BOTH scales under gated NMSE (and also ungated NMSE for reference)
    metrics = {}
    for name, scale, a, b in (
        ("ungated_calib", scale_u, a_u, b_u),
        ("gated_calib", scale_g, a_g, b_g),
    ):
        gated_err = eval_gated_nmse(
            xs, gates, w_backup, o_proj.bias, scale, w_q, x_q, qcfg.develop_dtype, device
        )
        ungated_err = eval_ungated_nmse(
            xs, w_backup, o_proj.bias, scale, w_q, x_q, qcfg.develop_dtype, device
        )
        metrics[name] = {
            "alpha": a,
            "beta": b,
            "gated_nmse": gated_err,
            "ungated_nmse": ungated_err,
            "scale_mean": float(scale.mean()),
        }

    # baseline: no smooth (scale=1)
    ones = torch.ones_like(scale_u)
    metrics["no_smooth"] = {
        "alpha": None,
        "beta": None,
        "gated_nmse": eval_gated_nmse(xs, gates, w_backup, o_proj.bias, ones, w_q, x_q, qcfg.develop_dtype, device),
        "ungated_nmse": eval_ungated_nmse(xs, w_backup, o_proj.bias, ones, w_q, x_q, qcfg.develop_dtype, device),
        "scale_mean": 1.0,
    }

    return {
        "block": block_id,
        "n_forwards": len(xs),
        "metrics": metrics,
        "gated_nmse_delta": metrics["ungated_calib"]["gated_nmse"] - metrics["gated_calib"]["gated_nmse"],
        "gated_nmse_rel_improve": (
            (metrics["ungated_calib"]["gated_nmse"] - metrics["gated_calib"]["gated_nmse"])
            / max(metrics["ungated_calib"]["gated_nmse"], 1e-12)
        ),
    }


@torch.inference_mode()
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} blocks={BLOCKS}", flush=True)
    qcfg = load_quant_config()

    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe = pipe.to(device)

    results = []
    print(
        f"{'block':>5}  {'calib':>14}  {'(α,β)':>12}  {'gated_nmse':>12}  {'ungated_nmse':>12}",
        flush=True,
    )
    for bid in BLOCKS:
        r = run_one_block(pipe, qcfg, bid, device)
        results.append(r)
        for name in ("no_smooth", "ungated_calib", "gated_calib"):
            m = r["metrics"][name]
            ab = "—" if m["alpha"] is None else f"({m['alpha']:.1f},{m['beta']:.1f})"
            print(
                f"{bid:5d}  {name:>14}  {ab:>12}  {m['gated_nmse']:12.6e}  {m['ungated_nmse']:12.6e}",
                flush=True,
            )
        print(
            f"       delta_gated_nmse(ungated_calib - gated_calib)="
            f"{r['gated_nmse_delta']:.6e}  rel_improve={r['gated_nmse_rel_improve']*100:.2f}%",
            flush=True,
        )
        print(flush=True)

    out = OUT_DIR / "o_proj_gated_nmse.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
