#!/usr/bin/env python3
"""Gated error using the calibrator's exact OutputsError path (fixes prior offline NMSE bug)."""
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
    xs, tembs = [], []
    orig_block, orig_o = block.forward, o_proj.forward

    def block_fwd(hidden_states, encoder_hidden_states, temb, rotary_emb):
        latest["t"] = temb.detach()
        return orig_block(hidden_states, encoder_hidden_states, temb, rotary_emb)

    def o_fwd(x):
        xs.append(x.detach().to(dtype=torch.bfloat16).cpu())
        tembs.append(latest["t"].float().cpu())
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
    return xs, tembs


def choose_ab(calib, scale):
    best = scale.detach().float().cpu().reshape(-1)
    chosen, best_diff = None, 1e9
    for a_span, b_span in calib.span_pairs:
        a_span = a_span.detach().float().cpu().reshape(-1)
        b_span = b_span.detach().float().cpu().reshape(-1)
        for alpha, beta in calib.alpha_beta_pairs:
            if alpha == 0 and beta == 0:
                cand = torch.ones_like(a_span)
            else:
                cand = get_smooth_scale(alpha_base=a_span, beta_base=b_span, alpha=alpha, beta=beta).reshape(-1)
            diff = (cand - best).abs().max().item()
            if diff < best_diff:
                best_diff, chosen = diff, (float(alpha), float(beta))
    return chosen


@torch.inference_mode()
def eval_with_calibrator_path(
    calib: SmoothLinearCalibrator,
    scale: torch.Tensor,
    o_proj,
    eval_module,
    eval_inputs: TensorsCache,
    develop_dtype,
):
    """Exact OutputsError + NMSE using calibrator hooks (same as search)."""
    # Parse ipts like calibrate does
    ipts = calib._parse_ipts(eval_inputs, set_device=True)
    # FP reference through eval_module (gated or not)
    orig_opts = {}
    ref_sq = 0.0
    for i in range(len(ipts.front().data)):
        ipt = ipts.extract(i, {})
        y = eval_module(*ipt.args, **ipt.kwargs)
        y = y[0] if not isinstance(y, torch.Tensor) else y
        orig_opts[i] = y.detach()
        ref_sq += float(y.float().pow(2).sum().item())

    # Apply candidate scale via calibrator internals
    calib.candidate = scale.to(device=o_proj.weight.device, dtype=develop_dtype)
    wgts = [o_proj.weight]
    mods = [o_proj]
    # save & restore
    backup = o_proj.weight.data.detach().clone()
    calib._process_wgts_centric_mod(wgts=wgts, mods=mods)
    sse = 0.0
    for i in range(len(ipts.front().data)):
        ipt = ipts.extract(i, {})
        y = eval_module(*ipt.args, **ipt.kwargs)
        y = y[0] if not isinstance(y, torch.Tensor) else y
        diff = (y - orig_opts[i].to(device=y.device)).float()
        sse += float(diff.pow(2).sum().item())
    calib._recover_mod()
    o_proj.weight.data.copy_(backup)

    outputs_error = sse  # degree=2 sum, same as search
    gated_nmse = sse / max(ref_sq, 1e-12)
    return {"outputs_error_sse": outputs_error, "nmse": gated_nmse, "ref_sq": ref_sq}


@torch.inference_mode()
def run_block(pipe, qcfg, block_id, device):
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

    xs, tembs = capture(pipe, block_id, device)
    act_inputs = make_act_inputs(xs, device)
    eval_inputs = make_eval_inputs(xs, tembs, device)
    w_backup = o_proj.weight.detach().cpu().clone()

    def find_scale(gated_obj: bool):
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        eval_mod = wrap_wan_gated(o_proj, block, "msa") if gated_obj else o_proj
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
            eval_inputs=eval_inputs if gated_obj else act_inputs,
            eval_module=eval_mod,
            eval_kwargs={},
        )
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        ab = choose_ab(calib, scale)
        return scale.detach().cpu(), ab, calib

    scale_u, ab_u, calib_u = find_scale(False)
    scale_g, ab_g, calib_g = find_scale(True)

    # Evaluate BOTH scales under GATED eval_module, using a fresh calibrator for hooks
    def make_calib():
        return SmoothLinearCalibrator(
            config=qcfg.smooth.proj,
            weight_quantizer=Quantizer(qcfg.wgts, key=attn_struct.out_proj_key, low_rank=qcfg.wgts.low_rank),
            input_quantizer=Quantizer(qcfg.ipts, channels_dim=-1, key=attn_struct.out_proj_key),
            develop_dtype=qcfg.develop_dtype,
        )

    # Need spans/setup: run a dummy reset by calibrating briefly? Easier: reuse calib_g after
    # re-calling _reset via a partial calibrate setup.
    # Re-init calib_g state by running calibrate's reset only — call calibrate again is expensive.
    # Instead: create calib, call calibrate with gated once already done — use calib_g which still
    # has span state from last calibrate. But candidate processing needs x_quantizer etc. which remain.

    gated_eval = wrap_wan_gated(o_proj, block, "msa")
    # Re-bind quantizers on calib_g (still valid after calibrate)
    o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))

    # Force calib into a state ready for _process_wgts_centric_mod: needs needs_w_quant flags etc.
    # These are set during calibrate(); calib_g should still have them.
    m_u = eval_with_calibrator_path(calib_g, scale_u, o_proj, gated_eval, eval_inputs, qcfg.develop_dtype)
    m_g = eval_with_calibrator_path(calib_g, scale_g, o_proj, gated_eval, eval_inputs, qcfg.develop_dtype)
    ones = torch.ones_like(scale_u)
    m_1 = eval_with_calibrator_path(calib_g, ones, o_proj, gated_eval, eval_inputs, qcfg.develop_dtype)

    return {
        "block": block_id,
        "ungated_calib": {"alpha": ab_u[0], "beta": ab_u[1], **m_u},
        "gated_calib": {"alpha": ab_g[0], "beta": ab_g[1], **m_g},
        "no_smooth": m_1,
        "gated_nmse_improve_vs_ungated_calib": (m_u["nmse"] - m_g["nmse"]) / max(m_u["nmse"], 1e-12),
        "gated_sse_improve_vs_ungated_calib": (m_u["outputs_error_sse"] - m_g["outputs_error_sse"])
        / max(m_u["outputs_error_sse"], 1e-12),
    }


@torch.inference_mode()
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qcfg = load_quant_config()
    print(f"device={device} blocks={BLOCKS}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe = pipe.to(device)

    results = []
    print(
        f"{'block':>5} {'calib':>14} {'(α,β)':>10} {'gated_SSE':>14} {'gated_NMSE':>14}",
        flush=True,
    )
    for bid in BLOCKS:
        r = run_block(pipe, qcfg, bid, device)
        results.append(r)
        for name, key in (("no_smooth", "no_smooth"), ("ungated_calib", "ungated_calib"), ("gated_calib", "gated_calib")):
            m = r[key]
            ab = "—" if "alpha" not in m else f"({m['alpha']:.1f},{m['beta']:.1f})"
            print(
                f"{bid:5d} {name:>14} {ab:>10} {m['outputs_error_sse']:14.6e} {m['nmse']:14.6e}",
                flush=True,
            )
        print(
            f"      gated_calib vs ungated_calib: NMSE improve {r['gated_nmse_improve_vs_ungated_calib']*100:.2f}%, "
            f"SSE improve {r['gated_sse_improve_vs_ungated_calib']*100:.2f}%",
            flush=True,
        )
        print(flush=True)

    out = OUT_DIR / "o_proj_gated_nmse_fixed.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
