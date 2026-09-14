#!/usr/bin/env python3
"""Report Smooth (alpha, beta) chosen under ungated vs gated OutputsError."""
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


def match_alpha_beta(calibrator, best_scale: torch.Tensor) -> dict:
    """Map best_scale back to (alpha, beta) [+ span mode] under Layer granularity."""
    best = best_scale.detach().float().cpu().reshape(-1)
    hits = []
    for span_id, (a_span, b_span) in enumerate(calibrator.span_pairs):
        a_span = a_span.detach().float().cpu().reshape(-1)
        b_span = b_span.detach().float().cpu().reshape(-1)
        modes = calibrator.span_mode_pairs[span_id]
        for alpha, beta in calibrator.alpha_beta_pairs:
            if alpha == 0 and beta == 0:
                cand = torch.ones_like(a_span)
            else:
                cand = get_smooth_scale(alpha_base=a_span, beta_base=b_span, alpha=alpha, beta=beta).reshape(-1)
            # exact / near-exact match
            if torch.allclose(cand, best, rtol=1e-4, atol=1e-5):
                hits.append(
                    {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "span_modes": [m.name for m in modes],
                        "max_abs_diff": 0.0,
                    }
                )
            else:
                diff = (cand - best).abs().max().item()
                hits.append(
                    {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "span_modes": [m.name for m in modes],
                        "max_abs_diff": float(diff),
                    }
                )
    hits.sort(key=lambda h: h["max_abs_diff"])
    best_hit = hits[0]
    # uniqueness: how many within 1e-4 of best
    near = [h for h in hits if h["max_abs_diff"] <= max(1e-4, best_hit["max_abs_diff"] + 1e-12)]
    return {"best": best_hit, "num_near_ties": len(near), "top3": hits[:3]}


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

    xs, tembs = capture(pipe, block_id, device)
    act_inputs = make_act_inputs(xs, device)
    eval_inputs = make_eval_inputs(xs, tembs, device)
    w_backup = o_proj.weight.detach().cpu().clone()

    def run(gated: bool):
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
        # undo in-place smooth mutation if any happened during search hooks
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        match = match_alpha_beta(calib, scale if scale is not None else calib.best_scale)
        return {
            "alpha": match["best"]["alpha"],
            "beta": match["best"]["beta"],
            "span_modes": match["best"]["span_modes"],
            "match_max_abs_diff": match["best"]["max_abs_diff"],
            "num_near_ties": match["num_near_ties"],
            "top3": match["top3"],
            "scale_mean": float(scale.float().mean()),
            "scale_std": float(scale.float().std()),
        }

    u = run(False)
    g = run(True)
    return {
        "block": block_id,
        "n_forwards": len(xs),
        "ungated": u,
        "gated": g,
        "same_alpha_beta": u["alpha"] == g["alpha"] and u["beta"] == g["beta"] and u["span_modes"] == g["span_modes"],
    }


@torch.inference_mode()
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} blocks={BLOCKS}", flush=True)
    qcfg = load_quant_config()
    print(
        f"smooth strategy grids: num_grids={qcfg.smooth.proj.num_grids} "
        f"alpha={qcfg.smooth.proj.alpha} beta={qcfg.smooth.proj.beta} "
        f"granularity={qcfg.smooth.proj.granularity}",
        flush=True,
    )
    pairs = qcfg.smooth.proj.get_alpha_beta_pairs()
    print(f"candidate (alpha,beta) count={len(pairs)} first/last={pairs[0]}/{pairs[-1]}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe = pipe.to(device)

    results = []
    for bid in BLOCKS:
        print(f"\n===== block {bid} =====", flush=True)
        r = run_one_block(pipe, qcfg, bid, device)
        results.append(r)
        print(
            f"ungated: alpha={r['ungated']['alpha']:.4f} beta={r['ungated']['beta']:.4f} "
            f"modes={r['ungated']['span_modes']} scale_mean={r['ungated']['scale_mean']:.4f}",
            flush=True,
        )
        print(
            f"gated:   alpha={r['gated']['alpha']:.4f} beta={r['gated']['beta']:.4f} "
            f"modes={r['gated']['span_modes']} scale_mean={r['gated']['scale_mean']:.4f}",
            flush=True,
        )
        print(f"same_alpha_beta={r['same_alpha_beta']}", flush=True)

    out = OUT_DIR / "o_proj_alpha_beta.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
