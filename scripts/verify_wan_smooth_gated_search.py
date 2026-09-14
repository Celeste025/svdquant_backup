#!/usr/bin/env python3
"""Verify: under GATED OutputsError, does gated-chosen (α,β) beat ungated-chosen?

Uses the calibrator's OWN error path (not a reimplemented NMSE).
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


def record_search_errors(calib: SmoothLinearCalibrator):
    """Wrap ask/tell to record (alpha,beta,error) for every candidate."""
    records = []
    orig_tell = calib.tell

    def tell(error):
        alpha_beta_id, span_pair_id = calib._split_candidate_id(calib.candidate_id)
        alpha, beta = calib.alpha_beta_pairs[alpha_beta_id]
        modes = calib.span_mode_pairs[span_pair_id]
        err = float(sum(e.float().sum().item() for e in error))
        records.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "span_modes": [m.name for m in modes],
                "error": err,
                "candidate_id": int(calib.candidate_id),
            }
        )
        return orig_tell(error)

    calib.tell = tell  # type: ignore[method-assign]
    return records


def match_pair(records, alpha, beta, rtol=1e-6):
    hits = [r for r in records if abs(r["alpha"] - alpha) <= rtol and abs(r["beta"] - beta) <= rtol]
    if not hits:
        return None
    # if multiple span modes, take min error
    return min(hits, key=lambda r: r["error"])


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

    def calibrate(gated: bool):
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        eval_mod = wrap_wan_gated(o_proj, block, "msa") if gated else o_proj
        calib = SmoothLinearCalibrator(
            config=qcfg.smooth.proj,
            weight_quantizer=Quantizer(qcfg.wgts, key=attn_struct.out_proj_key, low_rank=qcfg.wgts.low_rank),
            input_quantizer=Quantizer(qcfg.ipts, channels_dim=-1, key=attn_struct.out_proj_key),
            develop_dtype=qcfg.develop_dtype,
        )
        records = record_search_errors(calib)
        scale = calib.calibrate(
            x_wgts=[o_proj.weight],
            x_acts=act_inputs,
            x_mods=[o_proj],
            eval_inputs=eval_inputs if gated else act_inputs,
            eval_module=eval_mod,
            eval_kwargs={},
        )
        o_proj.weight.copy_(w_backup.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype))
        # recover chosen pair from scale
        best = scale.detach().float().cpu().reshape(-1)
        chosen = None
        best_diff = 1e9
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
                    best_diff = diff
                    chosen = (float(alpha), float(beta))
        # best among recorded errors
        best_rec = min(records, key=lambda r: r["error"])
        return {
            "chosen_alpha": chosen[0],
            "chosen_beta": chosen[1],
            "chosen_match_diff": best_diff,
            "best_by_error": best_rec,
            "records": records,
            "scale_mean": float(scale.float().mean()),
        }

    ungated = calibrate(False)
    gated = calibrate(True)

    # Cross-check: in the GATED search table, what error did ungated's (α,β) get?
    u_ab = (ungated["chosen_alpha"], ungated["chosen_beta"])
    g_ab = (gated["chosen_alpha"], gated["chosen_beta"])
    u_in_gated_table = match_pair(gated["records"], *u_ab)
    g_in_gated_table = match_pair(gated["records"], *g_ab)
    u_in_ungated_table = match_pair(ungated["records"], *u_ab)
    g_in_ungated_table = match_pair(ungated["records"], *g_ab)

    # Consistency checks
    gated_search_ok = (
        g_in_gated_table is not None
        and u_in_gated_table is not None
        and g_in_gated_table["error"] <= u_in_gated_table["error"] + 1e-6
    )
    ungated_search_ok = (
        u_in_ungated_table is not None
        and g_in_ungated_table is not None
        and u_in_ungated_table["error"] <= g_in_ungated_table["error"] + 1e-6
    )

    return {
        "block": block_id,
        "ungated_choice": {"alpha": u_ab[0], "beta": u_ab[1], "scale_mean": ungated["scale_mean"]},
        "gated_choice": {"alpha": g_ab[0], "beta": g_ab[1], "scale_mean": gated["scale_mean"]},
        "gated_table_error_at_ungated_ab": u_in_gated_table["error"] if u_in_gated_table else None,
        "gated_table_error_at_gated_ab": g_in_gated_table["error"] if g_in_gated_table else None,
        "ungated_table_error_at_ungated_ab": u_in_ungated_table["error"] if u_in_ungated_table else None,
        "ungated_table_error_at_gated_ab": g_in_ungated_table["error"] if g_in_ungated_table else None,
        "gated_search_picks_lower_or_eq_than_ungated_ab": gated_search_ok,
        "ungated_search_picks_lower_or_eq_than_gated_ab": ungated_search_ok,
        "gated_best_by_error": gated["best_by_error"],
        "ungated_best_by_error": ungated["best_by_error"],
        # top5 gated table
        "gated_table_top5": sorted(gated["records"], key=lambda r: r["error"])[:5],
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
    for bid in BLOCKS:
        print(f"\n===== block {bid} =====", flush=True)
        r = run_block(pipe, qcfg, bid, device)
        results.append(r)
        print(
            f"ungated chose (α,β)=({r['ungated_choice']['alpha']:.1f},{r['ungated_choice']['beta']:.1f})",
            flush=True,
        )
        print(
            f"gated   chose (α,β)=({r['gated_choice']['alpha']:.1f},{r['gated_choice']['beta']:.1f})",
            flush=True,
        )
        print(
            f"[GATED search table] error@ungated_ab={r['gated_table_error_at_ungated_ab']:.6e}  "
            f"error@gated_ab={r['gated_table_error_at_gated_ab']:.6e}  "
            f"gated_wins={r['gated_search_picks_lower_or_eq_than_ungated_ab']}",
            flush=True,
        )
        print(
            f"[UNGATED search table] error@ungated_ab={r['ungated_table_error_at_ungated_ab']:.6e}  "
            f"error@gated_ab={r['ungated_table_error_at_gated_ab']:.6e}  "
            f"ungated_wins={r['ungated_search_picks_lower_or_eq_than_gated_ab']}",
            flush=True,
        )
        print("gated table top5:", flush=True)
        for t in r["gated_table_top5"]:
            print(f"  (α,β)=({t['alpha']:.1f},{t['beta']:.1f}) error={t['error']:.6e} modes={t['span_modes']}", flush=True)

    out = OUT_DIR / "o_proj_search_error_verify.json"
    # drop huge records if any nested - already only top5
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
