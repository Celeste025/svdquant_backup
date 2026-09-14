#!/usr/bin/env python3
"""Smoke: BF16 Wan2.1-1.3B + SageAttention2 (H20 / sm90).

- First-step A/B: SDPA vs Sage2 timing + latent/proj_out NMSE
- Full GIF with Sage2; side-by-side vs heldout BF16 GIF if present

Does not modify deepcompressor; monkeypatches F.scaled_dot_product_attention.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DEEPCOMPRESSOR_TRANSFORMER_ONLY", "0")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_FLOW_SHIFT", "3.0")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_GATED", "0")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import pyarrow  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_firststep_block_nmse import (  # noqa: E402
    WAN_NEGATIVE,
    WAN_PROMPT,
    StopAfterFirstStep,
    load_wan_bf16,
    metrics,
)
from infer_wan_bf16_vs_w4a4_one import _save_video  # noqa: E402
from exp_wan_keep_bf16_blocks_18_24 import make_side_by_side  # noqa: E402

SEED = 42
OUT = DATA_ROOT / "compare" / "wan_sageattn2_smoke"
PRIOR_BF16_GIF = DATA_ROOT / "compare" / "wan_bf16_vs_s16_vs_gated" / "heldout_bike" / f"bf16_seed{SEED}.gif"


def _require_sage() -> dict:
    import importlib.metadata as md

    try:
        from sageattention import sageattn
        from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_sm90
    except Exception as e:
        raise SystemExit(
            f"sageattention not available ({e}). "
            "Install from https://github.com/thu-ml/SageAttention (need 2.x with sm90)."
        ) from e
    ver = "unknown"
    try:
        ver = md.version("sageattention")
    except Exception:
        pass
    return {
        "version": ver,
        "sageattn": sageattn,
        "sageattn_sm90": sageattn_qk_int8_pv_fp8_cuda_sm90,
    }


@contextmanager
def patch_sdpa_sage2(sage_api, *, force_sm90_fp32: bool = True):
    """Replace F.scaled_dot_product_attention with SageAttention2 wrapper.

    On H20, default sageattn() already selects sm90 + pv_accum_dtype=fp32+fp32 (Sage2, not 2++).
    """
    orig = F.scaled_dot_product_attention
    stats = {"calls": 0, "fallback": 0, "fallback_reasons": {}}

    def _wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        stats["calls"] += 1
        # Sage2 does not implement dropout / arbitrary attn_mask
        if dropout_p and dropout_p > 0:
            stats["fallback"] += 1
            stats["fallback_reasons"]["dropout"] = stats["fallback_reasons"].get("dropout", 0) + 1
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        if attn_mask is not None:
            stats["fallback"] += 1
            stats["fallback_reasons"]["attn_mask"] = stats["fallback_reasons"].get("attn_mask", 0) + 1
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        try:
            sm_scale = scale
            if force_sm90_fp32 and torch.cuda.get_device_capability()[0] >= 9:
                return sage_api["sageattn_sm90"](
                    query,
                    key,
                    value,
                    tensor_layout="HND",
                    is_causal=bool(is_causal),
                    sm_scale=sm_scale,
                    pv_accum_dtype="fp32+fp32",
                )
            return sage_api["sageattn"](
                query,
                key,
                value,
                tensor_layout="HND",
                is_causal=bool(is_causal),
                sm_scale=sm_scale,
            )
        except Exception as e:
            stats["fallback"] += 1
            key_r = type(e).__name__
            stats["fallback_reasons"][key_r] = stats["fallback_reasons"].get(key_r, 0) + 1
            if stats["fallback"] <= 3:
                print(f"[sage2] fallback to SDPA: {e}", flush=True)
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

    F.scaled_dot_product_attention = _wrapped  # type: ignore[assignment]
    try:
        yield stats
    finally:
        F.scaled_dot_product_attention = orig


def run_first_step_capture(pipe, prompt: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (latents_after_step0, proj_out, seconds)."""
    store: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        store["proj_out"] = t.detach().to("cpu", torch.bfloat16).contiguous().clone()

    h = pipe.transformer.proj_out.register_forward_hook(hook)
    latents_box: dict[str, torch.Tensor] = {}

    def callback(_pipe, index, _timestep, kwargs):
        if index == 0:
            lat = kwargs["latents"]
            latents_box["latents"] = lat.detach().to("cpu", torch.bfloat16).contiguous().clone()
            raise StopAfterFirstStep
        return kwargs

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        pipe(
            prompt,
            negative_prompt=WAN_NEGATIVE,
            height=480,
            width=832,
            num_frames=33,
            num_inference_steps=50,
            guidance_scale=6.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    except StopAfterFirstStep:
        pass
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    h.remove()
    assert "latents" in latents_box and "proj_out" in store
    return latents_box["latents"], store["proj_out"], elapsed


@torch.inference_mode()
def generate_gif(pipe, prompt: str, seed: int, out_path: Path) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe(
        prompt,
        negative_prompt=WAN_NEGATIVE,
        height=480,
        width=832,
        num_frames=33,
        num_inference_steps=50,
        guidance_scale=6.0,
        generator=gen,
    )
    frames = out.frames[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    _save_video(frames, out_path)
    info = {
        "path": str(out_path),
        "seconds": elapsed,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
        "n_frames": len(frames),
    }
    print(f"[gen] {out_path.name} {elapsed:.1f}s peak={info['peak_alloc_gib']:.2f}GiB", flush=True)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--prompt", type=str, default=WAN_PROMPT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-gif", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sage_api = _require_sage()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    print(f"=== Wan BF16 + SageAttention2 smoke | {gpu} sm{cap[0]}{cap[1]} | sage={sage_api['version']} ===", flush=True)

    print("[1] load BF16 Wan...", flush=True)
    pipe = load_wan_bf16()

    report: dict = {
        "prompt": args.prompt,
        "seed": args.seed,
        "gpu": gpu,
        "compute_capability": list(cap),
        "sageattention_version": sage_api["version"],
        "attn": "sageattn_qk_int8_pv_fp8_cuda_sm90 / pv_accum_dtype=fp32+fp32 (SageAttention2 on Hopper)",
    }

    # --- first-step SDPA ---
    print("[2a] first-step SDPA warmup + timed...", flush=True)
    run_first_step_capture(pipe, args.prompt, args.seed)  # warmup
    lat_sdpa, proj_sdpa, t_sdpa = run_first_step_capture(pipe, args.prompt, args.seed)
    print(f"  SDPA first-step {t_sdpa:.3f}s", flush=True)

    # --- first-step Sage2 ---
    print("[2b] first-step Sage2 warmup + timed...", flush=True)
    with patch_sdpa_sage2(sage_api) as st_warm:
        run_first_step_capture(pipe, args.prompt, args.seed)
    with patch_sdpa_sage2(sage_api) as st_timed:
        lat_sage, proj_sage, t_sage = run_first_step_capture(pipe, args.prompt, args.seed)
    print(f"  Sage2 first-step {t_sage:.3f}s  sdpa_calls={st_timed['calls']} fallback={st_timed['fallback']}", flush=True)

    m_lat = metrics(lat_sdpa.float(), lat_sage.float())
    m_proj = metrics(proj_sdpa.float(), proj_sage.float())
    report["first_step"] = {
        "sdpa_seconds": t_sdpa,
        "sage2_seconds": t_sage,
        "speedup": (t_sdpa / t_sage) if t_sage > 0 else None,
        "latent_vs_sdpa": m_lat,
        "proj_out_vs_sdpa": m_proj,
        "sage_patch_stats": st_timed,
    }
    print(
        f"  NMSE latent={100*m_lat['nmse']:.4f}%  proj_out={100*m_proj['nmse']:.4f}%  "
        f"speedup={report['first_step']['speedup']:.3f}x",
        flush=True,
    )

    # --- full GIF with Sage2 ---
    if not args.skip_gif:
        print("[3] full gen GIF with Sage2...", flush=True)
        gif_path = args.out_dir / f"sageattn2_bf16_seed{args.seed}.gif"
        with patch_sdpa_sage2(sage_api) as st_gif:
            report["gif"] = generate_gif(pipe, args.prompt, args.seed, gif_path)
            report["gif"]["sage_patch_stats"] = st_gif
        if PRIOR_BF16_GIF.is_file():
            montage = args.out_dir / f"side_by_side_bf16_vs_sage2_seed{args.seed}.gif"
            make_side_by_side([PRIOR_BF16_GIF, gif_path], ["BF16-SDPA", "BF16-Sage2"], montage)
            report["side_by_side"] = str(montage)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    out_json = args.out_dir / "report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"ALL DONE → {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
