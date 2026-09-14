#!/usr/bin/env python3
"""Wan W4A4 quantization error pattern analysis.

For sampled blocks, decompose errors on:
  - single Linears: q/k/v/o, ffn_up, ffn_down
  - whole modules: attn1, attn2, ffn (compose)

Reports:
  1) weight error (W_eff = W_q + lowrank vs W_fp)
  2) activation error (act-quantizer fake-quant vs BF16 input)
  3) matmul terms: weight-only / act-only / both vs BF16 matmul
  4) histograms + per-token RMSE reshaped to spatiotemporal (T,H,W)

Teacher-forced BF16 calib inputs (same protocol as local NMSE).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DEEPCOMPRESSOR_TRANSFORMER_ONLY", "1")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_FLOW_SHIFT", "3.0")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_GATED", "0")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import pyarrow  # noqa: F401,E402

REPO = Path(__file__).resolve().parents[1]
DIFFUSION = REPO / "third_party" / "deepcompressor" / "examples" / "diffusion"
sys.path.insert(0, str(REPO / "scripts"))

from collect_calib_local_block_nmse import (  # noqa: E402
    WAN_CALIB,
    WAN_CKPT,
    _parse_cfg,
    block_id_from_name,
    flat_sample_steps,
    list_selected_metas,
    log,
    move_ipt_to_device,
    pick_tensor,
    resolve_quant_module,
)

OUT_DEFAULT = DATA_ROOT / "compare" / "wan_quant_error_patterns"

# 480x832x33, patch (1,2,2) → latent tokens 9 x 30 x 52 = 14040
DEFAULT_THW = (9, 30, 52)
SAMPLE_BLOCKS = (0, 14, 28)


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    den = a.square().mean().clamp_min(1e-20)
    return float((a - b).square().mean() / den)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).square().mean().sqrt())


def get_branch(mod: nn.Module):
    for h in mod._forward_hooks.values():
        br = getattr(h, "branch", None)
        if br is not None:
            return br
    return None


def get_act_quantizer(mod: nn.Module):
    for h in mod._forward_pre_hooks.values():
        proc = getattr(h, "processor", None)
        if proc is not None and hasattr(proc, "process"):
            return proc
    return None


def effective_weight(q_linear: nn.Module) -> torch.Tensor:
    w = q_linear.weight.detach().float()
    br = get_branch(q_linear)
    if br is not None and hasattr(br, "get_effective_weight"):
        we = br.get_effective_weight()
        if we is not None:
            w = w + we.detach().float()
    return w


def reshape_token_map(err_bt_d: torch.Tensor, thw: tuple[int, int, int]) -> np.ndarray | None:
    """err: [B,T,D] or [T,D] → mean |err| over D → [t,h,w]."""
    t, h, w = thw
    x = err_bt_d.detach().float()
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2:
        return None
    n, d = x.shape
    if n != t * h * w:
        return None
    mag = x.norm(dim=-1) / max(d**0.5, 1.0)  # RMS over channels
    return mag.view(t, h, w).cpu().numpy()


def _to_cpu_f32_npy(t: torch.Tensor) -> np.ndarray:
    """Detach → float32 numpy; squeeze leading batch dim if B==1."""
    x = t.detach().float().cpu()
    if x.ndim >= 3 and x.shape[0] == 1:
        x = x[0]
    return x.numpy()


def save_elementwise(ew_dir: Path, module_name: str, tensors: dict[str, torch.Tensor | None]) -> dict:
    """Save raw elementwise tensors as float32 .npy; return manifest entries."""
    ew_dir.mkdir(parents=True, exist_ok=True)
    safe = module_name.replace(".", "_")
    meta = {}
    for key, ten in tensors.items():
        if ten is None:
            continue
        arr = _to_cpu_f32_npy(ten)
        path = ew_dir / f"{safe}__{key}.npy"
        np.save(path, arr)
        meta[key] = {"file": path.name, "shape": list(arr.shape), "dtype": str(arr.dtype), "nbytes": int(arr.nbytes)}
    return meta


def hist_stats(err: torch.Tensor) -> dict:
    e = err.detach().float().reshape(-1)
    # subsample for hist if huge
    if e.numel() > 2_000_000:
        idx = torch.randperm(e.numel())[:2_000_000]
        e = e[idx]
    abs_e = e.abs()
    return {
        "mean": float(e.mean()),
        "std": float(e.std()),
        "abs_mean": float(abs_e.mean()),
        "abs_p50": float(abs_e.median()),
        "abs_p90": float(abs_e.quantile(0.90)),
        "abs_p99": float(abs_e.quantile(0.99)),
        "nmse_vs_zero_ref_placeholder": None,
    }


@torch.inference_mode()
def analyze_linear(
    name: str,
    kind: str,
    fp_lin: nn.Linear,
    q_lin: nn.Linear,
    x: torch.Tensor,
    thw: tuple[int, int, int],
) -> dict:
    """x: [B,T,Din] on device."""
    device = x.device
    W_fp = fp_lin.weight.detach().float()
    b_fp = fp_lin.bias.detach().float() if fp_lin.bias is not None else None
    W_q = q_lin.weight.detach().float()
    W_eff = effective_weight(q_lin)
    b_q = q_lin.bias.detach().float() if q_lin.bias is not None else None
    br = get_branch(q_lin)
    aq = get_act_quantizer(q_lin)

    # weight errors
    w_err_q = W_q - W_fp
    w_err_eff = W_eff - W_fp
    w_stats = {
        "nmse_Wq_vs_Wfp": nmse(W_fp, W_q),
        "nmse_Weff_vs_Wfp": nmse(W_fp, W_eff),
        "rmse_Weff": rmse(W_fp, W_eff),
        "has_branch": br is not None,
        "has_act_quant": aq is not None,
        "shape": list(W_fp.shape),
    }

    x_f = x.float()
    # activation quant (fake quant → dequant)
    if aq is not None:
        try:
            x_q = aq.process(x).float()
        except Exception as e:
            log(f"  [warn] act quant failed on {name}: {e}")
            x_q = x_f
            w_stats["has_act_quant"] = False
    else:
        x_q = x_f

    a_err = x_q - x_f
    a_stats = {
        "nmse_Xq_vs_X": nmse(x_f, x_q),
        "rmse": rmse(x_f, x_q),
        **{f"err_{k}": v for k, v in hist_stats(a_err).items() if v is not None},
    }

    def lin(xf, W, b):
        y = F.linear(xf, W.to(device), b.to(device) if b is not None else None)
        return y

    y_fp = lin(x_f, W_fp, b_fp)
    y_w = lin(x_f, W_eff, b_q)  # weight(+branch) only
    y_a = lin(x_q, W_fp, b_fp)  # act only
    y_wa = lin(x_q, W_eff, b_q)  # both
    # full module (smooth/hooks as deployed)
    y_mod = pick_tensor(q_lin(x)).float()

    e_w = y_w - y_fp
    e_a = y_a - y_fp
    e_wa = y_wa - y_fp
    e_mod = y_mod - y_fp
    e_cross = e_wa - e_w - e_a  # residual interaction

    mat = {
        "nmse_weight_only": nmse(y_fp, y_w),
        "nmse_act_only": nmse(y_fp, y_a),
        "nmse_both_approx": nmse(y_fp, y_wa),
        "nmse_full_module": nmse(y_fp, y_mod),
        "nmse_cross_term": float(e_cross.square().mean() / y_fp.square().mean().clamp_min(1e-20)),
        "frac_var_weight": float(e_w.square().mean() / e_wa.square().mean().clamp_min(1e-20)),
        "frac_var_act": float(e_a.square().mean() / e_wa.square().mean().clamp_min(1e-20)),
    }

    token_maps = {
        "weight_only": reshape_token_map(e_w, thw),
        "act_only": reshape_token_map(e_a, thw),
        "both_approx": reshape_token_map(e_wa, thw),
        "full_module": reshape_token_map(e_mod, thw),
        "act_input_err": reshape_token_map(a_err, thw),
    }

    return {
        "name": name,
        "kind": kind,
        "block_id": block_id_from_name(name),
        "weight": w_stats,
        "activation": a_stats,
        "matmul": mat,
        "token_maps": token_maps,
        "weight_err_hist": hist_stats(w_err_eff),
        "out_err_hist_full": hist_stats(e_mod),
        # raw tensors for optional dump (stripped before JSON)
        "elementwise": {
            "w_err": w_err_eff,  # [Dout, Din]
            "a_err": a_err,  # [B, T, Din]
            "e_w": e_w,
            "e_a": e_a,
            "e_wa": e_wa,
            "e_mod": e_mod,  # [B, T, Dout]
        },
    }


@torch.inference_mode()
def analyze_module_out(
    name: str,
    kind: str,
    fp_mod: nn.Module,
    q_mod: nn.Module,
    args,
    kwargs,
    thw: tuple[int, int, int],
) -> dict:
    y_fp = pick_tensor(fp_mod(*args, **kwargs)).float()
    y_q = pick_tensor(q_mod(*args, **kwargs)).float()
    e = y_q - y_fp
    return {
        "name": name,
        "kind": kind,
        "block_id": block_id_from_name(name),
        "matmul": {"nmse_full_module": nmse(y_fp, y_q), "rmse": rmse(y_fp, y_q)},
        "out_err_hist_full": hist_stats(e),
        "token_maps": {"full_module": reshape_token_map(e, thw)},
        "weight": None,
        "activation": None,
        "elementwise": {"e_mod": e},
    }


def plot_summary(results: list[dict], out_dir: Path, thw: tuple[int, int, int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    linears = [r for r in results if r.get("weight") is not None]
    wholes = [r for r in results if r.get("weight") is None]

    # --- bar: mean NMSE by kind for weight / act / matmul terms ---
    kinds = []
    for r in linears:
        if r["kind"] not in kinds:
            kinds.append(r["kind"])
    metrics = [
        ("weight", "nmse_Weff_vs_Wfp", "#1f4e79"),
        ("activation", "nmse_Xq_vs_X", "#2a9d8f"),
        ("matmul", "nmse_weight_only", "#b85c38"),
        ("matmul", "nmse_act_only", "#e9a46c"),
        ("matmul", "nmse_both_approx", "#5c4d7a"),
        ("matmul", "nmse_full_module", "#8b3a2a"),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    x = np.arange(len(kinds), dtype=float)
    width = 0.12
    for i, (sec, key, color) in enumerate(metrics):
        ys = []
        for k in kinds:
            vals = [r[sec][key] * 100 for r in linears if r["kind"] == k and r.get(sec) and key in r[sec]]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + (i - 2.5) * width, ys, width, label=f"{sec}:{key.replace('nmse_', '')}", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=30, ha="right")
    ax.set_ylabel("mean NMSE (%)")
    ax.set_title("Wan — quant error decomposition by layer kind (sampled blocks)")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "linear_error_decomposition_bars.png")
    plt.close(fig)

    # whole-module NMSE bars
    if wholes:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        labels = [f"{r['block_id']}.{r['kind']}" for r in wholes]
        ys = [r["matmul"]["nmse_full_module"] * 100 for r in wholes]
        ax.bar(range(len(ys)), ys, color="#1f4e79")
        ax.set_xticks(range(len(ys)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("NMSE (%)")
        ax.set_title("Wan — whole attn / FFN output NMSE (teacher-forced)")
        ax.grid(True, axis="y", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / "whole_module_nmse_bars.png")
        plt.close(fig)

    # histograms for a few representative linears
    reps = [r for r in linears if r["kind"] in ("q", "o", "ffn_down", "v")][:6]
    if reps:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=140)
        for ax, r in zip(axes.ravel(), reps):
            # reconstruct rough hist from stored percentiles only — re-load not available
            # use out_err abs percentiles as stem plot
            h = r["out_err_hist_full"]
            ax.bar(
                ["mean", "p50", "p90", "p99"],
                [h["abs_mean"], h["abs_p50"], h["abs_p90"], h["abs_p99"]],
                color="#1f4e79",
            )
            ax.set_title(f"{r['block_id']} {r['kind']} |out|", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        for ax in axes.ravel()[len(reps) :]:
            ax.axis("off")
        fig.suptitle("Output error magnitude percentiles (full quant module)")
        fig.tight_layout()
        fig.savefig(out_dir / "out_err_percentiles.png")
        plt.close(fig)

    # spatiotemporal grids
    t, h, w = thw
    for r in results:
        maps = r.get("token_maps") or {}
        keys = [k for k, v in maps.items() if v is not None]
        if not keys:
            continue
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0 + 0.15 * t), dpi=140, squeeze=False)
        for ax, k in zip(axes[0], keys):
            arr = maps[k]
            # show mean over time as HxW, and also max over time strip
            m2 = arr.mean(axis=0)
            im = ax.imshow(m2, aspect="auto", cmap="magma")
            ax.set_title(f"{k}\n(mean_t)", fontsize=8)
            ax.set_xlabel("W")
            ax.set_ylabel("H")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{r['name']} token RMSE-like maps  THW={thw}", fontsize=10)
        fig.tight_layout()
        safe = r["name"].replace(".", "_")
        fig.savefig(out_dir / f"spatemp_{safe}.png")
        plt.close(fig)

        # time×space: flatten H*W
        if "full_module" in maps and maps["full_module"] is not None:
            arr = maps["full_module"]
            flat = arr.reshape(t, h * w)
            fig, ax = plt.subplots(figsize=(10, 2.8), dpi=140)
            im = ax.imshow(flat, aspect="auto", cmap="magma")
            ax.set_xlabel("spatial token (H*W)")
            ax.set_ylabel("time t")
            ax.set_title(f"{r['name']} full-module err  (t × spatial)")
            fig.colorbar(im, ax=ax, fraction=0.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"spatemp_txspatial_{safe}.png")
            plt.close(fig)

    log(f"[plots] wrote under {out_dir}")


def build_models(num_samples: int):
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq
    from deepcompressor.app.diffusion.quant.utils import get_needs_inputs_fn

    os.chdir(DIFFUSION)
    scratch = DATA_ROOT / "tmp" / "wan_err_pat_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    base = [
        "configs/model/wan2.1-1.3b.yaml",
        "configs/svdquant/int4.yaml",
        "configs/svdquant/wan_s16.yaml",
        f"--calib-path={WAN_CALIB}",
        f"--calib-num-samples={num_samples}",
        f"--output-root={scratch}",
        f"--cache-root={scratch}",
        "--skip-eval",
        "--skip-gen",
        "--eval-num-gpus=1",
    ]
    cfg = _parse_cfg(base)
    cfg.quant.calib.num_samples = num_samples
    cfg.quant.calib.path = str(WAN_CALIB)

    log("[load] quant from ckpt...")
    cfg_q = _parse_cfg([*base, f"--load-from={WAN_CKPT}"])
    cfg_q.quant.calib.num_samples = num_samples
    cfg_q.quant.calib.path = str(WAN_CALIB)
    quant_pipe = cfg_q.pipeline.build()
    quant_struct = DiffusionModelStruct.construct(quant_pipe)
    ptq(
        quant_struct,
        cfg_q.quant,
        cache=None,
        load_dirpath=str(WAN_CKPT),
        save_dirpath="",
        copy_on_save=False,
        save_model=False,
    )
    quant_nn = quant_struct.module.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    log("[load] bf16...")
    bf16_pipe = cfg.pipeline.build()
    bf16_struct = DiffusionModelStruct.construct(bf16_pipe)
    device = next(bf16_struct.module.parameters()).device
    base_needs = get_needs_inputs_fn(bf16_struct, cfg.quant)
    return cfg, bf16_struct, quant_nn, device, base_needs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--blocks", type=str, default=",".join(map(str, SAMPLE_BLOCKS)))
    ap.add_argument("--num-samples", type=int, default=8, help="calib files (keep small)")
    ap.add_argument("--thw", type=str, default="9,30,52")
    ap.add_argument("--max-batches", type=int, default=1, help="act-cache batches per layer group")
    ap.add_argument(
        "--dump-elementwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save full float32 elementwise err tensors under out_dir/elementwise/ (~17GiB for 3 blocks)",
    )
    cli = ap.parse_args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    blocks = [int(x) for x in cli.blocks.split(",") if x.strip() != ""]
    thw = tuple(int(x) for x in cli.thw.split(","))
    assert len(thw) == 3

    cfg, bf16_struct, quant_nn, device, base_needs = build_models(cli.num_samples)
    _ = base_needs  # kept for compatibility; needs() is custom below
    metas = list_selected_metas(WAN_CALIB, cli.num_samples, seed=0)
    sample_steps = flat_sample_steps(metas, cfg.quant.calib.batch_size)
    log(f"selected {len(metas)} calib files; steps={len(sample_steps)}; THW={thw}")

    # targets per block
    linear_kinds = [
        ("attn1.to_q", "q"),
        ("attn1.to_k", "k"),
        ("attn1.to_v", "v"),
        ("attn1.to_out.0", "o"),
        ("attn2.to_q", "q"),
        ("attn2.to_k", "k"),
        ("attn2.to_v", "v"),
        ("attn2.to_out.0", "o"),
        ("ffn.net.0.proj", "ffn_up"),
        ("ffn.net.2.linear", "ffn_down"),
    ]
    whole_kinds = [
        ("attn1", "attn1"),
        ("attn2", "attn2"),
        # FFN is NOT cacheable by ActivationCache — measured via up-proj inputs below
    ]
    ffn_compose_suf = "ffn"

    want_names: set[str] = set()
    for bi in blocks:
        for suf, _ in linear_kinds + whole_kinds:
            want_names.add(f"blocks.{bi}.{suf}")

    def needs(name: str, module) -> bool:
        # ActivationCache only supports Linear / Attention / Conv.
        # Register at least attn1 on EVERY block so the calib iterator's hook_args
        # stays complete (otherwise KeyError on unselected blocks).
        cls = module.__class__.__name__
        is_attn = "Attention" in cls
        is_lin = isinstance(module, nn.Linear)
        if not (is_attn or is_lin):
            return False
        if name in want_names:
            return True
        if is_attn and name.endswith(".attn1"):
            return True
        return False

    results: list[dict] = []
    # serialize token maps separately (numpy)
    maps_dir = cli.out_dir / "token_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    ew_dir = cli.out_dir / "elementwise"
    ew_manifest: dict[str, dict] = {}
    if cli.dump_elementwise:
        ew_dir.mkdir(parents=True, exist_ok=True)
        log(f"[elementwise] dumping float32 tensors under {ew_dir} (expect ~17GiB for 3 blocks)")

    def finalize_row(row: dict, dump_name: str | None = None) -> dict:
        """Save token maps + optional elementwise; return JSON-safe row (no tensors)."""
        name = dump_name or row["name"]
        for mk, arr in (row.get("token_maps") or {}).items():
            if arr is not None:
                np.save(maps_dir / f"{name.replace('.', '_')}_{mk}.npy", arr)
        ew_keys = list((row.get("elementwise") or {}).keys())
        if cli.dump_elementwise and row.get("elementwise"):
            ew_manifest[name] = save_elementwise(ew_dir, name, row["elementwise"])
            nbytes = sum(v["nbytes"] for v in ew_manifest[name].values())
            log(f"  [ew] {name} saved {nbytes/1024**2:.1f} MiB")
        # drop heavy payloads from in-memory row
        row["elementwise"] = None
        row["token_maps"] = {k: (None if v is None else list(v.shape)) for k, v in (row.get("token_maps") or {}).items()}
        return {
            **{k: v for k, v in row.items() if k != "elementwise"},
            "elementwise_keys": ew_keys,
        }

    log("[measure] iterating calib activations...")
    from deepcompressor.app.diffusion.nn.struct import DiffusionTransformerBlockStruct

    seen_blocks: set[int] = set()
    for layer_name, (layer, layer_cache, layer_kwargs) in cfg.quant.calib.build_loader().iter_layer_activations(
        bf16_struct,
        needs_inputs_fn=needs,
        skip_pre_modules=True,
        skip_post_modules=True,
    ):
        tblocks = []
        if isinstance(layer, DiffusionTransformerBlockStruct):
            tblocks = [layer]
        elif hasattr(layer, "iter_transformer_block_structs"):
            tblocks = list(layer.iter_transformer_block_structs())
        else:
            continue

        for tb in tblocks:
            # infer block index
            bid = None
            for attn in tb.iter_attention_structs():
                m = __import__("re").search(r"blocks\.(\d+)", attn.name)
                if m:
                    bid = int(m.group(1))
                    break
            if bid is None or bid not in blocks:
                continue
            if bid in seen_blocks:
                layer_cache.clear()
                continue
            # only process each selected block once
            log(f"=== block{bid} ===")
            seen_blocks.add(bid)

            # one activation batch is enough for patterns
            for suf, kind in linear_kinds:
                name = f"blocks.{bid}.{suf}"
                if name not in layer_cache or layer_cache[name].inputs is None:
                    log(f"  [skip-cache] {name}")
                    continue
                try:
                    fp_m = resolve_quant_module(bf16_struct.module, name)
                    q_m = resolve_quant_module(quant_nn, name)
                    q_m.to(device)
                    # move branches
                    for hook_dict in (q_m._forward_hooks, q_m._forward_pre_hooks):
                        for h in hook_dict.values():
                            br = getattr(h, "branch", None)
                            if isinstance(br, nn.Module):
                                br.to(device)
                    ipt = layer_cache[name].inputs.extract(0, {})
                    args, kwargs = move_ipt_to_device(ipt, device)
                    x = args[0] if args else kwargs.get("hidden_states")
                    if x is None:
                        log(f"  [skip] no tensor input for {name}")
                        continue
                    # take first sample only to limit memory
                    if x.shape[0] > 1:
                        x = x[:1]
                    row = analyze_linear(name, kind, fp_m, q_m, x, thw)
                    row_json = finalize_row(row)
                    results.append(row_json)
                    log(
                        f"  [{kind}] {name} W_eff={100*row['weight']['nmse_Weff_vs_Wfp']:.3f}% "
                        f"X={100*row['activation']['nmse_Xq_vs_X']:.3f}% "
                        f"w_only={100*row['matmul']['nmse_weight_only']:.3f}% "
                        f"a_only={100*row['matmul']['nmse_act_only']:.3f}% "
                        f"both={100*row['matmul']['nmse_both_approx']:.3f}% "
                        f"full={100*row['matmul']['nmse_full_module']:.3f}%"
                    )
                    # free elementwise tensors ASAP
                    del row
                    gc.collect()
                except torch.cuda.OutOfMemoryError as e:
                    log(f"  [OOM] {name}: {e}")
                    torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  [warn] {name}: {e}")
                    import traceback

                    traceback.print_exc()
                finally:
                    try:
                        q_m.to("cpu")
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

            for suf, kind in whole_kinds:
                name = f"blocks.{bid}.{suf}"
                if name not in layer_cache or layer_cache[name].inputs is None:
                    log(f"  [skip-cache] {name}")
                    continue
                try:
                    fp_m = resolve_quant_module(bf16_struct.module, name)
                    q_m = resolve_quant_module(quant_nn, name)
                    q_m.to(device)
                    for hook_dict in (getattr(q_m, "_forward_hooks", {}), getattr(q_m, "_forward_pre_hooks", {})):
                        for h in hook_dict.values():
                            br = getattr(h, "branch", None)
                            if isinstance(br, nn.Module):
                                br.to(device)
                    # Attention needs filtered kwargs
                    eval_kwargs = {}
                    for attn in tb.iter_attention_structs():
                        if attn.name == name:
                            eval_kwargs = attn.filter_kwargs(layer_kwargs)
                            break
                    ipt = layer_cache[name].inputs.extract(0, eval_kwargs)
                    args, kwargs = move_ipt_to_device(ipt, device)
                    # first sample only if batched
                    if args and torch.is_tensor(args[0]) and args[0].shape[0] > 1:
                        args = tuple(a[:1] if torch.is_tensor(a) and a.shape[0] > 1 else a for a in args)
                        kwargs = {k: (v[:1] if torch.is_tensor(v) and v.shape[0] > 1 else v) for k, v in kwargs.items()}
                    row = analyze_module_out(name, kind, fp_m, q_m, args, kwargs, thw)
                    full = 100 * row["matmul"]["nmse_full_module"]
                    results.append(finalize_row(row))
                    del row
                    gc.collect()
                    log(f"  [{kind}] {name} full_nmse={full:.3f}%")
                except torch.cuda.OutOfMemoryError as e:
                    log(f"  [OOM] {name}: {e}")
                    torch.cuda.empty_cache()
                except Exception as e:
                    log(f"  [warn] {name}: {e}")
                    import traceback

                    traceback.print_exc()
                finally:
                    try:
                        q_m.to("cpu")
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

            # FFN compose: reuse up-proj inputs (ActivationCache cannot wrap FeedForward)
            up_name = f"blocks.{bid}.ffn.net.0.proj"
            ffn_name = f"blocks.{bid}.{ffn_compose_suf}"
            if up_name in layer_cache and layer_cache[up_name].inputs is not None:
                try:
                    fp_ffn = resolve_quant_module(bf16_struct.module, ffn_name)
                    q_ffn = resolve_quant_module(quant_nn, ffn_name)
                    q_ffn.to(device)
                    for hook_dict in (getattr(q_ffn, "_forward_hooks", {}), getattr(q_ffn, "_forward_pre_hooks", {})):
                        for h in hook_dict.values():
                            br = getattr(h, "branch", None)
                            if isinstance(br, nn.Module):
                                br.to(device)
                    # also move child linears' branches
                    for _n, m in q_ffn.named_modules():
                        if isinstance(m, nn.Linear):
                            for hook_dict in (m._forward_hooks, m._forward_pre_hooks):
                                for h in hook_dict.values():
                                    br = getattr(h, "branch", None)
                                    if isinstance(br, nn.Module):
                                        br.to(device)
                    ipt = layer_cache[up_name].inputs.extract(0, {})
                    args, kwargs = move_ipt_to_device(ipt, device)
                    if args and torch.is_tensor(args[0]) and args[0].shape[0] > 1:
                        args = tuple(a[:1] if torch.is_tensor(a) and a.shape[0] > 1 else a for a in args)
                        kwargs = {k: (v[:1] if torch.is_tensor(v) and v.shape[0] > 1 else v) for k, v in kwargs.items()}
                    row = analyze_module_out(ffn_name, "ffn_compose", fp_ffn, q_ffn, args, kwargs, thw)
                    full = 100 * row["matmul"]["nmse_full_module"]
                    results.append(finalize_row(row, dump_name=ffn_name))
                    del row
                    gc.collect()
                    log(f"  [ffn_compose] {ffn_name} full_nmse={full:.3f}%")
                except Exception as e:
                    log(f"  [warn] {ffn_name}: {e}")
                    import traceback

                    traceback.print_exc()
                finally:
                    try:
                        q_ffn.to("cpu")
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
            else:
                log(f"  [skip-cache] ffn compose (need {up_name})")

            layer_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

            if seen_blocks.issuperset(blocks):
                break
        if seen_blocks.issuperset(blocks):
            break

    # reload maps for plotting
    for r in results:
        name = r["name"]
        tm = {}
        for mk, shape in (r.get("token_maps") or {}).items():
            p = maps_dir / f"{name.replace('.', '_')}_{mk}.npy"
            tm[mk] = np.load(p) if p.exists() else None
        r["token_maps"] = tm

    out_json = cli.out_dir / "wan_quant_error_patterns.json"
    # JSON without numpy arrays
    serial = []
    for r in results:
        serial.append(
            {
                **{k: v for k, v in r.items() if k != "token_maps"},
                "token_maps_shapes": {
                    k: (None if v is None else list(np.asarray(v).shape)) for k, v in (r.get("token_maps") or {}).items()
                },
            }
        )
    if cli.dump_elementwise:
        man_path = ew_dir / "manifest.json"
        total_nbytes = sum(m["nbytes"] for mods in ew_manifest.values() for m in mods.values())
        man_path.write_text(
            json.dumps(
                {
                    "dtype": "float32",
                    "layout": (
                        "Linear: w_err[Dout,Din], a_err[T,Din], e_w/e_a/e_wa/e_mod[T,Dout]; "
                        "whole: e_mod[T,D]. Leading B=1 squeezed."
                    ),
                    "total_nbytes": total_nbytes,
                    "total_gib": total_nbytes / 1024**3,
                    "modules": ew_manifest,
                },
                indent=2,
            )
        )
        log(f"[elementwise] manifest {man_path}  total={total_nbytes/1024**3:.2f} GiB")

    out_json.write_text(
        json.dumps(
            {
                "ckpt": str(WAN_CKPT),
                "calib": str(WAN_CALIB),
                "blocks": blocks,
                "thw": thw,
                "num_samples": cli.num_samples,
                "dump_elementwise": bool(cli.dump_elementwise),
                "elementwise_dir": str(ew_dir) if cli.dump_elementwise else None,
                "note": (
                    "W_eff = W_q + lowrank branch; act = DiffusionActivationQuantizer fake-quant; "
                    "matmul approx ignores smooth side-effects (full_module includes them). "
                    "Raw float32 errs under elementwise/ when dump_elementwise=true."
                ),
                "modules": serial,
            },
            indent=2,
        )
    )
    log(f"[json] {out_json}")

    plot_summary(results, cli.out_dir, thw)

    # concise markdown
    lines = ["# Wan quant error patterns", "", f"blocks={blocks} THW={thw}", ""]
    lines.append("| module | kind | W_eff% | Xq% | w_only% | a_only% | both% | full% |")
    lines.append("|--|--|--|--|--|--|--|--|")
    for r in results:
        if r.get("weight") is None:
            lines.append(
                f"| `{r['name']}` | {r['kind']} | — | — | — | — | — | {100*r['matmul']['nmse_full_module']:.3f} |"
            )
        else:
            lines.append(
                f"| `{r['name']}` | {r['kind']} | "
                f"{100*r['weight']['nmse_Weff_vs_Wfp']:.3f} | "
                f"{100*r['activation']['nmse_Xq_vs_X']:.3f} | "
                f"{100*r['matmul']['nmse_weight_only']:.3f} | "
                f"{100*r['matmul']['nmse_act_only']:.3f} | "
                f"{100*r['matmul']['nmse_both_approx']:.3f} | "
                f"{100*r['matmul']['nmse_full_module']:.3f} |"
            )
    (cli.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    log(f"[summary] {cli.out_dir / 'SUMMARY.md'}")
    log(f"ALL DONE → {cli.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
