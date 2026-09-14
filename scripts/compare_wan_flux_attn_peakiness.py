#!/usr/bin/env python3
"""Compare Wan vs Flux attention peakiness (entropy / max mass).

Does NOT materialize full T×T for all queries: samples N_QUERY rows per head,
computes softmax over keys, then reports:
  - mean max attention weight (higher → sharper)
  - mean entropy / log(T_k)  (lower → sharper; 1=uniform)

Runs a short BF16 pipeline forward with hooks on selected blocks.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import pyarrow  # noqa: F401,E402

REPO = Path(__file__).resolve().parents[1]
DIFFUSION = REPO / "third_party" / "deepcompressor" / "examples" / "diffusion"
OUT_DEFAULT = DATA_ROOT / "compare" / "wan_flux_attn_peakiness"

N_QUERY = 128  # sampled query rows per head
PROMPT = "A person is riding a bike"
NEGATIVE_WAN = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _parse_cfg(argv: list[str]):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    sys.argv = ["compare_attn_peakiness.py", *argv]
    config, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args()
    return config


def peakiness_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float,
    n_query: int,
    rng: torch.Generator,
) -> dict:
    """q,k: [B, H, T, D] (already head-split)."""
    assert q.ndim == 4 and k.ndim == 4
    b, h, tq, d = q.shape
    tk = k.shape[2]
    n_q = min(n_query, tq)
    # sample query indices (CPU generator is portable across devices)
    idx = torch.randperm(tq, generator=rng)[:n_q].to(q.device)
    q_s = q.index_select(2, idx)  # [B,H,n_q,D]
    # scores [B,H,n_q,Tk] — may be large; compute in float32 chunks over heads if needed
    scores = torch.matmul(q_s.float(), k.float().transpose(-1, -2)) * float(scale)
    # numerical stability
    scores = scores - scores.amax(dim=-1, keepdim=True)
    p = scores.softmax(dim=-1)
    max_mass = p.amax(dim=-1)  # [B,H,n_q]
    ent = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)  # [B,H,n_q]
    log_tk = math.log(max(tk, 2))
    return {
        "t_q": int(tq),
        "t_k": int(tk),
        "n_heads": int(h),
        "n_query_sampled": int(n_q),
        "max_mass_mean": float(max_mass.mean().item()),
        "max_mass_p50": float(max_mass.median().item()),
        "max_mass_p90": float(max_mass.quantile(0.9).item()),
        "entropy_mean": float(ent.mean().item()),
        "entropy_norm_mean": float((ent / log_tk).mean().item()),  # 1 ≈ uniform
        "top1_over_uniform": float((max_mass.mean() / (1.0 / tk)).item()),
    }


class PeakinessCollector:
    """Tag Attention modules + patch SDPA so we see post-RoPE / joint Q,K."""

    def __init__(self, store: dict, n_query: int, seed: int):
        self.store = store
        self.n_query = n_query
        self.seed = seed
        self.current: str | None = None
        self._handles: list = []
        self._orig_sdpa = None

    def _pre(self, name: str):
        def hook(_m, _inp):
            self.current = name

        return hook

    def _post(self, _m, _inp, _out):
        self.current = None

    def _sdpa(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        name = self.current
        if name is not None and name not in self.store and query.ndim == 4 and key.ndim == 4:
            d = query.shape[-1]
            sc = (1.0 / math.sqrt(d)) if scale is None else float(scale)
            try:
                q = query.detach()
                k = key.detach()
                rng = torch.Generator()
                rng.manual_seed(self.seed)
                stats = peakiness_from_qk(q, k, scale=sc, n_query=self.n_query, rng=rng)
                self.store[name] = stats
                log(
                    f"  [{name}] Tq={stats['t_q']} Tk={stats['t_k']} H={stats['n_heads']} "
                    f"max_mass={stats['max_mass_mean']:.4f} "
                    f"H/logT={stats['entropy_norm_mean']:.4f}"
                )
            except torch.cuda.OutOfMemoryError:
                log(f"  [OOM] {name}: skip")
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"  [skip] {name}: {type(e).__name__}: {e}")
        return self._orig_sdpa(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    def install_named(self, module, name: str) -> None:
        self._handles.append(module.register_forward_pre_hook(self._pre(name)))
        self._handles.append(module.register_forward_hook(self._post))

    def patch_sdpa(self) -> None:
        self._orig_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._sdpa  # type: ignore[assignment]

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
            self._orig_sdpa = None


def install_wan_hooks(transformer, store: dict, block_ids: list[int], n_query: int, seed: int) -> PeakinessCollector:
    col = PeakinessCollector(store, n_query, seed)
    for i in block_ids:
        block = transformer.blocks[i]
        for attn_name in ("attn1", "attn2"):
            col.install_named(getattr(block, attn_name), f"block{i}.{attn_name}")
    col.patch_sdpa()
    return col


def install_flux_hooks(
    transformer, store: dict, dual_ids: list[int], single_ids: list[int], n_query: int, seed: int
) -> PeakinessCollector:
    col = PeakinessCollector(store, n_query, seed)
    for i in dual_ids:
        col.install_named(transformer.transformer_blocks[i].attn, f"dual{i}.attn")
    for i in single_ids:
        col.install_named(transformer.single_transformer_blocks[i].attn, f"single{i}.attn")
    col.patch_sdpa()
    return col


def load_wan_bf16():
    os.chdir(DIFFUSION)
    cfg = _parse_cfg(
        [
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/wan_smoke.yaml",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    log("[wan] build BF16...")
    return cfg.pipeline.build()


def load_flux_bf16():
    from diffusers import FluxPipeline

    log("[flux] build BF16 via FluxPipeline...")
    return FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")


@torch.inference_mode()
def run_wan(store: dict, block_ids: list[int], n_query: int, seed: int, steps: int):
    pipe = load_wan_bf16()
    col = install_wan_hooks(pipe.transformer, store, block_ids, n_query, seed)
    try:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        log(f"[wan] forward steps={steps} (first SDPA per module)...")
        pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_WAN,
            height=480,
            width=832,
            num_frames=33,
            num_inference_steps=steps,
            guidance_scale=6.0,
            generator=gen,
            output_type="latent",
        )
    finally:
        col.close()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def run_flux(store: dict, dual_ids: list[int], single_ids: list[int], n_query: int, seed: int, steps: int):
    pipe = load_flux_bf16()
    col = install_flux_hooks(pipe.transformer, store, dual_ids, single_ids, n_query, seed)
    try:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        log(f"[flux] forward steps={steps}...")
        pipe(
            prompt=PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=steps,
            guidance_scale=3.5,
            generator=gen,
            output_type="latent",
        )
    finally:
        col.close()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def summarize(store: dict, tag: str) -> dict:
    rows = []
    for name, st in store.items():
        rows.append({"name": name, **st})
    if not rows:
        return {"model": tag, "n": 0}
    max_m = [r["max_mass_mean"] for r in rows]
    ent_n = [r["entropy_norm_mean"] for r in rows]
    return {
        "model": tag,
        "n": len(rows),
        "max_mass_mean": float(np.mean(max_m)),
        "max_mass_median": float(np.median(max_m)),
        "entropy_norm_mean": float(np.mean(ent_n)),
        "entropy_norm_median": float(np.median(ent_n)),
        "modules": rows,
    }


def plot_compare(wan: dict, flux: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)

    def _xy(res, key):
        mods = res.get("modules") or []
        return list(range(len(mods))), [m[key] for m in mods], [m["name"] for m in mods]

    ax = axes[0]
    for res, color, lab in ((wan, "#1f4e79", "Wan"), (flux, "#b85c38", "Flux")):
        xs, ys, _ = _xy(res, "max_mass_mean")
        if ys:
            ax.plot(xs, ys, "o-", ms=4, lw=1.2, color=color, label=f"{lab} mean={np.mean(ys):.3f}")
    ax.set_title("Mean max attention mass (↑ sharper)")
    ax.set_xlabel("module index")
    ax.set_ylabel("max_j a_{ij}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for res, color, lab in ((wan, "#1f4e79", "Wan"), (flux, "#b85c38", "Flux")):
        xs, ys, _ = _xy(res, "entropy_norm_mean")
        if ys:
            ax.plot(xs, ys, "o-", ms=4, lw=1.2, color=color, label=f"{lab} mean={np.mean(ys):.3f}")
    ax.axhline(1.0, color="#888", ls=":", lw=1, label="uniform (=1)")
    ax.set_title("Normalized entropy H / log(T_k) (↓ sharper)")
    ax.set_xlabel("module index")
    ax.set_ylabel("entropy / log T_k")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Attention peakiness — Wan vs Flux (BF16, sampled queries)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=2, help="denoising steps (hooks fire early)")
    ap.add_argument("--n-query", type=int, default=N_QUERY)
    ap.add_argument("--wan-blocks", type=str, default="0,7,14,21,28")
    ap.add_argument("--flux-dual", type=str, default="0,5,10,15")
    ap.add_argument("--flux-single", type=str, default="0,10,20,30")
    ap.add_argument("--model", choices=["wan", "flux", "both"], default="both")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wan_blocks = [int(x) for x in args.wan_blocks.split(",") if x.strip() != ""]
    flux_dual = [int(x) for x in args.flux_dual.split(",") if x.strip() != ""]
    flux_single = [int(x) for x in args.flux_single.split(",") if x.strip() != ""]

    wan_store: dict = {}
    flux_store: dict = {}
    if args.model in ("wan", "both"):
        run_wan(wan_store, wan_blocks, args.n_query, args.seed, args.steps)
    if args.model in ("flux", "both"):
        run_flux(flux_store, flux_dual, flux_single, args.n_query, args.seed, args.steps)

    wan_sum = summarize(wan_store, "wan")
    flux_sum = summarize(flux_store, "flux")
    report = {
        "prompt": PROMPT,
        "seed": args.seed,
        "steps": args.steps,
        "n_query": args.n_query,
        "note": (
            "Sharper ⇒ higher max_mass, lower entropy_norm. "
            "Stats from first forward per module (typically early step / cond branch)."
        ),
        "wan": wan_sum,
        "flux": flux_sum,
    }
    out_json = args.out_dir / "attn_peakiness.json"
    out_json.write_text(json.dumps(report, indent=2))
    log(f"[json] {out_json}")

    log("\n=== SUMMARY ===")
    for tag, s in (("Wan", wan_sum), ("Flux", flux_sum)):
        if s.get("n", 0) == 0:
            log(f"{tag}: no modules captured")
            continue
        log(
            f"{tag}: n={s['n']}  max_mass mean/med={s['max_mass_mean']:.4f}/{s['max_mass_median']:.4f}  "
            f"H/logT mean/med={s['entropy_norm_mean']:.4f}/{s['entropy_norm_median']:.4f}"
        )
    if wan_sum.get("n") and flux_sum.get("n"):
        dm = wan_sum["max_mass_mean"] - flux_sum["max_mass_mean"]
        de = wan_sum["entropy_norm_mean"] - flux_sum["entropy_norm_mean"]
        log(f"Δ(Wan-Flux): max_mass={dm:+.4f}  H/logT={de:+.4f}")
        if dm > 0.02 and de < -0.02:
            log("→ Wan looks sharper on these probes.")
        elif dm < -0.02 and de > 0.02:
            log("→ Flux looks sharper on these probes.")
        else:
            log("→ No large consistent sharpness gap on these probes.")
        plot_compare(wan_sum, flux_sum, args.out_dir / "wan_vs_flux_attn_peakiness.png")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
