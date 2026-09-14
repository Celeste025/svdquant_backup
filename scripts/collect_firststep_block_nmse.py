#!/usr/bin/env python3
"""First-step BF16 vs W4A4 per-layer NMSE (Wan / Flux), natural forward.

Runs only denoising step 0 with the same prompt/seed for BF16 and PTQ W4A4,
hooks aligned layer outputs, and writes NMSE metrics JSON for plotting.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DEEPCOMPRESSOR_TRANSFORMER_ONLY", "0")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_FLOW_SHIFT", "3.0")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import torch  # noqa: E402
import pyarrow  # noqa: E402,F401

WAN_PROMPT = "A person is riding a bike"
WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)
FLUX_PROMPT = "A cat holding a sign that says hello world"
DEFAULT_SEED = 42


class StopAfterFirstStep(Exception):
    pass


def metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    x, y = x.float(), y.float()
    e = y - x
    p = x.square().mean().clamp_min(1e-20)
    return {
        "mse": float(e.square().mean()),
        "nmse": float(e.square().mean() / p),
        "cosine": float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)),
        "reference_rms": float(p.sqrt()),
        "error_rms": float(e.square().mean().sqrt()),
        "shape": list(x.shape),
    }


def _parse_cfg(argv: list[str]):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    sys.argv = ["collect_firststep_block_nmse.py", *argv]
    config, *_rest = DiffusionPtqRunConfig.get_parser().parse_known_args()
    return config


def _repo_diffusion() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "deepcompressor" / "examples" / "diffusion"


def _to_cpu_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.bfloat16).contiguous().clone()


def _register_store_hook(module, store: dict, key: str, pick_fn):
    def hook(_m, _i, out):
        store[key] = _to_cpu_bf16(pick_fn(out))

    return module.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# Wan
# ---------------------------------------------------------------------------


def load_wan_bf16():
    os.chdir(_repo_diffusion())
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
    print("[wan/bf16] building pipeline...", flush=True)
    return cfg.pipeline.build()


def load_wan_quant(ckpt_dir: Path):
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    os.chdir(_repo_diffusion())
    scratch = DATA_ROOT / "compare" / "firststep_block_nmse" / "ptq_load_scratch" / "wan"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = _parse_cfg(
        [
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/wan_smoke.yaml",
            f"--load-from={ckpt_dir}",
            f"--output-root={scratch}",
            f"--cache-root={scratch}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    print(f"[wan/quant] building + loading {ckpt_dir}...", flush=True)
    pipe = cfg.pipeline.build()
    model = DiffusionModelStruct.construct(pipe)
    ptq(
        model,
        cfg.quant,
        cache=None,
        load_dirpath=str(ckpt_dir),
        save_dirpath="",
        copy_on_save=False,
        save_model=False,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return pipe


def install_wan_hooks(transformer, store: dict) -> list:
    handles = []
    handles.append(
        _register_store_hook(
            transformer.patch_embedding,
            store,
            "00_patch_embedding",
            lambda out: out[0] if isinstance(out, tuple) else out,
        )
    )
    for i, block in enumerate(transformer.blocks):
        handles.append(
            _register_store_hook(
                block,
                store,
                f"{i + 1:02d}_block{i}",
                lambda out: out[0] if isinstance(out, tuple) else out,
            )
        )
    handles.append(
        _register_store_hook(
            transformer.proj_out,
            store,
            f"{len(transformer.blocks) + 1:02d}_proj_out",
            lambda out: out[0] if isinstance(out, tuple) else out,
        )
    )
    return handles


def run_wan_first_step(pipe, prompt: str, seed: int) -> None:
    def callback(_pipe, index, _timestep, kwargs):
        if index == 0:
            raise StopAfterFirstStep
        return kwargs

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


def collect_wan_acts(pipe, prompt: str, seed: int) -> dict[str, torch.Tensor]:
    store: dict[str, torch.Tensor] = {}
    handles = install_wan_hooks(pipe.transformer, store)
    run_wan_first_step(pipe, prompt, seed)
    for h in handles:
        h.remove()
    assert store, "Wan hooks captured nothing"
    return store


# ---------------------------------------------------------------------------
# Flux
# ---------------------------------------------------------------------------


def load_flux_bf16():
    from diffusers import FluxPipeline

    print("[flux/bf16] building FluxPipeline...", flush=True)
    return FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")


def load_flux_quant(ckpt_dir: Path):
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    os.chdir(_repo_diffusion())
    scratch = DATA_ROOT / "compare" / "firststep_block_nmse" / "ptq_load_scratch" / "flux"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = _parse_cfg(
        [
            "configs/model/flux.1-dev.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/fast.yaml",
            "configs/svdquant/lowmem.yaml",
            f"--calib-path={DATA_ROOT}/datasets/torch.bfloat16/flux.1-dev/fmeuler50-g3.5/qdiff/s64",
            "--calib-num-samples=16",
            f"--load-from={ckpt_dir}",
            f"--output-root={scratch}",
            f"--cache-root={scratch}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    )
    print(f"[flux/quant] building + loading {ckpt_dir}...", flush=True)
    pipe = cfg.pipeline.build()
    model = DiffusionModelStruct.construct(pipe)
    ptq(
        model,
        cfg.quant,
        cache=None,
        load_dirpath=str(ckpt_dir),
        save_dirpath="",
        copy_on_save=False,
        save_model=False,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return pipe


def install_flux_hooks(transformer, store: dict) -> list:
    handles = []
    idx = 0

    def pick_dual(out):
        # (encoder_hidden_states, hidden_states) → image stream
        if isinstance(out, tuple):
            return out[1]
        return out

    def pick_single(out):
        if isinstance(out, tuple):
            return torch.cat(out[:2], dim=1)
        return out

    handles.append(
        _register_store_hook(
            transformer.x_embedder,
            store,
            f"{idx:02d}_x_embedder",
            lambda out: out[0] if isinstance(out, tuple) else out,
        )
    )
    idx += 1
    for i, block in enumerate(transformer.transformer_blocks):
        handles.append(
            _register_store_hook(block, store, f"{idx:02d}_dual{i}", pick_dual)
        )
        idx += 1
    for i, block in enumerate(transformer.single_transformer_blocks):
        handles.append(
            _register_store_hook(block, store, f"{idx:02d}_single{i}", pick_single)
        )
        idx += 1
    handles.append(
        _register_store_hook(
            transformer.proj_out,
            store,
            f"{idx:02d}_proj_out",
            lambda out: out[0] if isinstance(out, tuple) else out,
        )
    )
    return handles


def run_flux_first_step(pipe, prompt: str, seed: int) -> None:
    def callback(_pipe, index, _timestep, kwargs):
        if index == 0:
            raise StopAfterFirstStep
        return kwargs

    try:
        pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    except StopAfterFirstStep:
        pass


def collect_flux_acts(pipe, prompt: str, seed: int) -> dict[str, torch.Tensor]:
    store: dict[str, torch.Tensor] = {}
    handles = install_flux_hooks(pipe.transformer, store)
    run_flux_first_step(pipe, prompt, seed)
    for h in handles:
        h.remove()
    assert store, "Flux hooks captured nothing"
    return store


# ---------------------------------------------------------------------------
# Compare + save
# ---------------------------------------------------------------------------


def compare_stores(bf16: dict[str, torch.Tensor], quant: dict[str, torch.Tensor]) -> list[dict]:
    keys = sorted(set(bf16) & set(quant))
    missing_b = sorted(set(quant) - set(bf16))
    missing_q = sorted(set(bf16) - set(quant))
    if missing_b or missing_q:
        print(f"[warn] key mismatch bf16_only={missing_q} quant_only={missing_b}", flush=True)
    rows = []
    for order, key in enumerate(keys):
        a, b = bf16[key], quant[key]
        if a.shape != b.shape:
            raise RuntimeError(f"shape mismatch at {key}: bf16={tuple(a.shape)} quant={tuple(b.shape)}")
        m = metrics(a, b)
        # optional CFG split for Wan-like batch=2 leading dim
        cfg = {}
        if a.ndim >= 2 and a.shape[0] == 2:
            cfg["cfg0"] = metrics(a[0:1], b[0:1])
            cfg["cfg1"] = metrics(a[1:2], b[1:2])
        if "patch_embedding" in key or "x_embedder" in key:
            stage = "embed"
        elif "_dual" in key or "_block" in key:
            stage = "block"
        elif "_single" in key:
            stage = "single"
        elif "proj_out" in key:
            stage = "proj_out"
        else:
            stage = "other"
        rows.append(
            {
                "order": order,
                "name": key,
                "stage": stage,
                **m,
                "cfg": cfg or None,
            }
        )
    rows.sort(key=lambda r: r["name"])
    # re-number by sorted name which is zero-padded
    for i, r in enumerate(rows):
        r["order"] = i
    return rows


def run_model(model: str, ckpt: Path, prompt: str, seed: int, out_dir: Path, save_acts: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if model == "wan":
        load_bf16, load_q, collect = load_wan_bf16, load_wan_quant, collect_wan_acts
        tag = "wan"
    else:
        load_bf16, load_q, collect = load_flux_bf16, load_flux_quant, collect_flux_acts
        tag = "flux"

    print(f"=== {tag} BF16 ===", flush=True)
    pipe = load_bf16()
    bf16_acts = collect(pipe, prompt, seed)
    print(f"[bf16] captured {len(bf16_acts)} tensors: {sorted(bf16_acts)}", flush=True)
    if save_acts:
        torch.save(bf16_acts, out_dir / f"{tag}_bf16_acts.pt")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"=== {tag} W4A4 ===", flush=True)
    pipe = load_q(ckpt)
    quant_acts = collect(pipe, prompt, seed)
    print(f"[quant] captured {len(quant_acts)} tensors: {sorted(quant_acts)}", flush=True)
    if save_acts:
        torch.save(quant_acts, out_dir / f"{tag}_quant_acts.pt")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    rows = compare_stores(bf16_acts, quant_acts)
    meta = {
        "model": tag,
        "prompt": prompt,
        "seed": seed,
        "ckpt": str(ckpt),
        "n_layers": len(rows),
        "rows": rows,
    }
    out_json = out_dir / f"{tag}_firststep_block_nmse.json"
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {out_json} ({len(rows)} layers)", flush=True)
    peak = max(rows, key=lambda r: r["nmse"])
    print(
        f"[summary] peak NMSE={100 * peak['nmse']:.3f}% @ {peak['name']}; "
        f"final={100 * rows[-1]['nmse']:.3f}% @ {rows[-1]['name']}",
        flush=True,
    )
    return out_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["wan", "flux", "both"], default="both")
    parser.add_argument(
        "--wan-ckpt",
        type=Path,
        default=DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16",
    )
    parser.add_argument(
        "--flux-ckpt",
        type=Path,
        default=DATA_ROOT / "ckpts" / "flux.1-dev-int4-s16",
    )
    parser.add_argument("--wan-prompt", type=str, default=WAN_PROMPT)
    parser.add_argument("--flux-prompt", type=str, default=FLUX_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_ROOT / "compare" / "firststep_block_nmse",
    )
    parser.add_argument("--save-acts", action="store_true")
    args = parser.parse_args()

    models = ["wan", "flux"] if args.model == "both" else [args.model]
    for m in models:
        if m == "wan":
            run_model(m, args.wan_ckpt, args.wan_prompt, args.seed, args.out_dir, args.save_acts)
        else:
            run_model(m, args.flux_ckpt, args.flux_prompt, args.seed, args.out_dir, args.save_acts)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
