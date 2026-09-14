#!/usr/bin/env python3
"""Compare official full SVD vs torch.svd_lowrank on one Flux smooth module.

Full search for transformer_blocks.0.attn.qkv_proj under both LowRankBranch backends.
Reports best (alpha, beta), official search objective error, and NMSE.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIFFUSION = REPO / "third_party" / "deepcompressor" / "examples" / "diffusion"

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("DEEPCOMPRESSOR_TRANSFORMER_ONLY", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

OUT_DIR = DATA_ROOT / "runs" / "profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "compare_svd_vs_svd_lowrank_qkv0.json"
OUT_LOG = OUT_DIR / "compare_svd_vs_svd_lowrank_qkv0.log"

import torch  # noqa: E402
import pyarrow  # noqa: E402,F401

_TRACK = {
    "best_alpha": None,
    "best_beta": None,
    "best_error": None,
    "best_span_pair": None,
    "history": [],
}
_ORIG_TELL = None


def log(msg: str) -> None:
    print(msg, flush=True)
    with OUT_LOG.open("a") as f:
        f.write(msg + "\n")


def reset_track() -> None:
    _TRACK["best_alpha"] = None
    _TRACK["best_beta"] = None
    _TRACK["best_error"] = None
    _TRACK["best_span_pair"] = None
    _TRACK["history"] = []


def install_tell_tracker() -> None:
    global _ORIG_TELL
    from deepcompressor.calib import smooth as smooth_mod

    if _ORIG_TELL is None:
        _ORIG_TELL = smooth_mod.SmoothCalibrator._tell

    def tracked_tell(self, error):
        alpha_beta_id, span_pair_id = self._split_candidate_id(self.candidate_id)
        alpha, beta = self.alpha_beta_pairs[alpha_beta_id]
        span_pair = self.span_mode_pairs[span_pair_id]
        prev = None if self.best_error is None else [float(x.item()) for x in self.best_error]
        _ORIG_TELL(self, error)
        new = [float(x.item()) for x in self.best_error]
        err0 = float(error[0].item()) if error and error[0] is not None else None
        became = prev != new
        _TRACK["history"].append(
            {"alpha": float(alpha), "beta": float(beta), "error": err0, "became_best": became}
        )
        if became:
            _TRACK["best_alpha"] = float(alpha)
            _TRACK["best_beta"] = float(beta)
            _TRACK["best_error"] = new[0] if new else None
            _TRACK["best_span_pair"] = [str(span_pair[0]), str(span_pair[1])]

    smooth_mod.SmoothCalibrator._tell = tracked_tell  # type: ignore[method-assign]


def install_official_svd() -> None:
    from deepcompressor.nn.patch import lowrank as lowrank_mod

    def reset_parameters(self, weight: torch.Tensor | None = None) -> None:
        if weight is None:
            if self.rank < 0:
                torch.nn.init.zeros_(self.a.weight)
            elif self.rank > 0:
                torch.nn.init.kaiming_uniform_(self.a.weight)
                torch.nn.init.zeros_(self.b.weight)
            return
        if weight.ndim >= 2:
            assert weight.shape[2:].numel() == 1
        weight = weight.view(weight.shape[0], -1)
        dtype = weight.dtype
        self.to(device=weight.device, dtype=dtype)
        assert self.in_features == weight.shape[1] and self.out_features == weight.shape[0]
        if self.rank < 0:
            self.a.weight.data.copy_(weight)
        elif self.rank > 0:
            u, s, vh = torch.linalg.svd(weight.double())
            us = u[:, : self.rank] * s[: self.rank]
            vh = vh[: self.rank]
            self.a.weight.data.copy_(vh.to(dtype))
            self.b.weight.data.copy_(us.to(dtype))

    lowrank_mod.LowRankBranch.reset_parameters = reset_parameters  # type: ignore[method-assign]
    log("[patch] LowRankBranch -> official torch.linalg.svd(weight.double())")


def install_svd_lowrank(q_extra: int = 8, niter: int = 2) -> None:
    from deepcompressor.nn.patch import lowrank as lowrank_mod

    def reset_parameters(self, weight: torch.Tensor | None = None) -> None:
        if weight is None:
            if self.rank < 0:
                torch.nn.init.zeros_(self.a.weight)
            elif self.rank > 0:
                torch.nn.init.kaiming_uniform_(self.a.weight)
                torch.nn.init.zeros_(self.b.weight)
            return
        if weight.ndim >= 2:
            assert weight.shape[2:].numel() == 1
        weight = weight.view(weight.shape[0], -1)
        dtype = weight.dtype
        self.to(device=weight.device, dtype=dtype)
        out_features, in_features = weight.shape
        assert self.in_features == in_features and self.out_features == out_features
        if self.rank < 0:
            self.a.weight.data.copy_(weight)
        elif self.rank > 0:
            q = min(min(out_features, in_features), max(self.rank + q_extra, self.rank))
            U, S, V = torch.svd_lowrank(weight.float(), q=q, niter=niter)
            us = U[:, : self.rank] * S[: self.rank]
            vh = V[:, : self.rank].transpose(0, 1)
            self.a.weight.data.copy_(vh.to(dtype))
            self.b.weight.data.copy_(us.to(dtype))

    lowrank_mod.LowRankBranch.reset_parameters = reset_parameters  # type: ignore[method-assign]
    log(f"[patch] LowRankBranch -> torch.svd_lowrank(float32, q=rank+{q_extra}, niter={niter})")


@torch.inference_mode()
def measure_nmse(attn, scale: torch.Tensor, layer_cache, layer_kwargs, quant, module_key, config_wgts) -> float:
    """NMSE of attn outputs under smooth+W4A4-lowrank vs BF16, using current LowRankBranch impl."""
    from deepcompressor.calib.smooth import ActivationSmoother, smooth_upscale_param
    from deepcompressor.quantizer import Quantizer

    mods = list(attn.qkv_proj)
    saved = [m.weight.data.detach().clone() for m in mods]
    eval_inputs = layer_cache[attn.name].inputs
    eval_kwargs = attn.filter_kwargs(layer_kwargs)
    hooks = []

    try:
        refs = []
        for i in range(len(eval_inputs.front().data)):
            ipt = eval_inputs.extract(i, eval_kwargs)
            y = attn(*ipt.args, **ipt.kwargs)
            y = y[0] if not isinstance(y, torch.Tensor) else y
            refs.append(y.detach().float().cpu())

        up = scale.to(device=mods[0].weight.device, dtype=torch.float32)
        for m in mods:
            smooth_upscale_param(m.weight, up, channels_dim=1)

        w_quantizer = Quantizer(config_wgts, key=module_key, low_rank=quant.wgts.low_rank)
        x_quantizer = Quantizer(quant.ipts, channels_dim=-1, key=module_key)
        input_packager = x_quantizer.get_input_packager() if x_quantizer.is_enabled() else None
        wgts = [m.weight for m in mods]
        qtensors, branches = w_quantizer.quantize_with_low_rank(
            wgts, kernel=None, develop_dtype=quant.develop_dtype
        )
        for qt, br, m in zip(qtensors, branches, mods, strict=True):
            m.weight.data = qt.data
            hooks.append(br.as_hook(input_packager=input_packager).register(m))
            hooks.append(
                ActivationSmoother(up, channels_dim=-1, develop_dtype=quant.develop_dtype).as_hook().register(m)
            )
            if x_quantizer.is_enabled():
                hooks.append(x_quantizer.as_hook().register(m))

        num = 0.0
        den = 0.0
        for i in range(len(eval_inputs.front().data)):
            ipt = eval_inputs.extract(i, eval_kwargs)
            y = attn(*ipt.args, **ipt.kwargs)
            y = y[0] if not isinstance(y, torch.Tensor) else y
            ref = refs[i].to(device=y.device, dtype=torch.float32)
            yf = y.float()
            num += float((yf - ref).pow(2).sum().item())
            den += float(ref.pow(2).sum().item())
        return num / den if den > 0 else float("nan")
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        for m, w in zip(mods, saved, strict=True):
            m.weight.data.copy_(w)


def run_one(label: str, quant, attn, layer_cache, layer_kwargs) -> dict:
    from deepcompressor.calib.smooth import smooth_linear_modules
    from deepcompressor.quantizer import Quantizer

    reset_track()
    mods = list(attn.qkv_proj)
    saved = [m.weight.data.detach().clone() for m in mods]

    module_key = attn.qkv_proj_key
    config_wgts = quant.wgts
    if quant.enabled_extra_wgts and quant.extra_wgts.is_enabled_for(module_key):
        config_wgts = quant.extra_wgts

    log(f"=== RUN {label} ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scale = smooth_linear_modules(
        None,
        attn.qkv_proj,
        scale=None,
        config=quant.smooth.proj,
        weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=quant.wgts.low_rank),
        input_quantizer=Quantizer(quant.ipts, channels_dim=-1, key=module_key),
        inputs=layer_cache[attn.q_proj_name].inputs,
        eval_inputs=layer_cache[attn.name].inputs,
        eval_module=attn,
        eval_kwargs=attn.filter_kwargs(layer_kwargs),
        develop_dtype=quant.develop_dtype,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # smooth_linear_modules applied scale in-place; restore before NMSE / next run
    scale_cpu = scale.detach().float().cpu().clone()
    for m, w in zip(mods, saved, strict=True):
        m.weight.data.copy_(w)

    nmse = measure_nmse(attn, scale_cpu, layer_cache, layer_kwargs, quant, module_key, config_wgts)
    for m, w in zip(mods, saved, strict=True):
        m.weight.data.copy_(w)

    result = {
        "label": label,
        "wall_sec": dt,
        "best_alpha": _TRACK["best_alpha"],
        "best_beta": _TRACK["best_beta"],
        "best_span_pair": _TRACK["best_span_pair"],
        "best_objective_error": _TRACK["best_error"],
        "nmse": nmse,
        "scale_min": float(scale_cpu.min()),
        "scale_max": float(scale_cpu.max()),
        "scale_mean": float(scale_cpu.mean()),
        "scale": scale_cpu.tolist(),
        "num_candidates": len(_TRACK["history"]),
    }
    log(
        f"[{label}] wall={dt:.1f}s alpha={result['best_alpha']} beta={result['best_beta']} "
        f"obj_error={result['best_objective_error']} nmse={nmse:.8e} "
        f"scale=[{result['scale_min']:.4f},{result['scale_max']:.4f}]"
    )
    return result


def main() -> int:
    OUT_LOG.write_text("")
    os.chdir(DIFFUSION)
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.nn.struct import (
        DiffusionAttentionStruct,
        DiffusionFeedForwardStruct,
        DiffusionModelStruct,
        DiffusionTransformerBlockStruct,
    )
    from deepcompressor.app.diffusion.quant.utils import get_needs_inputs_fn
    from deepcompressor.utils import tools

    tools.logging.setup(path=str(OUT_LOG), level=tools.logging.INFO)
    install_tell_tracker()

    argv = [
        "configs/model/flux.1-dev.yaml",
        "configs/svdquant/int4.yaml",
        "configs/svdquant/fast.yaml",
        "configs/svdquant/lowmem.yaml",
        f"--calib-path={DATA_ROOT}/datasets/torch.bfloat16/flux.1-dev/fmeuler50-g3.5/qdiff/s64",
        "--calib-num-samples=64",
        f"--output-root={DATA_ROOT}/runs/profile",
        f"--cache-root={DATA_ROOT}/runs/profile",
        "--skip-eval",
        "--skip-gen",
        "--eval-num-gpus=1",
    ]
    sys.argv = ["compare_svd_methods_one_module.py", *argv]
    config, *_rest = DiffusionPtqRunConfig.get_parser().parse_known_args()

    pipeline = config.pipeline.build()
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "vae", "image_encoder", "controlnet"):
        if hasattr(pipeline, attr):
            mod = getattr(pipeline, attr)
            if isinstance(mod, torch.nn.Module):
                mod.to("cpu")
    torch.cuda.empty_cache()
    model = DiffusionModelStruct.construct(pipeline)
    quant = config.quant

    layer = layer_cache = layer_kwargs = None
    for layer_name, (layer, layer_cache, layer_kwargs) in quant.calib.build_loader().iter_layer_activations(
        model,
        needs_inputs_fn=get_needs_inputs_fn(model, quant),
        skip_pre_modules=True,
        skip_post_modules=True,
    ):
        log(f"collected layer {layer_name}")
        break

    attn = None
    for _k, _n, _m, parent, _ in layer.named_key_modules():
        if isinstance(parent, (DiffusionAttentionStruct, DiffusionFeedForwardStruct)):
            block = parent.parent
            assert isinstance(block, DiffusionTransformerBlockStruct)
            attn = block.attn_structs[0]
            break
    assert attn is not None
    log(f"module={attn.name}.qkv_proj shapes={[tuple(m.weight.shape) for m in attn.qkv_proj]}")

    install_official_svd()
    official = run_one("official_full_svd", quant, attn, layer_cache, layer_kwargs)

    install_svd_lowrank()
    lowrank = run_one("svd_lowrank", quant, attn, layer_cache, layer_kwargs)

    s0 = torch.tensor(official["scale"])
    s1 = torch.tensor(lowrank["scale"])
    alpha_same = (
        official["best_alpha"] == lowrank["best_alpha"] and official["best_beta"] == lowrank["best_beta"]
    )
    summary = {
        "module": f"{attn.name}.qkv_proj",
        "alpha_beta_identical": alpha_same,
        "official": {k: v for k, v in official.items() if k != "scale"},
        "svd_lowrank": {k: v for k, v in lowrank.items() if k != "scale"},
        "scale_max_abs_diff": float((s0 - s1).abs().max()),
        "scale_rel_l2_diff": float((s0 - s1).norm() / (s0.norm() + 1e-12)),
        "objective_error_delta_lowrank_minus_official": (
            (lowrank["best_objective_error"] or 0) - (official["best_objective_error"] or 0)
        ),
        "nmse_delta_lowrank_minus_official": lowrank["nmse"] - official["nmse"],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    log("=" * 60)
    log(f"alpha_beta_identical = {alpha_same}")
    log(
        f"official   : alpha={official['best_alpha']} beta={official['best_beta']} "
        f"obj_error={official['best_objective_error']} nmse={official['nmse']:.8e}"
    )
    log(
        f"svd_lowrank: alpha={lowrank['best_alpha']} beta={lowrank['best_beta']} "
        f"obj_error={lowrank['best_objective_error']} nmse={lowrank['nmse']:.8e}"
    )
    log(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        with OUT_LOG.open("a") as f:
            f.write(traceback.format_exc())
        raise
