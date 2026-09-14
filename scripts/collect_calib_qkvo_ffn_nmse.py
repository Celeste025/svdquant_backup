#!/usr/bin/env python3
"""Teacher-forced calib local NMSE for Q/K/V/O linears and composed FFN (up+act+down).

Complements ``collect_calib_local_block_nmse.py`` (whole attn / ffn_up / ffn_down):
  - per-proj: q, k, v, o (and add_* for cross/joint)
  - ffn_compose: full FeedForward (or Flux-single virtual up→act→down)

Same PTQ-aligned calib sampling as the parent script.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from collections import defaultdict
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
import torch.nn as nn  # noqa: E402
import pyarrow  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
DIFFUSION = REPO / "third_party" / "deepcompressor" / "examples" / "diffusion"
OUT_ROOT = DATA_ROOT / "compare" / "calib_qkvo_ffn_nmse"

from collect_calib_local_block_nmse import (  # noqa: E402
    FLUX_CALIB,
    FLUX_CKPT,
    FLUX_NUM_SAMPLES,
    FLUX_SAMPLE_BATCH,
    FLUX_SAMPLE_SIZE,
    WAN_CALIB,
    WAN_CKPT,
    WAN_NUM_SAMPLES,
    WAN_SAMPLE_BATCH,
    WAN_SAMPLE_SIZE,
    _parse_cfg,
    block_id_from_name,
    flat_sample_steps,
    list_selected_metas,
    log,
    measure_eval_module,
    resolve_quant_module,
)

# Keep add_* / ctx distinct — do NOT fold into main-stream q/k/v/o or ffn_compose.
KIND_ALIASES = {
    "q": "q",
    "k": "k",
    "v": "v",
    "o": "o",
    "add_q": "add_q",
    "add_k": "add_k",
    "add_v": "add_v",
    "add_o": "add_o",
    "ffn_compose": "ffn_compose",
    "ffn_compose_ctx": "ffn_compose_ctx",
}


def _kind_from_module_name(name: str, fallback: str) -> str:
    """Prefer path suffix over struct field (Wan cross to_k was mislabeled add_k)."""
    n = name.replace("\\", "/")
    if n.endswith("to_add_out") or n.endswith(".to_add_out"):
        return "add_o"
    if "proj_out.linears.0" in n:  # Flux-single image o half
        return "o"
    if n.endswith("to_out.0") or n.endswith(".to_out.0"):
        return "o"
    if n.endswith("add_q_proj"):
        return "add_q"
    if n.endswith("add_k_proj"):
        return "add_k"
    if n.endswith("add_v_proj"):
        return "add_v"
    if n.endswith("to_q"):
        return "q"
    if n.endswith("to_k"):
        return "k"
    if n.endswith("to_v"):
        return "v"
    return fallback


class _ComposeFFN(nn.Module):
    """up → act → down for Flux-single virtual FFN (modules live on the block)."""

    def __init__(self, up: nn.Module, act: nn.Module, down: nn.Module):
        super().__init__()
        self.up = up
        self.act = act
        self.down = down

    def forward(self, x):
        return self.down(self.act(self.up(x)))


def _row(name: str, kind: str, stats: dict) -> dict:
    return {
        "name": name,
        "block_id": block_id_from_name(name),
        "kind": kind,
        "kind_canon": KIND_ALIASES.get(kind, kind),
        **stats["aligned"],
        "mean_batch_nmse_full": stats["mean_batch_nmse_full"],
        "n_batches_full": stats["n_batches_full"],
        "n_samples_seen": stats.get("n_samples_seen"),
        "step_align_ok": stats.get("step_align_ok"),
        "n_steps_seen": len(stats.get("steps_seen") or []),
    }


def _measure_one(
    name: str,
    kind: str,
    fp_mod: nn.Module,
    q_mod: nn.Module,
    eval_inputs,
    eval_kwargs: dict,
    sample_batch_size: int,
    sample_size: int,
    sample_steps: list,
    device,
    modules_out: list,
    per_layer_per_step: dict,
    measure: str = "linear_isolated",
) -> None:
    _move_mod_and_branches(q_mod, device)
    try:
        stats = measure_eval_module(
            fp_mod,
            q_mod,
            eval_inputs,
            eval_kwargs,
            sample_batch_size=sample_batch_size,
            sample_size=sample_size,
            sample_steps=sample_steps,
        )
        row = _row(name, kind, stats)
        row["measure"] = measure
        modules_out.append(row)
        per_layer_per_step[name] = stats["per_step"]
        log(
            f"  [{kind}] {name} nmse={100*row['nmse']:.4f}% "
            f"({measure}) sum_ratio={100*row['nmse_sum_ratio']:.4f}% n_act={row['n_act_batches']}"
        )
    finally:
        q_mod.to("cpu")
        torch.cuda.empty_cache()


def _move_mod_and_branches(mod: nn.Module, device) -> None:
    """Move module and any AccumBranchHook LowRankBranch payloads."""
    mod.to(device)
    for hook_dict in (getattr(mod, "_forward_hooks", {}), getattr(mod, "_forward_pre_hooks", {})):
        for h in hook_dict.values():
            br = getattr(h, "branch", None)
            if isinstance(br, nn.Module):
                br.to(device)


def _register_out_hook(mod: nn.Module, bucket: dict, key: str):
    def _hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        bucket[key] = t.detach()

    return mod.register_forward_hook(_hook)


@torch.inference_mode()
def measure_attn_projs_in_forward(
    fp_attn: nn.Module,
    q_attn: nn.Module,
    proj_specs: list[tuple[str, str, nn.Module, nn.Module]],
    eval_inputs,
    eval_kwargs: dict,
    sample_batch_size: int,
    sample_size: int,
    sample_steps: list,
    device,
) -> dict[str, dict]:
    """Compare QKVO (and add_*) by hooking outputs inside a full Attention forward.

    Isolated Linear forwards miss SVDQuant fused-qkv / smooth / branch context and can
    produce spuriously huge NMSE; in-attention hooks match the whole-attn protocol.
    """
    from collect_calib_local_block_nmse import repartition_eval_inputs, nmse_tensors, move_ipt_to_device

    # Accumulated sum-sq over aligned (sample_size) path
    acc: dict[str, dict] = {
        name: {"sum_sq_err": 0.0, "sum_ref_sq": 0.0, "batch_nmses": []} for name, _k, _f, _q in proj_specs
    }

    def _lead_bs(ipt) -> int:
        if ipt.args and torch.is_tensor(ipt.args[0]):
            return int(ipt.args[0].shape[0])
        for v in ipt.kwargs.values():
            if torch.is_tensor(v):
                return int(v.shape[0])
        raise RuntimeError("no tensor lead in eval inputs")

    def _slice_ipt(ipt, start: int, end: int):
        args = []
        for a in ipt.args:
            if torch.is_tensor(a) and a.shape[0] >= end:
                args.append(a[start:end])
            else:
                args.append(a)
        kwargs = {}
        for k, v in ipt.kwargs.items():
            if torch.is_tensor(v) and v.shape[0] >= end:
                kwargs[k] = v[start:end]
            else:
                kwargs[k] = v
        return args, kwargs

    def _forward_pair_collect(ei_cache, idx: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        ipt = ei_cache.extract(idx, eval_kwargs)
        bs = _lead_bs(ipt)
        # per-sample to limit memory; concat proj outs
        fp_parts: dict[str, list] = {n: [] for n, _, _, _ in proj_specs}
        q_parts: dict[str, list] = {n: [] for n, _, _, _ in proj_specs}
        for s in range(bs):
            args_s, kwargs_s = _slice_ipt(ipt, s, s + 1)
            args_s = [a.to(device, non_blocking=True) if torch.is_tensor(a) else a for a in args_s]
            kwargs_s = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in kwargs_s.items()
            }
            fp_bucket, q_bucket = {}, {}
            handles = []
            for name, _kind, fp_proj, q_proj in proj_specs:
                handles.append(_register_out_hook(fp_proj, fp_bucket, name))
                handles.append(_register_out_hook(q_proj, q_bucket, name))
            try:
                _ = fp_attn(*args_s, **kwargs_s)
                _ = q_attn(*args_s, **kwargs_s)
            finally:
                for h in handles:
                    h.remove()
            for name, _, _, _ in proj_specs:
                if name not in fp_bucket or name not in q_bucket:
                    raise RuntimeError(f"hook miss for {name}: fp={list(fp_bucket)} q={list(q_bucket)}")
                fp_parts[name].append(fp_bucket[name].float().cpu())
                q_parts[name].append(q_bucket[name].float().cpu())
            del args_s, kwargs_s, fp_bucket, q_bucket
            torch.cuda.empty_cache()
        out = {}
        for name, _, _, _ in proj_specs:
            out[name] = (torch.cat(fp_parts[name], 0), torch.cat(q_parts[name], 0))
        return out

    # Aligned sample_size path (matches parent script's primary NMSE)
    ei = repartition_eval_inputs(eval_inputs, sample_batch_size, sample_size)
    n_act = len(ei.front().data)
    for i in range(n_act):
        pairs = _forward_pair_collect(ei, i)
        for name, (yf, yq) in pairs.items():
            m = nmse_tensors(yf, yq)
            acc[name]["sum_sq_err"] += float(m["sum_sq_err"])
            acc[name]["sum_ref_sq"] += float(m["sum_ref_sq"])
            acc[name]["batch_nmses"].append(m["nmse"])
        torch.cuda.empty_cache()

    results = {}
    for name, kind, _, _ in proj_specs:
        a = acc[name]
        sum_ref = a["sum_ref_sq"]
        results[name] = {
            "kind": kind,
            "aligned": {
                "nmse": float(np.mean(a["batch_nmses"])) if a["batch_nmses"] else float("nan"),
                "nmse_sum_ratio": (a["sum_sq_err"] / max(sum_ref, 1e-20)) if sum_ref == sum_ref else float("nan"),
                "sum_sq_err": a["sum_sq_err"],
                "rmse_like": a["sum_sq_err"] ** 0.5 if a["sum_sq_err"] == a["sum_sq_err"] else float("nan"),
                "sum_ref_sq": sum_ref,
                "n_act_batches": n_act,
            },
            "mean_batch_nmse_full": float(np.mean(a["batch_nmses"])) if a["batch_nmses"] else float("nan"),
            "n_batches_full": n_act,
            "n_samples_seen": None,
            "step_align_ok": None,
            "steps_seen": [],
            "per_step": {},
        }
    return results


def _attn_proj_pair_specs(attn, quant_attn_mod: nn.Module) -> list[tuple[str, str, nn.Module, nn.Module]]:
    """(name, kind, fp_proj, quant_proj) resolved against quant Attention module."""
    qmap = dict(quant_attn_mod.named_modules())
    # also allow absolute names via root later
    specs = []

    def _q_resolve(rname: str, abs_name: str) -> nn.Module | None:
        if rname and rname in qmap:
            return qmap[rname]
        # strip attn prefix: blocks.0.attn1.to_q -> to_q
        short = abs_name.split(".")[-1] if abs_name else ""
        if short in qmap:
            return qmap[short]
        # to_out.0
        if abs_name.endswith("to_out.0") and "to_out.0" in qmap:
            return qmap["to_out.0"]
        if abs_name.endswith("to_out.0"):
            # nested
            for k, m in qmap.items():
                if k.endswith("to_out.0") or k == "0" and "to_out" in abs_name:
                    pass
            if "to_out" in qmap and hasattr(qmap["to_out"], "__getitem__"):
                try:
                    return qmap["to_out"][0]
                except Exception:
                    pass
        return None

    pairs = [
        ("q_proj", "q", attn.q_proj, attn.q_proj_rname, attn.q_proj_name),
        ("k_proj", "k", attn.k_proj, attn.k_proj_rname, attn.k_proj_name),
        ("v_proj", "v", attn.v_proj, attn.v_proj_rname, attn.v_proj_name),
        ("add_q_proj", "add_q", attn.add_q_proj, attn.add_q_proj_rname, attn.add_q_proj_name),
        ("add_k_proj", "add_k", attn.add_k_proj, attn.add_k_proj_rname, attn.add_k_proj_name),
        ("add_v_proj", "add_v", attn.add_v_proj, attn.add_v_proj_rname, attn.add_v_proj_name),
        ("o_proj", "o", attn.o_proj, attn.o_proj_rname, attn.o_proj_name),
        ("add_o_proj", "add_o", attn.add_o_proj, attn.add_o_proj_rname, attn.add_o_proj_name),
    ]
    for _field, kind, fp_m, rname, abs_name in pairs:
        if fp_m is None or not abs_name:
            continue
        qm = _q_resolve(rname, abs_name)
        if qm is None:
            # Flux-single o_proj lives on parent; skip here (handled separately if needed)
            continue
        kind_fix = _kind_from_module_name(abs_name, kind)
        specs.append((abs_name, kind_fix, fp_m, qm))
    return specs


def _attn_proj_targets(attn) -> list[tuple[str, str, nn.Module, str]]:
    """Legacy list kept for reference; QKVO now measured via in-attention hooks."""
    out: list[tuple[str, str, nn.Module, str]] = []
    if attn.q_proj is not None and attn.q_proj_name:
        out.append((attn.q_proj_name, "q", attn.q_proj, attn.q_proj_name))
    if attn.k_proj is not None and attn.k_proj_name:
        out.append((attn.k_proj_name, "k", attn.k_proj, attn.q_proj_name))
    if attn.v_proj is not None and attn.v_proj_name:
        out.append((attn.v_proj_name, "v", attn.v_proj, attn.q_proj_name))
    add_cache = attn.add_k_proj_name or attn.add_q_proj_name
    if attn.add_q_proj is not None and attn.add_q_proj_name:
        out.append((attn.add_q_proj_name, "add_q", attn.add_q_proj, add_cache))
    if attn.add_k_proj is not None and attn.add_k_proj_name:
        out.append((attn.add_k_proj_name, "add_k", attn.add_k_proj, add_cache))
    if attn.add_v_proj is not None and attn.add_v_proj_name:
        out.append((attn.add_v_proj_name, "add_v", attn.add_v_proj, add_cache))
    if attn.o_proj is not None and attn.o_proj_name:
        out.append((attn.o_proj_name, "o", attn.o_proj, attn.o_proj_name))
    if attn.add_o_proj is not None and attn.add_o_proj_name:
        out.append((attn.add_o_proj_name, "add_o", attn.add_o_proj, attn.add_o_proj_name))
    return out


def _build_flux_single_compose(block_mod: nn.Module) -> _ComposeFFN:
    down = block_mod.proj_out.linears[1]
    return _ComposeFFN(block_mod.proj_mlp, block_mod.act_mlp, down)


@torch.inference_mode()
def run_model(
    model: str,
    calib_path: Path,
    ckpt: Path,
    num_samples: int,
    sample_size: int,
    sample_batch_size: int,
    out_dir: Path,
) -> dict:
    from deepcompressor.app.diffusion.nn.struct import (
        DiffusionModelStruct,
        DiffusionTransformerBlockStruct,
    )
    from deepcompressor.app.diffusion.ptq import ptq
    from deepcompressor.app.diffusion.quant.utils import get_needs_inputs_fn

    os.chdir(DIFFUSION)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model == "wan":
        base_argv = [
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/wan_s16.yaml",
            f"--calib-path={calib_path}",
            f"--calib-num-samples={num_samples}",
            f"--output-root={out_dir / 'scratch'}",
            f"--cache-root={out_dir / 'scratch'}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]
    else:
        base_argv = [
            "configs/model/flux.1-dev.yaml",
            "configs/svdquant/int4.yaml",
            "configs/svdquant/flux_s16.yaml",
            f"--calib-path={calib_path}",
            f"--calib-num-samples={num_samples}",
            f"--output-root={out_dir / 'scratch'}",
            f"--cache-root={out_dir / 'scratch'}",
            "--skip-eval",
            "--skip-gen",
            "--eval-num-gpus=1",
        ]

    def _force_sample_knobs(c):
        c.quant.calib.num_samples = num_samples
        c.quant.calib.path = str(calib_path)
        for attr in ("proj", "attn"):
            sc = getattr(c.quant.smooth, attr, None)
            if sc is not None and hasattr(sc, "sample_size"):
                sc.sample_size = sample_size
                sc.sample_batch_size = sample_batch_size

    log(f"=== {model}: parse config ===")
    cfg = _parse_cfg(base_argv)
    _force_sample_knobs(cfg)

    metas = list_selected_metas(calib_path, num_samples, seed=0)
    loader_bs = cfg.quant.calib.batch_size
    sample_steps = flat_sample_steps(metas, loader_bs)
    log(
        f"[{model}] selected {len(metas)} files; loader_bs={loader_bs}; "
        f"n_sample_steps={len(sample_steps)}; sample_size={sample_size}"
    )
    (out_dir / "selected_files.json").write_text(
        json.dumps({"metas": metas, "sample_steps": sample_steps, "loader_batch_size": loader_bs}, indent=2)
    )

    log(f"[{model}] load quant to CPU from {ckpt}...")
    cfg_q = _parse_cfg([*base_argv, f"--load-from={ckpt}"])
    _force_sample_knobs(cfg_q)
    quant_pipe = cfg_q.pipeline.build()
    quant_struct = DiffusionModelStruct.construct(quant_pipe)
    ptq(
        quant_struct,
        cfg_q.quant,
        cache=None,
        load_dirpath=str(ckpt),
        save_dirpath="",
        copy_on_save=False,
        save_model=False,
    )
    quant_nn = quant_struct.module
    quant_nn.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    bf16_pipe = cfg.pipeline.build()
    bf16_struct = DiffusionModelStruct.construct(bf16_pipe)
    device = next(bf16_struct.module.parameters()).device
    base_needs = get_needs_inputs_fn(bf16_struct, cfg.quant)
    quant_mods = dict(quant_nn.named_modules())

    def needs(name: str, module) -> bool:
        if base_needs(name, module):
            return True
        # ActivationCache only supports Linear / Attention / Conv — never ShiftedLinear etc.
        if not isinstance(module, nn.Linear):
            # Still allow Attention modules (whole-attn hooks used by default calib).
            cls = module.__class__.__name__
            if "Attention" not in cls:
                return False
            if name.endswith(".attn") or name.endswith(".attn1") or name.endswith(".attn2"):
                return True
            return False
        if name.endswith(".to_q") or name.endswith(".to_k") or name.endswith(".to_v"):
            return True
        if name.endswith("to_out.0") or name.endswith(".to_out.0"):
            return True
        if name.endswith("add_q_proj") or name.endswith("add_k_proj") or name.endswith("add_v_proj"):
            return True
        if name.endswith("to_add_out"):
            return True
        if name.endswith(".proj_mlp"):
            return True
        # Flux-single o_proj half of ConcatLinear (must be bare Linear, not ShiftedLinear).
        if name.endswith("proj_out.linears.0") or name.endswith("proj_out.linears.0.linear"):
            return True
        if name.endswith("proj_out.linears.1.linear"):
            return True
        return False

    modules_out: list[dict] = []
    per_layer_per_step: dict[str, dict] = {}

    log(f"[{model}] online measure QKVO + FFN compose...")
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
            log(f"  [skip] {layer_name} type={type(layer).__name__}")
            continue

        for tb in tblocks:
            # --- QKVO / add_*: hook inside full Attention forward (not isolated Linear) ---
            for attn in tb.iter_attention_structs():
                if attn.name not in layer_cache or layer_cache[attn.name].inputs is None:
                    log(f"  [skip-cache] attn {attn.name}")
                    continue
                try:
                    q_attn = resolve_quant_module(quant_nn, attn.name)
                except KeyError:
                    log(f"  [skip-miss] attn {attn.name}")
                    continue

                # Projections that live on the Attention module itself
                specs = _attn_proj_pair_specs(attn, q_attn)
                # Flux-single o_proj sits on the block (rname starts with '.'); measure later in isolation
                external_o = (
                    attn.o_proj is not None
                    and attn.o_proj_name
                    and (attn.o_proj_rname.startswith(".") or attn.o_proj not in set(attn.module.modules()))
                )
                if external_o:
                    specs = [s for s in specs if s[1] != "o"]

                # o / add_o must be teacher-forced on BF16 *inputs to the Linear*
                # (isolated). Measuring them via in-attn output hooks compares
                # o(AV_fp) vs o(AV_q) ≡ whole-attn NMSE — that was the bug.
                hook_specs = [s for s in specs if s[1] not in ("o", "add_o")]
                isolated_out_specs = [s for s in specs if s[1] in ("o", "add_o")]

                eval_kwargs = attn.filter_kwargs(layer_kwargs)
                try:
                    _move_mod_and_branches(q_attn, device)
                    for _n, _k, _fp, qp in hook_specs:
                        _move_mod_and_branches(qp, device)
                    if hook_specs:
                        res_map = measure_attn_projs_in_forward(
                            attn.module,
                            q_attn,
                            hook_specs,
                            layer_cache[attn.name].inputs,
                            eval_kwargs,
                            sample_batch_size,
                            sample_size,
                            sample_steps,
                            device,
                        )
                        for name, payload in res_map.items():
                            kind = _kind_from_module_name(name, payload["kind"])
                            row = {
                                "name": name,
                                "block_id": block_id_from_name(name),
                                "kind": kind,
                                "kind_canon": KIND_ALIASES.get(kind, kind),
                                **payload["aligned"],
                                "mean_batch_nmse_full": payload["mean_batch_nmse_full"],
                                "n_batches_full": payload["n_batches_full"],
                                "n_samples_seen": payload.get("n_samples_seen"),
                                "step_align_ok": payload.get("step_align_ok"),
                                "n_steps_seen": len(payload.get("steps_seen") or []),
                                "measure": "attn_hook",
                            }
                            modules_out.append(row)
                            per_layer_per_step[name] = payload.get("per_step") or {}
                            log(
                                f"  [{kind}] {name} nmse={100*row['nmse']:.4f}% "
                                f"(in-attn hook) n_act={row['n_act_batches']}"
                            )

                    def _measure_isolated_out(name: str, kind: str, fp_m: nn.Module, q_m: nn.Module) -> None:
                        cache_key = name
                        if cache_key not in layer_cache or layer_cache[cache_key].inputs is None:
                            log(f"  [skip-cache] isolated {kind} {name}")
                            return
                        _measure_one(
                            name,
                            kind,
                            fp_m,
                            q_m,
                            layer_cache[cache_key].inputs,
                            {},
                            sample_batch_size,
                            sample_size,
                            sample_steps,
                            device,
                            modules_out,
                            per_layer_per_step,
                            measure="linear_isolated",
                        )

                    for name, kind, fp_m, q_m in isolated_out_specs:
                        _measure_isolated_out(name, kind, fp_m, q_m)

                    if external_o and attn.o_proj_name in quant_mods:
                        _measure_isolated_out(
                            attn.o_proj_name,
                            "o",
                            attn.o_proj,
                            quant_mods[attn.o_proj_name],
                        )
                except torch.cuda.OutOfMemoryError as exc:
                    log(f"  [OOM] attn {attn.name}: {exc}")
                except Exception as exc:
                    log(f"  [warn] attn {attn.name}: {exc}")
                    traceback.print_exc()
                finally:
                    q_attn.to("cpu")
                    for _n, _k, _fp, qp in hook_specs:
                        try:
                            qp.to("cpu")
                        except Exception:
                            pass
                    torch.cuda.empty_cache()

            # --- composed FFN (and context FFN if present) ---
            for ffn, kind in (
                (getattr(tb, "ffn_struct", None), "ffn_compose"),
                (getattr(tb, "add_ffn_struct", None), "ffn_compose_ctx"),
            ):
                if ffn is None:
                    continue
                up_name = getattr(ffn, "up_proj_name", None)
                if not up_name or up_name not in layer_cache or layer_cache[up_name].inputs is None:
                    log(f"  [skip-cache] ffn compose up={up_name}")
                    continue
                eval_inputs = layer_cache[up_name].inputs

                # Real FeedForward (Wan / Flux dual) vs Flux-single virtual Sequential.
                fp_ffn = ffn.module
                compose_name = ffn.name or f"{tb.name}.ffn_compose"
                # Empty rname ⇒ FFN is the block itself (Flux single); Sequential is virtual.
                use_virtual = (getattr(ffn, "rname", None) == "") or isinstance(fp_ffn, nn.Sequential)

                try:
                    if use_virtual:
                        fp_block = tb.module
                        if not hasattr(fp_block, "proj_mlp"):
                            log(f"  [skip-ffn] cannot compose virtual FFN for {tb.name}")
                            continue
                        q_block = resolve_quant_module(quant_nn, tb.name)
                        fp_wrap = _build_flux_single_compose(fp_block)
                        q_wrap = _build_flux_single_compose(q_block)
                        compose_name = f"{tb.name}.ffn_compose"
                        _move_mod_and_branches(q_block, device)
                        q_wrap.to(device)
                        stats = measure_eval_module(
                            fp_wrap,
                            q_wrap,
                            eval_inputs,
                            {},
                            sample_batch_size=sample_batch_size,
                            sample_size=sample_size,
                            sample_steps=sample_steps,
                        )
                        row = _row(compose_name, kind, stats)
                        row["measure"] = "ffn_compose"
                        modules_out.append(row)
                        per_layer_per_step[compose_name] = stats["per_step"]
                        log(
                            f"  [{kind}] {compose_name} nmse={100*row['nmse']:.4f}% "
                            f"(virtual up→act→down)"
                        )
                        q_block.to("cpu")
                    else:
                        if not ffn.name or ffn.name not in quant_mods:
                            log(f"  [skip-ffn] missing quant module {ffn.name}")
                            continue
                        _measure_one(
                            compose_name,
                            kind,
                            fp_ffn,
                            quant_mods[ffn.name],
                            eval_inputs,
                            {},
                            sample_batch_size,
                            sample_size,
                            sample_steps,
                            device,
                            modules_out,
                            per_layer_per_step,
                            measure="ffn_compose",
                        )
                except torch.cuda.OutOfMemoryError as exc:
                    log(f"  [OOM] {compose_name}: {exc}")
                except Exception as exc:
                    log(f"  [warn] {compose_name}: {exc}")
                    traceback.print_exc()
                finally:
                    torch.cuda.empty_cache()

        layer_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    by_kind: dict[str, list[float]] = defaultdict(list)
    for r in modules_out:
        by_kind[r.get("kind_canon", r["kind"])].append(r["nmse"])
    kind_summary = {
        k: {
            "n": len(vs),
            "nmse_mean": float(np.mean(vs)),
            "nmse_max": float(np.max(vs)),
            "nmse_median": float(np.median(vs)),
        }
        for k, vs in sorted(by_kind.items())
    }
    log(f"[{model}] measured={len(modules_out)} kind_summary={json.dumps(kind_summary)}")

    result = {
        "model": model,
        "ckpt": str(ckpt),
        "calib_path": str(calib_path),
        "num_samples": num_samples,
        "sample_size": sample_size,
        "sample_batch_size": sample_batch_size,
        "loader_batch_size": loader_bs,
        "n_modules": len(modules_out),
        "modules": modules_out,
        "kind_summary": kind_summary,
        "per_layer_per_step": per_layer_per_step,
        "note": (
            "Teacher-forced local NMSE. "
            "q/k/v(/add_*) via in-Attention output hooks (same Attention inputs; "
            "avoids broken isolated fused-qkv). "
            "o/add_o via linear_isolated on BF16-cached Linear inputs "
            "(NOT in-attn hooks — those equalled whole-attn NMSE). "
            "ffn_compose = full FeedForward (up+act+down); "
            "ffn_compose_ctx kept separate for Flux dual context FFN."
        ),
    }
    out_json = out_dir / f"{model}_qkvo_ffn_nmse.json"
    out_json.write_text(json.dumps(result, indent=2))
    log(f"[{model}] wrote {out_json}")

    plot_model_kind_curves(result, out_dir / f"{model}_qkvo_ffn_by_kind.png")
    del bf16_pipe, bf16_struct, quant_pipe, quant_struct
    gc.collect()
    torch.cuda.empty_cache()
    return result


def plot_model_kind_curves(res: dict, out_png: Path) -> None:
    mods = res.get("modules") or []
    if not mods:
        return
    kinds = sorted({r["kind"] for r in mods})
    n = len(kinds)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), dpi=150, squeeze=False)
    for ax, kind in zip(axes.ravel(), kinds):
        sub = [r for r in mods if r["kind"] == kind]
        # stable block order
        sub = sorted(sub, key=lambda r: (block_id_from_name(r["name"]), r["name"]))
        ys = [100 * r["nmse"] for r in sub]
        ax.plot(range(len(ys)), ys, "o-", ms=3, lw=1.2)
        ax.set_title(f"{res['model'].upper()} {kind}  mean={float(np.mean(ys)):.3f}%")
        ax.set_ylabel("NMSE (%)")
        ax.set_xlabel("module index")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")


def plot_wan_vs_flux(wan: dict, flux: dict, out_dir: Path) -> None:
    """Overlay Wan vs Flux for canonical kinds q/k/v/o/ffn_compose."""
    canon_order = ["q", "k", "v", "o", "ffn_compose"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=150)
    axes = axes.ravel()
    summary_lines = ["# QKVO + FFN-compose calib local NMSE", ""]

    for ax, canon in zip(axes, canon_order):
        for tag, res, color in (("Wan", wan, "#1f4e79"), ("Flux", flux, "#b85c38")):
            sub = [r for r in res["modules"] if r.get("kind_canon") == canon]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: (r["block_id"], r["name"]))
            ys = [100 * r["nmse"] for r in sub]
            ax.plot(range(len(ys)), ys, "o-", ms=2.5, lw=1.1, label=f"{tag} (n={len(ys)})", color=color)
            summary_lines.append(
                f"- {tag} {canon}: n={len(ys)} mean={float(np.mean(ys)):.4f}% "
                f"median={float(np.median(ys)):.4f}% max={float(np.max(ys)):.4f}%"
            )
        ax.set_title(canon)
        ax.set_ylabel("NMSE (%)")
        ax.set_xlabel("module index (model order)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, frameon=False)

    # Summary bar: mean NMSE by canon kind
    ax = axes[5]
    x = np.arange(len(canon_order))
    w = 0.35
    for i, (tag, res, color) in enumerate((("Wan", wan, "#1f4e79"), ("Flux", flux, "#b85c38"))):
        means = []
        for canon in canon_order:
            vs = [r["nmse"] for r in res["modules"] if r.get("kind_canon") == canon]
            means.append(100 * float(np.mean(vs)) if vs else float("nan"))
        ax.bar(x + (i - 0.5) * w, means, width=w, label=tag, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(canon_order)
    ax.set_ylabel("mean NMSE (%)")
    ax.set_title("mean NMSE by kind")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Calib local NMSE — Q/K/V/O projs & composed FFN (teacher-forced, W4A4)", fontsize=12)
    fig.tight_layout()
    out_png = out_dir / "wan_vs_flux_qkvo_ffn_nmse.png"
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")

    # Also dump a compact boxplot-style distribution
    fig2, ax2 = plt.subplots(figsize=(10, 4.5), dpi=150)
    data, labels, colors = [], [], []
    for canon in canon_order:
        for tag, res, color in (("Wan", wan, "#1f4e79"), ("Flux", flux, "#b85c38")):
            vs = [100 * r["nmse"] for r in res["modules"] if r.get("kind_canon") == canon]
            if not vs:
                continue
            data.append(vs)
            labels.append(f"{tag}\n{canon}")
            colors.append(color)
    bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax2.set_ylabel("NMSE (%)")
    ax2.set_title("Wan vs Flux — QKVO / FFN-compose NMSE distribution")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    out_box = out_dir / "wan_vs_flux_qkvo_ffn_nmse_box.png"
    fig2.savefig(out_box)
    plt.close(fig2)
    log(f"[plot] {out_box}")

    summary_lines.append("")
    summary_lines.append(f"- overlay: `{out_png}`")
    summary_lines.append(f"- boxplot: `{out_box}`")
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    log(f"[summary] {out_dir / 'SUMMARY.md'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["wan", "flux", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing JSONs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wan_res = flux_res = None
    if args.plot_only:
        wp = args.out_dir / "wan" / "wan_qkvo_ffn_nmse.json"
        fp = args.out_dir / "flux" / "flux_qkvo_ffn_nmse.json"
        if wp.is_file():
            wan_res = json.loads(wp.read_text())
        if fp.is_file():
            flux_res = json.loads(fp.read_text())
    else:
        if args.model in ("wan", "both"):
            wan_res = run_model(
                "wan",
                WAN_CALIB,
                WAN_CKPT,
                WAN_NUM_SAMPLES,
                WAN_SAMPLE_SIZE,
                WAN_SAMPLE_BATCH,
                args.out_dir / "wan",
            )
        if args.model in ("flux", "both"):
            flux_res = run_model(
                "flux",
                FLUX_CALIB,
                FLUX_CKPT,
                FLUX_NUM_SAMPLES,
                FLUX_SAMPLE_SIZE,
                FLUX_SAMPLE_BATCH,
                args.out_dir / "flux",
            )

    if wan_res is None:
        wp = args.out_dir / "wan" / "wan_qkvo_ffn_nmse.json"
        if wp.is_file():
            wan_res = json.loads(wp.read_text())
    if flux_res is None:
        fp = args.out_dir / "flux" / "flux_qkvo_ffn_nmse.json"
        if fp.is_file():
            flux_res = json.loads(fp.read_text())

    if wan_res and flux_res:
        plot_wan_vs_flux(wan_res, flux_res, args.out_dir)

    log(f"ALL DONE → {args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
