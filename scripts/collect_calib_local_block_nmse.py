#!/usr/bin/env python3
"""Teacher-forced local calib NMSE: BF16 layer inputs vs final W4A4 modules.

Aligns with PTQ sampling (NOT full cache):
  Wan ungated s16: num_samples=64, sample_size=32, sample_batch_size=4  (from wan_s16_ptq.log)
  Flux s16:        num_samples=16, sample_size=-1, sample_batch_size=16

Records per-module / per-block aggregate NMSE and layer×denoising-step NMSE.
Does not retrain or modify deepcompressor.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
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

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pyarrow  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]
DIFFUSION = REPO / "third_party" / "deepcompressor" / "examples" / "diffusion"
OUT_ROOT = DATA_ROOT / "compare" / "calib_local_block_nmse"

WAN_CALIB = DATA_ROOT / "datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16"
FLUX_CALIB = DATA_ROOT / "datasets/torch.bfloat16/flux.1-dev/fmeuler50-g3.5/qdiff/s16"
WAN_CKPT = DATA_ROOT / "ckpts/wan2.1-1.3b-int4-s16"
FLUX_CKPT = DATA_ROOT / "ckpts/flux.1-dev-int4-s16"

# Hard-aligned to the ckpt PTQ logs (yaml may have drifted to sample_size=-1).
WAN_NUM_SAMPLES = 64
WAN_SAMPLE_SIZE = 32
WAN_SAMPLE_BATCH = 4
FLUX_NUM_SAMPLES = 16
FLUX_SAMPLE_SIZE = -1
FLUX_SAMPLE_BATCH = 16

STEP_RE = re.compile(r"^(?P<prefix>.+)-(?P<step>\d{5})-(?P<guid>\d+)\.pt$")


def log(msg: str) -> None:
    print(msg, flush=True)


def _parse_cfg(argv: list[str]):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    sys.argv = ["collect_calib_local_block_nmse.py", *argv]
    config, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args()
    return config


def parse_step_guid(name: str) -> tuple[int | None, int | None]:
    m = STEP_RE.match(name)
    if not m:
        return None, None
    return int(m.group("step")), int(m.group("guid"))


def list_selected_metas(calib_path: Path, num_samples: int, seed: int = 0) -> list[dict]:
    """Mirror DiffusionDataset file select + DiffusionCalibDataset data shuffle."""
    path = calib_path
    if (path / "caches").is_dir():
        path = path / "caches"
    filenames = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
    if num_samples > 0 and num_samples < len(filenames):
        random.Random(seed).shuffle(filenames)
        filenames = filenames[:num_samples]
        filenames = sorted(filenames)
    metas = []
    for fn in filenames:
        step, guid = parse_step_guid(fn)
        metas.append({"file": fn, "step": step, "guidance": guid, "path": str(path / fn)})
    random.Random(seed).shuffle(metas)
    return metas


def batches_to_steps(metas: list[dict], batch_size: int) -> list[list[int | None]]:
    """Per dataloader batch (drop_last), the step ids of samples in that batch."""
    n = (len(metas) // batch_size) * batch_size
    out = []
    for i in range(0, n, batch_size):
        out.append([metas[j]["step"] for j in range(i, i + batch_size)])
    return out


def flat_sample_steps(metas: list[dict], batch_size: int) -> list[int | None]:
    """Flattened step ids in dataloader order (drop_last), matching act-cache sample order."""
    return [st for batch in batches_to_steps(metas, batch_size) for st in batch]


def pick_tensor(out) -> torch.Tensor:
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        # rare
        out = next(iter(out.values()))
    return out


def nmse_tensors(ref: torch.Tensor, hyp: torch.Tensor) -> dict:
    a = ref.float()
    b = hyp.float()
    d = (a - b).pow(2)
    ref_sq = a.pow(2)
    sum_sq = float(d.sum().item())
    sum_ref = float(ref_sq.sum().item())
    mean_sq = float(d.mean().item())
    mean_ref = float(ref_sq.mean().clamp_min(1e-20).item())
    return {
        "nmse": mean_sq / mean_ref,
        "mse": mean_sq,
        "sum_sq_err": sum_sq,
        "rmse_like": sum_sq**0.5,
        "sum_ref_sq": sum_ref,
    }


def move_ipt_to_device(ipt, device):
    """ModuleForwardInput → same structure on device."""
    args = []
    for a in ipt.args:
        if torch.is_tensor(a):
            args.append(a.to(device=device, non_blocking=True))
        else:
            args.append(a)
    kwargs = {}
    for k, v in ipt.kwargs.items():
        if torch.is_tensor(v):
            kwargs[k] = v.to(device=device, non_blocking=True)
        elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            kwargs[k] = type(v)(x.to(device=device, non_blocking=True) if torch.is_tensor(x) else x for x in v)
        else:
            kwargs[k] = v
    return args, kwargs


def repartition_eval_inputs(eval_inputs, sample_batch_size: int, sample_size: int):
    from deepcompressor.data.cache import TensorsCache

    return TensorsCache(
        {
            key: ipt.repartition(
                max_batch_size=sample_batch_size,
                max_size=sample_size,
                standardize=False,
                reshape=False,
            )
            for key, ipt in eval_inputs.items()
        }
    )


@torch.inference_mode()
def measure_eval_module(
    fp_mod: torch.nn.Module,
    q_mod: torch.nn.Module,
    eval_inputs,
    eval_kwargs: dict,
    sample_batch_size: int,
    sample_size: int,
    sample_steps: list[int | None],
) -> dict:
    """Overall (sample_size-aligned) + per-step NMSE.

    Only shrinks the *sample* batch on GPU (process 1 calib sample at a time).
    Does NOT chunk tokens within a sample.

    Per-step labels follow flat dataloader sample order (`sample_steps`), matched to
    activation-cache batches by walking input batch sizes (cache may concat/repartition
    differently than loader_bs). Per-sample NMSE is taken on each microbatch forward so
    output tensors without a leading batch dim still map to a timestep.
    """
    device = next(q_mod.parameters()).device

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

    def _lead_bs(ipt) -> int:
        if ipt.args and torch.is_tensor(ipt.args[0]):
            return int(ipt.args[0].shape[0])
        for v in ipt.kwargs.values():
            if torch.is_tensor(v):
                return int(v.shape[0])
        raise RuntimeError("no tensor lead in eval inputs")

    def _forward_pair(ei_cache, idx: int, steps_for_batch: list[int | None] | None):
        # Keep full cache on CPU; upload only one sample (batch slice) at a time.
        ipt = ei_cache.extract(idx, eval_kwargs)
        bs = _lead_bs(ipt)
        if steps_for_batch is not None and len(steps_for_batch) != bs:
            raise RuntimeError(
                f"step/batch mismatch: input_bs={bs} n_steps={len(steps_for_batch)} "
                f"(sample_steps cursor misaligned with act cache)"
            )
        fp_parts, q_parts = [], []
        sample_nmses: list[float] = []
        for s in range(bs):
            args_s, kwargs_s = _slice_ipt(ipt, s, s + 1)
            args_s = [a.to(device, non_blocking=True) if torch.is_tensor(a) else a for a in args_s]
            kwargs_s = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in kwargs_s.items()
            }
            yf = pick_tensor(fp_mod(*args_s, **kwargs_s))
            yq = pick_tensor(q_mod(*args_s, **kwargs_s))
            yf_c = yf.detach().float().cpu()
            yq_c = yq.detach().float().cpu()
            if yf_c.shape != yq_c.shape:
                raise RuntimeError(f"shape mismatch fp={tuple(yf_c.shape)} q={tuple(yq_c.shape)}")
            sample_nmses.append(nmse_tensors(yf_c, yq_c)["nmse"])
            fp_parts.append(yf_c)
            q_parts.append(yq_c)
            del yf, yq, args_s, kwargs_s
            torch.cuda.empty_cache()
        # Cat only when batch dims agree; otherwise keep list (aligned path uses batch nmse only).
        try:
            y_fp = torch.cat(fp_parts, 0)
            y_q = torch.cat(q_parts, 0)
            m = nmse_tensors(y_fp, y_q)
        except Exception:
            m = {
                "nmse": float(np.mean(sample_nmses)) if sample_nmses else float("nan"),
                "sum_sq_err": float("nan"),
                "sum_ref_sq": float("nan"),
            }
            y_fp = y_q = None
        return m, sample_nmses, steps_for_batch

    n_batches = len(eval_inputs.front().data)
    per_step_acc: dict[int, list[float]] = defaultdict(list)
    batch_nmses = []
    used_steps = set()
    cursor = 0
    step_align_ok = True

    for i in range(n_batches):
        ipt0 = eval_inputs.extract(i, eval_kwargs)
        bs = _lead_bs(ipt0)
        del ipt0
        if cursor + bs <= len(sample_steps):
            steps_for_batch = sample_steps[cursor : cursor + bs]
        else:
            step_align_ok = False
            steps_for_batch = [None] * bs
            log(
                f"  [warn] sample_steps exhausted at batch {i}: cursor={cursor} bs={bs} "
                f"len(sample_steps)={len(sample_steps)}; per-step labels disabled for remainder"
            )
        m, sample_nmses, steps_for_batch = _forward_pair(eval_inputs, i, steps_for_batch)
        batch_nmses.append(m["nmse"])
        if steps_for_batch is not None:
            for sm, st in zip(sample_nmses, steps_for_batch):
                if st is not None:
                    per_step_acc[int(st)].append(sm)
                    used_steps.add(int(st))
        cursor += bs
        torch.cuda.empty_cache()

    if cursor != len(sample_steps):
        step_align_ok = False
        log(
            f"  [warn] act-cache sample count ({cursor}) != sample_steps ({len(sample_steps)}); "
            f"per-step coverage may be partial"
        )

    ei = repartition_eval_inputs(eval_inputs, sample_batch_size, sample_size)
    sum_sq = 0.0
    sum_ref = 0.0
    aligned_batch_nmses = []
    n_act = len(ei.front().data)
    for i in range(n_act):
        m, _sample_nmses, _ = _forward_pair(ei, i, None)
        aligned_batch_nmses.append(m["nmse"])
        if m.get("sum_sq_err") == m.get("sum_sq_err"):  # not NaN
            sum_sq += float(m["sum_sq_err"])
            sum_ref += float(m["sum_ref_sq"])
        torch.cuda.empty_cache()

    aligned = {
        "nmse": float(np.mean(aligned_batch_nmses)) if aligned_batch_nmses else float("nan"),
        "nmse_sum_ratio": sum_sq / max(sum_ref, 1e-20) if sum_ref == sum_ref else float("nan"),
        "sum_sq_err": sum_sq,
        "rmse_like": sum_sq**0.5 if sum_sq == sum_sq else float("nan"),
        "sum_ref_sq": sum_ref,
        "n_act_batches": n_act,
    }
    per_step = {
        str(st): {
            "nmse": float(np.mean(vs)),
            "n_files": len(vs),
            "nmse_std": float(np.std(vs)) if len(vs) > 1 else 0.0,
        }
        for st, vs in sorted(per_step_acc.items())
    }
    return {
        "aligned": aligned,
        "per_step": per_step,
        "n_batches_full": n_batches,
        "mean_batch_nmse_full": float(np.mean(batch_nmses)) if batch_nmses else float("nan"),
        "steps_seen": sorted(used_steps),
        "n_samples_seen": cursor,
        "step_align_ok": step_align_ok and bool(per_step),
    }


def resolve_quant_module(quant_root: torch.nn.Module, name: str) -> torch.nn.Module:
    mods = dict(quant_root.named_modules())
    if name not in mods:
        raise KeyError(f"quant module missing: {name}")
    return mods[name]


def block_id_from_name(name: str) -> str:
    # blocks.21.attn1 -> block21; transformer_blocks.3.attn -> dual3; single_transformer_blocks.5 -> single5
    m = re.search(r"blocks\.(\d+)", name)
    if m and "single_transformer_blocks" not in name and "transformer_blocks" not in name:
        return f"block{m.group(1)}"
    m = re.search(r"single_transformer_blocks\.(\d+)", name)
    if m:
        return f"single{m.group(1)}"
    m = re.search(r"transformer_blocks\.(\d+)", name)
    if m:
        return f"dual{m.group(1)}"
    return name.split(".")[0]


@torch.inference_mode()
def collect_fp_outputs(fp_mod, eval_inputs, eval_kwargs: dict, device) -> list[torch.Tensor]:
    """Run FP module batch-by-batch; return CPU bf16 outputs."""
    outs = []
    for i in range(len(eval_inputs.front().data)):
        ipt = eval_inputs.extract(i, eval_kwargs)
        args, kwargs = move_ipt_to_device(ipt, device)
        y = pick_tensor(fp_mod(*args, **kwargs))
        outs.append(y.detach().to("cpu", torch.bfloat16).contiguous().clone())
        del y, args, kwargs
        if i % 2 == 1:
            torch.cuda.empty_cache()
    return outs


@torch.inference_mode()
def measure_from_cached_fp(
    q_mod: torch.nn.Module,
    eval_inputs,
    eval_kwargs: dict,
    fp_outs: list[torch.Tensor],
    sample_batch_size: int,
    sample_size: int,
    batch_steps: list[list[int | None]],
) -> dict:
    """Compare quant module vs cached FP outputs. Streams one batch; microbatches if needed."""
    device = next(q_mod.parameters()).device

    def _q_forward(ei_cache, idx: int) -> torch.Tensor:
        ipt = ei_cache.extract(idx, eval_kwargs)
        args, kwargs = move_ipt_to_device(ipt, device)
        # Microbatch along dim0 if activation is huge
        hs = None
        if args and torch.is_tensor(args[0]):
            hs = args[0]
        elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            hs = kwargs["hidden_states"]
        if hs is not None and hs.numel() > 20_000_000:  # ~40MB bf16 threshold → microbatch
            bs = hs.shape[0]
            chunks = []
            for s in range(bs):
                a2 = []
                for a in args:
                    if torch.is_tensor(a) and a.shape[0] == bs:
                        a2.append(a[s : s + 1])
                    else:
                        a2.append(a)
                k2 = {}
                for k, v in kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == bs:
                        k2[k] = v[s : s + 1]
                    else:
                        k2[k] = v
                y = pick_tensor(q_mod(*a2, **k2))
                chunks.append(y)
                del y
            out = torch.cat(chunks, dim=0)
            del args, kwargs, chunks
            return out
        y = pick_tensor(q_mod(*args, **kwargs))
        del args, kwargs
        return y

    n_batches = len(eval_inputs.front().data)
    assert n_batches == len(fp_outs)
    per_step_acc: dict[int, list[float]] = defaultdict(list)
    batch_nmses = []
    used_steps = set()

    for i in range(n_batches):
        y_q = _q_forward(eval_inputs, i)
        y_fp = fp_outs[i].to(device=y_q.device, dtype=torch.float32)
        y_q_f = y_q.float()
        m = nmse_tensors(y_fp, y_q_f)
        batch_nmses.append(m["nmse"])
        steps = batch_steps[i] if i < len(batch_steps) else [None] * y_fp.shape[0]
        if y_fp.shape[0] == len(steps):
            for sidx in range(y_fp.shape[0]):
                sm = nmse_tensors(y_fp[sidx : sidx + 1], y_q_f[sidx : sidx + 1])
                st = steps[sidx]
                if st is not None:
                    per_step_acc[int(st)].append(sm["nmse"])
                    used_steps.add(int(st))
        del y_q, y_fp, y_q_f
        torch.cuda.empty_cache()

    # Aligned subset: repartition inputs AND fp outs consistently via same indices
    ei = repartition_eval_inputs(eval_inputs, sample_batch_size, sample_size)
    # Map aligned batches back: repartition strides data list; recompute by running quant on ei
    # and matching fp by re-forward is unavailable — instead re-extract fp from original by stride.
    # Mirror TensorCache.repartition for list of fp_outs (same shapes assumed).
    fp_aligned = _repartition_tensor_list(fp_outs, sample_batch_size, sample_size)
    assert len(fp_aligned) == len(ei.front().data)

    sum_sq = 0.0
    sum_ref = 0.0
    aligned_batch_nmses = []
    for i in range(len(fp_aligned)):
        y_q = _q_forward(ei, i)
        y_fp = fp_aligned[i].to(device=y_q.device, dtype=torch.float32)
        m = nmse_tensors(y_fp, y_q.float())
        aligned_batch_nmses.append(m["nmse"])
        sum_sq += m["sum_sq_err"]
        sum_ref += m["sum_ref_sq"]
        del y_q, y_fp
        torch.cuda.empty_cache()

    aligned = {
        "nmse": float(np.mean(aligned_batch_nmses)) if aligned_batch_nmses else float("nan"),
        "nmse_sum_ratio": sum_sq / max(sum_ref, 1e-20),
        "sum_sq_err": sum_sq,
        "rmse_like": sum_sq**0.5,
        "sum_ref_sq": sum_ref,
        "n_act_batches": len(fp_aligned),
    }
    per_step = {
        str(st): {
            "nmse": float(np.mean(vs)),
            "n_files": len(vs),
            "nmse_std": float(np.std(vs)) if len(vs) > 1 else 0.0,
        }
        for st, vs in sorted(per_step_acc.items())
    }
    return {
        "aligned": aligned,
        "per_step": per_step,
        "n_batches_full": n_batches,
        "mean_batch_nmse_full": float(np.mean(batch_nmses)) if batch_nmses else float("nan"),
        "steps_seen": sorted(used_steps),
    }


def _repartition_tensor_list(
    data: list[torch.Tensor], max_batch_size: int, max_size: int
) -> list[torch.Tensor]:
    """Mirror TensorCache.repartition on a list of same-shaped batch tensors."""
    assert data and all(t.shape == data[0].shape for t in data)
    out = list(data)
    if max_batch_size > 0:
        bs = out[0].shape[0]
        if bs > max_batch_size:
            out = [
                x[i * max_batch_size : (i + 1) * max_batch_size]
                for x in out
                for i in range(int(bs // max_batch_size))
            ]
        bs = out[0].shape[0]
        if max_size > 0 and bs * len(out) > max_size:
            assert max_size >= bs
            out = out[:: int(len(out) // (max_size // bs))]
    return out


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
    batch_steps = batches_to_steps(metas, loader_bs)
    sample_steps = flat_sample_steps(metas, loader_bs)
    log(
        f"[{model}] selected {len(metas)} files; loader_bs={loader_bs}; "
        f"n_batches={len(batch_steps)}; n_sample_steps={len(sample_steps)}; sample_size={sample_size}"
    )
    (out_dir / "selected_files.json").write_text(
        json.dumps(
            {
                "metas": metas,
                "batch_steps": batch_steps,
                "sample_steps": sample_steps,
                "loader_batch_size": loader_bs,
            },
            indent=2,
        )
    )

    # Quant first → keep on CPU so BF16 can own the GPU for act collection.
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
    log(f"[{model}] quant on CPU; build BF16 on GPU...")

    bf16_pipe = cfg.pipeline.build()
    bf16_struct = DiffusionModelStruct.construct(bf16_pipe)
    device = next(bf16_struct.module.parameters()).device
    base_needs = get_needs_inputs_fn(bf16_struct, cfg.quant)

    def needs(name: str, module) -> bool:
        if base_needs(name, module):
            return True
        # Flux single blocks set parallel=True, so default needs_inputs caches the
        # whole block instead of Attention. Force-cache attn for local NMSE.
        if "single_transformer_blocks" in name and name.endswith(".attn"):
            return True
        return False

    modules_out: list[dict] = []
    per_layer_per_step: dict[str, dict] = {}
    skipped_attn: list[str] = []

    log(f"[{model}] online measure (no disk dump of activations)...")
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
            targets: list[tuple[str, torch.nn.Module, dict, str]] = []
            for attn in tb.iter_attention_structs():
                if attn.name not in layer_cache or layer_cache[attn.name].inputs is None:
                    skipped_attn.append(attn.name)
                    continue
                kind = (
                    "attn2"
                    if "attn2" in attn.name or (hasattr(attn, "is_cross_attn") and attn.is_cross_attn())
                    else "attn1"
                )
                if "single_transformer_blocks" in attn.name:
                    kind = "attn"
                targets.append((attn.name, attn.module, attn.filter_kwargs(layer_kwargs), kind))

            ffn = getattr(tb, "ffn_struct", None)
            if ffn is not None:
                for attr, kind in (("down_proj_name", "ffn_down"), ("up_proj_name", "ffn_up")):
                    n = getattr(ffn, attr, None)
                    mod = getattr(ffn, attr.replace("_name", ""), None)
                    if n and mod is not None and n in layer_cache and layer_cache[n].inputs is not None:
                        targets.append((n, mod, {}, kind))

            for name, fp_mod, eval_kwargs, kind in targets:
                q_mod = None
                try:
                    q_mod = resolve_quant_module(quant_nn, name)
                    q_mod.to(device)
                    stats = measure_eval_module(
                        fp_mod,
                        q_mod,
                        layer_cache[name].inputs,
                        eval_kwargs,
                        sample_batch_size=sample_batch_size,
                        sample_size=sample_size,
                        sample_steps=sample_steps,
                    )
                    row = {
                        "name": name,
                        "block_id": block_id_from_name(name),
                        "kind": kind,
                        **stats["aligned"],
                        "mean_batch_nmse_full": stats["mean_batch_nmse_full"],
                        "n_batches_full": stats["n_batches_full"],
                        "n_samples_seen": stats.get("n_samples_seen"),
                        "step_align_ok": stats.get("step_align_ok"),
                        "n_steps_seen": len(stats.get("steps_seen") or []),
                    }
                    modules_out.append(row)
                    per_layer_per_step[name] = stats["per_step"]
                    log(
                        f"  [{name}] nmse={100*row['nmse']:.4f}% "
                        f"sum_ratio={100*row['nmse_sum_ratio']:.4f}% n_act={row['n_act_batches']} "
                        f"n_steps={row['n_steps_seen']} align={row['step_align_ok']}"
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    log(f"  [OOM] {name}: cannot fit one full sample on GPU without offload/token-chunk. Skipping. ({exc})")
                    traceback.print_exc()
                except Exception as exc:
                    log(f"  [warn] {name}: {exc}")
                    traceback.print_exc()
                finally:
                    if q_mod is not None:
                        q_mod.to("cpu")
                    torch.cuda.empty_cache()

        # drop layer caches ASAP
        layer_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    if skipped_attn:
        log(f"[{model}] WARNING: {len(skipped_attn)} attn modules missing from cache (first5={skipped_attn[:5]})")
    n_single_attn = sum(1 for r in modules_out if r["kind"] == "attn" and "single" in r["block_id"])
    n_dual_attn = sum(1 for r in modules_out if "dual" in r["block_id"] and r["kind"] in ("attn", "attn1"))
    log(f"[{model}] measured modules={len(modules_out)} dual_attn={n_dual_attn} single_attn={n_single_attn}")

    # aggregate by block
    by_block: dict[str, list[dict]] = defaultdict(list)
    for r in modules_out:
        by_block[r["block_id"]].append(r)
    blocks = []
    for bid, rows in by_block.items():
        nmses = [r["nmse"] for r in rows]
        blocks.append(
            {
                "block_id": bid,
                "n_modules": len(rows),
                "nmse_mean": float(np.mean(nmses)),
                "nmse_max": float(np.max(nmses)),
                "modules": [r["name"] for r in rows],
            }
        )

    def block_sort_key(b):
        m = re.search(r"(\d+)", b["block_id"])
        prefix = 0 if b["block_id"].startswith("dual") else (1 if b["block_id"].startswith("single") else 2)
        if b["block_id"].startswith("block"):
            prefix = 0
        return (prefix, int(m.group(1)) if m else 0)

    blocks.sort(key=block_sort_key)

    # sample_size hit steps: steps present in first n_act batches after repartition stride approx
    # Record all steps in selected metas + steps that appear in per_layer tables
    all_steps = sorted({m["step"] for m in metas if m["step"] is not None})
    hit_steps = sorted(
        {
            int(st)
            for layer in per_layer_per_step.values()
            for st in layer.keys()
        }
    )

    result = {
        "model": model,
        "ckpt": str(ckpt),
        "calib_path": str(calib_path),
        "num_samples": num_samples,
        "sample_size": sample_size,
        "sample_batch_size": sample_batch_size,
        "loader_batch_size": loader_bs,
        "seed": 0,
        "n_modules": len(modules_out),
        "modules": modules_out,
        "blocks": blocks,
        "per_layer_per_step": per_layer_per_step,
        "selected_files": [m["file"] for m in metas],
        "selected_steps": all_steps,
        "sample_size_hit_steps": hit_steps,
        "note": "Teacher-forced local NMSE on BF16-collected layer inputs vs final W4A4 modules from ckpt.",
    }
    out_json = out_dir / f"{model}_calib_local_nmse.json"
    out_json.write_text(json.dumps(result, indent=2))
    log(f"[{model}] wrote {out_json}")

    plot_block_curve(blocks, out_dir / f"{model}_calib_local_block_nmse.png", model)
    plot_step_heatmap(per_layer_per_step, blocks, out_dir / f"{model}_calib_local_layer_step_heatmap.png", model)

    del bf16_pipe, bf16_struct, quant_pipe, quant_struct
    gc.collect()
    torch.cuda.empty_cache()
    return result


def plot_block_curve(blocks: list[dict], out_png: Path, model: str) -> None:
    if not blocks:
        return
    xs = np.arange(len(blocks))
    mean = [100 * b["nmse_mean"] for b in blocks]
    mx = [100 * b["nmse_max"] for b in blocks]
    labels = [b["block_id"] for b in blocks]
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
    ax.plot(xs, mean, marker="o", ms=3.5, lw=1.6, label="mean over modules", color="#1f4e79")
    ax.plot(xs, mx, marker="x", ms=3.5, lw=1.0, label="max over modules", color="#a33", alpha=0.8)
    ax.set_title(f"{model.upper()} calib local block NMSE (teacher-forced, PTQ-aligned samples)")
    ax.set_ylabel("NMSE (%)")
    ax.set_xlabel("Block")
    step = max(1, len(labels) // 16)
    ticks = list(range(0, len(labels), step))
    if ticks[-1] != len(labels) - 1:
        ticks.append(len(labels) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    peak = int(np.argmax(mx))
    ax.annotate(
        f"max {mx[peak]:.2f}%\n{labels[peak]}",
        xy=(xs[peak], mx[peak]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#833"),
    )
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")


def plot_step_heatmap(per_layer_per_step: dict, blocks: list[dict], out_png: Path, model: str) -> None:
    # Pick one representative module per block (max nmse module name from blocks list order)
    if not per_layer_per_step:
        return
    # Use up to ~30 layers: prefer *attn* names ending patterns
    layer_names = []
    for b in blocks:
        # choose module with largest aligned nmse among this block — need lookup
        cands = [n for n in b["modules"] if n in per_layer_per_step]
        if not cands:
            continue
        # prefer attn1 / attn over ffn for readability
        pref = [n for n in cands if "attn1" in n or n.endswith(".attn") or ".attn." in n]
        layer_names.append((pref[0] if pref else cands[0]))
    if not layer_names:
        layer_names = list(per_layer_per_step.keys())[:30]

    all_steps = sorted({int(st) for d in per_layer_per_step.values() for st in d.keys()})
    if not all_steps:
        return
    mat = np.full((len(layer_names), len(all_steps)), np.nan, dtype=np.float64)
    for i, ln in enumerate(layer_names):
        d = per_layer_per_step.get(ln, {})
        for j, st in enumerate(all_steps):
            if str(st) in d:
                mat[i, j] = 100.0 * d[str(st)]["nmse"]

    fig, ax = plt.subplots(figsize=(14, max(4, 0.28 * len(layer_names))), dpi=140)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(f"{model.upper()} calib local NMSE (%) by layer × denoising step")
    ax.set_xlabel("denoising step index (from cache filename)")
    ax.set_ylabel("module")
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([n.split(".", 1)[-1] if n.startswith("blocks.") or "blocks." in n else n for n in layer_names], fontsize=7)
    step_ticks = list(range(0, len(all_steps), max(1, len(all_steps) // 16)))
    ax.set_xticks(step_ticks)
    ax.set_xticklabels([all_steps[i] for i in step_ticks], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="NMSE %")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")


def plot_overlay(wan_blocks: list[dict], flux_blocks: list[dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    if wan_blocks:
        ax.plot(
            range(len(wan_blocks)),
            [100 * b["nmse_max"] for b in wan_blocks],
            marker="o",
            ms=3,
            label="Wan max-module",
            color="#1f4e79",
        )
    if flux_blocks:
        ax.plot(
            range(len(flux_blocks)),
            [100 * b["nmse_max"] for b in flux_blocks],
            marker="o",
            ms=3,
            label="Flux max-module",
            color="#b85c38",
        )
    ax.set_title("Calib local block NMSE (max over modules) — Wan vs Flux")
    ax.set_ylabel("NMSE (%)")
    ax.set_xlabel("Block index (model-specific order)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")


def write_summary(wan: dict | None, flux: dict | None, out_dir: Path) -> None:
    lines = ["# Calib local NMSE summary", ""]
    for tag, res in (("wan", wan), ("flux", flux)):
        if not res:
            continue
        blocks = res["blocks"]
        peak = max(blocks, key=lambda b: b["nmse_max"]) if blocks else None
        lines.append(f"## {tag}")
        lines.append(
            f"- samples: num_samples={res['num_samples']}, sample_size={res['sample_size']}, "
            f"loader_bs={res['loader_batch_size']}"
        )
        lines.append(f"- modules: {res['n_modules']}")
        if peak:
            lines.append(
                f"- peak block: {peak['block_id']} max_nmse={100*peak['nmse_max']:.3f}% "
                f"mean={100*peak['nmse_mean']:.3f}%"
            )
        # step with highest mean nmse across layers
        step_scores: dict[str, list[float]] = defaultdict(list)
        for layer, d in res["per_layer_per_step"].items():
            for st, info in d.items():
                step_scores[st].append(info["nmse"])
        if step_scores:
            worst = max(step_scores.items(), key=lambda kv: float(np.mean(kv[1])))
            best = min(step_scores.items(), key=lambda kv: float(np.mean(kv[1])))
            lines.append(
                f"- timestep mean-NMSE: worst step={worst[0]} ({100*float(np.mean(worst[1])):.3f}%), "
                f"best step={best[0]} ({100*float(np.mean(best[1])):.3f}%)"
            )
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    log(f"[summary] {out_dir / 'SUMMARY.md'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["wan", "flux", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wan_res = flux_res = None
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

    # When re-running one side, reuse the other side's JSON for overlay/summary.
    if wan_res is None:
        wp = args.out_dir / "wan" / "wan_calib_local_nmse.json"
        if wp.is_file():
            wan_res = json.loads(wp.read_text())
            log(f"[summary] reuse existing {wp}")
    if flux_res is None:
        fp = args.out_dir / "flux" / "flux_calib_local_nmse.json"
        if fp.is_file():
            flux_res = json.loads(fp.read_text())
            log(f"[summary] reuse existing {fp}")

    if wan_res and flux_res:
        plot_overlay(wan_res["blocks"], flux_res["blocks"], args.out_dir / "wan_vs_flux_calib_local_nmse.png")
    write_summary(wan_res, flux_res, args.out_dir)

    # Extra: Wan NMSE vs timestep curve when per-step data exists
    if wan_res and wan_res.get("per_layer_per_step"):
        plot_nmse_vs_timestep(wan_res, args.out_dir / "wan_nmse_vs_timestep.png")
    if flux_res and flux_res.get("per_layer_per_step"):
        plot_nmse_vs_timestep(flux_res, args.out_dir / "flux_nmse_vs_timestep.png")

    log(f"ALL DONE → {args.out_dir}")
    return 0


def plot_nmse_vs_timestep(res: dict, out_png: Path) -> None:
    pls = res.get("per_layer_per_step") or {}
    by_step: dict[int, list[float]] = defaultdict(list)
    for stepmap in pls.values():
        for st, info in stepmap.items():
            nmse = info["nmse"] if isinstance(info, dict) else float(info)
            by_step[int(st)].append(nmse)
    if not by_step:
        return
    steps = sorted(by_step)
    means = [100 * float(np.mean(by_step[s])) for s in steps]
    maxes = [100 * float(np.max(by_step[s])) for s in steps]
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(steps, means, "o-", lw=1.8, ms=5, label="mean over modules", color="#1f4e79")
    ax.plot(steps, maxes, "x--", lw=1.0, ms=4, label="max over modules", color="#a33", alpha=0.85)
    ax.axhline(float(np.mean(means)), color="#888", ls=":", lw=1, label=f"overall mean {float(np.mean(means)):.3f}%")
    ax.set_xlabel("Denoising step index (from calib filename)")
    ax.set_ylabel("Calib local NMSE (%)")
    ax.set_title(f"{res.get('model', '?').upper()} — calib local NMSE vs timestep")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
