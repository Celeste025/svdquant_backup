#!/usr/bin/env python3
"""Student-forced local NMSE on first denoising step (Wan).

Runs the *quant* model for step-0 with the real gen prompt/seed, captures each
target module's *quant-propagated inputs*, then compares:

  y_q = quant_module(x_q)
  y_fp = bf16_module(x_q)   # same inputs
  NMSE(y_fp, y_q)

vs teacher-forced calib local NMSE (BF16-collected inputs). If student-forced
NMSE >> calib, activations have drifted / calib overfit; if similar, e2e error
is mostly accumulation.
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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pyarrow  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from collect_firststep_block_nmse import (  # noqa: E402
    WAN_NEGATIVE,
    WAN_PROMPT,
    DEFAULT_SEED,
    load_wan_bf16,
    load_wan_quant,
    run_wan_first_step,
)

OUT_DEFAULT = DATA_ROOT / "compare" / "wan_quant_input_local_nmse"
WAN_CKPT = DATA_ROOT / "ckpts/wan2.1-1.3b-int4-s16"
CALIB_JSON = DATA_ROOT / "compare/calib_local_block_nmse/wan/wan_calib_local_nmse.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_tensor(out) -> torch.Tensor:
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = next(iter(out.values()))
    return out


def nmse(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    a, b = ref.float(), hyp.float()
    return float((a - b).pow(2).mean() / a.pow(2).mean().clamp_min(1e-20))


def to_cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", torch.bfloat16).contiguous().clone()
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu_tree(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_cpu_tree(v) for k, v in obj.items()}
    return obj


def to_device_tree(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device_tree(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device_tree(v, device) for k, v in obj.items()}
    return obj


def list_wan_targets(transformer) -> list[tuple[str, str, torch.nn.Module]]:
    """(name, kind, module) matching calib probe set."""
    out = []
    for i, block in enumerate(transformer.blocks):
        out.append((f"blocks.{i}.attn1", "attn1", block.attn1))
        out.append((f"blocks.{i}.attn2", "attn2", block.attn2))
        # Wan FFN: net.0.proj = up, net.2.linear = down (ShiftedLinear)
        ffn = block.ffn
        up = ffn.net[0].proj
        down = ffn.net[2].linear if hasattr(ffn.net[2], "linear") else ffn.net[2]
        out.append((f"blocks.{i}.ffn.net.0.proj", "ffn_up", up))
        out.append((f"blocks.{i}.ffn.net.2.linear", "ffn_down", down))
    return out


@torch.inference_mode()
def capture_quant_inputs(quant_pipe, targets: list[tuple[str, str, torch.nn.Module]], prompt: str, seed: int):
    """One first-step forward on quant; store CPU bf16 inputs per target name."""
    store: dict[str, dict] = {}
    handles = []

    for name, _kind, mod in targets:

        def make_pre(n):
            def pre_hook(_m, args, kwargs):
                # Keep only first capture (CFG may call modules twice; last write wins —
                # for step0 we want the actual call; overwriting with latest is fine).
                store[n] = {"args": to_cpu_tree(args), "kwargs": to_cpu_tree(kwargs)}

            return pre_hook

        handles.append(mod.register_forward_pre_hook(make_pre(name), with_kwargs=True))

    log(f"[capture] running quant first step; watching {len(targets)} modules...")
    run_wan_first_step(quant_pipe, prompt, seed)
    for h in handles:
        h.remove()
    missing = [n for n, _, _ in targets if n not in store]
    if missing:
        raise RuntimeError(f"missing inputs for {len(missing)} modules e.g. {missing[:5]}")
    log(f"[capture] got inputs for {len(store)} modules")
    return store


@torch.inference_mode()
def measure_on_inputs(
    name: str,
    kind: str,
    q_mod: torch.nn.Module,
    fp_mod: torch.nn.Module,
    ipt: dict,
    device: torch.device,
) -> dict:
    args = to_device_tree(ipt["args"], device)
    kwargs = to_device_tree(ipt["kwargs"], device)
    q_mod.to(device)
    fp_mod.to(device)
    try:
        y_q = pick_tensor(q_mod(*args, **kwargs)).float().cpu()
        y_fp = pick_tensor(fp_mod(*args, **kwargs)).float().cpu()
        if y_q.shape != y_fp.shape:
            raise RuntimeError(f"{name}: shape q={tuple(y_q.shape)} fp={tuple(y_fp.shape)}")
        val = nmse(y_fp, y_q)
        return {
            "name": name,
            "kind": kind,
            "block_id": f"block{name.split('.')[1]}" if name.startswith("blocks.") else name,
            "nmse": val,
            "shape": list(y_fp.shape),
        }
    finally:
        q_mod.to("cpu")
        fp_mod.to("cpu")
        del args, kwargs
        torch.cuda.empty_cache()


def load_calib_map(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    d = json.loads(path.read_text())
    return {m["name"]: float(m["nmse"]) for m in d.get("modules", [])}


def plot_compare(rows: list[dict], out_png: Path) -> None:
    # one panel per kind: block index vs student / calib nmse
    kinds = ["attn1", "attn2", "ffn_down", "ffn_up"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=140, sharex=True)
    for ax, kind in zip(axes.ravel(), kinds):
        sub = [r for r in rows if r["kind"] == kind]
        sub = sorted(sub, key=lambda r: int(r["block_id"].replace("block", "")))
        xs = [int(r["block_id"].replace("block", "")) for r in sub]
        stu = [100 * r["nmse"] for r in sub]
        cal = [100 * r["calib_nmse"] if r.get("calib_nmse") is not None else np.nan for r in sub]
        ax.plot(xs, stu, "o-", ms=3.5, lw=1.5, color="#c0392b", label="student-forced (quant inputs)")
        if any(c == c for c in cal):
            ax.plot(xs, cal, "s--", ms=3.0, lw=1.2, color="#1f4e79", label="calib teacher-forced")
        ax.set_title(kind)
        ax.set_ylabel("NMSE (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    axes[1, 0].set_xlabel("block")
    axes[1, 1].set_xlabel("block")
    fig.suptitle(
        "Wan step-0 local NMSE: quant-propagated inputs vs calib BF16 inputs\n"
        f"prompt/seed = gen setting; modules = attn1/attn2/ffn",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"[plot] {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=WAN_CKPT)
    parser.add_argument("--prompt", type=str, default=WAN_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calib-json", type=Path, default=CALIB_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    calib_map = load_calib_map(args.calib_json)
    log(f"[calib] loaded {len(calib_map)} module NMSEs from {args.calib_json}")

    log("[load] quant on GPU...")
    quant_pipe = load_wan_quant(args.ckpt)
    q_targets = list_wan_targets(quant_pipe.transformer)
    log(f"[load] quant targets={len(q_targets)}")

    ipt_store = capture_quant_inputs(quant_pipe, q_targets, args.prompt, args.seed)

    # Free pipeline activations; keep modules for measure (named lookup).
    q_mods = {n: m for n, _k, m in q_targets}
    # Move whole quant transformer to CPU to free GPU for pairwise forwards
    log("[load] moving quant transformer to CPU; building BF16...")
    quant_pipe.transformer.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    bf16_pipe = load_wan_bf16()
    bf16_pipe.transformer.to("cpu")
    fp_targets = list_wan_targets(bf16_pipe.transformer)
    fp_mods = {n: m for n, _k, m in fp_targets}
    device = torch.device("cuda")

    rows = []
    for name, kind, _ in q_targets:
        try:
            row = measure_on_inputs(name, kind, q_mods[name], fp_mods[name], ipt_store[name], device)
            row["calib_nmse"] = calib_map.get(name)
            if row["calib_nmse"] is not None and row["calib_nmse"] > 0:
                row["ratio_vs_calib"] = row["nmse"] / row["calib_nmse"]
            else:
                row["ratio_vs_calib"] = None
            rows.append(row)
            cal_s = f"{100*row['calib_nmse']:.3f}%" if row["calib_nmse"] is not None else "NA"
            ratio_s = f"{row['ratio_vs_calib']:.2f}x" if row["ratio_vs_calib"] is not None else "NA"
            log(
                f"  [{name}] student={100*row['nmse']:.4f}% calib={cal_s} ratio={ratio_s}"
            )
        except torch.cuda.OutOfMemoryError as exc:
            log(f"  [OOM] {name}: {exc}")
            traceback.print_exc()
            torch.cuda.empty_cache()
        except Exception as exc:
            log(f"  [warn] {name}: {exc}")
            traceback.print_exc()
        # drop captured inputs ASAP
        ipt_store.pop(name, None)
        gc.collect()
        torch.cuda.empty_cache()

    # summary stats
    ratios = [r["ratio_vs_calib"] for r in rows if r.get("ratio_vs_calib") is not None]
    summary = {
        "prompt": args.prompt,
        "seed": args.seed,
        "ckpt": str(args.ckpt),
        "calib_json": str(args.calib_json),
        "n_modules": len(rows),
        "student_mean_nmse": float(np.mean([r["nmse"] for r in rows])) if rows else None,
        "calib_mean_nmse": float(np.mean([r["calib_nmse"] for r in rows if r.get("calib_nmse") is not None]))
        if any(r.get("calib_nmse") is not None for r in rows)
        else None,
        "ratio_mean": float(np.mean(ratios)) if ratios else None,
        "ratio_median": float(np.median(ratios)) if ratios else None,
        "ratio_p90": float(np.percentile(ratios, 90)) if ratios else None,
        "ratio_max": float(np.max(ratios)) if ratios else None,
        "note": (
            "Student-forced: first-step quant forward captures module inputs; "
            "NMSE between quant vs BF16 modules on those same inputs. "
            "Compare to teacher-forced calib local NMSE."
        ),
        "modules": rows,
    }
    out_json = args.out_dir / "wan_quant_input_local_nmse.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"[write] {out_json}")
    plot_compare(rows, args.out_dir / "wan_quant_input_vs_calib_nmse.png")

    md = [
        "# Wan quant-input (student-forced) local NMSE — step 0",
        "",
        f"- prompt: `{args.prompt}`",
        f"- seed: {args.seed}",
        f"- modules: {len(rows)}",
        f"- student mean NMSE: {100*summary['student_mean_nmse']:.4f}%" if summary["student_mean_nmse"] is not None else "",
        f"- calib mean NMSE: {100*summary['calib_mean_nmse']:.4f}%" if summary["calib_mean_nmse"] is not None else "",
        f"- ratio student/calib: mean={summary['ratio_mean']:.3f}x median={summary['ratio_median']:.3f}x "
        f"p90={summary['ratio_p90']:.3f}x max={summary['ratio_max']:.3f}x"
        if summary["ratio_mean"] is not None
        else "",
        "",
        "Interpretation: ratio≈1 → local error similar on drifted inputs (accumulation dominates); "
        "ratio≫1 → local modules much worse on quant-propagated activations (shift/overfit).",
        "",
    ]
    (args.out_dir / "SUMMARY.md").write_text("\n".join(md) + "\n")
    log(f"[summary] ratio mean={summary['ratio_mean']} median={summary['ratio_median']} max={summary['ratio_max']}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
