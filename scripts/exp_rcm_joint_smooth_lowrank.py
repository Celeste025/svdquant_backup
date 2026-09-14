#!/usr/bin/env python3
"""Jointly fine-tune rCM-Wan SVDQuant smoothing and LowRank branches.

This trainer starts from the existing real-NVFP4 checkpoint.  Its dynamic
adapter is deliberately *anchored*: at zero scale delta every patched Linear
returns the checkpoint's exact stored W4A4 weight.  Away from zero, it applies
the matched smooth-coordinate BF16 weight, the checkpoint's NVFP4 quantizer,
and adds only the resulting QDQ delta to the stored weight.  This makes the
zero-delta equivalence check a real correctness gate instead of an approximate
comparison against a second quantizer implementation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import WanPipeline, WanTransformer3DModel

from exp_rcm_blockwise_lowrank_pilot import (
    EPS, TEST_PROMPTS, TRAIN_PROMPTS, all_branches, branch_map, freeze_all,
    install_activation_ste, load_items, restore_activation_processors,
)
from exp_rcm_fullmodel_lowrank_denoiser import (
    as_tensor, branch_parameters, branch_state, capture_teacher, evaluate,
    export_by_block, student_forward, write_csv,
)
from exp_rcm_signed_alignment_audit import make_autograd_safe
from exp_rcm_trajectory_sensitivity_audit import json_default
from infer_rcm_wan_4step import REPO_ROOT, load_quantized_transformer


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data1/models/svdquant-wjq")
MODEL = DATA / "models/Wan2.1-T2V-1.3B-Diffusers"
RCM_TRANSFORMER = DATA / "models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer"
CKPT = DATA / "ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16"
CACHE = DATA / "datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches"
EXTRA_CACHE = DATA / "datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/blockwise_extra4_s16/caches"
EXTRA_PROMPTS = ("motion_hummingbird", "motion_motorcycle", "person_dance", "precision_circuit")
OUT = ROOT / "results/reports/rcm_joint_smooth_lowrank"

TARGET_SUFFIXES = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ffn.net.0.proj", "ffn.net.2",
)


def named_modules(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def target_names(model: nn.Module) -> list[str]:
    result = [name for name, module in model.named_modules()
              if isinstance(module, nn.Linear) and name.startswith("blocks.") and name.endswith(TARGET_SUFFIXES)]
    if len(result) != 300:
        raise RuntimeError(f"expected 300 quantized Linear modules, found {len(result)}")
    return result


def lowrank_branch(module: nn.Module) -> nn.Module:
    found = {id(getattr(h, "branch", None)): getattr(h, "branch", None)
             for h in list(module._forward_pre_hooks.values()) + list(module._forward_hooks.values())
             if type(getattr(h, "branch", None)).__name__ == "LowRankBranch"}
    found.pop(id(None), None)
    if len(found) != 1:
        raise RuntimeError(f"expected one LowRankBranch, got {len(found)}")
    return next(iter(found.values()))


def weight_config() -> Any:
    """Parse exactly the config used by infer_rcm_wan_4step.load_quantized_transformer."""
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    diffusion_root = REPO_ROOT / "third_party" / "deepcompressor" / "examples" / "diffusion"
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(diffusion_root)
        sys.argv = ["joint_smooth.py", "configs/model/wan2.1-1.3b.yaml",
                    "configs/svdquant/real_nvfp4.yaml", "configs/svdquant/wan_s16.yaml",
                    f"--pipeline-path={MODEL}", "--skip-eval", "--skip-gen"]
        cfg, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args()
        return cfg.quant
    finally:
        sys.argv, _ = old_argv, old_cwd
        os.chdir(old_cwd)


def find_smoothers(model: nn.Module) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Return smoother processor per target and its exact shared target set."""
    modules = named_modules(model)
    site_to_proc: dict[str, Any] = {}
    for site, module in modules.items():
        for hook in module._forward_pre_hooks.values():
            proc = getattr(hook, "processor", None)
            if type(proc).__name__ == "ActivationSmoother":
                if site in site_to_proc and site_to_proc[site] is not proc:
                    raise RuntimeError(f"multiple smoothers at {site}")
                site_to_proc[site] = proc
    per_target: dict[str, Any] = {}
    for block in range(30):
        p = f"blocks.{block}"
        rules = {
            f"{p}.attn1": [f"{p}.attn1.to_{x}" for x in ("q", "k", "v")],
            f"{p}.attn1.to_out.0": [f"{p}.attn1.to_out.0"],
            f"{p}.attn2": [f"{p}.attn2.to_q"],
            f"{p}.attn2.to_k": [f"{p}.attn2.to_k", f"{p}.attn2.to_v"],
            f"{p}.attn2.to_v": [f"{p}.attn2.to_k", f"{p}.attn2.to_v"],
            f"{p}.attn2.to_out.0": [f"{p}.attn2.to_out.0"],
            f"{p}.ffn.net.0.proj": [f"{p}.ffn.net.0.proj"],
            f"{p}.ffn.net.2": [f"{p}.ffn.net.2"],
        }
        for site, names in rules.items():
            if site not in site_to_proc:
                raise RuntimeError(f"missing ActivationSmoother at {site}")
            for name in names:
                per_target[name] = site_to_proc[site]
    groups: dict[int, list[str]] = defaultdict(list)
    for name, proc in per_target.items(): groups[id(proc.smooth_scale)].append(name)
    if len(groups) != 210 or set(per_target) != set(target_names(model)):
        raise RuntimeError(f"expected 210 groups / 300 targets, got {len(groups)} / {len(per_target)}")
    return per_target, {str(key): sorted(value) for key, value in groups.items()}


class DynamicSmooth(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module, checkpoint: Path, bound: float,
                 anchor_weights: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.student, self.teacher = student, teacher
        self.bound = float(bound)
        # When supplied by an anchored export, these exact static NVFP4 main
        # weights replace any freshly reconstructed PTQ weights.
        self.anchor_weights = anchor_weights
        self.target_to_proc, groups = find_smoothers(student)
        self.names = target_names(student)
        self.group_for_target: dict[str, int] = {}
        procs: dict[int, Any] = {id(proc.smooth_scale): proc for proc in self.target_to_proc.values()}
        self.initial = [procs[int(key)].smooth_scale.detach().float().clone() for key in groups]
        self.delta = nn.ParameterList([nn.Parameter(torch.zeros_like(scale)) for scale in self.initial])
        proc_to_group = {int(key): index for index, key in enumerate(groups)}
        for name, proc in self.target_to_proc.items(): self.group_for_target[name] = proc_to_group[id(proc.smooth_scale)]
        self.groups = [groups[key] for key in groups]
        if len(self.delta) != 210:
            raise RuntimeError("incorrect smoothing parameter count")
        self._patch_smoothers()
        self._patch_branches()
        self._patch_linears(checkpoint)

    def ratio(self, group: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # A bounded log delta avoids a scale collapse while remaining differentiable inside the interval.
        u = self.delta[group].clamp(math.log(1 / self.bound), math.log(self.bound))
        return u.exp().to(device=device, dtype=dtype)

    def scale(self, group: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.initial[group].to(device=device, dtype=dtype) * self.ratio(group, dtype, device)

    def _patch_smoothers(self) -> None:
        done: set[int] = set()
        for name, proc in self.target_to_proc.items():
            if id(proc) in done: continue
            done.add(id(proc)); group = self.group_for_target[name]
            def process(_proc: Any, tensor: torch.Tensor, _group=group, _self=self) -> torch.Tensor:
                s = _self.scale(_group, tensor.dtype, tensor.device)
                view = [1] * tensor.ndim; view[_proc.channels_dim] = -1
                return tensor.div(s.view(view)).to(dtype=tensor.dtype)
            proc.process = types.MethodType(process, proc)

    def _patch_branches(self) -> None:
        done: set[int] = set()
        modules = named_modules(self.student)
        for name in self.names:
            branch = lowrank_branch(modules[name])
            if id(branch) in done: continue
            done.add(id(branch)); group = self.group_for_target[name]; original = branch.forward
            # LowRank lives in the smoothed coordinate system.  Scaling its input by
            # s/s0 makes its effective weight acquire the same column transform as W.
            def forward(input: torch.Tensor, _orig=original, _group=group, _self=self) -> torch.Tensor:
                ratio = _self.ratio(_group, input.dtype, input.device)
                return _orig(input * ratio)
            branch.forward = forward

    def _patch_linears(self, checkpoint: Path) -> None:
        from deepcompressor.app.diffusion.quant.quantizer import DiffusionWeightQuantizer
        cfg = weight_config()
        wstates = torch.load(checkpoint / "wgts.pt", map_location="cpu", weights_only=False)
        smods, tmods = named_modules(self.student), named_modules(self.teacher)
        for name in self.names:
            module, raw = smods[name], tmods[name]
            if module.weight.shape != raw.weight.shape: raise RuntimeError(f"raw mapping mismatch: {name}")
            quantizer = DiffusionWeightQuantizer(cfg.wgts, develop_dtype=cfg.develop_dtype, key=name)
            quantizer.load_state_dict(wstates[name], device=module.weight.device)
            base = (self.anchor_weights[name].to(device=module.weight.device, dtype=module.weight.dtype).clone()
                    if self.anchor_weights is not None else module.weight.detach().clone())
            raw_weight = raw.weight.detach().clone()
            branch, group = lowrank_branch(module), self.group_for_target[name]
            # This is invariant across all updates.  It must not be replaced by
            # module.weight: low-rank subtraction and NVFP4 state make them differ.
            initial_s0 = self.initial[group].to(device=module.weight.device, dtype=torch.float32)
            initial_eff = branch.get_effective_weight().float()
            q_reference = quantizer.process(raw_weight.float() * initial_s0.unsqueeze(0) - initial_eff).float().detach().cpu()
            def forward(input: torch.Tensor, _base=base, _raw=raw_weight, _branch=branch,
                        _quant=quantizer, _qref=q_reference, _group=group, _module=module, _self=self) -> torch.Tensor:
                ratio = _self.ratio(_group, torch.float32, input.device)
                s0 = _self.initial[_group].to(device=input.device, dtype=torch.float32)
                # PTQ smooths first, then calibrates/subtracts LowRank.  The branch is
                # column-scaled with the same ratio, so main and residual paths remain matched.
                eff = _branch.get_effective_weight().float()
                candidate = _raw.float() * (s0 * ratio).unsqueeze(0) - eff * ratio.unsqueeze(0)
                q_candidate = _quant.process(candidate).float()
                q_ste = candidate + (q_candidate - candidate).detach()
                weight = _base.float() + (q_ste - _qref.to(device=input.device, dtype=torch.float32))
                return F.linear(input, weight.to(dtype=input.dtype), _module.bias)
            module.forward = forward

    def manifest(self) -> list[dict[str, Any]]:
        return [{"group": i, "size": int(self.initial[i].numel()), "layers": names}
                for i, names in enumerate(self.groups)]


def _quantizer_processor_entries(model: nn.Module) -> list[dict[str, Any]]:
    """Collect shared activation-QDQ processors together with stable hook sites."""
    entries: dict[int, dict[str, Any]] = {}
    for module_name, module in model.named_modules():
        for kind, hooks in (("pre", module._forward_pre_hooks), ("post", module._forward_hooks)):
            for hook in hooks.values():
                proc = getattr(hook, "processor", None)
                if proc is None or "Quantizer" not in type(proc).__name__ or not hasattr(proc, "state_dict"):
                    continue
                entry = entries.setdefault(id(proc), {"type": type(proc).__name__,
                    "key": getattr(proc, "key", ""), "sites": [], "processor": proc})
                entry["sites"].append((module_name, kind))
    result = []
    for entry in entries.values():
        proc = entry.pop("processor")
        result.append({**entry, "state": proc.state_dict(device="cpu")})
    return result


def _restore_quantizer_processors(model: nn.Module, entries: list[dict[str, Any]]) -> None:
    modules = named_modules(model)
    for entry in entries:
        module_name, kind = entry["sites"][0]
        module = modules[module_name]
        hooks = module._forward_pre_hooks if kind == "pre" else module._forward_hooks
        matches = [getattr(hook, "processor", None) for hook in hooks.values()
                   if type(getattr(hook, "processor", None)).__name__ == entry["type"]
                   and getattr(getattr(hook, "processor", None), "key", "") == entry["key"]]
        if len(matches) != 1:
            raise RuntimeError(f"cannot uniquely restore quantizer {entry['key']} at {module_name}")
        device = module.weight.device if hasattr(module, "weight") else next(module.parameters()).device
        matches[0].load_state_dict(entry["state"], device=device)


def save_base_anchor(path: Path, student: nn.Module) -> dict[str, Any]:
    """Persist every non-deterministic standard-SVDQuant runtime anchor.

    The public rCM checkpoint lacks branch.pt; reloading it otherwise reruns
    low-rank calibration.  Dynamic QDQ deltas are anchored to these exact
    main weights, branches and smooth scales, so all three are immutable.
    """
    modules = named_modules(student)
    per_target, _ = find_smoothers(student)
    groups: dict[tuple[str, ...], torch.Tensor] = {}
    for name, proc in per_target.items():
        key = tuple(sorted(n for n, p in per_target.items() if p is proc))
        groups.setdefault(key, proc.smooth_scale.detach().cpu().clone())
    target_weight_keys = {f"{name}.weight" for name in target_names(student)}
    # SmoothQuant can rewrite adjacent non-target parameters; persist them too.
    non_target_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()
                        if key not in target_weight_keys and not key.startswith("_low_rank_branches")}
    anchor = {
        "format": "rcm_joint_svdquant_anchor_v2",
        "weights": {name: modules[name].weight.detach().cpu().clone() for name in target_names(student)},
        "non_target_state": non_target_state,
        "lowrank": export_by_block(student),
        "smooth_groups": [{"layers": list(key), "scale": scale} for key, scale in groups.items()],
        "quantizer_processors": _quantizer_processor_entries(student),
    }
    torch.save(anchor, path)
    return anchor


def load_base_anchor(student: nn.Module, path: Path) -> dict[str, torch.Tensor]:
    """Restore the baseline branch/smooth state before constructing DynamicSmooth."""
    anchor = torch.load(path, map_location="cpu", weights_only=False)
    if anchor.get("format") != "rcm_joint_svdquant_anchor_v2":
        raise RuntimeError("invalid joint SVDQuant anchor")
    missing, unexpected = student.load_state_dict(anchor["non_target_state"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith("_low_rank_branches")]
    if unexpected:
        raise RuntimeError(f"unexpected non-target anchor keys: {unexpected[:3]}")
    for index, branches in anchor["lowrank"].items():
        maps = branch_map(student.blocks[int(index)])
        for name, values in branches.items():
            maps[name].load_state_dict(values)
    _restore_quantizer_processors(student, anchor.get("quantizer_processors", []))
    per_target, _ = find_smoothers(student)
    current = {tuple(sorted(n for n, p in per_target.items() if p is proc)): proc
               for proc in {id(proc): proc for proc in per_target.values()}.values()}
    saved = {tuple(entry["layers"]): entry["scale"] for entry in anchor["smooth_groups"]}
    if set(saved) != set(current):
        raise RuntimeError("anchor smoothing manifest mismatch")
    for key, proc in current.items():
        proc.smooth_scale.copy_(saved[key].to(device=proc.smooth_scale.device, dtype=proc.smooth_scale.dtype))
    if set(anchor["weights"]) != set(target_names(student)):
        raise RuntimeError("anchor target weight manifest mismatch")
    return anchor["weights"]


def max_nmse(model: nn.Module, records: list[dict[str, Any]], device: torch.device) -> float:
    _, rows = evaluate(model, records, device)
    return max(row["nmse"] for row in rows)


def save_state(path: Path, dynamic: DynamicSmooth, student: nn.Module, optimizer: Any, epoch: int,
               anchor_path: Path) -> None:
    torch.save({"epoch": epoch, "lowrank": export_by_block(student),
                "smooth_log_delta": [x.detach().cpu() for x in dynamic.delta],
                "groups": dynamic.manifest(), "anchor": str(anchor_path),
                "optimizer": optimizer.state_dict()}, path)


def plot_curve(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([x["epoch"] for x in rows], [x["train_nmse"] for x in rows], marker="o", label="train")
    ax.plot([x["epoch"] for x in rows], [x["test_nmse"] for x in rows], marker="o", label="held-out")
    ax.set(xlabel="epoch", ylabel="teacher-forced v NMSE", yscale="log"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out, dpi=180); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lowrank-lr", type=float, default=1e-5)
    ap.add_argument("--prox-lambda", type=float, default=1e-3,
                    help="relative proximal L2 coefficient for LowRank and log-smoothing deltas")
    ap.add_argument("--smooth-lr", type=float, default=5e-7); ap.add_argument("--bound", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=20260908); ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", type=Path, default=CACHE); ap.add_argument("--extra-cache-dir", type=Path, default=EXTRA_CACHE)
    ap.add_argument("--checkpoint", type=Path, default=CKPT); ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true", help="2 samples, block-0/full-forward equivalence and one update")
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed); device = torch.device(args.device)
    train_items = load_items(args.cache_dir, TRAIN_PROMPTS) + load_items(args.extra_cache_dir, EXTRA_PROMPTS)
    test_items = load_items(args.cache_dir, TEST_PROMPTS)
    if len(train_items) != 32 or len(test_items) != 16: raise RuntimeError(f"cache count mismatch {len(train_items)} / {len(test_items)}")
    if args.smoke: train_items, test_items = train_items[:2], test_items[:1]
    teacher_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    teacher_pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    teacher_pipe.to(device)
    teacher_pipe.text_encoder.to("cpu"); teacher_pipe.vae.to("cpu"); freeze_all(teacher_pipe.transformer)
    records = capture_teacher(teacher_pipe.transformer, train_items + test_items, device)
    train_records, test_records = records[:len(train_items)], records[len(train_items):]
    student_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    student_pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    student_pipe.to(device)
    student_pipe.text_encoder.to("cpu"); student_pipe.vae.to("cpu"); load_quantized_transformer(student_pipe, args.checkpoint, MODEL)
    student = student_pipe.transformer; freeze_all(student); student.enable_gradient_checkpointing(); safe = make_autograd_safe(student)
    anchor_path = args.output_dir / "base_svdquant_anchor.pt"
    save_base_anchor(anchor_path, student)
    # Compare against the untouched checkpoint before installing the dynamic adapter.
    gate_records = train_records[:2] + test_records[:1]
    baseline_gate, _ = evaluate(student, gate_records, device)
    baseline, _ = evaluate(student, train_records + test_records, device)
    dynamic = DynamicSmooth(student, teacher_pipe.transformer, args.checkpoint, args.bound).to(device)
    # The adapter owns detached raw target weights; release the full teacher before student evaluation/backward.
    dynamic.teacher = None
    teacher_pipe.to("cpu"); del teacher_pipe; gc.collect(); torch.cuda.empty_cache()
    # Exact activation forward with identity backward; install after dynamic smooth patches.
    originals, ste_count = install_activation_ste(student)
    zero, _ = evaluate(student, gate_records, device)
    zero_rel = abs(zero["err2"] - baseline_gate["err2"]) / max(baseline_gate["ref2"], EPS)
    if zero_rel >= 5e-5:
        raise RuntimeError(f"zero-delta dynamic adapter is not equivalent: NMSE delta={zero_rel}")
    lr_params, owners = branch_parameters(student)
    # Anchor the joint update around the standard-SVDQuant solution rather than zero.
    lr_initial = [param.detach().clone() for param in lr_params]
    params = [{"params": lr_params, "lr": args.lowrank_lr}, {"params": list(dynamic.delta), "lr": args.smooth_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    config = {"args": vars(args), "safe": safe, "ste_quantizers": ste_count,
              "groups": dynamic.manifest(), "zero_delta_nmse": zero_rel, "baseline": baseline,
              "base_anchor": str(anchor_path), "prox_lambda": args.prox_lambda}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=json_default))
    curve: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        order = list(range(len(train_records))); random.Random(args.seed + epoch).shuffle(order)
        vals, regularizers, ratios = [], [], []
        for pos, idx in enumerate(order):
            rec = train_records[idx]; opt.zero_grad(set_to_none=True)
            output = student_forward(student, rec, device).float(); target = rec["target"].to(device).float()
            data_loss = (output - target).square().sum() / target.square().sum().clamp_min(EPS)
            branch_prox = sum((param.float() - initial.float()).square().mean() /
                              initial.float().square().mean().clamp_min(EPS)
                              for param, initial in zip(lr_params, lr_initial, strict=True)) / len(lr_params)
            smooth_prox = sum(delta.float().square().mean() for delta in dynamic.delta) / len(dynamic.delta)
            regularizer = branch_prox + smooth_prox
            regularizer_term = args.prox_lambda * regularizer
            loss = data_loss + regularizer_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lr_params + list(dynamic.delta), 1.0); opt.step()
            vals.append(float(data_loss.detach()))
            regularizers.append(float(regularizer.detach()))
            ratios.append(float((regularizer_term.detach() / data_loss.detach().clamp_min(EPS))))
            del output, target, loss
        row = {"epoch": epoch + 1, "online_train_nmse": sum(vals) / len(vals),
               "prox_regularizer": sum(regularizers) / len(regularizers),
               "prox_term_to_data_ratio": sum(ratios) / len(ratios)}
        curve.append(row); print(json.dumps(row), flush=True); write_csv(args.output_dir / "training_curve.csv", curve)
        save_state(args.output_dir / f"epoch_{epoch + 1:02d}.pt", dynamic, student, opt, epoch + 1, anchor_path)
    final_train, train_rows = evaluate(student, train_records, device); final_test, test_rows = evaluate(student, test_records, device)
    # Plot the online objective only; full metrics are deliberately start/end only.
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot([x["epoch"] for x in curve], [x["online_train_nmse"] for x in curve], marker="o")
    ax.set(xlabel="epoch", ylabel="online teacher-forced v NMSE", yscale="log"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.output_dir / "loss_curve.png", dpi=180); plt.close(fig)
    save_state(args.output_dir / "final_joint_state.pt", dynamic, student, opt, args.epochs, anchor_path)
    write_csv(args.output_dir / "per_sample_nmse.csv", [{"split": "train", **x} for x in train_rows] + [{"split": "heldout", **x} for x in test_rows])
    summary = {"baseline": baseline, "final_train": final_train, "final_test": final_test,
               "train_improvement": 1 - final_train["nmse"] / baseline["nmse"],
               "heldout_improvement": 1 - final_test["nmse"] / baseline["nmse"], "epochs": args.epochs}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    restore_activation_processors(originals); print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__": main()
