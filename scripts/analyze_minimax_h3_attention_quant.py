#!/usr/bin/env python3
"""Offline tensor/error study for a capture made by capture_minimax_h3_attention.py.

Raw P is intentionally never materialized.  Score rows are processed in small
chunks and P heatmaps are 128x128 tile summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FP4 = torch.tensor([-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.])


def save_heatmap(image: torch.Tensor, path: Path, *, log10: bool = False) -> None:
    """Dependency-light PNG heatmap; numeric values remain in report.json."""
    x = image.float().cpu().numpy()
    if log10:
        x = np.log10(np.maximum(x, 1e-12))
    lo, hi = np.nanpercentile(x, (1, 99))
    u8 = np.clip((x - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
    cv2.imwrite(str(path), colored)


def shape_tree(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {k: shape_tree(v) for k, v in value.items()}
    return str(type(value))


def scale_summary(value):
    if isinstance(value, torch.Tensor):
        x = value.float().flatten()
        return {"min": float(x.min()), "p01": float(torch.quantile(x, .01)),
                "median": float(torch.median(x)), "p99": float(torch.quantile(x, .99)),
                "max": float(x.max()), "zero_rate": float((x == 0).float().mean())}
    if isinstance(value, dict):
        return {k: scale_summary(v) for k, v in value.items()}
    return str(type(value))


def save_distribution(values: torch.Tensor, path: Path) -> None:
    """Write a log-count histogram without adding matplotlib to the H3 env."""
    x = values.detach().float().flatten().cpu().numpy()
    limit = np.percentile(np.abs(x), 99.9) if x.size else 1.0
    counts, _ = np.histogram(np.clip(x, -limit, limit), bins=160, range=(-limit, limit))
    canvas = np.full((300, 640, 3), 255, dtype=np.uint8)
    height = np.log1p(counts) / max(np.log1p(counts).max(), 1e-12) * 250
    width = canvas.shape[1] / len(counts)
    for i, y in enumerate(height.astype(int)):
        cv2.rectangle(canvas, (round(i * width), 280 - y), (round((i + 1) * width), 280), (120, 40, 160), -1)
    cv2.line(canvas, (320, 20), (320, 285), (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def metric(x: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    e = (x.float() - ref.float()).flatten()
    r = ref.float().flatten()
    den = r.square().sum().clamp_min(1e-30)
    return {"nmse": float(e.square().sum() / den), "rel_l2": float(e.norm() / den.sqrt()),
            "cosine": float(F.cosine_similarity(x.float().flatten(), r, dim=0)),
            "max_abs": float(e.abs().max()), "p99_abs": float(torch.quantile(e.abs(), .99)),
            "p999_abs": float(torch.quantile(e.abs(), .999)),
            "zero_rate": float((x == 0).float().mean())}


def fp8(x: torch.Tensor) -> torch.Tensor:
    return x.float().to(torch.float8_e4m3fn).float()


def nearest_fp4(x: torch.Tensor) -> torch.Tensor:
    levels = FP4.to(x.device)
    return levels[(x.float().unsqueeze(-1) - levels).abs().argmin(dim=-1)]


def int8_qdq(x: torch.Tensor, token_group: int) -> tuple[torch.Tensor, dict]:
    """Logical Sage2 Q/K proxy: symmetric INT8 per token tile and head."""
    h, n, d = x.shape
    pad = (-n) % token_group
    z = F.pad(x.float(), (0, 0, 0, pad)).reshape(h, -1, token_group, d)
    scale = z.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-7) / 127
    q = (z / scale).round().clamp(-127, 127)
    return (q * scale).reshape(h, n + pad, d)[:, :n].to(x.dtype), {"scale": scale.squeeze().cpu()}


def sa2_operands(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # The real SA2++ output is evaluated separately.  This proxy exposes its
    # public INT8-QK / FP8-V numerical ingredients without claiming to expose
    # the opaque per-thread kernel registers.
    qc, qs = int8_qdq(q, 32)
    k_center = k - k.mean(dim=1, keepdim=True)
    kc, ks = int8_qdq(k_center, 64)
    vt = v.transpose(1, 2).float()  # [H,D,N], per-channel scale over sequence
    scale = vt.abs().amax(dim=-1, keepdim=True).clamp_min(1e-7) / 2.25
    vc = (fp8(vt / scale) * scale).transpose(1, 2).to(v.dtype)
    return {"q": qc, "k": kc, "v": vc}, {"q": qs, "k": ks, "v_scale": scale.squeeze().cpu()}


def fp4_qdq(x: torch.Tensor, *, group: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    if shape[-1] % group:
        raise ValueError(f"last dimension {shape[-1]} is not divisible by {group}")
    z = x.float().reshape(*shape[:-1], -1, group)
    scale = fp8(z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 6)
    out = nearest_fp4(z / scale) * scale
    return out.reshape(shape).to(x.dtype), scale.squeeze(-1)


def sa3_operands(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q_group = 128
    if q.shape[1] % q_group:
        q_pad = F.pad(q, (0, 0, 0, (-q.shape[1]) % q_group))
    else:
        q_pad = q
    qm = q_pad.reshape(q.shape[0], -1, q_group, q.shape[-1]).mean(dim=2, keepdim=True)
    q_center = (q_pad.reshape(q.shape[0], -1, q_group, q.shape[-1]) - qm).reshape_as(q_pad)[:, :q.shape[1]]
    k_center = k - k.mean(dim=1, keepdim=True)
    qc, sq = fp4_qdq(q_center)
    kc, sk = fp4_qdq(k_center)
    # V's FP4 groups lie along the K/token axis after the kernel's transpose.
    vc_t, sv = fp4_qdq(v.transpose(1, 2))
    qmean = qm.squeeze(2).repeat_interleave(q_group, dim=1)[:, :q.shape[1]]
    return {"q": qc, "q_functional": qc + qmean, "k": kc, "v": vc_t.transpose(1, 2).contiguous()}, {
        "q_scale": sq.cpu(), "k_scale": sk.cpu(), "v_scale": sv.cpu(), "q_mean": qm.squeeze(2).cpu(),
    }


def p_sa3_qdq(p: torch.Tensor) -> torch.Tensor:
    """SA3 P representation, restored to probability units after C cancels."""
    n = p.shape[-1]
    padded = F.pad(p.float(), (0, (-n) % 16))
    z = padded.reshape(*p.shape[:-1], -1, 16)
    pmax = z.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    # Kernel encodes C*P, C=448*6; scale is 448*pmax.  Divide C back only for
    # diagnostics, because online-softmax's numerator/denominator cancel it.
    scale = fp8(448.0 * pmax)
    encoded = nearest_fp4((448.0 * 6.0 * z) / scale) * scale / (448.0 * 6.0)
    return encoded.reshape(*p.shape[:-1], -1)[..., :n]


def p_stats_and_maps(q: torch.Tensor, k: torch.Tensor, backends: dict[str, tuple[torch.Tensor, torch.Tensor]],
                     heads: list[int], out_dir: Path, chunk: int = 64, tile: int = 128) -> dict:
    """Stream P comparisons and save block-average/max maps for selected heads."""
    h, n, _ = q.shape
    grids = {head: {"mean": torch.zeros(n // tile, n // tile), "max": torch.zeros(n // tile, n // tile)} for head in heads}
    accum = {name: {"l1": 0., "kl": 0., "count": 0, "topk": 0.} for name in backends}
    kt = k.transpose(-1, -2).float()
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        score = torch.matmul(q[:, start:stop].float(), kt) * (q.shape[-1] ** -.5)
        p = score.softmax(dim=-1)
        for name, (qb, kb) in backends.items():
            pb = (torch.matmul(qb[:, start:stop].float(), kb.transpose(-1, -2).float()) * (q.shape[-1] ** -.5)).softmax(dim=-1)
            if name == "sage3_mirror":
                pb = p_sa3_qdq(pb)
            elif name == "sage2_mirror":
                pb = fp8(pb)
            a = accum[name]
            a["l1"] += float((pb - p).abs().sum())
            a["kl"] += float((p * (p.clamp_min(1e-30).log() - pb.clamp_min(1e-30).log())).sum())
            a["count"] += p.numel()
            a["topk"] += float((torch.topk(p, 8, dim=-1).indices == torch.topk(pb, 8, dim=-1).indices).float().mean())
        for head in heads:
            for r in range(start // tile, (stop - 1) // tile + 1):
                rs, re = max(start, r * tile), min(stop, (r + 1) * tile)
                local = p[head, rs - start:re - start].reshape(re - rs, n // tile, tile)
                grids[head]["mean"][r] = local.mean(dim=(0, 2)).cpu()
                grids[head]["max"][r] = local.amax(dim=(0, 2)).cpu()
    for name, value in accum.items():
        value["mean_l1"] = value.pop("l1") / value["count"]
        value["mean_kl"] = value.pop("kl") / (h * n)
        value["topk_overlap"] = value.pop("topk") / (n // chunk + bool(n % chunk))
        value.pop("count")
    for head, maps in grids.items():
        for kind, image in maps.items():
            save_heatmap(image, out_dir / f"p_head{head}_{kind}.png", log10=True)
    return accum


def operand_heatmap(name: str, x: torch.Tensor, out_dir: Path, tile: int = 128) -> None:
    h, n, d = x.shape
    z = x.float().reshape(h, n // tile, tile, d)
    maps = {"absmax": z.abs().amax(dim=(2, 3)), "rms": z.square().mean(dim=(2, 3)).sqrt(),
            "outlier_rate": (z.abs() > 3 * z.square().mean(dim=(2, 3), keepdim=True).sqrt()).float().mean(dim=(2, 3))}
    for kind, image in maps.items():
        save_heatmap(image, out_dir / f"{name}_{kind}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/research/E089_h3_attention")
    p.add_argument("--sage3-path", type=Path, default=Path("/data1/models/svdquant-wjq/third_party/SageAttention-official-20261004/sageattention3_blackwell"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    t, meta = data["tensors"], data["metadata"]
    cu = t["cu_seqlens"].tolist()
    segments = [(a, b) for a, b in zip(cu[:-1], cu[1:]) if b > a]
    seg_index, (start, stop) = max(enumerate(segments), key=lambda x: x[1][1] - x[1][0])
    q, k, v = (t[key][start:stop].transpose(0, 1).contiguous().to(args.device) for key in ("q", "k", "v"))
    reference_saved = t["attention_output"][start:stop].transpose(0, 1).contiguous().to(args.device)
    ref = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=float(t["softmax_scale"])).squeeze(0)
    replay = metric(ref, reference_saved)
    q2, sa2_scales = sa2_operands(q, k, v)
    q3, sa3_scales = sa3_operands(q, k, v)
    q_groups = F.pad(q, (0, 0, 0, (-q.shape[1]) % 128)).reshape(q.shape[0], -1, 128, q.shape[-1])
    q_center = (q_groups - q_groups.mean(dim=2, keepdim=True)).reshape(q.shape[0], -1, q.shape[-1])[:, :q.shape[1]]
    operand = {"sage2_mirror": {key: metric(q2[key], (k - k.mean(1, keepdim=True)) if key == "k" else {"q": q, "v": v}[key]) for key in q2},
               "sage3_mirror": {"q": metric(q3["q"], q_center), "k": metric(q3["k"], k - k.mean(1, keepdim=True)), "v": metric(q3["v"], v)}}
    for tensor_name, ref_tensor in (("q", q), ("k", k), ("v", v)):
        save_distribution(ref_tensor, args.output_dir / f"{tensor_name}_bf16_distribution.png")
        for backend, operands in (("sage2_mirror", q2), ("sage3_mirror", q3)):
            if tensor_name not in operands:
                continue
            baseline = (k - k.mean(1, keepdim=True)) if tensor_name == "k" else q_center if backend == "sage3_mirror" and tensor_name == "q" else ref_tensor
            save_distribution(operands[tensor_name] - baseline, args.output_dir / f"{tensor_name}_{backend}_residual_distribution.png")
    # q-centering reference above is only used for operand error; functional SA3
    # correctness is assessed by actual kernel output below.
    heat_n = q.shape[1] // 128 * 128
    for name, x in (("q", q[:, :heat_n]), ("k", k[:, :heat_n]), ("v", v[:, :heat_n])):
        operand_heatmap(name, x.cpu(), args.output_dir)
    # Determine representative heads from SA3 Q proxy error, then stream P.
    head_err = (q3["q"].float() - q.float()).square().mean(dim=(1, 2))
    heads = sorted(set([0, int(head_err.argsort()[len(head_err)//2]), int(head_err.argmax())]))
    p_report = p_stats_and_maps(q[:, :heat_n], k[:, :heat_n], {"sage2_mirror": (q2["q"][:, :heat_n], q2["k"][:, :heat_n]), "sage3_mirror": (q3["q_functional"][:, :heat_n], q3["k"][:, :heat_n])}, heads, args.output_dir)
    actual: dict[str, object] = {}
    with torch.inference_mode():
        try:
            from sageattention import sageattn
            o2 = sageattn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), tensor_layout="HND", sm_scale=float(t["softmax_scale"])).squeeze(0)
            actual["sage2pp"] = metric(o2, ref)
        except Exception as exc:
            actual["sage2pp_error"] = repr(exc)
        try:
            sys.path.insert(0, str(args.sage3_path))
            from sageattn3 import sageattn3_blackwell
            o3 = sageattn3_blackwell(q.unsqueeze(0), k.unsqueeze(0).clone(), v.unsqueeze(0), is_causal=False).squeeze(0)
            actual["sage3"] = metric(o3, ref)
        except Exception as exc:
            actual["sage3_error"] = repr(exc)
    # Post-projection error when an actual backend output was available.
    weight = t["out_proj_weight"].to(args.device)
    for label, out in (("sage2pp", locals().get("o2")), ("sage3", locals().get("o3"))):
        if isinstance(out, torch.Tensor):
            projected = F.linear(out.transpose(0, 1).reshape(stop - start, -1), weight)
            actual[label + "_out_proj"] = metric(projected, t["out_proj_output"][start:stop].to(args.device))
    rows = [{"backend": backend, "tensor": tensor, **values} for backend, values in operand.items() for tensor, values in values.items()]
    rows += [{"backend": backend, "tensor": "attention_output", **values} for backend, values in actual.items() if isinstance(values, dict)]
    with (args.output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row})); writer.writeheader(); writer.writerows(rows)
    report = {"state": "complete", "capture": str(args.capture), "capture_metadata": meta,
              "selected_segment": {"index": seg_index, "start": start, "stop": stop, "length": stop-start,
                                   "p_heatmap_length": heat_n, "p_heatmap_tail_excluded": (stop-start)-heat_n},
              "reference_replay": replay, "representative_heads": heads, "operand_error": operand,
              "p_error": p_report, "actual_kernel_error": actual,
              "scale_shapes": {"sage2": shape_tree(sa2_scales), "sage3": shape_tree(sa3_scales)},
              "scale_summary": {"sage2": scale_summary(sa2_scales), "sage3": scale_summary(sa3_scales)}}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
