#!/usr/bin/env python3
"""Smoke-test native Sage3 mixed PV against its FP4 control on captured H3 QKV."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/sage3_native_mixed"),
                "/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe"]
import fp4attn_mixed_cuda
import fp4attn_cuda
import sage3_mixed_router_cuda
from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                           scale_and_quant_fp4_permute,
                           scale_and_quant_fp4_transpose)


def nmse(a, b):
    return ((a.float() - b.float()).square().sum() /
            b.float().square().sum()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/research/E114")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    capture = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
                   "h3_p1_s0_b25_480x832_73f_qkv.pt")
    payload = torch.load(capture, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:args.tokens].transpose(0, 1)[:args.heads]
               .unsqueeze(0).contiguous().cuda() for name in ("q", "k", "v"))
    with torch.inference_mode():
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packed = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1], ds,
                  args.tokens, None, 128 ** -0.5, False, True, True)
        fp8_v, fp8_scale, route, relsq = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
        control = fp4attn_mixed_cuda.fwd(*packed)[0]
        installed = fp4attn_cuda.fwd(*packed)[0]
        none = fp4attn_mixed_cuda.fwd_mixed(*packed, fp8_v, fp8_scale,
                                           torch.zeros_like(route))[0]
        all8_result = fp4attn_mixed_cuda.fwd_mixed(*packed, fp8_v, fp8_scale,
                                                    torch.ones_like(route))
        all8 = all8_result[0]
        selected = fp4attn_mixed_cuda.fwd_mixed(*packed, fp8_v, fp8_scale, route)[0]
        reference_prob = reference_out = None
        if args.tokens <= 1024:
            reference_scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (128 ** -0.5)
            reference_prob = torch.softmax(reference_scores, dim=-1)
            reference_out = torch.matmul(reference_prob, v.float())
        torch.cuda.synchronize()
        report = {"shape": list(q.shape), "padded_tokens": vp.shape[2],
                  "selected_tiles": int(route.sum().item()),
                  "all_tiles": route.numel(),
                  "control_vs_installed_nmse": nmse(control, installed),
                  "none_vs_control_nmse": nmse(none, control),
                  "all8_vs_control_nmse": nmse(all8, control),
                  "selected_vs_control_nmse": nmse(selected, control),
                  "control_rms": control.float().square().mean().sqrt().item(),
                  "all8_rms": all8.float().square().mean().sqrt().item(),
                  "all8_max_abs": all8.float().abs().max().item(),
                  "all8_cosine": torch.nn.functional.cosine_similarity(
                      all8.float().flatten(), control.float().flatten(), dim=0).item(),
                  "outputs_finite": all(torch.isfinite(t).all().item()
                                        for t in (control, none, all8, selected))}
        if reference_out is not None:
            report["control_vs_bf16_nmse"] = nmse(control, reference_out)
            report["all8_vs_bf16_nmse"] = nmse(all8, reference_out)
            report["selected_vs_bf16_nmse"] = nmse(selected, reference_out)
        if len(all8_result) > 2 and args.tokens == 128:
            staged = all8_result[2].view(torch.float8_e4m3fn)[0].float()
            v_dequant = fp8_v[0, 0].float() * fp8_scale[0, 0, :, 0, None]
            reconstructed_pv = staged @ v_dequant.T
            row_cosine = torch.nn.functional.cosine_similarity(
                reconstructed_pv, all8[0, 0].float(), dim=1)
            row_cosine_control = torch.nn.functional.cosine_similarity(
                reconstructed_pv, control[0, 0].float(), dim=1)
            staged_probability = staged / staged.sum(dim=1, keepdim=True)
            stage_col = torch.arange(128, device=staged.device)
            true_col = ((stage_col // 32) * 32 +
                        ((stage_col % 8) // 2) * 8 +
                        ((stage_col % 32) // 8) * 2 +
                        stage_col % 2)
            corrected_staged = torch.empty_like(staged)
            corrected_staged[:, true_col] = staged
            corrected_probability = corrected_staged / corrected_staged.sum(dim=1, keepdim=True)
            corrected_pv = corrected_staged @ v_dequant.T
            p8_row0_top = torch.topk(staged_probability[0], 5)
            p8_prob_row_nmse = ((staged_probability - reference_prob[0, 0]).square().sum(dim=1) /
                                 reference_prob[0, 0].square().sum(dim=1))
            all8_out_row_nmse = ((all8[0, 0].float() - reference_out[0, 0]).square().sum(dim=1) /
                                 reference_out[0, 0].square().sum(dim=1))
            worst_prob_rows = torch.topk(p8_prob_row_nmse, 5)
            worst_out_rows = torch.topk(all8_out_row_nmse, 5)
            probability_similarity = torch.nn.functional.normalize(staged_probability, dim=1) @ \
                torch.nn.functional.normalize(reference_prob[0, 0], dim=1).T
            best_ref_row = probability_similarity.argmax(dim=1)
            column_similarity = torch.nn.functional.normalize(staged_probability.T, dim=1) @ \
                torch.nn.functional.normalize(reference_prob[0, 0].T, dim=1).T
            best_ref_col = column_similarity.argmax(dim=1)
            report["p8_staging"] = {
                "nonzero_fraction": (staged != 0).float().mean().item(),
                "max": staged.max().item(),
                "mean": staged.mean().item(),
                "rms": staged.square().mean().sqrt().item(),
                "row0_first16": staged[0, :16].tolist(),
                "row0_argmax": int(staged[0].argmax().item()),
                "pv_vs_kernel_cosine": torch.nn.functional.cosine_similarity(
                    reconstructed_pv.flatten(), all8[0, 0].float().flatten(), dim=0).item(),
                "pv_vs_control_cosine": torch.nn.functional.cosine_similarity(
                    reconstructed_pv.flatten(), control[0, 0].float().flatten(), dim=0).item(),
                "pv_vs_kernel_row_cosine_mean": row_cosine.mean().item(),
                "pv_vs_kernel_row_cosine_min": row_cosine.min().item(),
                "pv_vs_kernel_row_cosine_max": row_cosine.max().item(),
                "pv_vs_control_row_cosine_mean": row_cosine_control.mean().item(),
                "row0_pv_first8": reconstructed_pv[0, :8].tolist(),
                "row0_kernel_first8": all8[0, 0, 0, :8].float().tolist(),
                "row0_control_first8": control[0, 0, 0, :8].float().tolist(),
                "row0_bf16_first8": reference_out[0, 0, 0, :8].tolist(),
                "row0_reference_argmax": int(reference_prob[0, 0, 0].argmax().item()),
                "row0_reference_peak": reference_prob[0, 0, 0].max().item(),
                "row0_p8_top5": [(int(i), float(x)) for x, i in zip(
                    p8_row0_top.values, p8_row0_top.indices)],
                "p8_probability_vs_bf16_nmse": nmse(
                    staged_probability, reference_prob[0, 0]),
                "corrected_p8_probability_vs_bf16_nmse": nmse(
                    corrected_probability, reference_prob[0, 0]),
                "corrected_pv_vs_bf16_row_cosine_mean": torch.nn.functional.cosine_similarity(
                    corrected_pv, reference_out[0, 0], dim=1).mean().item(),
                "staged_row_sum_min": staged.sum(dim=1).min().item(),
                "staged_row_sum_max": staged.sum(dim=1).max().item(),
                "staged_zero_rows": int((staged.sum(dim=1) == 0).sum().item()),
                "staged_top1_first32": staged.argmax(dim=1)[:32].tolist(),
                "reference_top1_first32": reference_prob[0, 0].argmax(dim=1)[:32].tolist(),
                "selected_row_top5": {
                    str(row): {
                        "p8": [(int(i), float(x)) for x, i in zip(
                            *torch.topk(staged_probability[row], 5))],
                        "bf16": [(int(i), float(x)) for x, i in zip(
                            *torch.topk(reference_prob[0, 0, row], 5))],
                        "saturated_count": int((staged[row] == 448).sum().item()),
                    } for row in (2, 14, 30, 94)
                },
                "worst_prob_rows": [(int(i), float(x), int(staged_probability[i].argmax()),
                                     int(reference_prob[0, 0, i].argmax()),
                                     int(best_ref_row[i]), float(probability_similarity[i].max()))
                                    for x, i in zip(worst_prob_rows.values, worst_prob_rows.indices)],
                "worst_output_rows": [(int(i), float(x)) for x, i in zip(
                    worst_out_rows.values, worst_out_rows.indices)],
                "best_reference_row_first32": best_ref_row[:32].tolist(),
                "best_reference_row_64to96": best_ref_row[64:96].tolist(),
                "best_reference_col_first64": best_ref_col[:64].tolist(),
                "best_reference_col_similarity_mean": column_similarity.max(dim=1).values.mean().item(),
            }
            matches = []
            staged_all = all8_result[2].view(torch.float8_e4m3fn)
            for scratch_block in range(args.heads):
                for head in range(args.heads):
                    v_head = fp8_v[0, head].float() * fp8_scale[0, head, :, 0, None]
                    pv = staged_all[scratch_block].float() @ v_head.T
                    cos = torch.nn.functional.cosine_similarity(
                        pv.flatten(), all8[0, head].float().flatten(), dim=0).item()
                    matches.append((cos, scratch_block, head))
            report["p8_staging"]["best_pv_kernel_matches"] = sorted(matches, reverse=True)[:5]
        print(json.dumps(report, indent=2), flush=True)
        if (report["control_vs_installed_nmse"] != 0 or
                report["none_vs_control_nmse"] != 0 or
                report["all8_vs_control_nmse"] > 0.05 or
                not report["outputs_finite"]):
            raise RuntimeError("native mixed core failed smoke gate")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {key: value for key, value in report.items() if key != "p8_staging"}
        output = args.output_dir / f"native_validation_h{args.heads}_k{args.tokens}.json"
        output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
