#!/usr/bin/env python3
"""First-step Wan error with blocks 18--24 kept in BF16 (rest fake W4A4).

Reuses existing BF16 reference tensors; only the mixed quant run is collected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from collect_timestep_activation_outliers import (
    WAN_MODEL,
    WAN_NEGATIVE,
    WAN_PROMPT,
    seed_everything,
)
from compare_wan_svdquant_fake import load_calibrated_transformer, make_pipeline


ROOT = Path("outputs/quant_error_diagnosis/block_propagation")
CACHE = Path("outputs/wan_svdquant_calib_large/svdquant_large_calibrated.pt")
KEEP_BF16_BLOCKS = tuple(range(18, 25))
KEEP_PATTERNS = tuple(f"blocks.{i}." for i in KEEP_BF16_BLOCKS)
SPARSE_BLOCKS = tuple(list(range(0, 30, 3)) + [29])
DENSE_BLOCKS = tuple(range(18, 25)) + (27, 29)
TAG = "keep_bf16_b18_24"


class StopAfterFirstStep(Exception):
    pass


def metrics(reference: torch.Tensor, quantized: torch.Tensor) -> dict:
    reference = reference.float()
    quantized = quantized.float()
    error = quantized - reference
    power = reference.square().mean().clamp_min(1e-20)
    return {
        "mse": float(error.square().mean()),
        "nmse": float(error.square().mean() / power),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.flatten(), quantized.flatten(), dim=0
            )
        ),
        "reference_rms": float(power.sqrt()),
        "error_rms": float(error.square().mean().sqrt()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="cuda device id for cpu_offload"
    )
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)

    sparse_ref = torch.load(
        ROOT / "wan_firststep_bf16.pt", map_location="cpu", weights_only=False
    )["outputs"]
    dense_ref = torch.load(
        ROOT / "wan_firststep_dense_head_bf16.pt",
        map_location="cpu",
        weights_only=False,
    )["outputs"]

    pipe = make_pipeline(WAN_MODEL)
    pipe.transformer.to("cpu")
    converted, _ = load_calibrated_transformer(
        pipe.transformer,
        CACHE,
        quantize_activation=True,
        keep_bf16_patterns=KEEP_PATTERNS,
    )
    print(
        json.dumps(
            {
                "keep_bf16_blocks": list(KEEP_BF16_BLOCKS),
                "quantized_linears": len(converted),
                "keep_patterns": list(KEEP_PATTERNS),
            },
            indent=2,
        ),
        flush=True,
    )

    sparse_calls = {b: 0 for b in SPARSE_BLOCKS}
    dense_calls: dict[str, int] = {}
    dense_outputs: dict[str, torch.Tensor] = {}
    sparse_reports: list[dict] = []
    dense_reports: list[dict] = []
    handles = []

    def record_dense(stage: str, tensor: torch.Tensor) -> None:
        branch = dense_calls.get(stage, 0)
        dense_calls[stage] = branch + 1
        if branch >= 2:
            return
        key = f"{stage}_branch{branch}"
        cpu = tensor.detach().to("cpu", torch.bfloat16).clone()
        dense_outputs[key] = cpu
        report = metrics(dense_ref[key], cpu)
        report.update({"stage": stage, "cfg_branch": branch})
        dense_reports.append(report)

    for block in SPARSE_BLOCKS:
        module = pipe.transformer.blocks[block]

        def sparse_hook(_module, _inputs, output, *, block=block):
            branch = sparse_calls[block]
            sparse_calls[block] += 1
            if branch >= 2:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            key = f"block{block}_branch{branch}"
            report = metrics(sparse_ref[key], tensor.detach().to("cpu", torch.bfloat16))
            report.update(
                {
                    "block": block,
                    "cfg_branch": branch,
                    "kept_bf16": block in KEEP_BF16_BLOCKS,
                }
            )
            sparse_reports.append(report)

        handles.append(module.register_forward_hook(sparse_hook))

    for block in DENSE_BLOCKS:
        module = pipe.transformer.blocks[block]

        def dense_block_hook(_module, _inputs, output, *, block=block):
            record_dense(
                f"block{block}",
                output[0] if isinstance(output, tuple) else output,
            )

        handles.append(module.register_forward_hook(dense_block_hook))

    handles.append(
        pipe.transformer.norm_out.register_forward_hook(
            lambda _m, _i, out: record_dense("norm_out", out)
        )
    )
    handles.append(
        pipe.transformer.proj_out.register_forward_pre_hook(
            lambda _m, inputs: record_dense("post_timestep_modulation", inputs[0])
        )
    )
    handles.append(
        pipe.transformer.proj_out.register_forward_hook(
            lambda _m, _i, out: record_dense("proj_out", out)
        )
    )

    pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    seed_everything(44)

    def callback(_pipe, index, _timestep, kwargs):
        if index == 0:
            raise StopAfterFirstStep
        return kwargs

    try:
        pipe(
            prompt=WAN_PROMPT,
            negative_prompt=WAN_NEGATIVE,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=50,
            guidance_scale=6.0,
            generator=torch.Generator(device="cuda").manual_seed(44),
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    except StopAfterFirstStep:
        pass
    finally:
        for handle in handles:
            handle.remove()

    for stage in ("proj_out",):
        ref_guided = dense_ref[f"{stage}_branch0"].float() + 6.0 * (
            dense_ref[f"{stage}_branch1"].float()
            - dense_ref[f"{stage}_branch0"].float()
        )
        quant_guided = dense_outputs[f"{stage}_branch0"].float() + 6.0 * (
            dense_outputs[f"{stage}_branch1"].float()
            - dense_outputs[f"{stage}_branch0"].float()
        )
        report = metrics(ref_guided, quant_guided)
        report.update({"stage": f"{stage}_cfg6_guided", "cfg_branch": None})
        dense_reports.append(report)

    sparse_path = ROOT / f"wan_firststep_metrics_{TAG}.json"
    dense_path = ROOT / f"wan_firststep_dense_head_metrics_{TAG}.json"
    sparse_path.write_text(json.dumps(sparse_reports, indent=2), encoding="utf-8")
    dense_path.write_text(json.dumps(dense_reports, indent=2), encoding="utf-8")

    # Compact CFG / head summary for the console.
    import numpy as np

    def mean_stage(stage: str) -> dict:
        rows = [r for r in dense_reports if r["stage"] == stage]
        return {
            "mse": float(np.mean([r["mse"] for r in rows])),
            "nmse": float(np.mean([r["nmse"] for r in rows])),
        }

    guided = next(r for r in dense_reports if r["stage"] == "proj_out_cfg6_guided")
    branch = [r for r in dense_reports if r["stage"] == "proj_out"]
    summary = {
        "tag": TAG,
        "keep_bf16_blocks": list(KEEP_BF16_BLOCKS),
        "quantized_linears": len(converted),
        "block29": mean_stage("block29"),
        "norm_out": mean_stage("norm_out"),
        "post_timestep_modulation": mean_stage("post_timestep_modulation"),
        "proj_out": mean_stage("proj_out"),
        "proj_out_cfg6_guided": {
            "mse": guided["mse"],
            "nmse": guided["nmse"],
        },
        "cfg_amplify_mse": guided["mse"]
        / float(np.mean([r["mse"] for r in branch])),
        "cfg_amplify_nmse": guided["nmse"]
        / float(np.mean([r["nmse"] for r in branch])),
        "sparse_path": str(sparse_path),
        "dense_path": str(dense_path),
    }
    (ROOT / f"wan_{TAG}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
