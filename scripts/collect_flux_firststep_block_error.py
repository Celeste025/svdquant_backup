#!/usr/bin/env python3
"""Measure first-step cumulative block error for true Nunchaku FLUX W4A4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from collect_timestep_activation_outliers import FLUX_PROMPT, seed_everything


ROOT = Path("outputs/quant_error_diagnosis/block_propagation")
DUAL = tuple(list(range(0, 19, 3)) + [18])
SINGLE = tuple(list(range(0, 38, 3)) + [37])


class StopAfterFirstStep(Exception):
    pass


def metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    x, y = x.float(), y.float()
    e = y - x
    p = x.square().mean().clamp_min(1e-20)
    return {
        "mse": float(e.square().mean()),
        "nmse": float(e.square().mean() / p),
        "cosine": float(
            torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)
        ),
        "reference_rms": float(p.sqrt()),
        "error_rms": float(e.square().mean().sqrt()),
    }


def run_first_step(pipe):
    seed_everything(44)

    def callback(_pipe, index, _timestep, kwargs):
        if index == 0:
            raise StopAfterFirstStep
        return kwargs

    try:
        pipe(
            FLUX_PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(44),
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    except StopAfterFirstStep:
        pass


def install_adaln_split_capture(norm_out_module, store: dict):
    """Split AdaLayerNormContinuous into LN-only and post-modulation tensors.

    Matches Wan head probes:
      norm_out                  = LayerNorm(x)
      post_timestep_modulation  = LN(x) * (1+scale) + shift
    """
    original = norm_out_module.forward

    def forward(x: torch.Tensor, conditioning_embedding: torch.Tensor):
        emb = norm_out_module.linear(
            norm_out_module.silu(conditioning_embedding).to(x.dtype)
        )
        scale, shift = torch.chunk(emb, 2, dim=1)
        ln = norm_out_module.norm(x)
        store["norm_out"] = ln.detach().to("cpu", torch.bfloat16).clone()
        out = ln * (1 + scale)[:, None, :] + shift[:, None, :]
        store["post_timestep_modulation"] = out.detach().to(
            "cpu", torch.bfloat16
        ).clone()
        return out

    norm_out_module.forward = forward  # type: ignore[method-assign]
    return original


@torch.inference_mode()
def collect_bf16():
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    outputs = {}
    handles = []
    for block in DUAL:
        def hook(_m, _i, out, *, block=block):
            # Diffusers joint block returns (encoder_hidden_states, hidden_states).
            outputs[f"dual{block}_context"] = out[0].detach().to("cpu", torch.bfloat16).clone()
            outputs[f"dual{block}_image"] = out[1].detach().to("cpu", torch.bfloat16).clone()
        handles.append(pipe.transformer.transformer_blocks[block].register_forward_hook(hook))
    for block in SINGLE:
        def hook(_m, _i, out, *, block=block):
            # Newer Diffusers single-stream blocks return (context, image);
            # store the concatenated tensor so quant replay (combined) still matches.
            if isinstance(out, tuple):
                value = torch.cat(out[:2], dim=1)
            else:
                value = out
            outputs[f"single{block}"] = value.detach().to("cpu", torch.bfloat16).clone()
        handles.append(pipe.transformer.single_transformer_blocks[block].register_forward_hook(hook))
    original_norm = install_adaln_split_capture(pipe.transformer.norm_out, outputs)
    handles.append(
        pipe.transformer.proj_out.register_forward_hook(
            lambda _m, _i, out: outputs.update(
                {"proj_out": out.detach().to("cpu", torch.bfloat16).clone()}
            )
        )
    )
    pipe.enable_model_cpu_offload(gpu_id=0)
    run_first_step(pipe)
    for h in handles:
        h.remove()
    pipe.transformer.norm_out.forward = original_norm
    torch.save({"dual": DUAL, "single": SINGLE, "outputs": outputs}, ROOT / "flux_firststep_bf16.pt")
    print(f"saved {len(outputs)} reference tensors")
    assert "norm_out" in outputs and "post_timestep_modulation" in outputs


@torch.inference_mode()
def collect_quant():
    from diffusers import FluxPipeline
    from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        "mit-han-lab/svdq-int4-flux.1-dev", device="cuda:0"
    )
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    reference = torch.load(
        ROOT / "flux_firststep_bf16.pt", map_location="cpu", weights_only=False
    )["outputs"]
    captured = {}

    def prehook(_module, args, kwargs):
        if kwargs:
            values = (
                kwargs["hidden_states"],
                kwargs["temb"],
                kwargs["encoder_hidden_states"],
                kwargs["image_rotary_emb"],
            )
        else:
            values = args[:4]
        captured["hidden"], captured["temb"], captured["context"], captured["rope"] = [
            x.detach().clone() if torch.is_tensor(x) else x for x in values
        ]

    handle = transformer.transformer_blocks[0].register_forward_pre_hook(
        prehook, with_kwargs=True
    )
    original_norm = install_adaln_split_capture(transformer.norm_out, captured)
    head_handles = [
        transformer.proj_out.register_forward_hook(
            lambda _m, _i, out: captured.update(
                {"proj_out": out.detach().to("cpu", torch.bfloat16).clone()}
            )
        ),
    ]
    run_first_step(pipe)
    handle.remove()
    for head_handle in head_handles:
        head_handle.remove()
    transformer.norm_out.forward = original_norm
    wrapper = transformer.transformer_blocks[0]
    m = wrapper.m
    hidden = captured["hidden"].to("cuda", torch.bfloat16)
    context = captured["context"].to("cuda", torch.bfloat16)
    temb = captured["temb"].to("cuda", torch.bfloat16)
    rope = captured["rope"].to("cuda")
    txt = context.shape[1]
    img = hidden.shape[1]
    rope = rope.reshape(1, txt + img, *rope.shape[3:])
    rope_context, rope_img = rope[:, :txt], rope[:, txt:]
    reports = []
    for block in range(19):
        hidden, context = m.forward_layer(
            block, hidden, context, temb, rope_img, rope_context
        )
        if block in DUAL:
            for stream, value in (("image", hidden), ("context", context)):
                result = metrics(reference[f"dual{block}_{stream}"], value.cpu())
                result.update({"stage": "dual", "block": block, "stream": stream})
                reports.append(result)
    combined = torch.cat([context, hidden], dim=1)
    for block in range(38):
        combined = m.forward_single_layer(block, combined, temb, rope)
        if block in SINGLE:
            reference_combined = reference[f"single{block}"]
            for stream, ref_value, quant_value in (
                ("combined", reference_combined, combined.cpu()),
                (
                    "image",
                    reference_combined[:, txt:],
                    combined[:, txt:].cpu(),
                ),
            ):
                result = metrics(ref_value, quant_value)
                result.update({"stage": "single", "block": block, "stream": stream})
                reports.append(result)
    for stage in ("norm_out", "post_timestep_modulation", "proj_out"):
        result = metrics(reference[stage], captured[stage])
        result.update({"stage": stage, "block": None, "stream": "image"})
        reports.append(result)
    (ROOT / "flux_firststep_metrics.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(json.dumps(reports, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bf16", "quant"), required=True)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    collect_bf16() if args.mode == "bf16" else collect_quant()


if __name__ == "__main__":
    main()
