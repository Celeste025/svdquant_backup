#!/usr/bin/env python3
"""Stream a large, spatiotemporally stratified Wan calibration set to memmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from calibrate_wan_svdquant import module_groups
from compare_wan_svdquant_fake import DEFAULT_MODEL, make_pipeline, seed_everything


DEFAULT_PROMPTS = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "examples/wan/assets/vbench_subset_8.json"
)
DEFAULT_INFER = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "examples/wan/configs/infer.yaml"
)


class StepController:
    def __init__(self, selected_steps: set[int], calls_per_step: int = 2) -> None:
        self.selected_steps = selected_steps
        self.calls_per_step = calls_per_step
        self.call_index = 0
        self.active = False
        self.step_index = -1

    def reset(self) -> None:
        self.call_index = 0
        self.active = False
        self.step_index = -1

    def pre(self, module: nn.Module, inputs: tuple, kwargs: dict) -> None:
        self.step_index = self.call_index // self.calls_per_step
        self.active = self.step_index in self.selected_steps

    def post(self, module: nn.Module, inputs: tuple, kwargs: dict, output) -> None:
        self.call_index += 1


class MemmapSampler:
    def __init__(
        self,
        output_dir: Path,
        group_index: int,
        names: list[str],
        total_calls: int,
        tokens_per_call: int,
        controller: StepController,
    ) -> None:
        self.output_dir = output_dir
        self.group_index = group_index
        self.names = names
        self.total_calls = total_calls
        self.tokens_per_call = tokens_per_call
        self.controller = controller
        self.array: np.memmap | None = None
        self.path = output_dir / f"group_{group_index:03d}.bf16.mmap"
        self.rows_per_call = 0
        self.offset = 0
        self.in_features = 0

    def __call__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if not self.controller.active:
            return
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        count = min(self.tokens_per_call, x.shape[0])
        if self.array is None:
            self.rows_per_call = count
            self.in_features = x.shape[-1]
            shape = (self.total_calls * count, self.in_features)
            self.array = np.memmap(self.path, dtype=np.uint16, mode="w+", shape=shape)
        if count != self.rows_per_call or x.shape[-1] != self.in_features:
            raise RuntimeError(
                f"shape changed for group {self.group_index}: "
                f"{tuple(x.shape)} vs rows={self.rows_per_call}, dim={self.in_features}"
            )
        positions = torch.linspace(0, x.shape[0] - 1, count, device=x.device).long()
        sample = x.index_select(0, positions).to(device="cpu", dtype=torch.bfloat16).contiguous()
        # NumPy has no native bfloat16; store the raw 16-bit representation.
        raw = sample.view(torch.uint16).numpy()
        self.array[self.offset : self.offset + count] = raw
        self.offset += count

    def finish(self) -> dict:
        if self.array is None:
            raise RuntimeError(f"group {self.group_index} collected no inputs")
        self.array.flush()
        expected = self.total_calls * self.rows_per_call
        if self.offset != expected:
            raise RuntimeError(f"group {self.group_index}: wrote {self.offset}, expected {expected}")
        return {
            "group_index": self.group_index,
            "names": self.names,
            "path": self.path.name,
            "shape": [expected, self.in_features],
            "rows_per_call": self.rows_per_call,
            "selected_calls": self.total_calls,
            "storage_dtype": "bfloat16_raw_uint16",
            "bytes": self.path.stat().st_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/wan_svdquant_calib_large")
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--tokens-per-call", type=int, default=2048)
    parser.add_argument("--num-selected-steps", type=int, default=10)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts.read_text())
    if len(prompts) < 8:
        raise RuntimeError(f"need at least 8 prompts, got {len(prompts)}")
    prompts = prompts[:8]
    infer_text = DEFAULT_INFER.read_text()
    marker = "negative_prompt: >-\n"
    negative_prompt = infer_text.split(marker, 1)[1].replace("\n  ", " ").strip()
    selected_steps = {
        int(round(x))
        for x in np.linspace(0, args.steps - 1, args.num_selected_steps)
    }
    if len(selected_steps) != args.num_selected_steps:
        raise RuntimeError(f"selected timestep collision: {sorted(selected_steps)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = make_pipeline(args.model)
    modules = dict(pipe.transformer.named_modules())
    controller = StepController(selected_steps)
    pre_handle = pipe.transformer.register_forward_pre_hook(controller.pre, with_kwargs=True)
    post_handle = pipe.transformer.register_forward_hook(controller.post, with_kwargs=True)
    total_calls = len(prompts) * len(selected_steps) * 2
    samplers = []
    handles = []
    for index, names in enumerate(module_groups()):
        module = modules[names[0]]
        sampler = MemmapSampler(
            args.output_dir,
            index,
            names,
            total_calls,
            args.tokens_per_call,
            controller,
        )
        samplers.append(sampler)
        handles.append(module.register_forward_pre_hook(sampler))

    pipe.enable_model_cpu_offload()
    run_records = []
    for prompt_index, item in enumerate(prompts):
        seed = args.base_seed + prompt_index
        controller.reset()
        seed_everything(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print(
            f"[prompt {prompt_index + 1}/8 seed={seed}] {item['prompt_en']}",
            flush=True,
        )
        pipe(
            prompt=item["prompt_en"],
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            output_type="latent",
        )
        if controller.call_index != args.steps * 2:
            raise RuntimeError(
                f"prompt {prompt_index} made {controller.call_index} transformer calls, "
                f"expected {args.steps * 2}"
            )
        run_records.append(
            {"index": prompt_index, "seed": seed, "prompt": item["prompt_en"]}
        )

    for handle in handles:
        handle.remove()
    pre_handle.remove()
    post_handle.remove()
    groups = [sampler.finish() for sampler in samplers]
    total_bytes = sum(group["bytes"] for group in groups)
    metadata = {
        "format": "wan-svdquant-memmap-v1",
        "model": str(args.model),
        "prompt_file": str(args.prompts),
        "prompts": run_records,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "flow_shift": 3.0,
        "selected_step_indices": sorted(selected_steps),
        "tokens_per_spatiotemporal_call": args.tokens_per_call,
        "num_groups": len(groups),
        "groups": groups,
        "total_bytes": total_bytes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
