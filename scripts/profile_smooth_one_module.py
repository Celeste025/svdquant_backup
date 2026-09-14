#!/usr/bin/env python3
"""Profile one Flux smooth module (transformer_blocks.0.attn.qkv_proj) on an idle GPU.

Breaks wall time into:
  - model/calib load
  - collect acts for block 0
  - baseline original outputs (OutputsError)
  - per-candidate: low-rank/SVD+quant (_process) vs attn forward eval
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
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
OUT_JSON = OUT_DIR / "smooth_qkv_proj_block0.json"
OUT_LOG = OUT_DIR / "smooth_qkv_proj_block0.log"

# Match _deepcompressor_entry.py: import torch/pyarrow before deepcompressor.
import torch  # noqa: E402,F401
import pyarrow  # noqa: E402,F401


class Timer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.marks: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1
            self.marks.append((name, dt))
            msg = f"[TIMER] {name}: {dt:.3f}s (count={self.counts[name]}, total={self.totals[name]:.3f}s)"
            print(msg, flush=True)
            with OUT_LOG.open("a") as f:
                f.write(msg + "\n")

    def summary(self) -> dict:
        return {
            "totals_sec": dict(self.totals),
            "counts": dict(self.counts),
            "marks": [{"name": n, "sec": s} for n, s in self.marks],
        }


def install_hooks(timer: Timer) -> None:
    import torch
    from deepcompressor.calib import search as search_mod
    from deepcompressor.calib import smooth as smooth_mod
    from deepcompressor.nn.patch.lowrank import LowRankBranch
    from deepcompressor.quantizer.processor import Quantizer

    orig_svd = torch.linalg.svd

    def timed_svd(A, *args, **kwargs):
        # LowRankBranch uses weight.double() SVD — usually on CUDA for Flux weights
        name = "candidate.torch_linalg_svd"
        if torch.is_tensor(A):
            name = f"candidate.torch_linalg_svd[{tuple(A.shape)}/{A.dtype}/{A.device}]"
        with timer.section(name):
            return orig_svd(A, *args, **kwargs)

    torch.linalg.svd = timed_svd  # type: ignore[assignment]

    orig_reset = LowRankBranch.reset_parameters

    def timed_reset(self, weight=None):
        with timer.section("candidate.LowRankBranch.reset_parameters"):
            return orig_reset(self, weight)

    LowRankBranch.reset_parameters = timed_reset  # type: ignore[method-assign]

    orig_qwl = Quantizer.quantize_with_low_rank

    def timed_qwl(self, *args, **kwargs):
        with timer.section("candidate.quantize_with_low_rank"):
            return orig_qwl(self, *args, **kwargs)

    Quantizer.quantize_with_low_rank = timed_qwl  # type: ignore[method-assign]

    orig_process = smooth_mod.SmoothLinearCalibrator._process_wgts_centric_mod

    def timed_process(self, *args, **kwargs):
        with timer.section("candidate.process_wgts_centric_mod"):
            return orig_process(self, *args, **kwargs)

    smooth_mod.SmoothLinearCalibrator._process_wgts_centric_mod = timed_process  # type: ignore

    orig_recover = search_mod.SearchBasedCalibrator._recover_mod

    def timed_recover(self, *args, **kwargs):
        with timer.section("candidate.recover_mod"):
            return orig_recover(self, *args, **kwargs)

    search_mod.SearchBasedCalibrator._recover_mod = timed_recover  # type: ignore

    orig_cal_wgts = search_mod.SearchBasedCalibrator._calibrate_wgts

    def timed_cal_wgts(self, wgts, ipts, eval_module, mods, orig_wgts, orig_ipts, eval_kwargs, **kwargs):
        from deepcompressor.calib.config import SearchBasedCalibObjective
        from deepcompressor.data.cache import TensorsCache
        import gc
        import psutil

        if self.objective != SearchBasedCalibObjective.OutputsError:
            with timer.section("calibrate_wgts.other_objective"):
                return orig_cal_wgts(
                    self, wgts, ipts, eval_module, mods, orig_wgts, orig_ipts, eval_kwargs, **kwargs
                )

        _state_dict = []
        if orig_wgts is not None:
            _state_dict = [(p, p.data) for p, _ in orig_wgts]
            for p, w in orig_wgts:
                p.data = w.to(device=p.data.device)
        if orig_ipts is None:
            orig_ipts = ipts
        assert isinstance(orig_ipts, TensorsCache)
        orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
        n = len(orig_ipts.front().data)
        with timer.section("baseline.original_outputs_eval"):
            for i in range(n):
                ipt = orig_ipts.extract(i, eval_kwargs)
                y = eval_module(*ipt.args, **ipt.kwargs)
                y = y[0] if not isinstance(y, torch.Tensor) else y
                orig_opts[(i,)] = y.to(device=self.opts_device or y.device, non_blocking=True)
                del ipt, y
        for p, s in _state_dict:
            p.data = s
        del orig_wgts, orig_ipts, _state_dict
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.debug(
            f"+ finished calculating the original outputs, ram usage: {psutil.virtual_memory().percent}"
        )

        cand_i = 0
        while not self.is_done():
            self.ask()
            cand_i += 1
            with timer.section("candidate.total"):
                self._process_wgts_centric_mod(wgts=wgts, mods=mods, **kwargs)
                e = [None]
                with timer.section("candidate.eval_module_forwards"):
                    for i in range(len(ipts.front().data)):
                        ipt = ipts.extract(i, eval_kwargs)
                        y = eval_module(*ipt.args, **ipt.kwargs)
                        y = y[0] if not isinstance(y, torch.Tensor) else y
                        y = (y - orig_opts[(i,)].to(device=y.device, non_blocking=True)).to(self.develop_dtype)
                        y = y.pow_(self.config.degree).sum().view(-1)
                        if e[0] is None:
                            e[0] = y
                        else:
                            e[0].add_(y)
                        del ipt, y
                self._recover_mod()
            self.tell(e)
            if cand_i <= 3 or cand_i % 5 == 0:
                svd_total = sum(v for k, v in timer.totals.items() if k.startswith("candidate.torch_linalg_svd"))
                print(
                    f"[TIMER] finished candidate {cand_i}: "
                    f"process={timer.totals['candidate.process_wgts_centric_mod']:.2f}s "
                    f"lowrank_wrap={timer.totals.get('candidate.quantize_with_low_rank', 0):.2f}s "
                    f"svd={svd_total:.2f}s "
                    f"eval={timer.totals['candidate.eval_module_forwards']:.2f}s",
                    flush=True,
                )
        return self.get_best()

    search_mod.SearchBasedCalibrator._calibrate_wgts = timed_cal_wgts  # type: ignore


def main() -> int:
    OUT_LOG.write_text("")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"log -> {OUT_LOG}")
    print(f"json -> {OUT_JSON}")

    os.chdir(DIFFUSION)

    import torch
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.quant.utils import get_needs_inputs_fn
    from deepcompressor.calib.smooth import smooth_linear_modules
    from deepcompressor.quantizer import Quantizer
    from deepcompressor.utils import tools

    tools.logging.setup(path=str(OUT_LOG), level=tools.logging.INFO)
    timer = Timer()
    install_hooks(timer)

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
    # DiffusionPtqRunConfig parser expects sys.argv style via parse_known_args without program name sometimes
    sys.argv = ["profile_smooth_one_module.py", *argv]
    config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    print("unused_cfgs", unused_cfgs, "unused_args", unused_args, "unknown", unknown_args)

    with timer.section("load.pipeline_build"):
        pipeline = config.pipeline.build()
        for attr in (
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
            "image_encoder",
            "controlnet",
        ):
            if hasattr(pipeline, attr):
                mod = getattr(pipeline, attr)
                if isinstance(mod, torch.nn.Module):
                    mod.to("cpu")
        torch.cuda.empty_cache()
        model = DiffusionModelStruct.construct(pipeline)

    quant = config.quant
    assert quant.enabled_smooth

    # Collect activations for first block only, then smooth only qkv_proj
    with timer.section("collect.acts_block0"):
        loader = quant.calib.build_loader()
        layer = None
        layer_cache = None
        layer_kwargs = None
        for layer_name, (layer, layer_cache, layer_kwargs) in loader.iter_layer_activations(
            model,
            needs_inputs_fn=get_needs_inputs_fn(model, quant),
            skip_pre_modules=True,
            skip_post_modules=True,
        ):
            print(f"collected first layer: {layer_name}", flush=True)
            break
        assert layer is not None

    # Find first transformer block attn
    from deepcompressor.app.diffusion.nn.struct import (
        DiffusionAttentionStruct,
        DiffusionFeedForwardStruct,
        DiffusionTransformerBlockStruct,
    )

    attn = None
    for module_key, module_name, module, parent, _ in layer.named_key_modules():
        if isinstance(parent, (DiffusionAttentionStruct, DiffusionFeedForwardStruct)):
            block = parent.parent
            assert isinstance(block, DiffusionTransformerBlockStruct)
            attn = block.attn_structs[0]
            break
    assert attn is not None
    print(f"profiling module: {attn.name}.qkv_proj", flush=True)
    print(
        f"weight shapes: {[tuple(m.weight.shape) for m in attn.qkv_proj]}",
        flush=True,
    )
    print(
        f"smooth.proj num_grids={quant.smooth.proj.num_grids} allow_low_rank={quant.smooth.proj.allow_low_rank}",
        flush=True,
    )
    print(
        f"calib num_samples={quant.calib.num_samples} batch_size={quant.calib.batch_size}",
        flush=True,
    )

    module_key = attn.qkv_proj_key
    config_wgts = quant.wgts
    if quant.enabled_extra_wgts and quant.extra_wgts.is_enabled_for(module_key):
        config_wgts = quant.extra_wgts

    with timer.section("smooth.qkv_proj_only"):
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
        print(f"got scale shape={tuple(scale.shape)} min={scale.min().item():.4f} max={scale.max().item():.4f}", flush=True)

    result = {
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "module": f"{attn.name}.qkv_proj",
        "weight_shapes": [list(m.weight.shape) for m in attn.qkv_proj],
        "num_grids": quant.smooth.proj.num_grids,
        "allow_low_rank": quant.smooth.proj.allow_low_rank,
        "calib_num_samples": quant.calib.num_samples,
        "timers": timer.summary(),
        "cuda_mem_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "cuda_mem_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print("=" * 60)
    print("SUMMARY totals_sec:")
    for k, v in sorted(timer.totals.items(), key=lambda kv: -kv[1]):
        print(f"  {k:45s} {v:8.2f}s  x{timer.counts[k]}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        with OUT_LOG.open("a") as f:
            f.write(traceback.format_exc())
        raise
