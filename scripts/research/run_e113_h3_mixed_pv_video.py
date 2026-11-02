#!/usr/bin/env python3
"""E113: matched BF16-linear H3 generation with Sage3 or mixed P/V attention."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research")]
SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
sys.path.insert(0, str(SAGE))

from sageattn3 import sageattn3_blackwell
from diffsynth.core.vram.layers import AutoWrappedLinear
from h3_native_fp4_attention import H3AttentionRouter
from h3_native_nvfp4 import TARGET_NAMES
from minimax_h3_svdquant_common import (
    MiniMaxH3Pipeline, ModelConfig, disk_config, load_h3_pipeline,
    local_h3_files, tree_device,
)
from run_e105_h3_output_interventions import case_info, split_output
from e113_native_mixed_attention import native_mixed_sage3_attention

ARMS = ("sage3", "native_fp4", "native_mixed")
CASES = ("race", "clap0")
DATA = Path("/data1/models/svdquant-wjq/research/20261020/E113_h3_mixed_pv_generation")
REPORT = ROOT / "results/research/E113"
PLAN = ROOT / "research_state/06_experiments/decision_value_plan_E113_h3_mixed_pv_generation.md"


class DenoiseComplete(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(x: torch.Tensor) -> str:
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temp.replace(path)


class Router(H3AttentionRouter):
    def __init__(self, dit, arm):
        require(arm in ARMS, "invalid attention arm")
        self.arm = arm
        import sageattn3.api as sage_api
        backend = sage_api if arm == "sage3" else sys.modules[native_mixed_sage3_attention.__module__]
        super().__init__(dit, "block_mean", _official=backend)
        self.manifest["e113_arm"] = arm
        self.manifest["mixed_threshold"] = (
            0.10 if arm == "native_mixed" else 0.50 if arm == "native_fp4" else None)
        self.manifest["main_attention"] = (
            "Sage3 SM110 native" if arm == "sage3" else
            "official Sage3 preprocess + native Triton NVFP4 QK/PV and FP8 PV tile core")

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        log = self._active_log.get()
        block = self._active_main.get()
        require(log is not None and softmax_scale == 128 ** -0.5,
                "attention called outside route or scale changed")
        expected = log.expected_refiner_cu if block is None else log.expected_cu
        require(q.shape == k.shape == v.shape == (expected[-1], 56, 128),
                "QKV shape changed")
        require(all(x.dtype == torch.bfloat16 and x.is_cuda for x in (q, k, v)),
                "QKV dtype/device changed")
        log.remember_cu(cu_seqlens, expected, "refiner" if block is None else "main")
        if block is None:
            log.refiner_helpers += 1
            log.original_bf16_segments += 1
            return self.original_helper(q, k, v, cu_seqlens, softmax_scale)
        n, total = expected[1:]
        Q, K, V = (x[:n].transpose(0, 1).unsqueeze(0).contiguous() for x in (q, k, v))
        if self.arm == "sage3":
            result = sageattn3_blackwell(Q, K.clone(), V, is_causal=False,
                                         per_block_mean=True)
        else:
            result = native_mixed_sage3_attention(
                Q, K, V, threshold=0.10 if self.arm == "native_mixed" else 0.50)
        require(result.shape == Q.shape and result.dtype == torch.bfloat16,
                "attention output shape/dtype changed")
        out = torch.empty_like(q)
        out[:n] = result[0].transpose(0, 1)
        out[n:] = self.original_helper(q[n:], k[n:], v[n:],
                                       log.padding_cpu_cu, softmax_scale)
        log.main_rows.append({"block": block, "valid_length": n,
                              "padding_length": total - n, "arm": self.arm})
        log.fp4_calls += 1
        log.original_bf16_segments += 1
        return out


def checked_linears(dit) -> dict:
    dtypes = {}
    offloaded = 0
    for name in TARGET_NAMES:
        module = dit.get_submodule(name)
        require(isinstance(module, torch.nn.Linear),
                f"non-linear target: {name}, type={type(module).__name__}")
        if isinstance(module, AutoWrappedLinear):
            require(module.computation_dtype == torch.bfloat16 and not module.enable_fp8
                    and len(module.lora_A_weights) == 0,
                    f"non-BF16 compute branch: {name}")
            if module.disk_offload:
                require(module.weight.device.type == "meta" and module.disk_map is not None,
                        f"offload state changed: {name}")
                offloaded += 1
        else:
            require(module.weight.dtype == torch.bfloat16,
                    f"non-BF16 linear weight: {name}")
        dtypes[name] = "torch.bfloat16"
    require(len(dtypes) == 200, "target linear count changed")
    return {"count": len(dtypes), "computation_dtype": "torch.bfloat16",
            "disk_offloaded_count": offloaded}


def attention_contract(kwargs) -> dict:
    return {
        "expected_cu": kwargs["packed_seq_params"]["cu_seqlens_q"].detach().cpu().tolist(),
        "expected_refiner_cu": kwargs["refiner_packed_seq_params"]["cu_seqlens_q"].detach().cpu().tolist(),
    }


@torch.inference_mode()
def preflight(case: str):
    info = case_info(case)
    capture = (Path("/data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases")
               / case / "bf16_teacher_calls.pt")
    require(capture.is_file(), "missing E101 BF16 DiT call cache")
    report_path = REPORT / case / "attention_preflight_native.json"
    require(not report_path.exists(), "preflight report already exists")
    calls = torch.load(capture, map_location="cpu", weights_only=False)
    require(len(calls) == 20, "incomplete cached trajectory")
    pipe = load_h3_pipeline(full=False, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    linears = checked_linears(pipe.dit)
    args = tree_device(calls[0]["args"], "cuda")
    kwargs = tree_device(calls[0]["kwargs"], "cuda")
    contract = attention_contract(kwargs)
    report = {"format": "e113-attention-preflight-v1", "case": case,
              "prompt_id": info["prompt_id"], "seed": info["seed"],
              "cached_calls": str(capture), "contract": contract,
              "linears": linears, "arms": {}}
    outputs = {}
    for arm in ARMS:
        router = Router(pipe.dit, arm)
        try:
            torch.cuda.synchronize()
            started = time.monotonic()
            with router.forward_context(**contract, diagnostics=False) as routed:
                prediction = split_output(pipe.dit(*args, **kwargs))
                torch.cuda.synchronize()
            require(all(torch.isfinite(x).all().item() for x in prediction),
                    "nonfinite DiT output")
            outputs[arm] = tuple(x.float() for x in prediction)
            report["arms"][arm] = {
                "seconds": time.monotonic() - started,
                "attention": {k: v for k, v in routed.summary.items() if k != "main_rows"},
                "output_sha256": {name: tensor_sha256(x) for name, x in
                                  zip(("video", "audio"), prediction, strict=True)},
            }
            print(f"PREFLIGHT {case}/{arm}: {report['arms'][arm]['seconds']:.2f} s", flush=True)
        finally:
            router.close()
    for left, right in (("native_fp4", "sage3"), ("native_mixed", "native_fp4"),
                        ("native_mixed", "sage3")):
        if left in outputs and right in outputs:
            report[f"{left}_vs_{right}_output_nmse"] = {
                name: float((l - r).square().sum() / r.square().sum())
                for name, l, r in zip(("video", "audio"), outputs[left], outputs[right], strict=True)
            }
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["status"] = "complete"
    save_json(report_path, report)
    print(json.dumps({"status": "complete", "preflight": str(report_path),
                      "mixed_vs_sage3_output_nmse": report["native_mixed_vs_sage3_output_nmse"],
                      "peak_allocated_gib": report["peak_allocated_gib"]}, indent=2), flush=True)


@torch.inference_mode()
def denoise(case: str, arm: str):
    info = case_info(case)
    report_path = REPORT / case / f"{arm}_denoise.json"
    stage = DATA / case / arm
    terminal = stage / "final_latents.pt"
    require(PLAN.is_file() and not report_path.exists() and not stage.exists(),
            "plan missing or arm output already exists")
    stage.mkdir(parents=True, exist_ok=False)
    report = {"format": "e113-h3-attention-denoise-v1", "status": "running",
              "case": case, "arm": arm, "prompt_id": info["prompt_id"],
              "seed": info["seed"], "prompt": info["prompt"],
              "settings": {"height": 576, "width": 1024, "frames": 124,
                           "steps": 20, "cfg_scale": 1.0, "flow_shift": 12.0,
                           "audio_flow_shift": 3.0, "rand_device": "cpu", "tiled": True},
              "plan_sha256": sha256(PLAN), "steps": [], "started_unix": time.time()}
    save_json(report_path, report)
    pipe = load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    report["linears"] = checked_linears(pipe.dit)
    router = Router(pipe.dit, arm)
    report["attention_router"] = router.manifest
    save_json(report_path, report)
    original_forward = pipe.dit.forward
    original_step = pipe.step
    original_loader = pipe.load_models_to_device
    final = {}
    initial = {}

    def observed_forward(_self, *args, **kwargs):
        index = len(report["steps"])
        require(index < 20, "too many DiT calls")
        require(torch.cuda.mem_get_info()[0] > 5 * 1024**3,
                "less than 5 GiB GPU memory free before DiT")
        contract = attention_contract(kwargs)
        torch.cuda.synchronize()
        started = time.monotonic()
        with router.forward_context(**contract, diagnostics=False) as routed:
            output = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
        video, audio = split_output(output)
        require(torch.isfinite(video).all().item() and torch.isfinite(audio).all().item(),
                "nonfinite DiT output")
        entry = {"step": index, "dit_seconds": time.monotonic() - started,
                 "attention_main_calls": routed.summary["main_helper_calls"],
                 "attention_refiner_calls": routed.summary["refiner_helper_calls"],
                 "main_tokens": routed.summary["expected_cu"][1],
                 "video_output_energy": float(video.float().square().mean()),
                 "audio_output_energy": float(audio.float().square().mean())}
        if index == 0:
            entry["output_sha256"] = {"video": tensor_sha256(video),
                                      "audio": tensor_sha256(audio)}
            entry["attention_main_rows"] = routed.summary["main_rows"]
        report["steps"].append(entry)
        save_json(report_path, report)
        print(f"{case}/{arm} step {index+1}/20: DiT {entry['dit_seconds']:.2f} s", flush=True)
        return output

    def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
        name = "video" if scheduler is pipe.scheduler else "audio" if scheduler is pipe.scheduler_audio else None
        require(name is not None and progress_id == len(report["steps"]) - 1,
                "scheduler order changed")
        if progress_id == 0:
            initial[name] = tensor_sha256(latents)
        updated = original_step(scheduler, latents, progress_id,
                                noise_pred=noise_pred, **kwargs)
        require(torch.isfinite(updated).all().item(), "nonfinite recurrent latent")
        if progress_id == 19:
            final[name + "_latents"] = updated.detach().cpu()
        return updated

    def stop_before_decode(_self, names):
        if list(names) == ["video_vae"] and len(report["steps"]) == 20:
            require(set(final) == {"video_latents", "audio_latents"},
                    "terminal latents incomplete")
            raise DenoiseComplete()
        return original_loader(names)

    pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
    pipe.step = types.MethodType(observed_step, pipe)
    pipe.load_models_to_device = types.MethodType(stop_before_decode, pipe)
    try:
        try:
            pipe(prompt=info["prompt"], seed=info["seed"], height=576, width=1024,
                 num_frames=124, num_inference_steps=20, cfg_scale=1.0,
                 flow_shift=12.0, audio_flow_shift=3.0, rand_device="cpu", tiled=True)
        except DenoiseComplete:
            pass
        else:
            raise RuntimeError("did not stop at VAE boundary")
        require(len(report["steps"]) == 20 and set(initial) == {"video", "audio"},
                "incomplete 20-step rollout")
        with terminal.open("xb") as handle:
            torch.save(final, handle)
        report.update(status="denoise_complete", initial_latent_sha256=initial,
                      terminal_latents=str(terminal.resolve()),
                      terminal_sha256=sha256(terminal),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      completed_unix=time.time())
        save_json(report_path, report)
        print(f"DENOISE COMPLETE {case}/{arm}: {terminal}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error), failed_unix=time.time())
        save_json(report_path, report)
        raise
    finally:
        router.close()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def decode(case: str, arm: str):
    from diffsynth.utils.data.audio_video import write_video_audio
    import av

    report_path = REPORT / case / f"{arm}_denoise.json"
    row = json.loads(report_path.read_text())
    require(row["status"] == "denoise_complete" and row["case"] == case and row["arm"] == arm,
            "complete denoise report missing")
    terminal = Path(row["terminal_latents"])
    require(sha256(terminal) == row["terminal_sha256"], "terminal checkpoint changed")
    video_path = DATA / case / arm / f"{case}_{arm}_576x1024_124f_20steps.mp4"
    report_out = REPORT / case / f"{arm}_video.json"
    require(not video_path.exists() and not report_out.exists(), "video output exists")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(path=local_h3_files("FL2VA/video_vae/source/model.safetensors"), **disk_config()),
            ModelConfig(path=local_h3_files("FL2VA/audio_vae/model.safetensors"), **disk_config()),
        ], processor_config=None, vram_limit=28.0)
    require(pipe.dit is None and pipe.video_vae is not None and pipe.audio_vae is not None,
            "decode must load only VAEs")
    latents = torch.load(terminal, map_location="cpu", weights_only=True)
    pipe.load_models_to_device(["video_vae"])
    decoded = pipe.video_vae.decode_video(latents["video_latents"].cuda(),
                                          dtype=torch.bfloat16, tiled=True,
                                          tile_size=256, tile_overlap=64)
    require(torch.isfinite(decoded).all().item(), "nonfinite decoded video")
    video = pipe.vae_output_to_video(decoded, min_value=0, max_value=1)
    del decoded
    pipe.load_models_to_device(["audio_vae"])
    waveform = pipe.audio_vae.decode_audio(latents["audio_latents"].cuda(), dtype=torch.bfloat16)
    audio = pipe.output_audio_format_check(waveform)
    require(len(video) == 124 and all(x.size == (1024, 576) for x in video)
            and torch.isfinite(audio).all().item(), "decoded output invalid")
    write_video_audio(video=video, audio=audio, output_path=str(video_path),
                      fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    with av.open(str(video_path)) as container:
        frames = sum(1 for _ in container.decode(video=0))
        vs = container.streams.video[0]
        require(frames == 124 and vs.width == 1024 and vs.height == 576
                and len(container.streams.audio) == 1, "media validation failed")
    media = {"format": "e113-h3-attention-video-v1", "case": case, "arm": arm,
             "video": str(video_path.resolve()), "sha256": sha256(video_path),
             "bytes": video_path.stat().st_size, "frames": frames,
             "width": 1024, "height": 576, "fps": 24,
             "audio_streams": 1, "denoise_report": str(report_path.resolve()),
             "denoise_report_sha256": sha256(report_path)}
    save_json(report_out, media)
    print(f"VIDEO COMPLETE {case}/{arm}: {video_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    require((args.phase == "preflight") == (args.arm is None),
            "--arm required only for denoise/decode")
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.phase == "preflight":
        preflight(args.case)
    elif args.phase == "denoise":
        denoise(args.case, args.arm)
    else:
        decode(args.case, args.arm)


if __name__ == "__main__":
    main()
