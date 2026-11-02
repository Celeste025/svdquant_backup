#!/usr/bin/env python3
"""E126: five MixAttention VBench prompts, true BF16/Sage3/split attention."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time
import types

import torch

import run_e121_h3_split_generation as e121


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E126"
DATA = Path("/data1/models/svdquant-wjq/research/20261023/E126_h3_vbench5_attention")
MANIFEST = REPORT / "manifest.json"
SETTINGS = {"height": 768, "width": 1344, "frames": 124, "steps": 50,
            "cfg_scale": 1.0, "flow_shift": 12.0, "audio_flow_shift": 3.0,
            "rand_device": "cpu", "tiled": True, "fps": 24}


def case_info(case: str) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    e121.base.require(manifest["format"] == "e126-h3-vbench5-attention-manifest-v1"
                      and manifest["settings"] == SETTINGS, "E126 manifest changed")
    rows = [row for row in manifest["prompts"] if row["id"] == case]
    e121.base.require(len(rows) == 1 and rows[0]["seed"] == 42, "Unknown case")
    return {"prompt_id": case, "seed": 42, "prompt": rows[0]["prompt"]}


def prepare() -> None:
    e121.check_backend()
    e121.base.REPORT = REPORT
    e121.base.DATA = DATA
    e121.base.PLAN = MANIFEST
    e121.base.Router = e121.Router
    e121.base.case_info = case_info
    importlib.import_module("diffsynth.core.attention.attention").ATTENTION_IMPLEMENTATION = "torch"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class PreflightComplete(Exception):
    pass


@torch.inference_mode()
def preflight(case: str) -> None:
    info = case_info(case)
    destination = REPORT / case / "preflight_split_mixed.json"
    e121.base.require(not destination.exists(), "E126 preflight already exists")
    pipe = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    linears = e121.base.checked_linears(pipe.dit)
    router = e121.Router(pipe.dit, "split_mixed")
    original_forward = pipe.dit.forward
    result = {"format": "e126-h3-vbench5-preflight-v1", "case": case,
              "prompt": info["prompt"], "seed": info["seed"],
              "settings": SETTINGS, "linears": linears}

    def observed_forward(_self, *args, **kwargs):
        contract = e121.base.attention_contract(kwargs)
        torch.cuda.synchronize()
        started = time.monotonic()
        with router.forward_context(**contract, diagnostics=False) as routed:
            output = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
        video, audio = e121.base.split_output(output)
        e121.base.require(torch.isfinite(video).all().item()
                          and torch.isfinite(audio).all().item(), "nonfinite preflight output")
        result.update(dit_seconds=time.monotonic() - started,
                      main_tokens=routed.summary["expected_cu"][1],
                      total_tokens=routed.summary["expected_cu"][-1],
                      attention_main_calls=routed.summary["main_helper_calls"],
                      attention_refiner_calls=routed.summary["refiner_helper_calls"],
                      video_output_energy=float(video.float().square().mean()),
                      audio_output_energy=float(audio.float().square().mean()),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      free_gib=torch.cuda.mem_get_info()[0] / 1024**3)
        raise PreflightComplete()

    pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
    try:
        try:
            pipe(prompt=info["prompt"], seed=info["seed"],
                 height=SETTINGS["height"], width=SETTINGS["width"],
                 num_frames=SETTINGS["frames"], num_inference_steps=SETTINGS["steps"],
                 cfg_scale=SETTINGS["cfg_scale"], flow_shift=SETTINGS["flow_shift"],
                 audio_flow_shift=SETTINGS["audio_flow_shift"],
                 rand_device="cpu", tiled=True)
        except PreflightComplete:
            pass
        else:
            raise RuntimeError("Preflight did not stop after one DiT call")
        e121.base.require(result["attention_main_calls"] == 50
                          and result["attention_refiner_calls"] == 2,
                          "Unexpected attention call count")
        e121.base.save_json(destination, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        router.close()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def compare_first(case: str) -> None:
    """Compare all arms on the exact same first DiT call for an outlier audit."""
    info = case_info(case)
    destination = REPORT / case / "paired_first_dit.json"
    e121.base.require(not destination.exists(), "Paired first-DiT report already exists")
    pipe = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    original_forward = pipe.dit.forward
    report = {"format": "e126-h3-paired-first-dit-v1", "case": case,
              "prompt": info["prompt"], "seed": 42, "settings": SETTINGS,
              "arms": {}}

    def observed_forward(_self, *args, **kwargs):
        contract = e121.base.attention_contract(kwargs)
        outputs = {}
        for arm in e121.ARMS:
            router = e121.Router(pipe.dit, arm)
            try:
                torch.cuda.synchronize()
                started = time.monotonic()
                with router.forward_context(**contract, diagnostics=False) as routed:
                    prediction = e121.base.split_output(original_forward(*args, **kwargs))
                    torch.cuda.synchronize()
                e121.base.require(all(torch.isfinite(x).all().item() for x in prediction),
                                  f"Nonfinite paired DiT output: {arm}")
                outputs[arm] = tuple(x.float().clone() for x in prediction)
                report["arms"][arm] = {
                    "dit_seconds": time.monotonic() - started,
                    "main_tokens": routed.summary["expected_cu"][1],
                    "attention_main_calls": routed.summary["main_helper_calls"],
                    "video_output_energy": float(prediction[0].float().square().mean()),
                    "audio_output_energy": float(prediction[1].float().square().mean()),
                    "output_sha256": {name: e121.base.tensor_sha256(x)
                                      for name, x in zip(("video", "audio"), prediction)},
                }
                print(f"PAIRED FIRST {case}/{arm}: {report['arms'][arm]['dit_seconds']:.2f}s",
                      flush=True)
            finally:
                router.close()
        report["paired_nmse"] = {
            f"{arm}_vs_bf16": {
                name: float((candidate - reference).square().sum(dtype=torch.float64) /
                            reference.square().sum(dtype=torch.float64))
                for name, candidate, reference in
                zip(("video", "audio"), outputs[arm], outputs["bf16"])}
            for arm in ("sage3_fixed", "split_mixed")}
        report["paired_nmse"]["split_mixed_vs_sage3_fixed"] = {
            name: float((candidate - reference).square().sum(dtype=torch.float64) /
                        reference.square().sum(dtype=torch.float64))
            for name, candidate, reference in zip(
                ("video", "audio"), outputs["split_mixed"], outputs["sage3_fixed"])}
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        raise PreflightComplete()

    pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
    try:
        try:
            pipe(prompt=info["prompt"], seed=42, height=SETTINGS["height"],
                 width=SETTINGS["width"], num_frames=SETTINGS["frames"],
                 num_inference_steps=SETTINGS["steps"], cfg_scale=1.0,
                 flow_shift=12.0, audio_flow_shift=3.0,
                 rand_device="cpu", tiled=True)
        except PreflightComplete:
            pass
        else:
            raise RuntimeError("Paired first-DiT comparison did not stop")
        e121.base.save_json(destination, report)
        print(json.dumps(report["paired_nmse"], indent=2), flush=True)
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def denoise(case: str, arm: str) -> None:
    info = case_info(case)
    report_path = REPORT / case / f"{arm}_denoise.json"
    stage = DATA / case / arm
    terminal = stage / "final_latents.pt"
    e121.base.require(MANIFEST.is_file() and not report_path.exists()
                      and not stage.exists(), "Plan missing or arm output already exists")
    stage.mkdir(parents=True, exist_ok=False)
    report = {"format": "e126-h3-attention-denoise-v1", "status": "running",
              "case": case, "arm": arm, "prompt": info["prompt"], "seed": 42,
              "settings": SETTINGS, "plan_sha256": e121.base.sha256(MANIFEST),
              "steps": [], "started_unix": time.time()}
    e121.base.save_json(report_path, report)
    pipe = e121.base.load_h3_pipeline(full=True, vram_limit_gib=18.0)
    pipe.load_models_to_device(["dit"])
    pipe.dit.eval()
    report["linears"] = e121.base.checked_linears(pipe.dit)
    router = e121.Router(pipe.dit, arm)
    report["attention_router"] = router.manifest
    e121.base.save_json(report_path, report)
    original_forward = pipe.dit.forward
    original_step = pipe.step
    original_loader = pipe.load_models_to_device
    final = {}
    initial = {}

    def observed_forward(_self, *args, **kwargs):
        index = len(report["steps"])
        e121.base.require(index < SETTINGS["steps"], "Too many DiT calls")
        e121.base.require(torch.cuda.mem_get_info()[0] > 5 * 1024**3,
                          "Less than 5 GiB free before DiT")
        contract = e121.base.attention_contract(kwargs)
        torch.cuda.synchronize()
        started = time.monotonic()
        with router.forward_context(**contract, diagnostics=False) as routed:
            output = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
        video, audio = e121.base.split_output(output)
        e121.base.require(torch.isfinite(video).all().item()
                          and torch.isfinite(audio).all().item(), "Nonfinite DiT output")
        entry = {"step": index, "dit_seconds": time.monotonic() - started,
                 "attention_main_calls": routed.summary["main_helper_calls"],
                 "attention_refiner_calls": routed.summary["refiner_helper_calls"],
                 "main_tokens": routed.summary["expected_cu"][1],
                 "video_output_energy": float(video.float().square().mean()),
                 "audio_output_energy": float(audio.float().square().mean())}
        if index == 0:
            entry["attention_main_rows"] = routed.summary["main_rows"]
        report["steps"].append(entry)
        e121.base.save_json(report_path, report)
        print(f"{case}/{arm} step {index + 1}/{SETTINGS['steps']}: DiT {entry['dit_seconds']:.2f} s", flush=True)
        return output

    def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
        name = ("video" if scheduler is pipe.scheduler else
                "audio" if scheduler is pipe.scheduler_audio else None)
        e121.base.require(name is not None and progress_id == len(report["steps"]) - 1,
                          "Scheduler order changed")
        if progress_id == 0:
            initial[name] = e121.base.tensor_sha256(latents)
        updated = original_step(scheduler, latents, progress_id,
                                noise_pred=noise_pred, **kwargs)
        e121.base.require(torch.isfinite(updated).all().item(), "Nonfinite latent")
        if progress_id == SETTINGS["steps"] - 1:
            final[name + "_latents"] = updated.detach().cpu()
        return updated

    def stop_before_decode(_self, names):
        if list(names) == ["video_vae"] and len(report["steps"]) == SETTINGS["steps"]:
            e121.base.require(set(final) == {"video_latents", "audio_latents"},
                              "Incomplete terminal latents")
            raise e121.base.DenoiseComplete()
        return original_loader(names)

    pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
    pipe.step = types.MethodType(observed_step, pipe)
    pipe.load_models_to_device = types.MethodType(stop_before_decode, pipe)
    try:
        try:
            pipe(prompt=info["prompt"], seed=42, height=SETTINGS["height"],
                 width=SETTINGS["width"], num_frames=SETTINGS["frames"],
                 num_inference_steps=SETTINGS["steps"], cfg_scale=1.0,
                 flow_shift=12.0, audio_flow_shift=3.0, rand_device="cpu", tiled=True)
        except e121.base.DenoiseComplete:
            pass
        else:
            raise RuntimeError("Did not stop at VAE boundary")
        e121.base.require(len(report["steps"]) == SETTINGS["steps"]
                          and set(initial) == {"video", "audio"},
                          "Incomplete 50-step rollout")
        with terminal.open("xb") as handle:
            torch.save(final, handle)
        report.update(status="denoise_complete", initial_latent_sha256=initial,
                      terminal_latents=str(terminal.resolve()),
                      terminal_sha256=e121.base.sha256(terminal),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      completed_unix=time.time())
        e121.base.save_json(report_path, report)
        print(f"DENOISE COMPLETE {case}/{arm}: {terminal}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error), failed_unix=time.time())
        e121.base.save_json(report_path, report)
        raise
    finally:
        router.close()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def decode(case: str, arm: str) -> None:
    from diffsynth.utils.data.audio_video import write_video_audio
    import av

    report_path = REPORT / case / f"{arm}_denoise.json"
    row = json.loads(report_path.read_text())
    e121.base.require(row["status"] == "denoise_complete" and row["case"] == case
                      and row["arm"] == arm, "Complete denoise report missing")
    terminal = Path(row["terminal_latents"])
    e121.base.require(e121.base.sha256(terminal) == row["terminal_sha256"],
                      "Terminal checkpoint changed")
    video_path = DATA / case / arm / f"{case}_{arm}_768x1344_124f_50steps.mp4"
    media_report = REPORT / case / f"{arm}_video.json"
    e121.base.require(not video_path.exists() and not media_report.exists(),
                      "Video output already exists")
    pipe = e121.base.MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            e121.base.ModelConfig(path=e121.base.local_h3_files("FL2VA/video_vae/source/model.safetensors"),
                                  **e121.base.disk_config()),
            e121.base.ModelConfig(path=e121.base.local_h3_files("FL2VA/audio_vae/model.safetensors"),
                                  **e121.base.disk_config()),
        ], processor_config=None, vram_limit=28.0)
    e121.base.require(pipe.dit is None and pipe.video_vae is not None
                      and pipe.audio_vae is not None, "Decoder loaded other models")
    latents = torch.load(terminal, map_location="cpu", weights_only=True)
    pipe.load_models_to_device(["video_vae"])
    decoded = pipe.video_vae.decode_video(latents["video_latents"].cuda(),
                                          dtype=torch.bfloat16, tiled=True,
                                          tile_size=256, tile_overlap=64)
    e121.base.require(torch.isfinite(decoded).all().item(), "Nonfinite video decode")
    video = pipe.vae_output_to_video(decoded, min_value=0, max_value=1)
    del decoded
    pipe.load_models_to_device(["audio_vae"])
    waveform = pipe.audio_vae.decode_audio(latents["audio_latents"].cuda(),
                                           dtype=torch.bfloat16)
    audio = pipe.output_audio_format_check(waveform)
    e121.base.require(len(video) == SETTINGS["frames"]
                      and all(frame.size == (SETTINGS["width"], SETTINGS["height"])
                              for frame in video)
                      and torch.isfinite(audio).all().item(), "Invalid decoded media")
    write_video_audio(video=video, audio=audio, output_path=str(video_path),
                      fps=SETTINGS["fps"], audio_sample_rate=pipe.audio_vae.sample_rate)
    with av.open(str(video_path)) as container:
        count = sum(1 for _ in container.decode(video=0))
        stream = container.streams.video[0]
        e121.base.require(count == 124 and stream.width == 1344
                          and stream.height == 768 and len(container.streams.audio) == 1,
                          "MP4 validation failed")
    result = {"format": "e126-h3-attention-video-v1", "case": case, "arm": arm,
              "video": str(video_path.resolve()), "sha256": e121.base.sha256(video_path),
              "bytes": video_path.stat().st_size, "frames": count,
              "width": 1344, "height": 768, "fps": 24, "audio_streams": 1,
              "denoise_report": str(report_path.resolve()),
              "denoise_report_sha256": e121.base.sha256(report_path)}
    e121.base.save_json(media_report, result)
    print(f"VIDEO COMPLETE {case}/{arm}: {video_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--arm", choices=e121.ARMS)
    parser.add_argument("--phase", choices=("preflight", "compare_first", "denoise", "decode"), required=True)
    args = parser.parse_args()
    info = case_info(args.case)
    e121.base.require(bool(info["prompt"]), "Empty prompt")
    e121.base.require((args.phase in ("preflight", "compare_first")) == (args.arm is None),
                      "Preflight/compare have no arm; denoise/decode require an arm")
    prepare()
    if args.phase == "preflight":
        preflight(args.case)
        return
    if args.phase == "compare_first":
        compare_first(args.case)
        return
    output = REPORT / args.case / f"{args.arm}_{'denoise' if args.phase == 'denoise' else 'video'}.json"
    if output.exists():
        row = json.loads(output.read_text())
        complete = ((args.phase == "denoise" and row.get("status") == "denoise_complete")
                    or (args.phase == "decode" and row.get("frames") == 124))
        if complete:
            print(f"SKIP complete {args.case}/{args.arm}/{args.phase}", flush=True)
            return
        raise RuntimeError(f"Existing incomplete report requires inspection: {output}")
    if args.phase == "denoise":
        denoise(args.case, args.arm)
    else:
        decode(args.case, args.arm)


if __name__ == "__main__":
    main()
