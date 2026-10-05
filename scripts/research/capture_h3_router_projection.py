#!/usr/bin/env python3
"""E035: one BF16 teacher DiT and three same-input native SVD QKV replays.

Only capture: no sparse attention, sampler, video, or projection-input cropping.
The original Comfy attention forward owns QKV layout, QK norm, and RoPE.
"""
from __future__ import annotations
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E035'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E035/capture')
PLAN = ROOT/'research_state/06_experiments/E035_sparse_router_projection_plan.md'
CASE = 'e010_p036_s14'
BLOCKS = (0, 24, 48)
N, T, H, D, WIDTH = 22539, 22592, 56, 128, 5376
PREFIX = dict(text=[0, 813], audio=[813, 1227], video=[1227, N], padding=[N, T])
VIDEO_GRID = [37, 18, 32]
SEGMENTS = [813, 414, VIDEO_GRID]


def construct_call(base, value):
    """Delegate input construction to the unchanged pipeline model_fn."""
    pipe = base.inherited.MiniMaxH3Pipeline(device='cpu', torch_dtype=base.torch.bfloat16)
    packed = base.make_packed(pipe, value['embedding'], value['text_token_tags'], value['state'])
    captured = []
    class ReachedDiT(Exception):
        pass
    def stub(*args, **kwargs):
        captured.append(dict(args=args, kwargs=kwargs))
        raise ReachedDiT()
    try:
        base.model_fn_minimax_h3(dit=stub,
            video_latents=value['state']['video']['latents_before'],
            audio_latents=value['state']['audio']['latents_before'], packed=packed,
            prompt_embeds=value['embedding'],
            timestep_video=value['state']['video']['timestep'].reshape(1),
            timestep_audio=value['state']['audio']['timestep'].reshape(1))
    except ReachedDiT:
        pass
    assert len(captured) == 1
    return packed, captured[0]


def binding(base, report, budget):
    import h3_native_nvfp4 as native
    import h3_nvfp4_zero_sf_compat as compat
    import h3_nvfp4_fastpack as fast
    import wan_native_nvfp4 as packet_module
    from minimax_h3_svdquant_common import H3_DIT_PATH
    torch = base.torch
    manifest = json.loads(base.MANIFEST.read_text())
    case = next(c for c in manifest['cases'] if c['id'] == CASE)
    reference_path = ROOT/'results/research/E014/E014_evaluate_bf16.json'
    previous = json.loads(reference_path.read_text())
    assert previous['status'] == 'complete' and previous['arm'] == 'bf16'
    historical = next(c for c in previous['cases'] if c['id'] == CASE)
    assert historical['status'] == 'complete'
    for ref in [case['prepared'], *case['state'].values(), historical['artifact']]:
        budget(); base.verify_file(ref)
    value = base.load_case(case)
    assert list(value['state']['video']['latents_before'].shape) == [1,24,37,36,64]
    packed, call = construct_call(base, value)
    assert packed['cu_seqlens'].tolist() == [0, N, T]
    for key, name in [('text_pos', 'text'), ('audio_pos', 'audio'), ('img_pos', 'video')]:
        assert torch.equal(packed[key], torch.arange(*PREFIX[name]))
    actual = base.tree_signature(call)
    assert actual == historical['actual_dit_inputs'], 'Original teacher input reconstruction changed'
    model_path = Path(H3_DIT_PATH)
    asset = previous['asset_binding']['assets'][str(model_path)]
    st = model_path.stat()
    assert st.st_size == asset['bytes'] and st.st_mtime_ns == asset['mtime_ns']
    sdpa = dict(flash=torch.backends.cuda.flash_sdp_enabled(), math=torch.backends.cuda.math_sdp_enabled(),
                mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(), cudnn=torch.backends.cuda.cudnn_sdp_enabled())
    assert sdpa == previous['inherited_binding']['sdpa_enabled']
    comfy = importlib.import_module('diffsynth.models.minimax_h3_dit_comfy')
    attention = importlib.import_module('diffsynth.core.attention.attention')
    assert attention.ATTENTION_IMPLEMENTATION == 'torch'
    sources = {str(Path(p).resolve()): base.file_record(p) for p in [
        __file__, PLAN, base.__file__, base.inherited.__file__,
        ROOT/'scripts/minimax_h3_svdquant_common.py',
        ROOT/'scripts/research/bench_h3_native_nvfp4.py',
        ROOT/'scripts/research/probe_h3_conditional_response.py',
        native.__file__, compat.__file__, fast.__file__, packet_module.__file__,
        comfy.__file__, importlib.import_module('diffsynth.models.minimax_h3_dit').__file__,
        importlib.import_module('diffsynth.pipelines.minimax_h3_audio_video').__file__, attention.__file__]}
    for path, row in sources.items():
        if path in previous['sources']:
            assert row['sha256'] == previous['sources'][path]['sha256'], path
    export_dir = Path(manifest['svd_export_dir'])
    export = json.loads((export_dir/'manifest.json').read_text())
    assert export['status'] == 'complete' and export['target_count'] == export['exact_roundtrip_count'] == 200
    assert base.sha256(export_dir/'manifest.json') == previous['inherited_binding']['export_manifest']['sha256']
    layers = []
    for block in BLOCKS:
        budget()
        name = f'blocks.{block}.attn.qkv_proj'
        row = next(r for r in export['layers'] if r['name'] == name)
        ref = base.file_record(export_dir/row['file'])
        assert ref['sha256'] == row['file_sha256'] and row['roundtrip']['exact']
        payload = torch.load(ref['file'], map_location='cpu', weights_only=True, mmap=True)
        assert payload['name'] == name and payload['format'] == native.FORMAT and payload['recipe'] == native.RECIPE
        assert tuple(payload['shape']) == (3*H*D, WIDTH)
        assert payload['tensors']['smooth'].numel() == WIDTH
        assert payload['tensors']['bias'] is None
        layers.append(dict(block=block, name=name, artifact=ref, shape=list(payload['shape'])))
    report.update(sources=sources, plan=base.file_record(PLAN), source_manifest=base.file_record(base.MANIFEST),
        source_report=base.file_record(reference_path), source_case=case,
        historical_reference=historical['artifact'], export_manifest=base.file_record(export_dir/'manifest.json'),
        projection_exports=layers, input_signature=base.tree_signature(value), actual_dit_inputs=actual,
        input_reconstruction_exact=True, sdpa_enabled=sdpa, torch=torch.__version__,
        attention_implementation=attention.ATTENTION_IMPLEMENTATION,
        model_asset=dict(file=str(model_path), **asset, verification='Inherited full SHA; current size/mtime only'),
        geometry=dict(valid_length=N, total_tokens=T, heads=H, head_dim=D, width=WIDTH,
                      cu_seqlens=[0,N,T], scale=D**-.5, prefix=PREFIX,
                      video_grid=VIDEO_GRID, segments=SEGMENTS, token_order='Original packed order; no VSA tile reorder'))
    return call, historical


def numeric(torch, actual, expected):
    """Small historical-output drift only; no historical output equality gate."""
    a, b = actual.reshape(-1), expected.reshape(-1)
    e2 = r2 = 0.; max_abs = 0.
    for lo in range(0, a.numel(), 1048576):
        x, y = a[lo:lo+1048576].double(), b[lo:lo+1048576].double()
        delta = x-y
        e2 += float(delta.square().sum()); r2 += float(y.square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
    return dict(error_energy=e2, reference_energy=r2, nmse=e2/r2 if r2 else None, max_abs=max_abs)


def run(args, base, report, call, historical, budget):
    torch = base.torch
    from minimax_h3_svdquant_common import tree_cpu, tree_device
    from h3_native_nvfp4 import NativeH3Linear
    from h3_nvfp4_zero_sf_compat import pack_activation_fast, collect_fastpack_checks
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.get_device_capability() == (12, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(min(1., 60*2**30/torch.cuda.get_device_properties(0).total_memory))
    report['device'] = dict(name=torch.cuda.get_device_name(), visible_devices='0', capability=[12,0])
    def guard():
        budget(); base.inherited.memory_guard()
    args.data_dir.mkdir(parents=True, exist_ok=False)
    raw_call_path = args.data_dir/'teacher_call.pt'
    torch.save(call, raw_call_path)
    report['teacher_call'] = base.file_record(raw_call_path)
    pipe = base.inherited.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit']); pipe.dit.eval()
    report['resident_conversion'] = base.make_h3_resident(pipe.dit)
    report['loading_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    guard()
    comfy = importlib.import_module('diffsynth.models.minimax_h3_dit_comfy')
    original_helper = comfy._sdpa_varlen_attention
    captured, handles, current = {}, [], {'block':None, 'arm':'bf16'}
    audit = base.RuntimeAudit()
    raw_outputs = []
    class CapturedProjection(Exception):
        pass
    def observe(q, k, v, cu_seqlens, softmax_scale):
        block = current['block']
        if block is not None:
            target = captured[block]; arm = current['arm']
            assert arm not in target and cu_seqlens.tolist() == [0,N,T] and float(softmax_scale) == D**-.5
            assert all(t.shape == (T,H,D) and t.dtype == torch.bfloat16 and bool(t.isfinite().all()) for t in (q,k,v))
            target[arm] = {name:t.detach().cpu().contiguous() for name,t in zip(('q','k','v'),(q,k,v))}
            if arm == 'native':
                raise CapturedProjection()
        return original_helper(q,k,v,cu_seqlens=cu_seqlens,softmax_scale=softmax_scale)
    def pre_for(block):
        def pre(module, inputs, kwargs):
            assert current['block'] is None and block not in captured
            assert module.forward.__func__.__name__ == '_comfy_attention_forward'
            x = inputs[0]
            assert x.shape == (T,WIDTH) and x.dtype == torch.bfloat16 and bool(x.isfinite().all())
            captured[block] = dict(x=x.detach().cpu().contiguous(), kwargs=tree_cpu(kwargs))
            current['block'] = block
        return pre
    def end_attention(module, inputs, output):
        current['block'] = None
    def capture_raw(module, inputs, output):
        raw_outputs.append(tree_cpu(output))
    try:
        comfy._sdpa_varlen_attention = observe
        for block in BLOCKS:
            attn = pipe.dit.blocks[block].attn
            handles += [attn.register_forward_pre_hook(pre_for(block),with_kwargs=True),
                        attn.register_forward_hook(end_attention)]
        handles.append(pipe.dit.register_forward_hook(capture_raw))
        gpu_call = tree_device(call,'cuda')
        audit.phase = 'resident_bf16'; report['attempted_dit_calls'] += 1
        with audit.installed():
            outputs = pipe.dit(*gpu_call['args'], **gpu_call['kwargs'])
            torch.cuda.synchronize()
        report['complete_dit_calls'] = 1
        report['teacher_runtime'] = dict(audit.row())
        assert {k:audit.row()[k] for k in ('sdpa_calls','scaled_mm_calls','disk_loads')} == dict(sdpa_calls=102,scaled_mm_calls=0,disk_loads=0)
        assert len(raw_outputs) == 1 and set(captured) == set(BLOCKS)
        old = torch.load(historical['artifact']['file'],map_location='cpu',weights_only=True,mmap=True)
        raw = dict(zip(('video','audio'),raw_outputs[0]))
        assert all(bool(t.isfinite().all()) for t in raw.values())
        report['historical_output_drift'] = {m:numeric(torch,raw[m],old['raw_outputs'][m]) for m in raw}
        report['teacher_raw_outputs'] = base.tree_signature(raw)
        teacher_output_path = args.data_dir/'teacher_outputs.pt'
        torch.save(raw,teacher_output_path); report['teacher_output_artifact'] = base.file_record(teacher_output_path)
        del outputs, gpu_call, old, raw, raw_outputs
    finally:
        for h in handles: h.remove()
        comfy._sdpa_varlen_attention = original_helper
    current['block'] = None
    guard(); base.save(report,args.output)
    for block in BLOCKS:
        guard(); attn = pipe.dit.blocks[block].attn; target = captured.pop(block)
        # Put it back only while the shared original-forward observer consumes it.
        captured[block] = target
        source = next(r for r in report['projection_exports'] if r['block'] == block)
        payload = torch.load(source['artifact']['file'],map_location='cpu',weights_only=True,mmap=True)
        replacement = NativeH3Linear.from_export(payload,device='cuda',activation_packer=pack_activation_fast,chunk_rows=1024)
        original = attn.qkv_proj
        x = target['x'].cuda(); kwargs = tree_device(target['kwargs'],'cuda')
        seen = []
        def projection_input(module, inputs):
            assert inputs[0].shape == (T,WIDTH)
            seen.append(base.tensor_record(inputs[0]))
        hook = replacement.register_forward_pre_hook(projection_input)
        current.update(block=block,arm='native'); audit.phase = 'native'
        caught = False
        try:
            attn.qkv_proj = replacement; comfy._sdpa_varlen_attention = observe
            report['attempted_native_projections'] += 1
            with audit.installed(), collect_fastpack_checks() as checks:
                try:
                    attn(x,**kwargs)
                except CapturedProjection:
                    caught = True
            torch.cuda.synchronize()
            assert caught and len(seen) == 1 and seen[0] == base.tensor_record(target['x'])
            assert checks.summary['checked_calls'] == 1
            report['complete_native_projections'] += 1
        finally:
            attn.qkv_proj = original; comfy._sdpa_varlen_attention = original_helper
            hook.remove(); current['block'] = None
        assert audit.row()['sdpa_calls'] == audit.row()['disk_loads'] == 0
        assert audit.row()['scaled_mm_calls'] == report['complete_native_projections']
        target.update(block=block, valid_length=N,total_tokens=T,scale=D**-.5,prefix=PREFIX,
                      video_grid=VIDEO_GRID,segments=SEGMENTS)
        path = args.data_dir/f'block_{block:02d}.pt'; torch.save(target,path)
        row = dict(block=block,status='complete',valid_length=N,total_tokens=T,heads=H,head_dim=D,
            scale=D**-.5,cu_seqlens=[0,N,T],capture=base.file_record(path),tensors=base.tree_signature(target),
            projection_input_unchanged=True,actual_projection_input=seen[0],fastpack_checks=checks.summary,
            helper_sentinel_stopped_before_attention_and_out_proj=caught)
        report['cases'].append(row); report['native_runtime'] = dict(audit.row())
        base.save(report,args.output)
        print(json.dumps(dict(block=block,status='captured',native_projections=report['complete_native_projections'])),flush=True)
        del replacement,payload,original,attn,x,kwargs,target,checks
        captured.pop(block); gc.collect(); torch.cuda.empty_cache()
    assert report['complete_dit_calls'] == 1 and report['complete_native_projections'] == 3
    report['actual_scaled_mm_calls'] = audit.row()['scaled_mm_calls']
    report['actual_sdpa_calls'] = report['teacher_runtime']['sdpa_calls'] + audit.row()['sdpa_calls']
    assert report['actual_scaled_mm_calls'] == 3 and report['actual_sdpa_calls'] == 102
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    del pipe; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    report['model_released_allocated_bytes'] = torch.cuda.memory_allocated()
    guard()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float)
    p.add_argument('--output',type=Path)
    p.add_argument('--data-dir',type=Path,default=DATA)
    args=p.parse_args(); args.output=args.output or RD/f'capture_{args.phase}.json'
    assert not args.output.exists(), 'Preserve previous attempts'
    started=time.time()
    report=dict(experiment='E035',phase=args.phase,status='running',cases=[],attempted_dit_calls=0,
                complete_dit_calls=0,attempted_native_projections=0,complete_native_projections=0,
                scope='Same actual teacher input; SVD native projection only; no attention replacement, sparse consumer, rollout or quality claim')
    def budget():
        if args.phase == 'run':
            assert args.deadline_unix and time.time()<args.deadline_unix, 'Original shared deadline expired'
    try:
        if args.phase=='check': assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        else: assert args.deadline_unix and 0<args.deadline_unix-started<=900
        # The frozen helper sets explicit torch attention/offline environment before DiffSynth imports.
        import probe_h3_plain_baseline as base
        torch=base.torch; torch.set_num_threads(6)
        with torch.inference_mode():
            call,historical=binding(base,report,budget)
            if args.phase=='check':
                assert not torch.cuda.is_initialized()
            else:
                checked=json.loads((RD/'capture_check.json').read_text())
                assert checked['status']=='complete' and checked['cuda_initialized'] is False
                for key in ('sources','source_manifest','source_report','export_manifest','projection_exports','actual_dit_inputs','model_asset','sdpa_enabled'):
                    assert checked[key]==report[key],key
                report['cpu_check']=base.file_record(RD/'capture_check.json')
                run(args,base,report,call,historical,budget)
        report.update(status='complete',cuda_initialized=torch.cuda.is_initialized())
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc())
        raise
    finally:
        report['seconds']=time.time()-started
        # Keep even import/setup failures; this helper has no heavy dependencies.
        from probe_h3_query_mean_k4 import save
        save(args.output,report)
        print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__': main()
