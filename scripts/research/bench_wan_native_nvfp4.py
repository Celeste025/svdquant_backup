#!/usr/bin/env python3
"""E007: one fixed full-shape rCM-Wan native NVFP4 correctness baseline.

The initial software packer is diagnostic only: no speed or quality claim.
Three identical-input arms, exact reconstructed weights and real QDQ bypass,
same-code representative main-branch gates, then all 300 native projections.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq')
RCM = DATA/'models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer'
BASE = DATA/'models/Wan2.1-T2V-1.3B-Diffusers'
CHECKPOINT = DATA/'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16'
CACHE = DATA/'datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches/0001-00000-0.pt'
REPRESENTATIVES = ('blocks.0.attn1.to_q', 'blocks.0.ffn.net.2',
                   'blocks.0.attn2.to_k', 'blocks.0.attn2.to_v')


def tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: tree(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree(v, device) for v in value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metric(got, ref):
    if got.shape != ref.shape:
        raise ValueError('metric tensors have different shapes')
    err2 = ref2 = max_abs = 0.
    left, right = got.reshape(-1), ref.reshape(-1)
    for begin in range(0, right.numel(), 1024**2):
        g, r = left[begin:begin+1024**2].float(), right[begin:begin+1024**2].float()
        if not torch.isfinite(g).all() or not torch.isfinite(r).all():
            raise RuntimeError('nonfinite tensor in correctness metric')
        d = g-r
        err2 += d.double().square().sum().item()
        ref2 += r.double().square().sum().item()
        max_abs = max(max_abs, d.abs().max().item())
    return {'err2': err2, 'ref2': ref2, 'nmse': err2/max(ref2, 1e-30),
            'max_abs': max_abs, 'numel': got.numel()}


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(output)


def inspect_contract():
    config = json.loads((RCM/'config.json').read_text())
    expected = {'num_layers': 30, 'num_attention_heads': 12, 'attention_head_dim': 128,
                'ffn_dim': 8960, 'in_channels': 16, 'out_channels': 16, 'patch_size': [1, 2, 2]}
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(f'unexpected rCM architecture {key}: {config[key]}')
    payload = torch.load(CACHE, map_location='cpu', weights_only=False)
    x, emb = payload['input_args'][0], payload['input_kwargs']['encoder_hidden_states']
    if list(x.shape) != [1, 16, 20, 60, 104] or list(emb.shape) != [1, 512, 4096]:
        raise RuntimeError('fixed original full-shape input contract changed')
    if x.dtype != torch.bfloat16 or emb.dtype != torch.bfloat16:
        raise RuntimeError('cache input and text embedding must be BF16')
    if payload['filename'] != '0001' or payload['step'] != 0 or payload['guidance'] != 0:
        raise RuntimeError('fixed calibration sample identity changed')
    return {'config': config, 'latent_shape': list(x.shape), 'text_shape': list(emb.shape),
            'patch_tokens': 20*30*52, 'input_sha256': tensor_sha(x),
            'text_embedding_sha256': tensor_sha(emb), 'timestep': payload['input_kwargs']['timestep'].tolist()}, payload


@torch.inference_mode()
def execute(args, report):
    sys.path.insert(0, str(ROOT/'scripts'))
    sys.path.insert(0, str(ROOT/'third_party/deepcompressor'))
    import diffusers
    from diffusers import WanPipeline, WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from infer_rcm_wan_4step import load_quantized_transformer
    from deepcompressor.utils.hooks.processor import ProcessHook
    from deepcompressor.quantizer.processor import Quantizer
    import wan_native_nvfp4 as native_lib
    if diffusers.__version__ != '0.33.1':
        raise RuntimeError('E007 requires the fixed historical Diffusers 0.33.1 environment')
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    def checkpoint():
        report['elapsed_seconds_not_benchmark'] = time.time()-started
        report['peak_gpu_gib_not_benchmark'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        if report['elapsed_seconds_not_benchmark'] > 3600:
            raise TimeoutError('E007 one-hour correctness budget exhausted')
        if report['peak_gpu_gib_not_benchmark'] > 60:
            raise MemoryError('E007 allocated-memory stop line exceeded; no token truncation allowed')
    contract, payload = inspect_contract()
    report['input_contract'] = contract
    paths = [Path(__file__), Path(native_lib.__file__), ROOT/'scripts/infer_rcm_wan_4step.py',
             ROOT/'scripts/collect_rcm_wan_calib.py',
             ROOT/'research_state/06_experiments/E007_wan_native_baseline_plan.md',
             Path(importlib.import_module('diffusers.models.transformers.transformer_wan').__file__),
             RCM/'config.json', RCM/'diffusion_pytorch_model.safetensors', CACHE]
    paths += [CHECKPOINT/f'{name}.pt' for name in ('model', 'scale', 'wgts', 'smooth', 'branch')]
    print('Hashing fixed rCM input, model, checkpoint, and source provenance', flush=True)
    report['files'] = {str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size,
                              'resolved_path': str(p.resolve())} for p in paths}
    report.update(torch=torch.__version__, torch_file=torch.__file__, cuda=torch.version.cuda,
                  diffusers=diffusers.__version__, device=torch.cuda.get_device_name(),
                  sdpa_backend='torch SDPBackend.FLASH_ATTENTION forced identically for every arm')
    call = {'input_args': tree(payload['input_args'], 'cuda'),
            'input_kwargs': tree(payload['input_kwargs'], 'cuda')}
    cached_output = payload['outputs'][0]
    original_sdpa, original_mm = F.scaled_dot_product_attention, F.scaled_mm
    audit = {'phase': '', 'sdpa': {}, 'native_mm': {}, 'sdpa_dtypes': set(), 'sdpa_shapes': set()}
    def sdpa(q, k, v, *pos, **kw):
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            raise RuntimeError('non-BF16 actual SDPA QKV in E007')
        audit['sdpa'][audit['phase']] = audit['sdpa'].get(audit['phase'], 0)+1
        audit['sdpa_dtypes'].add(tuple(str(t.dtype) for t in (q, k, v)))
        audit['sdpa_shapes'].add(tuple(tuple(t.shape) for t in (q, k, v)))
        return original_sdpa(q, k, v, *pos, **kw)
    def scaled_mm(*pos, **kw):
        audit['native_mm'][audit['phase']] = audit['native_mm'].get(audit['phase'], 0)+1
        return original_mm(*pos, **kw)
    F.scaled_dot_product_attention, F.scaled_mm = sdpa, scaled_mm
    report['stages'] = {}

    def run(model, phase, references=None, store=False, exact=False):
        audit['phase'] = phase
        stage = {'status': 'running', 'blocks': []}
        report['stages'][phase] = stage
        outputs, handles = {}, []
        for bi, block in enumerate(model.blocks):
            def post(_module, _inp, out, bi=bi):
                if not torch.is_tensor(out) or list(out.shape) != [1, 31200, 1536]:
                    raise RuntimeError(f'full 31200-token block shape changed at block {bi}')
                if not torch.isfinite(out).all():
                    raise RuntimeError(f'nonfinite {phase} block {bi}')
                cpu = out.detach().cpu()
                row = {'block': bi, 'shape': list(cpu.shape), 'sha256': tensor_sha(cpu)}
                if references:
                    row['comparisons'] = {name: metric(cpu, refs[bi]) for name, refs in references.items()}
                    if exact and any(m['err2'] != 0 for m in row['comparisons'].values()):
                        raise RuntimeError(f'{phase} zero-change block replay failed at {bi}')
                if store:
                    outputs[bi] = cpu
                stage['blocks'].append(row)
                checkpoint()
            handles.append(block.register_forward_hook(post))
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                result = model(*call['input_args'], **call['input_kwargs'])[0].cpu()
        finally:
            for handle in handles:
                handle.remove()
        if len(stage['blocks']) != 30 or audit['sdpa'].get(phase, 0) != 60:
            raise RuntimeError(f'{phase} did not execute exactly 30 blocks and 60 SDPA calls')
        if not torch.isfinite(result).all():
            raise RuntimeError(f'nonfinite {phase} denoiser endpoint')
        stage.update(status='complete', endpoint_shape=list(result.shape),
                     endpoint_sha256=tensor_sha(result), actual_sdpa_calls=audit['sdpa'][phase],
                     actual_native_mm_calls=audit['native_mm'].get(phase, 0))
        checkpoint()
        return result, outputs

    try:
        print('Loading actual rCM BF16 transformer only', flush=True)
        teacher = WanTransformer3DModel.from_pretrained(RCM, torch_dtype=torch.bfloat16).cuda().eval()
        if len(teacher.blocks) != 30 or any(type(b.attn1.processor) is not WanAttnProcessor2_0 for b in teacher.blocks):
            raise RuntimeError('unexpected rCM model/attention processor')
        report['teacher_parameter_count'] = sum(p.numel() for p in teacher.parameters())
        teacher_out, teacher_blocks = run(teacher, 'bf16', store=True)
        report['bf16_vs_cached_reference'] = metric(teacher_out, cached_output)
        if report['bf16_vs_cached_reference']['nmse'] > 1e-6:
            raise RuntimeError('BF16/cache disagreement exceeds fixed provenance gate')
        repeat, _ = run(teacher, 'bf16_zero_replay', {'bf16': teacher_blocks}, exact=True)
        report['bf16_zero_replay_endpoint'] = metric(repeat, teacher_out)
        if report['bf16_zero_replay_endpoint']['err2'] != 0:
            raise RuntimeError('BF16 endpoint not exactly repeatable')
        del repeat
        # The same object is quantized after the BF16 references are captured.
        pipe = WanPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16,
            transformer=teacher, text_encoder=None, tokenizer=None, vae=None)
        load_quantized_transformer(pipe, CHECKPOINT, BASE)
        quantized = pipe.transformer.eval()
        targets = {name: module for name, module in quantized.named_modules()
                   if name.startswith('blocks.') and isinstance(module, torch.nn.Linear)}
        hooks = {}
        for name, module in targets.items():
            matches = [h for h in module._forward_pre_hooks.values()
                       if isinstance(h, ProcessHook) and isinstance(h.processor, Quantizer)]
            if len(matches) != 1:
                raise RuntimeError(f'expected one activation quantizer at {name}, got {len(matches)}')
            hooks[name] = matches[0]
        if len(targets) != 300 or len(hooks) != 300:
            raise RuntimeError('expected 300 historical block linear/activation targets')
        report['target_names'] = list(targets)
        representative_inputs = {}
        full_pack_inputs = {}
        saved_funcs = {}
        for name in REPRESENTATIVES:
            h = hooks[name]
            saved_funcs[name] = h.func
            def capture(x, name=name, h=h):
                flat = x.reshape(-1, x.shape[-1])
                if flat.shape[0] < 512:
                    raise RuntimeError('representative layer has fewer than fixed 512 rows')
                representative_inputs[name] = flat[:512].detach().cpu().clone()
                if name != 'blocks.0.attn2.to_v':
                    full_pack_inputs[name] = x.detach().cpu().clone()
                old = saved_funcs[name]
                return h.processor.process(x) if old is None else old(x)
            h.func = capture
        print('Old QDQ full forward and pre-QDQ representative activation capture', flush=True)
        try:
            qdq_out, qdq_blocks = run(quantized, 'legacy_qdq', {'bf16': teacher_blocks}, store=True)
        finally:
            for name, func in saved_funcs.items():
                hooks[name].func = func
        report['legacy_qdq_vs_bf16'] = metric(qdq_out, teacher_out)
        report['representative_inputs'] = {name: {'shape': list(x.shape), 'sha256': tensor_sha(x),
            'location': 'actual smoothed high-precision input entering the original activation Quantizer'}
            for name, x in representative_inputs.items()}
        pack_inputs_path = DATA/'research/20261002/E007/pack_inputs.pt'
        pack_inputs_path.parent.mkdir(parents=True, exist_ok=True)
        if pack_inputs_path.exists():
            raise FileExistsError(f'refusing to overwrite the captured packing oracle inputs: {pack_inputs_path}')
        torch.save({'sample': str(CACHE), 'sample_sha256': report['files'][str(CACHE)]['sha256'],
                    'location': 'actual activation Quantizer entry, after original smoothing and LR capture; not QDQ output',
                    'inputs': full_pack_inputs}, pack_inputs_path)
        report['full_pre_qdq_pack_inputs'] = {'path': str(pack_inputs_path), 'sha256': sha256(pack_inputs_path),
            'shapes': {name: list(x.shape) for name, x in full_pack_inputs.items()}}
        del full_pack_inputs
        checkpoint()
        print('Converting all 300 projections with exact saved-weight roundtrip', flush=True)
        conversion = native_lib.convert_wan_transformer_to_native(quantized, CHECKPOINT,
            activation_packer='legacy', chunk_rows=args.chunk_rows)
        report['conversion'] = conversion
        if conversion['target_count'] != 300 or conversion['exact_roundtrip_count'] != 300 or any(
                not row['exact'] or row['changed_elements'] != 0 for row in conversion['weights']):
            raise RuntimeError('not all 300 weight roundtrips were exact')
        # Conversion itself must reject any target weight mismatch.
        native_modules = {name: module for name, module in quantized.named_modules()
                          if isinstance(module, native_lib.NativeWanLinear)}
        if set(native_modules) != set(targets):
            raise RuntimeError('native conversion did not replace exactly the original 300 targets')
        del targets, hooks, saved_funcs, h
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint()
        print('Converted native bypass: actual restored legacy QDQ complete forward', flush=True)
        with native_lib.native_bypass(quantized):
            bypass_out, _ = run(quantized, 'native_bypass', {'legacy_qdq': qdq_blocks}, exact=True)
        report['native_bypass_vs_legacy_qdq'] = metric(bypass_out, qdq_out)
        if report['native_bypass_vs_legacy_qdq']['err2'] != 0:
            raise RuntimeError('native bypass endpoint differs from old QDQ')
        if audit['native_mm'].get('native_bypass', 0) != 0:
            raise RuntimeError('bypass unexpectedly called native GEMM')
        del bypass_out
        report['representative_main_contracts'] = []
        print('Four fixed N512 same-code main-branch correctness gates', flush=True)
        for name in REPRESENTATIVES:
            module = native_modules[name]
            x = representative_inputs[name].cuda()
            audit['phase'] = 'representative_main'
            packet = module.pack_input(x)
            reference = module.main_from_packet(packet, mode='packed_qdq', include_bias=False)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                got = module.main_from_packet(packet, mode='native', include_bias=False)
                torch.cuda.synchronize()
            kernels = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
            if not any('sm120' in k and 'e2m1' in k for k in kernels):
                raise RuntimeError(f'no native FP4 GEMM profiler evidence at {name}')
            row = {'layer': name, 'rows': 512, 'same_packet_object': True, 'bias_and_lr_excluded': True,
                   'metric': metric(got, reference), 'native_cuda_kernel_names': kernels}
            report['representative_main_contracts'].append(row)
            checkpoint()
            if row['metric']['nmse'] > 1e-4:
                raise RuntimeError(f'same-code native main-branch gate failed at {name}: {row["metric"]}')
            del x, packet, reference, got
        print('All 30 blocks / 300 native main projections on the fixed 31200-token input', flush=True)
        native_out, _ = run(quantized, 'native_full', {'bf16': teacher_blocks, 'legacy_qdq': qdq_blocks})
        if audit['native_mm'].get('native_full', 0) != 300:
            raise RuntimeError('expected exactly 300 native scaled_mm calls in the full transformer')
        report['native_vs_bf16'] = metric(native_out, teacher_out)
        report['native_vs_legacy_qdq'] = metric(native_out, qdq_out)
        report['runtime_audit'] = {'sdpa_calls': audit['sdpa'], 'native_mm_calls': audit['native_mm'],
            'sdpa_dtypes': [list(v) for v in sorted(audit['sdpa_dtypes'])],
            'sdpa_qkv_shapes': [[list(s) for s in v] for v in sorted(audit['sdpa_shapes'])]}
        report['status'] = 'complete'
        checkpoint()
        print(json.dumps({'status': report['status'], 'native_vs_bf16': report['native_vs_bf16'],
            'native_vs_legacy_qdq': report['native_vs_legacy_qdq'],
            'peak_gpu_gib_not_benchmark': report['peak_gpu_gib_not_benchmark'],
            'elapsed_seconds_not_benchmark': report['elapsed_seconds_not_benchmark']}, ensure_ascii=False), flush=True)
    finally:
        F.scaled_dot_product_attention, F.scaled_mm = original_sdpa, original_mm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-input', action='store_true', help='CPU-only architecture/cache contract inspection')
    parser.add_argument('--chunk-rows', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E007_wan_native_correctness.json')
    args = parser.parse_args()
    if args.inspect_input:
        contract, _ = inspect_contract()
        print(json.dumps(contract, indent=2))
        return
    report = {'experiment': 'E007', 'status': 'partial', 'sample': str(CACHE),
        'teacher': str(RCM), 'checkpoint': str(CHECKPOINT),
        'limitations': ['one original PTQ-calibration raw call, no held-out or quality claim',
            'full denoiser only; no decoded video or free rollout',
            'software packing correctness run; no speed or deployment-memory claim',
            'same-code native arithmetic may differ from BF16-QDQ GEMM; full output need not match bitwise']}
    try:
        execute(args, report)
    except Exception:
        report['status'] = 'failed_no_correctness_claim'
        report['error'] = traceback.format_exc()
        save(report, args.output)
        raise


if __name__ == '__main__':
    main()
