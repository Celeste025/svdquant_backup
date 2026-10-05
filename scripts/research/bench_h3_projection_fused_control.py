#!/usr/bin/env python3
"""Prepared E054: three H3 communication arms with one fused SVD consumer.

Reuses the frozen E039 packet/wire contracts. No custom kernel, no full DiT,
and no claim that these three arms exhaust the best distributed baseline.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import bench_h3_output_projection_parallel as base

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('bf16_return', 'fp8_return', 'fp4_side')
WARMUPS, REPEATS = 3, 10


def fp8_encode_eager(x):
    """The E039 per-source/destination E4M3 message, with a FP32 scale."""
    import torch
    value = x.float()
    scale = (value.abs().amax().clamp_min(1e-12) / 448.).reshape(1)
    codes = (value / scale).clamp(-448., 448.).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8), scale


def fp8_restore_smooth_eager(c0, s0, c1, s1, smooth):
    """Keep the BF16 reconstruction point before original BF16 smoothing."""
    import torch
    left = (c0.view(torch.float8_e4m3fn).float() * s0).bfloat16()
    right = (c1.view(torch.float8_e4m3fn).float() * s1).bfloat16()
    return torch.cat((left, right), dim=1) / smooth


class CompiledFP8:
    def __init__(self):
        import torch
        import torch._dynamo
        torch._dynamo.config.suppress_errors = False
        self.receipt = dict(backend='torch._inductor.compile', fullgraph=True,
            dynamic=False, suppress_errors=False, emulate_precision_casts=True,
            compile_invocations=0, graphs=[], encode_calls=0,
            restore_smooth_calls=0, fallback=False)

        def backend(graph, inputs):
            self.receipt['compile_invocations'] += 1
            self.receipt['graphs'].append(dict(code=graph.code,
                tensor_inputs=[dict(shape=list(x.shape), dtype=str(x.dtype))
                    for x in inputs if torch.is_tensor(x)]))
            # Compiler-generated pointwise/reduction kernels; no hand-written
            # FP8 kernel. Retain intermediate precision casts explicitly.
            return torch._inductor.compile(graph, inputs,
                options={'emulate_precision_casts': True})

        self.encoder = torch.compile(fp8_encode_eager, backend=backend,
            fullgraph=True, dynamic=False)
        self.restorer = torch.compile(fp8_restore_smooth_eager, backend=backend,
            fullgraph=True, dynamic=False)

    def encode(self, x):
        self.receipt['encode_calls'] += 1
        return self.encoder(x)

    def restore(self, received, smooth):
        self.receipt['restore_smooth_calls'] += 1
        a, b = received
        return self.restorer(a['codes'], a['scale'], b['codes'], b['scale'], smooth)


class Execution(base.Execution):
    def __init__(self, rank, full, half, budget, compiler):
        super().__init__(rank, full, half, budget)
        self.compiler = compiler
        self.counts.update(fused_main_up=0, dynamic_up_rescale=0)
        self.dynamic_l1 = []

    def fused(self, packet, down):
        import torch
        from flashinfer.gemm import mm_nvfp4_svdquant
        self.counts['fused_main_up'] += 1
        self.counts['dynamic_up_rescale'] += 1
        alpha = packet.global_scale.reshape(1) * self.full.weight_global.reshape(1)
        l1 = (self.full.lr_b.float() / alpha).to(torch.bfloat16).contiguous()
        self.dynamic_l1.append(l1)
        return mm_nvfp4_svdquant(packet.packed, self.full.weight_packed,
            packet.swizzled_scales.view(torch.uint8),
            self.full.weight_scales_swizzled.view(torch.uint8), alpha,
            down.contiguous(), l1, backend='cute-dsl', enable_pdl=False)

    def project_smoothed(self, x):
        packet = self.pack(x)
        down = self.down(x, self.full)
        return dict(output=self.fused(packet, down), down=down,
            activation_globals=packet.global_scale)

    def project_full(self, o):
        self.counts['smooth'] += 1
        return self.project_smoothed(o / self.full.smooth)

    def execute(self, o, arm):
        import torch
        from h3_native_nvfp4 import PackedNVFP4
        if arm == 'bf16_return':
            # Base execute calls the overridden project_full; no base native
            # main/up operation is used in this arm.
            return super().execute(o, arm)
        self.counts['boundary'] += 1
        if arm == 'fp8_return':
            parts = []
            for start, length in zip(base.STARTS, base.REAL):
                self.counts['fp8_quant'] += 1
                codes, scale = self.compiler.encode(o[start:start + length])
                parts.append(dict(codes=codes, scale=scale))
            received = self.a2a(parts, arm)
            self.counts['fp8_dequant'] += 2
            self.counts['smooth'] += 1
            x = self.compiler.restore(received, self.full.smooth)
            return self.project_smoothed(x)
        if arm != 'fp4_side':
            raise ValueError(arm)
        self.counts['smooth'] += 1
        x = o / self.half.smooth
        globals_ = self.destination_globals(x)
        parts = []
        for destination, (start, length) in enumerate(zip(base.STARTS, base.REAL)):
            part = x[start:start + length].contiguous()
            packet = self.pack(part, globals_[destination:destination + 1])
            down = self.down(part, self.half)
            parts.append(dict(codes=packet.packed, sf=packet.swizzled_scales, down=down))
        received = self.a2a(parts, arm)
        codes, sf = base.join_activation(received, self.rank)
        down = (received[0]['down'].float() + received[1]['down'].float()).bfloat16()
        packet = PackedNVFP4(codes, None, globals_[self.rank:self.rank + 1], sf,
            (base.REAL[self.rank], base.DIN), 'E039_transport_with_common_fused_consumer')
        return dict(output=self.fused(packet, down), down=down,
            activation_globals=globals_)


def sources(args, report):
    import torch
    import h3_native_nvfp4 as native
    import h3_nvfp4_fastpack as fast
    import wan_native_nvfp4 as packet
    paths = dict(runner=Path(__file__), wire_and_packet_helpers=Path(base.__file__),
        native=Path(native.__file__), packer=Path(fast.__file__),
        packet=Path(packet.__file__), plan=args.plan)
    spec = importlib.util.find_spec('flashinfer')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('FlashInfer is not installed in the selected environment')
    paths['fused_api'] = Path(next(iter(spec.submodule_search_locations))) / 'gemm/gemm_svdquant.py'
    report['sources'] = {k: base.common.record(p) for k, p in paths.items()}
    report['input'] = base.common.record(args.input)
    report['weight'] = base.common.record(base.WEIGHT)
    report['versions'] = {name: metadata.version(name)
        for name in ('torch', 'flashinfer-python', 'nvidia-cutlass-dsl')}
    payload = torch.load(base.WEIGHT, map_location='cpu', weights_only=True, mmap=True)
    assert tuple(payload['shape']) == (base.DOUT, base.DIN)
    assert payload['tensors']['bias'] is None
    report['geometry'] = dict(real_total=base.REAL_TOTAL, main_valid=base.N,
        heads=base.H, head_dim=base.D, input_features=base.DIN,
        output_features=base.DOUT, rank=base.R, token_starts=list(base.STARTS),
        real_token_lengths=list(base.REAL), valid_token_lengths=list(base.VALID),
        scale_physical_rows=list(base.PHYSICAL), model_padding_rows=53)
    return payload


def cpu_check(args, report, payload):
    import torch
    base.cpu_check(report, payload)
    raw = torch.load(args.input, map_location='cpu', weights_only=True, mmap=True)
    assert raw.shape == (base.REAL_TOTAL, base.H, base.D) and raw.dtype == torch.bfloat16
    assert bool(raw.isfinite().all())
    # This is a format/schema check, never a substitute for actual GPU parity.
    x = torch.linspace(-3., 3., 128).reshape(4, 32).bfloat16()
    codes, scale = fp8_encode_eager(x)
    restored = fp8_restore_smooth_eager(codes, scale, codes, scale,
        torch.ones(64, dtype=torch.bfloat16))
    assert codes.dtype == torch.uint8 and scale.dtype == torch.float32
    assert restored.shape == (4, 64) and restored.dtype == torch.bfloat16
    assert bool(restored.isfinite().all())
    assert not torch.cuda.is_initialized()
    report['cpu_check'].update(input_schema=True, fp8_eager_schema=True,
        fused_shape_compatible=(base.DOUT % 32 == base.DIN % 32 == 0 and base.R == 32),
        compiler_not_called=True, GPU_parity_not_tested=True)
    report.update(status='complete', cuda_initialized=False, prepared_only=True)


def compare(a, b):
    from probe_h3_fused_up import metric
    return metric(a, b)


def run(args, report, payload, budget):
    import torch
    import torch.distributed as dist
    from h3_native_nvfp4 import NativeH3Linear
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    assert rank == local and int(os.environ['WORLD_SIZE']) == 2
    assert len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == 2
    torch.cuda.set_device(local)
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(min(1., 60 * 2**30 /
        torch.cuda.get_device_properties(local).total_memory))
    dist.init_process_group('nccl', timeout=timedelta(seconds=max(1.,
        args.deadline_unix - time.time())), device_id=torch.device('cuda', local))
    report.update(rank=rank, world_size=2, environment=dict(torch=torch.__version__,
        cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()),
        device=torch.cuda.get_device_name(), visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        tf32=False, p2p_access=bool(torch.cuda.can_device_access_peer(local, 1-local))))
    raw = torch.load(args.input, map_location='cpu', weights_only=True, mmap=True)
    o = raw[:, rank*(base.H//2):(rank+1)*(base.H//2)].reshape(base.REAL_TOTAL, base.HALF).contiguous().cuda()
    full = NativeH3Linear.from_export(payload, device='cuda')
    half = base.half_weight(payload, rank).cuda()
    del raw, payload
    compiler = CompiledFP8()
    engine = Execution(rank, full, half, budget, compiler)
    report['actual_counts'] = engine.counts
    report['fp8_compiler'] = compiler.receipt
    report['control_barrier_calls'] = 0
    directory = args.data / f'rank{rank}'
    assert not directory.exists(), 'Preserve previous data, including failed runs'
    directory.mkdir(parents=True)
    entries = {arm: dict(arm=arm, warmups=[], repeats=[]) for arm in ARMS}
    report['benchmarks'] = list(entries.values())

    def boundary():
        budget()
        dist.barrier()
        report['control_barrier_calls'] += 1
        torch.cuda.synchronize()

    def memory():
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())

    def checked_product(product):
        assert product['output'].shape == (base.REAL[rank], base.DOUT)
        assert product['output'].dtype == torch.bfloat16
        assert all(bool(value.isfinite().all()) for value in product.values())
        assert all(bool(value.isfinite().all()) for value in engine.dynamic_l1)
        flags = torch.stack(engine.flags).cpu()
        assert not bool(flags[:, 0].ne(0).any()), 'Invalid original H3 packer input or scale'
        return dict(calls=len(engine.flags), invalid_calls=0,
            calls_with_nonzero_input_zero_sf=int(flags[:, 1].ne(0).sum()))

    def invoke(arm, index, order, position, measured):
        boundary()
        torch.cuda.reset_peak_memory_stats()
        before, old = memory(), dict(engine.counts)
        engine.collectives, engine.flags, engine.dynamic_l1 = [], [], []
        compile_before = compiler.receipt['compile_invocations']
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        product = engine.execute(o, arm)
        end.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - started) * 1000
        after = memory()
        budget()
        assert after['peak_allocated'] <= 60 * 2**30
        compiler_delta = compiler.receipt['compile_invocations'] - compile_before
        if measured and compiler_delta:
            raise RuntimeError('Compilation occurred in a measured boundary; result invalid')
        checks = checked_product(product)
        receipt = dict(round=index, order=list(order), position=position, wall_ms=wall,
            cuda_ms=begin.elapsed_time(end), before=before, after=after,
            incremental_peak_allocated=after['peak_allocated']-before['allocated'],
            compiler_invocations=compiler_delta,
            actual_counts={k: engine.counts[k]-old[k] for k in old},
            collectives=engine.collectives, pack_flag_checks=checks)
        entries[arm]['repeats' if measured else 'warmups'].append(receipt)
        if measured and index == REPEATS-1:
            output = product['output'].cpu()
            state = {k: product[k].cpu() for k in ('down', 'activation_globals')}
            path, state_path = directory/f'{arm}_output.pt', directory/f'{arm}_state.pt'
            torch.save(output, path)
            torch.save(state, state_path)
            entries[arm]['output'] = dict(artifact=base.common.record(path), tensor=base.common.trecord(output))
            entries[arm]['state'] = dict(artifact=base.common.record(state_path),
                tensors={k: base.common.trecord(v) for k, v in state.items()})
        del product
        engine.dynamic_l1 = []
        base.common.save(args.output, report)

    boundary()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report['resident_baseline'] = memory()
    prepare_started = time.time()
    report['prepare'] = dict(status='running', started_unix=prepare_started,
        scope='JIT, compiler warmup, profiling and numerical controls; excluded from steady-state medians, included in the one 1200s job deadline.')
    for warm in range(WARMUPS):
        for position, arm in enumerate(ARMS):
            invoke(arm, warm, ARMS, position, False)
    assert compiler.receipt['compile_invocations'] > 0

    # One eager-versus-compiled FP8 numerical check, before any measured timing.
    parts, eager = [], []
    for start, length in zip(base.STARTS, base.REAL):
        source = o[start:start+length]
        c, s = compiler.encode(source)
        ec, es = fp8_encode_eager(source)
        parts.append(dict(codes=c, scale=s))
        eager.append(dict(codes=ec, scale=es))
    local_part = parts[rank]
    local_eager = eager[rank]
    actual_x = compiler.restore([local_part, local_part], full.smooth)
    eager_x = fp8_restore_smooth_eager(local_eager['codes'], local_eager['scale'],
        local_eager['codes'], local_eager['scale'], full.smooth)
    report['fp8_compiled_vs_eager_diagnostic'] = dict(
        scope='Actual local head half duplicated solely to exercise full-width decode/smooth; not a model output or communication arm.',
        smoothed_nmse=compare(actual_x, eager_x),
        encode_scale_nmse=[compare(p['scale'], e['scale']) for p, e in zip(parts, eager)],
        numerical_drift_is_not_byte_equality_gate=True)
    assert bool(actual_x.isfinite().all())
    del parts, eager, local_part, local_eager, actual_x, eager_x, source, c, s, ec, es

    report['prepare_profiles'] = []
    for arm in ARMS:
        boundary()
        engine.collectives, engine.flags, engine.dynamic_l1 = [], [], []
        old = dict(engine.counts)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as prof:
            product = engine.execute(o, arm)
            torch.cuda.synchronize()
        checks = checked_product(product)
        kernels = sorted({event.name for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA})
        assert kernels, 'No actual CUDA kernels observed'
        report['prepare_profiles'].append(dict(arm=arm, kernels=kernels,
            actual_counts={k: engine.counts[k]-old[k] for k in old}, pack_flag_checks=checks))
        del product
        engine.dynamic_l1 = []
        budget()
    report['prepare'].update(status='complete', seconds=time.time()-prepare_started,
        compiler_invocations=compiler.receipt['compile_invocations'])
    report['after_prepare_baseline'] = memory()
    base.common.save(args.output, report)
    compile_frozen = compiler.receipt['compile_invocations']
    benchmark_started = time.time()
    for repeat in range(REPEATS):
        shift = repeat % len(ARMS)
        order = ARMS[shift:] + ARMS[:shift]
        for position, arm in enumerate(order):
            invoke(arm, repeat, order, position, True)
    assert compiler.receipt['compile_invocations'] == compile_frozen
    report['measured_phase_seconds'] = time.time()-benchmark_started
    expected = (WARMUPS + 1 + REPEATS) * len(ARMS)
    assert engine.counts['boundary'] == engine.counts['fused_main_up'] == expected
    assert engine.counts['dynamic_up_rescale'] == expected
    assert engine.counts['native_gemm'] == engine.counts['lr_up'] == 0
    assert engine.counts['all_to_all'] == expected
    assert engine.counts['all_reduce'] == WARMUPS + 1 + REPEATS
    assert engine.counts['reduce_scatter'] == 0
    for entry in entries.values():
        entry.update(status='complete', median_wall_ms=statistics.median(r['wall_ms'] for r in entry['repeats']),
            median_cuda_ms=statistics.median(r['cuda_ms'] for r in entry['repeats']))
    boundary()
    report.update(status='complete', actual_dit_calls=0, cuda_initialized=True,
        compiled_fp8_no_fallback=True, measured_compile_invocations=0)
    del o, full, half, engine, compiler
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report['released_allocated_bytes'] = torch.cuda.memory_allocated()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'run'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--input', type=Path, default=base.INPUT)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous result or failure'
    started = time.time()
    if args.phase == 'run':
        assert args.deadline_unix and 0 < args.deadline_unix-started <= 1200
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''

    def budget():
        if args.deadline_unix and time.time() >= args.deadline_unix:
            raise TimeoutError('E054 total job deadline includes JIT and preparation')

    report = dict(experiment='E054', phase=args.phase, status='running',
        rank=int(os.environ.get('RANK', '0')), world_size=2, arms=list(ARMS),
        actual_dit_calls=0, warmups_per_arm=WARMUPS, repeats_per_arm=REPEATS,
        profiles_per_arm=1, expected_fused_calls_per_rank=42,
        timing_scope='Resident BF16 head-owned O through smoothing, statistics, quantization, wire copies, communication, low-rank down, dynamic alpha and BF16 L1 rescale, fused main/up to token-owned BF16 output. Preparation/JIT, control barriers, validation readback and disk saving outside measured boundaries; all inside 1200s total job deadline.',
        numerical_contracts=dict(common_consumer='FlashInfer mm_nvfp4_svdquant cute-dsl; alpha=gA*gW and L1=BF16(B/alpha) recomputed each boundary. Different L1 rounding and epilogue order from E039; report drift, not byte equality.',
            packet='Unchanged E039 legacy packer, globals, SF physical layout, real model padding and original BF16 division smoothing.',
            fp8='E039 per-source/destination scalar E4M3 protocol; torch.compile with explicit intermediate precision casts fuses encode and two-source BF16 decode/concat/smooth. Numerical compiler drift recorded.',
            side='Unchanged partial BF16 down and FP32 sum -> BF16 at owner; same destination-global FP4 packet.'),
        scientific_scope='One actual H3 boundary, three common-fused-consumer arms. Not a full-model quality/latency result and not an exhaustive best-system comparison.')
    try:
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        payload = sources(args, report)
        if args.phase == 'check':
            cpu_check(args, report, payload)
        else:
            with torch.inference_mode():
                run(args, report, payload, budget)
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.time()-started
        base.common.save(args.output, report)
        print(json.dumps(dict(rank=report['rank'], status=report['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
