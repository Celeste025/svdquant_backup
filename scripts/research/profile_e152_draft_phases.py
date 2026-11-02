#!/usr/bin/env python3
"""Measure E151 native DraftAttention stages with CUDA events; prefix overlaps."""
import json
import statistics
from pathlib import Path
import torch
import bench_e142_attention_tradeoff as b
import anemoi.layers.attention.mpa.executor as ex

ROOT = b.ROOT
DATA = Path('/data1/models/svdquant-wjq/research/20261101/E151_long_h3_attention')
OUT = ROOT / 'results/research/E152'
NAMES = {
    '_prepare_h3_sm120_inputs': 'qkv_pack_quant_pool',
    'sm120_h3_draft_probability': 'draft_probability',
    '_route_sm120_probability': 'select_precision_route',
    'sm120_h3_materialize_route': 'route_materialization',
    '_run_sm120_phases': 'mixed_attention_main',
    'sm120_q64_prefix_int8_attention': 'prefix_attention_parallel',
    'assemble_h3_k64_output': 'output_assembly',
}

@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    scales = json.loads((ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales']
    report = {'gpu': torch.cuda.get_device_name(), 'frames': {}, 'description': 'CUDA event spans around actual API functions; prefix stream overlaps main stream; stage sums are not critical-path latency. Full timing without wrappers.', 'timing': '25 warmups, 6 batches of 12 full calls; 8 warmups + 30 per-call stage profiles'}
    originals = {name: getattr(ex, name) for name in NAMES}
    for frames in (124, 362):
        p = torch.load(DATA/f'race_768p_{frames}f_step0_block0.pt', map_location='cpu', weights_only=True, mmap=True)
        q, k, v = [p[n].cuda().transpose(1, 2).contiguous() for n in ('q','k','v')]
        layout = b.VisualLayout(tuple(p['video_grid']), p['prefix_tokens'])
        sparse = b.SparseConfig(query_block_size=64, sparsity_ratio=.75, maxpool_weight=.2)
        quant = b.QuantConfig(nvfp4_ratio=.6, int8_ratio=.4, prefix_kv_precision='int8', prefix_query_precision='int8')
        cal = b.NVFP4Calibration(*(float(scales[key][0]) for key in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
        def call():
            return b.anemoi_attention(q,k,v,layout=layout,layer=0,sparse_config=sparse,quant_config=quant,calibration=cal)
        full = b.measure(call, lambda:None)
        collected = {name: [] for name in NAMES.values()}
        pending = []
        def wrap(func, label):
            def instrumented(*args, **kwargs):
                a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); result = func(*args, **kwargs); z.record()
                pending.append((label, a, z))
                return result
            return instrumented
        for name, original in originals.items():
            setattr(ex, name, wrap(original, NAMES[name]))
        for i in range(38):
            pending.clear(); call(); torch.cuda.synchronize()
            if i >= 8:
                for label, a, z in pending: collected[label].append(a.elapsed_time(z))
        for name, original in originals.items(): setattr(ex, name, original)
        row = {'tokens': q.shape[1], 'prefix_tokens': p['prefix_tokens'], 'full': full, 'stages': {name: {'mean_ms': statistics.mean(xs), 'std_ms': statistics.stdev(xs), 'samples': len(xs)} for name, xs in collected.items()}}
        report['frames'][str(frames)] = row
        (OUT/'phase_profile.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'frames':frames, **row}),flush=True)

if __name__ == '__main__': main()
