#!/usr/bin/env python3
"""Fixed-checkpoint H3 block-0 diagnostic; not a video-quality experiment.

Preserves the full packed text/audio/video sequence at calibration resolution.
Changes only the low-rank branch input precision, holding all stored factors,
smoothing, quantizer rounding and BF16 attention fixed. Old calibration states
were optimized with the legacy bug; this is not a recalibrated SVDQuant result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import importlib
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
from minimax_h3_svdquant_common import (TARGET_SUFFIXES, load_h3_pipeline,
    raw_call_to_block0, tree_device, nvfp4_qdq, install_runtime_hooks)


def summarize(got, ref, indices=None):
    if indices is not None:
        got, ref = got[indices], ref[indices]
    diff = got.float() - ref.float()
    e = diff.square().sum(dtype=torch.float64).item()
    r = ref.float().square().sum(dtype=torch.float64).item()
    return {'err2': e, 'ref2': r, 'nmse': e/max(r, 1e-30),
            'max_abs': diff.abs().amax().item(), 'numel': ref.numel()}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts', default='1,20')
    p.add_argument('--steps', default='0,19')
    p.add_argument('--state', type=Path, default=ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt')
    p.add_argument('--cache', type=Path, default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    state = torch.load(args.state, map_location='cpu', weights_only=False)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(['dit'])
    block = pipe.dit.blocks[0]
    originals, weights, plain, residual = {}, {}, {}, {}
    # Ordinary Linear replacements make it impossible for disk offload to silently
    # reload the BF16 weight over an intervention.
    for suffix in TARGET_SUFFIXES:
        old = block.get_submodule(suffix)
        w, bias = old.load_from_disk(torch.bfloat16, 'cuda', assign=False)
        linear = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None,
                           device='cuda', dtype=torch.bfloat16)
        linear.weight.copy_(w)
        if bias is not None:
            linear.bias.copy_(bias)
        parent, attr = suffix.rsplit('.', 1)
        setattr(block.get_submodule(parent), attr, linear)
        originals[suffix] = linear
        weights[suffix] = w.clone()
        plain[suffix] = nvfp4_qdq(w)
        s = state['layers']['blocks.0.' + suffix]
        smooth, a, b = [s[k].to(w) for k in ('smooth', 'final_a', 'final_b')]
        residual[suffix] = (nvfp4_qdq(w*smooth-b@a), smooth, a, b)
    rows = []
    report = {'purpose':'baseline-correctness', 'model':'MiniMax-H3 pruned BF16',
              'block':0, 'attention_backend':importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION, 'state':str(args.state), 'state_config':state['config'],
              'cache':str(args.cache), 'torch':torch.__version__,
              'device':torch.cuda.get_device_name(), 'native_fp4':False,
              'common_source_sha256':hashlib.sha256((ROOT/'scripts/minimax_h3_svdquant_common.py').read_bytes()).hexdigest(),
              'limitations':['calibration prompts, not heldout', 'block endpoint, not free rollout or video quality',
                              'fixed legacy-calibrated state, not corrected recalibration',
                              'existing quantizer tie rule retained to isolate hook input precision'], 'rows':rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pid in map(int, args.prompts.split(',')):
        for step in map(int, args.steps.split(',')):
            start = time.time()
            path = args.cache/f'p{pid}'/f'sample_p{pid}_s{step:02d}.pt'
            sample = torch.load(path,map_location='cpu',weights_only=False)
            hidden, kw = raw_call_to_block0(pipe.dit, sample)
            hidden, kw = hidden.cuda(), tree_device(kw, 'cuda')
            for suffix, linear in originals.items():
                linear.weight.copy_(weights[suffix])
            ref = block(hidden, **kw)
            replay = block(hidden, **kw)
            replay_metric = summarize(replay, ref)
            if replay_metric['nmse'] > 1e-10:
                raise RuntimeError(f'BF16 replay mismatch {replay_metric}')
            outputs = {}
            for mode in ('plain_w4a4','legacy_lr_quantized_input','corrected_lr_high_precision_input'):
                handles, runtimes = [], []
                try:
                    for suffix, linear in originals.items():
                        if mode == 'plain_w4a4':
                            linear.weight.copy_(plain[suffix])
                            runtimes.append(install_runtime_hooks(linear, None, None, None))
                        else:
                            qw, smooth, a, b = residual[suffix]
                            linear.weight.copy_(qw)
                            if mode.startswith('corrected'):
                                runtimes.append(install_runtime_hooks(linear, smooth, a, b))
                            else:
                                def pre(_m, inp, smooth=smooth):
                                    return (nvfp4_qdq(inp[0]/smooth), *inp[1:])
                                def post(_m, inp, out, a=a, b=b):
                                    return out + F.linear(F.linear(inp[0],a),b)
                                handles.extend([linear.register_forward_pre_hook(pre),linear.register_forward_hook(post)])
                    out = block(hidden, **kw)
                    outputs[mode] = out
                    row = {'prompt_id':pid, 'step':step, 'mode':mode,
                           'tokens':hidden.shape[0], 'bf16_replay':replay_metric,
                           'all':summarize(out,ref)}
                    for modality, key in [('video','img_pos_info'),('audio','audio_pos_info'),('text','text_pos_info')]:
                        indices = sample['input_kwargs'][key]['position_ids'].long().cuda()
                        row[modality] = summarize(out,ref,indices)
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                finally:
                    for h in handles: h.remove()
                    for r in runtimes: r.remove()
                    for suffix, linear in originals.items(): linear.weight.copy_(weights[suffix])
            effect = summarize(outputs['corrected_lr_high_precision_input'],outputs['legacy_lr_quantized_input'])
            print(json.dumps({'prompt_id':pid,'step':step,'hook_effect':effect,'seconds':time.time()-start}),flush=True)
            report['rows'] = rows
            report['status'] = 'partial'
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            del outputs,ref,replay,hidden,kw,sample
    report['status'] = 'complete'
    report['peak_gpu_gib'] = torch.cuda.max_memory_allocated()/1024**3
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
