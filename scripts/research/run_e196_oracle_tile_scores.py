#!/usr/bin/env python3
"""Compare exact-attention tile scores using the real SM120 mixed attention kernel."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
ANEMOI = ROOT / 'third_party/anemoi-code'
sys.path.insert(0, str(ANEMOI))
from anemoi import NVFP4Calibration, QuantConfig, SparseConfig, VisualLayout, anemoi_attention
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.routing import route_probability, _route_counts
import anemoi.layers.attention.mpa.executor as executor

RULES = ('qk_abs_mean', 'prob_mean', 'prob_rowmax_sum', 'qk_abs_top20_sum')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


@torch.inference_mode()
def scores(q, k, prefix, indices, valid, counts, block, heads_per_chunk=4):
    """Q,K [heads,tokens,dim]; compute FP32 logits and full-key softmax."""
    heads, _, dim = q.shape
    nb = counts.numel()
    assert indices.numel() == nb * block and valid.numel() == nb * block
    result = {name: torch.empty((heads, nb, nb), device=q.device) for name in RULES}
    for begin in range(0, heads, heads_per_chunk):
        end = min(heads, begin + heads_per_chunk)
        qvideo = q[begin:end, indices + prefix].float().view(end-begin, nb, block, dim)
        kvideo = k[begin:end, indices + prefix].float()
        keys = torch.cat((k[begin:end, :prefix].float(), kvideo), dim=1)
        key_valid = torch.cat((torch.ones(prefix, device=q.device, dtype=torch.bool), valid))
        for row in range(nb):
            nq = int(counts[row])
            logits = torch.bmm(qvideo[:, row, :nq], keys.transpose(1, 2)) * dim ** -0.5
            logits.masked_fill_(~key_valid.view(1, 1, -1), -torch.inf)
            probs = logits.softmax(dim=-1)
            assert torch.isfinite(probs).all()
            p = probs[:, :, prefix:].view(end-begin, nq, nb, block)
            result['prob_mean'][begin:end, row] = p.sum((1, 3)) / (nq * counts)
            result['prob_rowmax_sum'][begin:end, row] = p.amax(-1).sum(1)
            raw = logits[:, :, prefix:].view(end-begin, nq, nb, block)
            absolute = raw.abs().masked_fill(~valid.view(1, 1, nb, block), 0)
            result['qk_abs_mean'][begin:end, row] = absolute.sum((1, 3)) / (nq * counts)
            # Sort along the entire tile, not independently within each query row.
            flat = absolute.permute(0, 2, 1, 3).reshape(end-begin, nb, -1)
            result['qk_abs_top20_sum'][begin:end, row] = flat.topk(min(20, flat.size(-1)), dim=-1).values.sum(-1)
        print(f'scored heads {begin}:{end} / {heads}', flush=True)
    assert all(torch.isfinite(x).all() for x in result.values())
    return result


def self_test():
    torch.manual_seed(13)
    device = torch.device('cuda')
    q, k = (torch.randn(2, 10, 8, device=device) for _ in range(2))
    prefix, block = 2, 3
    indices = torch.tensor([0,1,2,3,4,0,5,6,7], device=device)
    valid = torch.tensor([1,1,1,1,1,0,1,1,1], device=device, dtype=torch.bool)
    counts = torch.tensor([3,2,3], device=device)
    got = scores(q, k, prefix, indices, valid, counts, block, 2)
    for h in range(2):
        for i in range(3):
            qi = indices[i*block:i*block+int(counts[i])] + prefix
            for j in range(3):
                kj = indices[j*block:j*block+int(counts[j])] + prefix
                logits = q[h, qi] @ k[h].T / math.sqrt(8)
                p = logits.softmax(-1)[:, kj]
                x = logits[:, kj].abs().flatten()
                expected = (x.mean(), p.mean(), p.amax(-1).sum(), x.topk(min(20,x.numel())).values.sum())
                for name, val in zip(RULES, expected):
                    torch.testing.assert_close(got[name][h,i,j], val, rtol=2e-5, atol=2e-6)
    print('SELF_TEST_OK: four scores, ragged tiles and full-key softmax', flush=True)


def nmse(output, reference, start, end):
    a, b = output[:, start:end].float(), reference[:, start:end].float()
    per_head_num = (a-b).double().square().sum(dim=(0,1,3))
    per_head_den = b.double().square().sum(dim=(0,1,3))
    per_head = (per_head_num/per_head_den).cpu()
    return {'nmse': (per_head_num.sum()/per_head_den.sum()).item(),
            'per_head_nmse': per_head.tolist(), 'mean_head_nmse': per_head.mean().item(),
            'median_head_nmse': per_head.median().item(), 'min_head_nmse': per_head.min().item(),
            'max_head_nmse': per_head.max().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--step', type=int)
    p.add_argument('--layer', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    self_test()
    if args.self_test:
        return
    assert args.output and args.step is not None and args.layer is not None
    assert not args.output.exists(), args.output
    manifest_path = ROOT/'results/research/E187/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entry = next(x for x in manifest['captures'] if x['step']==args.step and x['layer']==args.layer)
    path = Path(entry['path'])
    assert sha(path) == entry['sha256']
    data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    qh, kh, vh = (data[x].cuda().contiguous() for x in ('q','k','v'))
    assert list(qh.shape) == list(kh.shape) == list(vh.shape) == entry['shape_hnd']
    prefix, grid = int(data['prefix_tokens']), tuple(data['video_grid'])
    layout = materialize_ragged_2d_layout(qh.device, frames=grid[0], height=grid[1], width=grid[2], logical_block=128, enable_anchors=False)
    counts = layout.counts
    print(f'capture step={args.step} layer={args.layer} prefix={prefix} grid={grid} blocks={counts.numel()}', flush=True)
    real_scores = scores(qh[0], kh[0], prefix, layout.indices, layout.slot_valid, counts, 128)
    generator = torch.Generator(device=qh.device).manual_seed(196)
    real_scores['random'] = torch.rand(real_scores[RULES[0]].shape, device=qh.device, generator=generator)
    reference = F.scaled_dot_product_attention(qh, kh, vh).transpose(1,2).contiguous()
    q,k,v = (x.transpose(1,2).contiguous() for x in (qh,kh,vh))
    del qh,kh,vh
    scales = json.loads((ANEMOI/'anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales']
    cal = NVFP4Calibration(*(float(scales[key][args.layer]) for key in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
    video_layout = VisualLayout(grid, prefix)
    report = {'status':'running', 'step_zero_based':args.step, 'layer_zero_based':args.layer,
        'capture_sha256':entry['sha256'], 'manifest_sha256':sha(manifest_path), 'script_sha256':sha(Path(__file__)),
        'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'grid':grid, 'prefix_tokens':prefix,
        'tile_count_per_head':counts.numel()**2, 'video_tokens':math.prod(grid),
        'scores':'FP32 QK from captured BF16 inputs, scaled by 1/sqrt(D); full valid key softmax including prefix',
        'reference':'BF16 SDPA', 'random_seed':196, 'results':[]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for skip, nv in ((.75,.6),(0.,.9)):
        sparse = SparseConfig(query_block_size=128,sparsity_ratio=skip,maxpool_weight=.2)
        quant = QuantConfig(nvfp4_ratio=nv,int8_ratio=1-nv,prefix_kv_precision='int8',prefix_query_precision='int8')
        for name in ('draftmap',*RULES,'random'):
            seen = []
            original = executor._route_sm120_probability
            def replace(probability, anchors, anchor_ids, *, anchor_count, sparsity_ratio, ratios):
                assert anchors is None and anchor_ids is None and anchor_count == 0
                if name == 'draftmap':
                    plan = original(probability, anchors, anchor_ids, anchor_count=anchor_count, sparsity_ratio=sparsity_ratio, ratios=ratios)
                else:
                    score = real_scores[name].unsqueeze(0)
                    assert score.shape == probability.shape
                    plan = route_probability(score, None, anchor_count=0, prefix_blocks=0, sparsity_ratio=sparsity_ratio,
                        nvfp4_ratio=ratios[0], fp8_ratio=ratios[1]+ratios[2], fp16_ratio=ratios[3])
                    # Decode route and verify every selected block against stable ranking.
                    n4, n8 = plan.nvfp4_blocks_per_head, plan.fp8_blocks_per_head
                    order = score.flatten(-2).argsort(dim=-1, descending=True, stable=True)
                    ids = plan.block_ids.long(); cols = score.shape[-1]
                    actual = torch.zeros_like(score, dtype=torch.uint8)
                    positions = torch.arange(cols,device=score.device).view(1,1,1,-1)
                    phase = torch.where(positions < plan.nvfp4_counts.unsqueeze(-1), 1,
                        torch.where(positions < (plan.nvfp4_counts+plan.fp8_counts).unsqueeze(-1), 2, 0)).to(torch.uint8)
                    actual.scatter_(-1, ids, phase)
                    expected = torch.zeros_like(score.flatten(-2),dtype=torch.uint8)
                    expected.scatter_(-1,order[...,:n8],2)
                    expected.scatter_(-1,order[...,n8:n8+n4],1)
                    assert torch.equal(actual.flatten(-2),expected)
                seen.append(plan)
                return plan
            with patch.object(executor, '_route_sm120_probability', replace):
                out = anemoi_attention(q,k,v,layout=video_layout,layer=args.layer,sparse_config=sparse,quant_config=quant,calibration=cal)
            assert len(seen)==1 and torch.isfinite(out).all()
            plan = seen[0]; items = counts.numel()**2
            exp = _route_counts(items,sparsity_ratio=skip,nvfp4_ratio=nv,fp8_ratio=1-nv,fp16_ratio=0.)
            assert (plan.fp16_blocks_per_head,plan.fp8_blocks_per_head,plan.nvfp4_blocks_per_head)==exp
            row = {'skip_fraction':skip,'rule':name,'tiles_int8':exp[1],'tiles_nvfp4':exp[2],
                'actual_int8_fraction':exp[1]/items,'actual_nvfp4_fraction':exp[2]/items,
                'actual_skip_fraction':1-(exp[1]+exp[2])/items,
                'video':nmse(out,reference,prefix,out.size(1)),'prefix':nmse(out,reference,0,prefix)}
            report['results'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'step':args.step,'layer':args.layer,'budget':skip,'rule':name,'video_nmse':row['video']['nmse']}),flush=True)
            del out,seen,plan
    report['status']='complete'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('E196_COMPLETE',flush=True)

if __name__ == '__main__':
    main()
