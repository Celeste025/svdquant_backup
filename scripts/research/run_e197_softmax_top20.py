#!/usr/bin/env python3
"""E196 replay with a true softmax tile top-20 probability score."""
import math
import torch
import run_e196_oracle_tile_scores as base

@torch.inference_mode()
def scores(q, k, prefix, indices, valid, counts, block, heads_per_chunk=4):
    heads, _, dim = q.shape
    nb = counts.numel()
    result = torch.empty((heads, nb, nb), device=q.device, dtype=torch.float32)
    for begin in range(0, heads, heads_per_chunk):
        end = min(begin + heads_per_chunk, heads)
        qv = q[begin:end, indices + prefix].float().view(end-begin, nb, block, dim)
        kv = k[begin:end, indices + prefix].float()
        keys = torch.cat((k[begin:end, :prefix].float(), kv), dim=1)
        key_valid = torch.cat((torch.ones(prefix, dtype=torch.bool, device=q.device), valid))
        for row in range(nb):
            nq = int(counts[row])
            logits = torch.bmm(qv[:, row, :nq], keys.transpose(1, 2)) * dim ** -0.5
            logits.masked_fill_(~key_valid.view(1, 1, -1), -torch.inf)
            prob = logits.softmax(dim=-1)[:, :, prefix:].view(end-begin, nq, nb, block)
            flat = prob.permute(0, 2, 1, 3).reshape(end-begin, nb, -1)
            result[begin:end, row] = flat.topk(min(20, flat.size(-1)), dim=-1).values.sum(-1)
        print(f'scored heads {begin}:{end} / {heads}', flush=True)
    assert torch.isfinite(result).all()
    return {'prob_top20_sum': result}

def self_test():
    torch.manual_seed(197)
    for block, counts_list in ((3, [3, 2, 3]), (8, [8, 5])):
        prefix, nb, dim = 2, len(counts_list), 16
        total = sum(counts_list)
        q, k = (torch.randn(2, prefix+total, dim, device='cuda') for _ in range(2))
        ids, mask, offset = [], [], 0
        for count in counts_list:
            ids.extend(range(offset, offset+count)); ids.extend([0]*(block-count))
            mask.extend([True]*count+[False]*(block-count)); offset += count
        indices = torch.tensor(ids, device=q.device)
        valid = torch.tensor(mask, dtype=torch.bool, device=q.device)
        counts = torch.tensor(counts_list, device=q.device)
        got = scores(q,k,prefix,indices,valid,counts,block,2)['prob_top20_sum']
        for head in range(2):
            for i in range(nb):
                qi = indices[i*block:i*block+counts_list[i]] + prefix
                logits = (q[head,qi] @ k[head].T) / math.sqrt(dim)
                probs = logits.softmax(-1)
                for j in range(nb):
                    kj = indices[j*block:j*block+counts_list[j]] + prefix
                    vals = probs[:,kj].flatten()
                    expected = vals.topk(min(20,vals.numel())).values.sum()
                    torch.testing.assert_close(got[head,i,j],expected,rtol=2e-5,atol=2e-6)
    print('SELF_TEST_OK: softmax top-20, small/large ragged tiles and full-key normalization', flush=True)

if __name__ == '__main__':
    base.RULES = ('prob_top20_sum',)
    base.scores = scores
    base.self_test = self_test
    base.main()
