"""Correctness of experimental INT8 softmax route capture."""
import math
from unittest.mock import patch
import torch


def test_int8_probability_mean_ragged_prefix():
    from anemoi.layers.attention.mpa.probability_capture import int8_softmax_mean
    torch.manual_seed(19); device='cuda'; b,h,qb,vb,pb,d=1,2,2,2,1,128
    q=torch.randint(-30,31,(b,h,qb*128,d),device=device,dtype=torch.int8)
    k=torch.randint(-30,31,(b,h,(pb+2*vb)*64,d),device=device,dtype=torch.int8)
    qs=torch.rand((b,h,qb),device=device)*.01+.005
    ks=torch.rand((b,h,pb+2*vb),device=device)*.01+.005
    valid=torch.tensor([[7,64,64,37,0]],device=device,dtype=torch.int32)
    counts=torch.tensor([128,37],device=device,dtype=torch.int32)
    actual=int8_softmax_mean((q,qs,k,ks,None,None),valid,counts,pb).float()/1024
    qr=q.float()*qs.repeat_interleave(128,-1).unsqueeze(-1)
    kr=k.float()*ks.repeat_interleave(64,-1).unsqueeze(-1)
    logits=qr@kr.transpose(-2,-1)/math.sqrt(d)
    valid_mask=(torch.arange(64,device=device).view(1,64)<valid.view(-1,1)).flatten()
    logits.masked_fill_(~valid_mask.view(1,1,1,-1),-torch.inf)
    probs=logits.softmax(-1);expected=torch.empty_like(actual)
    for i in range(qb):
        for j in range(vb):
            expected[:,:,i,j]=probs[:,:,i*128:i*128+int(counts[i]),(pb+2*j)*64:(pb+2*j)*64+int(counts[j])].mean((-2,-1))
    torch.testing.assert_close(actual,expected,rtol=.002,atol=1e-7)

def test_probability_route_reuse_without_recompute():
    from anemoi import AttentionRouteCache, QuantConfig, SparseConfig, VisualLayout, anemoi_attention
    import anemoi.layers.attention.mpa.probability_capture as pc
    torch.manual_seed(18)
    q,k,v=[torch.randn((1,273,2,128),device='cuda',dtype=torch.bfloat16)*.3 for _ in range(3)]
    opts=dict(layout=VisualLayout((1,16,16),17),layer=0,sparse_config=SparseConfig(query_block_size=128,sparsity_ratio=.75),quant_config=QuantConfig(nvfp4_ratio=.6,int8_ratio=.4,prefix_kv_precision='int8',prefix_query_precision='int8'))
    cache=AttentionRouteCache(capture_mode='softmax_mean')
    a=anemoi_attention(q,k,v,**opts,first_step=True,route_cache=cache)
    with patch.object(pc,'int8_softmax_mean',side_effect=AssertionError('reuse recalculated probabilities')):
        b=anemoi_attention(q,k,v,**opts,first_step=False,route_cache=cache)
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and len(cache)==1
    assert AttentionRouteCache().capture_mode=='qk_max'

if __name__ == '__main__':
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit('SM120 CUDA required')
    test_int8_probability_mean_ragged_prefix()
    test_probability_route_reuse_without_recompute()
    print('2 probability capture tests passed')
