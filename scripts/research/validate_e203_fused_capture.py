import sys,json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs,_expand_q128_valid_k
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.probability_capture import int8_softmax_mean
from anemoi.layers.attention.mpa.probability_capture_fused import fused_softmax_mean
from anemoi.layers.attention.mpa.backends.sm120_q128 import sm120_q128_dense_int8_importance_attention
p=torch.load('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt',map_location='cpu',weights_only=True,mmap=True)
q,k,v=[p[n].cuda() for n in ('q','k','v')];prefix=p['prefix_tokens'];grid=p['video_grid'];lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=128,enable_anchors=False)
pb=(prefix+63)//64;pc=(prefix-torch.arange(pb,device=q.device,dtype=torch.int32)*64).clamp(0,64).view(1,-1);valid=_expand_q128_valid_k(pc,lay.counts.view(1,-1));ops=_prepare_h3_sm120_inputs(q,k,v,video_token_indices=lay.indices,video_slot_valid=lay.slot_valid,video_valid_counts=lay.counts,prefix_tokens=prefix,query_block_size=128,phases=('int8',),has_prefix_query_int8=True,has_pool=False,has_maxpool=False,global_scales=None)[7]
a,score=fused_softmax_mean(ops,valid,lay.counts,pb);ref=int8_softmax_mean(ops,valid,lay.counts,pb);out,_=sm120_q128_dense_int8_importance_attention(ops,valid)
assert torch.equal(a,out),'dense output changed'
err=(score.float()-ref.float()).abs();torch.testing.assert_close(score,ref,rtol=.003,atol=2e-5)
def bench(fn):
 fn();torch.cuda.synchronize();times=[]
 for _ in range(5):
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();fn();e.record();e.synchronize();times.append(s.elapsed_time(e))
 return sum(times)/len(times)
r={'dense_output_bitwise_equal':True,'score_max_absolute_error_scaled1024':err.max().item(),'fused_ms':bench(lambda:fused_softmax_mean(ops,valid,lay.counts,pb)),'old_native_plus_two_pass_ms':bench(lambda:(sm120_q128_dense_int8_importance_attention(ops,valid),int8_softmax_mean(ops,valid,lay.counts,pb)))}
(ROOT/'results/research/E203/result.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
