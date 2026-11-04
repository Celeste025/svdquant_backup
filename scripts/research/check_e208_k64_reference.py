import sys,json,math,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'third_party/anemoi-code')]
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.probability_capture_fused import fused_softmax_mean_k64,route_k64
from anemoi.layers.attention.mpa.routing import _route_counts
@torch.inference_mode()
def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(6)
 # Small rectangular route: decode precision and compare to explicit global sort.
 score=torch.arange(30,device='cuda',dtype=torch.float16).reshape(1,2,3,5)/30
 cols=torch.arange(5,device='cuda');plan=route_k64(score,cols,0.,(.9,.1,0.,0.));n16,n8,n4=_route_counts(15,sparsity_ratio=0.,nvfp4_ratio=.9,fp8_ratio=.1,fp16_ratio=0.)
 precision=torch.zeros_like(score,dtype=torch.uint8);position=torch.arange(5,device='cuda').view(1,1,1,-1);phase=torch.where(position<plan.nvfp4_counts[...,None],1,2).to(torch.uint8);precision.scatter_(-1,plan.block_ids.long(),phase)
 exp=torch.ones_like(score.flatten(-2),dtype=torch.uint8);exp.scatter_(-1,score.flatten(-2).argsort(dim=-1,descending=True,stable=True)[...,:n8],2);assert torch.equal(precision.flatten(-2),exp)
 p=torch.load('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt',map_location='cpu',weights_only=True,mmap=True)
 q,k,v=(p[n][:,:2].cuda().contiguous() for n in ('q','k','v'));prefix=p['prefix_tokens'];grid=p['video_grid'];lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=64,enable_anchors=False);pb=(prefix+63)//64
 pc=(prefix-torch.arange(pb,device=q.device,dtype=torch.int32)*64).clamp(0,64).view(1,-1);valid=torch.cat((pc,lay.counts.view(1,-1)),dim=1).contiguous()
 ops=_prepare_h3_sm120_inputs(q,k,v,video_token_indices=lay.indices,video_slot_valid=lay.slot_valid,video_valid_counts=lay.counts,prefix_tokens=prefix,query_block_size=64,phases=('int8',),has_prefix_query_int8=True,has_pool=False,has_maxpool=False,global_scales=None)[7]
 out,got,columns=fused_softmax_mean_k64(ops,valid,lay.counts,pb,64);q8,qs,k8,ks,*_=ops
 keys=k8.float()*ks.repeat_interleave(64,-1)[...,None];mask=(torch.arange(64,device=q.device)[None,:]<valid[0,:,None]).flatten();maxerror=0.
 for i in (0,len(lay.counts)//2,len(lay.counts)-1):
  nq=int(lay.counts[i]);qq=q8[:,:,i*64:i*64+nq].float()*qs[:,:,i,None,None];s=qq@keys.transpose(-2,-1)/math.sqrt(128);s.masked_fill_(~mask.view(1,1,1,-1),-torch.inf);prob=s.softmax(-1)
  for j,col in enumerate(columns.tolist()):
   nk=int(valid[0,pb+col]);ref=prob[:,:,:,(pb+col)*64:(pb+col)*64+nk].mean((-1,-2));act=got[:,:,i,j].float()/1024;torch.testing.assert_close(act,ref,rtol=.003,atol=2e-7);maxerror=max(maxerror,(act-ref).abs().max().item())
 r={'rectangular_route_matches_explicit_sort':True,'q64_probability_reference_checked_heads':2,'query_tiles':3,'max_absolute_error_unscaled':maxerror};(ROOT/'results/research/E208/check.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
if __name__=='__main__':main()
