"""Matched Q128/K128, Q128/K64 and Q64/K64 operator measurements."""
import json,sys,statistics,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from anemoi import AttentionRouteCache,SparseConfig,QuantConfig,NVFP4Calibration,VisualLayout,anemoi_attention
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.probability_capture_fused import fused_softmax_mean_k64

def nmse(x,y):return float((x.float()-y.float()).square().sum()/y.float().square().sum())
def bench(fn):
 for _ in range(2):fn()
 torch.cuda.synchronize();v=[]
 for _ in range(8):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();v.append(a.elapsed_time(b))
 return {'mean_ms':statistics.mean(v),'samples_ms':v}
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 path=ROOT/'results/research/E209/result.json';assert not path.exists()
 data=torch.load('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt',map_location='cpu',weights_only=True,mmap=True)
 hq,hk,hv=[data[n].cuda().contiguous() for n in ('q','k','v')];q,k,v=[x.transpose(1,2).contiguous() for x in (hq,hk,hv)];prefix=int(data['prefix_tokens']);grid=tuple(data['video_grid']);layer=int(data['layer'])
 scales=json.loads((ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales'];cal=NVFP4Calibration(*(float(scales[n][layer]) for n in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))

 report={'status':'running','scope':'original per-step DraftMap, full API; same E208 QKV/budget, no cached route','arms':{}}
 for qb in (128,64):
  kwargs=dict(layout=VisualLayout(grid,prefix),layer=layer,sparse_config=SparseConfig(query_block_size=qb,sparsity_ratio=0.,maxpool_weight=.2),quant_config=QuantConfig(nvfp4_ratio=.9,int8_ratio=.1,prefix_query_precision='int8',prefix_kv_precision='int8'),calibration=cal)
  def call():return anemoi_attention(q,k,v,**kwargs)
  out=call();assert torch.isfinite(out).all();del out
  first=bench(call);later=bench(call)
  row={'first':first,'later_rebuild':later,'persistent_cache_bytes':0}
  report['arms'][f'Q{qb}K{qb}']=row;print(qb,row,flush=True)
 report['status']='complete';path.write_text(json.dumps(report,indent=2)+'\n')
 print('E209_COMPLETE',flush=True)
if __name__=='__main__':main()
