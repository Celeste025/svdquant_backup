"""Controlled original-map ablation: retain identical routes while bypassing work."""
import json,sys,statistics,torch
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from anemoi import SparseConfig,QuantConfig,NVFP4Calibration,VisualLayout,anemoi_attention
import anemoi.layers.attention.mpa.executor as ex

@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 output=ROOT/'results/research/E212/result.json';assert not output.exists()
 data=torch.load('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt',map_location='cpu',weights_only=True,mmap=True)
 q,k,v=[data[n].cuda().transpose(1,2).contiguous() for n in ('q','k','v')];layer=int(data['layer']);layout=VisualLayout(tuple(data['video_grid']),int(data['prefix_tokens']))
 scales=json.loads((ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales'];cal=NVFP4Calibration(*(float(scales[n][layer]) for n in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
 original_prepare=ex._prepare_h3_sm120_inputs;original_score=ex.sm120_h3_draft_probability;original_route=ex._route_sm120_probability
 report={'status':'running','scope':'same real input, same original DraftMap and route, same route materializer and main kernel','rows':[]}
 for skip,nv in ((0.,.9),(.8,.5)):
  for qb in (128,64):
   cache={};stage=[0]
   def prepare(*a,**kw):
    if stage[0]>=1:kw.update(has_pool=False,has_maxpool=False)
    values=list(original_prepare(*a,**kw))
    if stage[0]==0:cache['pools']=(values[0],values[1],values[-2],values[-1])
    else:values[0],values[1],values[-2],values[-1]=cache['pools']
    return tuple(values)
   def score(*a,**kw):
    if stage[0]>=2:return cache['score']
    result=original_score(*a,**kw)
    if stage[0]==0:cache['score']=result
    return result
   def route(*a,**kw):
    if stage[0]>=3:return cache['route']
    result=original_route(*a,**kw)
    if stage[0]==0:cache['route']=result
    return result
   kwargs=dict(layout=layout,layer=layer,sparse_config=SparseConfig(query_block_size=qb,sparsity_ratio=skip,maxpool_weight=.2),quant_config=QuantConfig(nvfp4_ratio=nv,int8_ratio=1-nv,prefix_query_precision='int8',prefix_kv_precision='int8'),calibration=cal)
   def call():return anemoi_attention(q,k,v,**kwargs)
   with patch.object(ex,'_prepare_h3_sm120_inputs',prepare),patch.object(ex,'sm120_h3_draft_probability',score),patch.object(ex,'_route_sm120_probability',route):
    reference=call();torch.cuda.synchronize();errors=[]
    for st in (1,2,3):
     stage[0]=st;result=call();assert torch.isfinite(result).all();nmse=float((result.float()-reference.float()).square().sum()/reference.float().square().sum());assert nmse<1e-8,nmse;errors.append(nmse);del result
    del reference
    samples={i:[] for i in range(4)}
    for order in ((0,1,2,3),(3,2,1,0)):
     for st in order:
      stage[0]=st
      for _ in range(3):call()
      torch.cuda.synchronize()
      for _ in range(12):
       a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();call();b.record();b.synchronize();samples[st].append(a.elapsed_time(b))
   means={st:statistics.mean(x) for st,x in samples.items()}
   row={'skip':skip,'query_block':qb,'nmse_vs_original':errors,'stages':{str(st):{'mean_ms':means[st],'median_ms':statistics.median(samples[st]),'samples_ms':samples[st]} for st in range(4)},'pool_saved_ms':means[0]-means[1],'score_saved_ms':means[1]-means[2],'sort_saved_ms':means[2]-means[3],'total_saved_ms':means[0]-means[3],'total_saved_percent':100*(means[0]-means[3])/means[0]}
   report['rows'].append(row);output.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in row.items() if k!='stages'},'stages',means,flush=True)
 report['status']='complete';output.write_text(json.dumps(report,indent=2)+'\n');print('E212_COMPLETE',flush=True)
if __name__=='__main__':main()
