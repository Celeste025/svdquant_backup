import sys, importlib.util, json, statistics
from pathlib import Path
root=Path('/home/wjq/workspace/svdquant-exp')
spec=importlib.util.spec_from_file_location('validation',root/'scripts/research/validate_e154_exact_route_reuse.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch=m.torch
spans={}
for name in ['_prepare_h3_sm120_inputs','sm120_q128_dense_int8_importance_attention','_route_sm120_probability','_compact_exact_route','assemble_h3_k64_output']:
 original=getattr(m.executor,name)
 def wrap(*a,_fn=original,_name=name,**kw):
  begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  begin.record(); result=_fn(*a,**kw);end.record()
  spans.setdefault(_name,[]).append((begin,end));return result
 setattr(m.executor,name,wrap)
sys.argv=['profile','--query-block','128','--warmup','1','--repeats','3','--skip-fraction','0','--nvfp4-fraction','0.9','--output',str(root/'results/research/E156/gpu2/full_api.json')]
m.main();torch.cuda.synchronize()
report={}
for name,events in spans.items():
 values=[a.elapsed_time(b) for a,b in events]
 report[name]={'all_samples_ms':values,'last_three_mean_ms':statistics.mean(values[-3:])}
p=root/'results/research/E156/gpu2/phase_times.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
