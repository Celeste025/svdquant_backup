import json,sys,torch,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
s=(ROOT/'scripts/research/validate_e203_fused_capture.py').read_text();exec(s[:s.index('a,score=')])
from anemoi.layers.attention.mpa.probability_capture_fused import extension
ext=extension()
for _ in range(5):ext.capture(*ops,valid)
torch.cuda.synchronize();vals=[]
for _ in range(40):
 a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();ext.capture(*ops,valid);b.record();b.synchronize();vals.append(a.elapsed_time(b))
r={'mean_ms':statistics.mean(vals),'median_ms':statistics.median(vals),'samples_ms':vals};Path(sys.argv[1]).write_text(json.dumps(r,indent=2));print(r['mean_ms'],flush=True)
