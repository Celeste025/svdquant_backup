#!/usr/bin/env python3
"""E150-aligned H3 generation: probability route first-only/5-step or DraftMap baseline."""
import argparse, json
from pathlib import Path
import torch
import run_e159_route_reuse_whitebox as first
import run_e161_periodic_route_whitebox as periodic
import run_e162_proxy_new as proxy
import run_e150_whitebox_bf16 as run
from anemoi import AttentionRouteCache

ROOT=Path(__file__).resolve().parents[2]
class FirstProb(first.Router):
    def __init__(self,dit,arm):
        super().__init__(dit,arm)
        self.cache=AttentionRouteCache(capture_mode='softmax_mean')
        self.manifest.update(route_capture_mode='softmax_mean',first_step_policy='step 1 dense INT8 + probability map capture, later reuse; layer 0 BF16')
class PeriodicProb(periodic.Router):
    def __init__(self,dit,arm):
        super().__init__(dit,arm)
        self.cache=AttentionRouteCache(capture_mode='softmax_mean')
        self.manifest.update(route_capture_mode='softmax_mean',first_step_policy='steps 1,6,11,16 dense INT8 + probability map refresh, later reuse; layer 0 BF16')

def main():
    p=argparse.ArgumentParser();p.add_argument('--prompt-id',type=int,choices=(25,48,44,7),required=True);p.add_argument('--policy',choices=('first','periodic','baseline'),required=True);p.add_argument('--phase',choices=('denoise','decode'),required=True);a=p.parse_args()
    case=f'p{a.prompt_id:03d}';ref=json.loads((ROOT/'results/research/E150'/case/'bf16_denoise.json').read_text());assert ref['status']=='denoise_complete' and Path(ref['terminal_latents']).exists()
    if a.policy=='first':first.MODE='reuse';router=FirstProb
    elif a.policy=='periodic':periodic.MODE='reuse';router=PeriodicProb
    else:first.MODE='proxy';router=proxy.Router
    run.REPORT=ROOT/'results/research/E202'/a.policy;run.DATA=ROOT/'results/local/E202'/a.policy;run.ARM=a.policy;run.e121.Router=router
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    read=run.read_prompt;run.read_prompt=lambda pid:read(pid,Path('/home/wjq/workspace/178866172854036'))
    if a.phase=='decode':run.decode(a.prompt_id);return
    run.denoise(a.prompt_id)
    path=run.REPORT/case/'bf16_denoise.json';d=json.loads(path.read_text())
    assert d['prompt']==ref['prompt'] and d['seed']==ref['seed'] and d['settings']==ref['settings'] and d['frames']==ref['frames'] and d['initial_latent_sha256']==ref['initial_latent_sha256']
    assert len(d['steps'])==20 and all(x['attention_main_calls']==50 for x in d['steps'])
    if a.policy=='first':assert d['attention_router']['route_capture_mode']=='softmax_mean' and d['attention_router']['cached_routes']==49
    if a.policy=='periodic':assert d['attention_router']['route_capture_mode']=='softmax_mean' and d['attention_router']['cached_routes']==49 and d['attention_router']['refresh_steps_1based']==[1,6,11,16]
    x=torch.load(d['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].float();y=torch.load(ref['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].float();assert x.shape==y.shape
    d['comparison']={'bf16_reference':ref['terminal_latents'],'video_latent_nmse':float((x-y).double().square().sum()/y.double().square().sum()),'initial_noise_matches_bf16':True,'policy':a.policy}
    path.write_text(json.dumps(d,indent=2)+'\n');print(f'E202_DENOISE_COMPLETE {case} {a.policy}',flush=True)
if __name__=='__main__':main()
