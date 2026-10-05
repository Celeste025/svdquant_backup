#!/usr/bin/env python3
"""E080 capture on exact E079 Sage3+SVD trajectories; no model modification."""
import argparse, json, os, time, traceback
from pathlib import Path
import torch
import run_h3_sage3_action_video as prior
from probe_h3_plain_baseline import file_record, tree_signature
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research/E080/v2'
DATA=Path('/data1/models/svdquant-wjq/research/20261004/E080/v2')
PLAN=ROOT/'research_state/06_experiments/E080_pv_contrast_plan.md'

def main():
 p=argparse.ArgumentParser();p.add_argument('--replica',type=int,choices=[0,1],required=True);p.add_argument('--phase',choices=['check','capture'],required=True);p.add_argument('--deadline-unix',type=float)
 a=p.parse_args();phase=a.phase
 a.phase='check' if phase=='check' else 'denoise';a.arm='svd_sage3';a.manifest=prior.MANIFEST;a.data_dir=None;a.report_dir=None;a.prepare_report=None;a.output=OUT/f'{phase}_r{a.replica}.json';a.report_writable=False
 assert not a.output.exists(), 'Never overwrite an executed receipt'
 report=dict(experiment='E080',phase=phase,replica=a.replica,arm=a.arm,status='running',cases=[],captures=[],attempted_dit_calls=0,complete_dit_calls=0,actual_video_vae_calls=0,actual_audio_vae_calls=0,deadline_unix=a.deadline_unix)
 start=time.monotonic();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 try:
  manifest=prior.read_manifest(a,report)
  a.data_dir=DATA;a.report_dir=OUT
  bound,_=prior.prepare_bindings(a,manifest,report,required=True)
  for rec in manifest['frozen_runtime_sources']: assert file_record(rec['file'])==rec
  cases=[c for c in manifest['cases'] if c['replica']==a.replica and c['prompt_id']==161]
  bound=[b for b in bound if b['case_id']==cases[0]['case_id']]
  assert len(cases)==len(bound)==1
  manifest=dict(manifest,cases=cases)
  report.update(allocated_dit_calls=20,scope='Unchanged E079 clapping rollout, all56heads QKV/output capture at step14 blocks0,24',capture_source=file_record(__file__),plan=file_record(PLAN),prepared_cases=bound)
  if phase=='check':
   assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
  else:
   checked=json.loads((OUT/f'check_r{a.replica}.json').read_text())
   assert checked['status']=='complete' and checked['capture_source']==report['capture_source'] and checked['plan']==report['plan'] and checked['prepared_cases']==bound
   import probe_h3_sage3_interaction as sageprobe
   Original=sageprobe.Router
   class CaptureRouter(Original):
    def __init__(self,dit):
     self.capture_step=-1;super().__init__(dit)
    def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
     block=self._active_main.get()
     if block==0:self.capture_step+=1
     take=self.capture_step==14 and block in (0,24)
     if take:
      n=self._active_log.get().expected_cu[1]
      payload={name:x[:n].detach().cpu().contiguous() for name,x in zip(('q','k','v'),(q,k,v))}
     out=super()._dispatch(q,k,v,cu_seqlens,softmax_scale)
     if take:
      payload.update(output=out[:n].detach().cpu().contiguous(),case_id=cases[0]['case_id'],step=14,block=block,valid_length=n,video_start=n-21312,video_grid=[37,18,32],scale=softmax_scale)
      path=DATA/f'{cases[0]["case_id"]}_s14_b{block}.pt';assert not path.exists();torch.save(payload,path)
      report['captures'].append(dict(case_id=cases[0]['case_id'],step=14,block=block,artifact=file_record(path),tensors=tree_signature(payload)))
      prior.save(report,a.output);print(f'E080 capture {path}',flush=True)
     return out
   sageprobe.Router=CaptureRouter
   try:prior.denoise(a,manifest,bound,report)
   finally:sageprobe.Router=Original
   assert len(report['captures'])==2
   reference=json.loads((ROOT/f'results/research/E079/denoise_svd_sage3_r{a.replica}.json').read_text())
   source=next(c for c in reference['cases'] if c['case_id']==cases[0]['case_id'])
   assert report['cases'][0]['final_tensors']==source['final_tensors'], 'Capture changed final tensors'
   report.update(status='complete',final_replay_exact=True,reference_final_tensors=source['final_tensors'])
 except BaseException:
  report.update(status='failed_stop',error=traceback.format_exc());raise
 finally:
  report['seconds_total']=time.monotonic()-start
  if torch.cuda.is_initialized():report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
  if a.report_writable:prior.save(report,a.output)
  print(json.dumps(dict(status=report['status'],path=str(a.output))),flush=True)
if __name__=='__main__':main()
