#!/usr/bin/env python3
"""Pinned E024 public model assets; all downloads/partials stay on DATA1."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data1/models/svdquant-wjq')
SOURCE=ROOT/'research_state/00_state/fastwan_qad_external_baseline_sources.json'
RD=ROOT/'results/research/E024'
REV='621c6aeb900f9f9a2ebb9ea9ed74c0daf31d5e6a'
TAE_REV='011dfc2112197741c540e0bdd5b7b67bcc930771'
TARGET=DATA/'models/FastWan-QAD-1.3B-621c6aeb'
BASE=DATA/'models/Wan2.1-T2V-1.3B-Diffusers'
TAE=DATA/'models/TAEHV-011dfc21'
WORK=DATA/'tmp/E024-assets'
MANIFEST=RD/'assets_manifest.json'
TAE_FILES=[('LICENSE',1073,'6dc1edfbb0516416aa9ba7984116ec4281745d85'),
 ('README.md',10016,'7476cb2ae6ea2a88c6f3c9a062b8150eaa0cc219'),
 ('taehv.py',26475,'8514281c20318c71ccdf916e4383d9a65f825009'),
 ('taew2_1.pth',22678901,'d6ffa27358914606153bbb790bd1da1c5cf822fd')]


def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def hashes(path,git_blob=False):
 h=hashlib.sha256();g=hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0') if git_blob else None
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):
   h.update(chunk)
   if g is not None:g.update(chunk)
 return h.hexdigest(),g.hexdigest() if g is not None else None


def worker(deadline):
 inventory=json.loads(SOURCE.read_text());assert inventory['hf_revision']==REV
 assert not MANIFEST.exists() and not WORK.exists(),'Preserve previous attempt'
 if ATTEMPT!=3:assert all(not p.exists() or not any(p.iterdir()) for p in (TARGET,TAE)),'Preserve existing populated targets'
 for path in (TARGET,TAE,WORK):path.mkdir(parents=True,exist_ok=True)
 started=time.time();active={}
 report=dict(status='running',worker_pid=os.getpid(),start_epoch=started,deadline_epoch=deadline,
  target=str(TARGET),taehv_target=str(TAE),hf_revision=REV,taehv_revision=TAE_REV,inventory=str(SOURCE),
  policy='Public fixed-revision downloads only. Original source assets are read-only; matching full SHA256 LFS assets become symlinks. No GPU or environment installation.',
  prior_failed_attempts=[str(RD/'assets_manifest.json'),str(RD/'assets_manifest_attempt2.json')] if ATTEMPT==3 else ([str(RD/'assets_manifest.json')] if ATTEMPT==2 else []),
  files={},failures=[],download_attempts=[],active_downloads=active)
 if ATTEMPT==3:
  prior=json.loads((RD/'assets_manifest_attempt2.json').read_text());assert prior['status']=='failed_partial_preserved'
  for key,item in prior['files'].items():
   p=Path(item['path']);assert p.exists() and p.stat().st_size==item['bytes']
   if item['mode']=='verified_local_symlink':assert p.is_symlink() and p.resolve()==Path(item['source']).resolve()
   else:assert hashes(p)[0]==item['sha256']
   report['files'][key]=item
  report['resume_reason']='Detached tmux server lacked caller HTTP(S) proxy; this session explicitly inherits proxy variables. Existing verified assets preserved.'

 def checkpoint():
  report.update(seconds=time.time()-started)
  for entry in active.values():entry['partial_bytes']=Path(entry['partial_path']).stat().st_size if Path(entry['partial_path']).exists() else 0
  save(MANIFEST,report)

 def begin(url,key,expected_bytes):
  remaining=deadline-time.time()
  if remaining<=0:raise TimeoutError('Asset deadline expired')
  stem=key.replace('/','__');part=WORK/(stem+f'.attempt{ATTEMPT}.partial');log=WORK/(stem+f'.attempt{ATTEMPT}.stderr.log');stats=WORK/(stem+f'.attempt{ATTEMPT}.transfer.json')
  err=log.open('x');out=stats.open('x')
  cmd=[sys.executable,str(Path(__file__).resolve()),'--download-url',url,'--partial',str(part),'--deadline-unix',str(deadline)]
  p=subprocess.Popen(cmd,stdout=out,stderr=err)
  item=dict(key=key,url=url,pid=p.pid,expected_bytes=expected_bytes,partial_path=str(part),stderr=str(log),transfer_stats=str(stats),start_epoch=time.time(),attempt=ATTEMPT)
  active[key]=item;report['download_attempts'].append(item);checkpoint()
  print(json.dumps(dict(event='download_started',**item)),flush=True)
  return p,err,out,item

 def finish(job,destination,expected_sha=None,expected_git=None):
  p,err,out,item=job
  while p.poll() is None:
   if time.time()>=deadline:raise TimeoutError('Asset deadline expired')
   checkpoint();time.sleep(3)
  err.close();out.close();item.update(returncode=p.returncode,seconds=time.time()-item['start_epoch'])
  part=Path(item['partial_path']);item['actual_bytes']=part.stat().st_size if part.exists() else 0
  if p.returncode:raise RuntimeError(f'Download failed {item["key"]}: downloader exit {p.returncode}; log {item["stderr"]}')
  if item['actual_bytes']!=item['expected_bytes']:raise ValueError(f'Download size mismatch {item["key"]}: {item["actual_bytes"]}')
  actual_sha,actual_git=hashes(part,git_blob=expected_git is not None)
  item.update(actual_sha256=actual_sha,actual_git_blob_sha1=actual_git)
  if expected_sha and actual_sha!=expected_sha:raise ValueError(f'LFS SHA mismatch {item["key"]}')
  if expected_git and actual_git!=expected_git:raise ValueError(f'Pinned Git blob mismatch {item["key"]}')
  destination.parent.mkdir(parents=True,exist_ok=True);part.replace(destination)
  result=dict(path=str(destination),bytes=item['actual_bytes'],sha256=actual_sha,git_blob_sha1=actual_git,
   mode='downloaded',url=item['url'],expected_lfs_sha256=expected_sha,expected_git_blob_sha1=expected_git,status='verified')
  report['files'][item['key']]=result;active.pop(item['key']);checkpoint();print(json.dumps(dict(event='asset_verified',key=item['key'],**result)),flush=True)
  return result

 large=next(x for x in inventory['hf_files'] if x['path']=='transformer/diffusion_pytorch_model.safetensors')
 try:
  big=begin(f'https://huggingface.co/FastVideo/FastWan-QAD-1.3B/resolve/{REV}/{large["path"]}?download=true','hf/'+large['path'],large['bytes'])
  # Make the tiny pinned TAEHV source available to the separate environment worker.
  rel,size,git_sha=TAE_FILES[2]
  if 'taehv/'+rel not in report['files']:
   finish(begin(f'https://raw.githubusercontent.com/madebyollin/taehv/{TAE_REV}/{rel}','taehv/'+rel,size),TAE/rel,expected_git=git_sha)
  # Hashing local 23GB assets proceeds while the new transformer downloads.
  for item in sorted(inventory['hf_files'],key=lambda x:x['lfs_sha256'] is None):
   rel=item['path']
   if rel==large['path']:continue
   key='hf/'+rel;local=BASE/rel;destination=TARGET/rel
   if key in report['files']:continue
   if item['lfs_sha256'] and local.exists() and local.stat().st_size==item['bytes']:
    report['current_local_hash']=str(local);checkpoint()
    actual,_=hashes(local)
    if actual==item['lfs_sha256']:
     destination.parent.mkdir(parents=True,exist_ok=True);destination.symlink_to(local)
     report['files'][key]=dict(path=str(destination),bytes=local.stat().st_size,sha256=actual,mode='verified_local_symlink',source=str(local),
      expected_lfs_sha256=item['lfs_sha256'],status='verified')
     checkpoint();print(json.dumps(dict(event='local_reused',key=key,bytes=item['bytes'],sha256=actual)),flush=True);continue
    report['failures'].append(dict(type='local_source_sha_mismatch',path=str(local),actual_sha256=actual,expected_sha256=item['lfs_sha256'],action='source unchanged; fetch fixed official asset'))
   job=begin(f'https://huggingface.co/FastVideo/FastWan-QAD-1.3B/resolve/{REV}/{rel}?download=true',key,item['bytes'])
   finish(job,destination,expected_sha=item['lfs_sha256'])
  report.pop('current_local_hash',None)
  for rel,size,git_sha in TAE_FILES:
   if rel=='taehv.py':continue
   job=begin(f'https://raw.githubusercontent.com/madebyollin/taehv/{TAE_REV}/{rel}','taehv/'+rel,size)
   finish(job,TAE/rel,expected_git=git_sha)
  finish(big,TARGET/large['path'],expected_sha=large['lfs_sha256'])
  report.update(status='complete',verified_files=len(report['files']),
   total_hf_logical_bytes=sum(x['bytes'] for x in report['files'].values() if x['path'].startswith(str(TARGET))),
   downloaded_asset_bytes=sum(x['bytes'] for x in report['files'].values() if x['mode']=='downloaded'),
   symlink_reused_bytes=sum(x['bytes'] for x in report['files'].values() if x['mode']=='verified_local_symlink'))
  checkpoint();print(json.dumps({k:report[k] for k in ('status','verified_files','downloaded_asset_bytes','symlink_reused_bytes','seconds')}),flush=True)
 except BaseException:
  report.update(status='failed_partial_preserved',error=traceback.format_exc());checkpoint()
  for item in active.values():
   try:os.kill(item['pid'],signal.SIGTERM)
   except ProcessLookupError:pass
  raise


def supervise(seconds):
 RD.mkdir(parents=True,exist_ok=True)
 suffix='' if ATTEMPT==1 else f'_attempt{ATTEMPT}'
 path=RD/f'assets_supervisor{suffix}.json';log=RD/f'assets_download{suffix}.log'
 assert not path.exists() and not log.exists(),'Preserve previous supervisor'
 assert 0<seconds<=1800
 start=time.time();deadline=start+seconds
 env=os.environ.copy();env.update(TMPDIR=str(DATA/'tmp'),HF_HOME=str(DATA/'tmp/E024-hf-cache'),XDG_CACHE_HOME=str(DATA/'tmp/E024-xdg-cache'),CUDA_VISIBLE_DEVICES='')
 command=[sys.executable,str(Path(__file__).resolve()),'--deadline-unix',str(deadline),'--attempt',str(ATTEMPT)]
 state=dict(status='starting',start_epoch=start,deadline_epoch=deadline,command=command,log=str(log),supervisor_pid=os.getpid())
 save(path,state)
 with log.open('x') as stream:
  p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  state.update(status='running',worker_pid=p.pid);save(path,state)
  timed_out=False
  try:code=p.wait(timeout=seconds)
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(p.pid,signal.SIGTERM)
   try:code=p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
 state.update(status='deadline_terminated' if timed_out else ('complete' if code==0 else 'failed'),returncode=code,seconds=time.time()-start)
 save(path,state)
 if code:raise SystemExit(code)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--supervise',action='store_true');parser.add_argument('--seconds',type=int,default=1800);parser.add_argument('--deadline-unix',type=float)
 parser.add_argument('--attempt',type=int,choices=(1,2,3),default=1);parser.add_argument('--download-url');parser.add_argument('--partial',type=Path);args=parser.parse_args()
 ATTEMPT=args.attempt
 if ATTEMPT!=1:
  MANIFEST=RD/f'assets_manifest_attempt{ATTEMPT}.json';WORK=DATA/f'tmp/E024-assets-attempt{ATTEMPT}'
 if args.download_url:
  started=time.time();downloaded=0
  with urllib.request.urlopen(args.download_url,timeout=60) as response,args.partial.open('xb') as output:
   for block in iter(lambda:response.read(4*1024**2),b''):
    if time.time()>=args.deadline_unix:raise TimeoutError('Download deadline expired')
    output.write(block);downloaded+=len(block)
  print(json.dumps(dict(http_status=response.status,bytes=downloaded,seconds=time.time()-started)),flush=True)
  raise SystemExit(0)
 if args.supervise:supervise(args.seconds)
 else:worker(args.deadline_unix)
