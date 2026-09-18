#!/usr/bin/env python3
"""Build/copy the H3 VBench-51 contract and dispatch ConvRot generation."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SETTINGS={"seed":0,"height":576,"width":1024,"frames":124,"steps":20,"cfg_scale":1.0,"rand_device":"cpu","tiled":True,"fps":24}

def atomic(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--state',type=Path,required=True); p.add_argument('--gpu',default='2'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    # The packed state is 11 GB.  Its adjacent summary was written atomically by
    # the quantizer and carries all structural metadata needed before dispatch;
    # load the packed tensors exactly once, inside the generation worker.
    summary_path=a.state.parent/'summary.json'
    if not a.state.is_file() or not summary_path.is_file(): raise RuntimeError('missing ConvRot state or summary')
    state=json.loads(summary_path.read_text())
    if state.get('format')!='minimax-h3-convrot-nvfp4-v1' or state.get('targets')!=200: raise RuntimeError('invalid ConvRot summary')
    mpath=a.output/'manifest.json'
    if mpath.exists(): m=json.loads(mpath.read_text())
    else:
        src=json.loads((a.source/'manifest.json').read_text())
        if src.get('settings')!=SETTINGS or len(src.get('cases',[]))!=51: raise RuntimeError('source is not H3 VBench-51')
        m={'schema_version':1,'created_at':time.time(),'source_manifest':str(a.source/'manifest.json'),'settings':SETTINGS,'sampling':src.get('sampling'),'variants':{'bf16':{'source':str(a.source),'action':'copy-sha256-verified'},'w4a4':{'source':str(a.source),'action':'copy-sha256-verified'},'convrot':{**{k:v for k,v in state.items() if k in ('format','targets','rot_size','weight','activation','use_triton_kernel','source')},'state':str(a.state)}},'cases':src['cases']}; atomic(mpath,m)
    recorded=Path(m.get('variants',{}).get('convrot',{}).get('state','')).resolve()
    if m.get('settings')!=SETTINGS or len(m.get('cases',[]))!=51 or recorded!=a.state.resolve(): raise RuntimeError('existing manifest mismatch')
    copies=[]
    for c in m['cases'][:1] if a.smoke else m['cases']:
        for v in ('bf16','w4a4'):
            for ext in ('.mp4','.json'):
                src=a.source/'cases'/c['case_id']/f'{v}{ext}'; dst=a.output/'cases'/c['case_id']/src.name
                if not src.is_file(): raise FileNotFoundError(src)
                h=sha(src)
                if dst.exists() and sha(dst)!=h: raise RuntimeError(f'baseline checksum mismatch: {dst}')
                if not dst.exists(): dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(src.read_bytes())
                copies.append({'file':str(dst.relative_to(a.output)),'sha256':h})
    atomic(a.output/'copy_status.json',{'state':'complete','source':str(a.source),'smoke':a.smoke,'artifacts':copies,'time':time.time()})
    diffsynth_root=Path(os.environ.get('DIFFSYNTH_ROOT', ROOT/'third_party'/'DiffSynth-Studio'))
    if not diffsynth_root.is_dir(): raise FileNotFoundError(f'DiffSynth-Studio not found: {diffsynth_root}')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu,'DIFFSYNTH_SKIP_DOWNLOAD':'True','PYTHONPATH':f'{ROOT}/scripts:{diffsynth_root}','TOKENIZERS_PARALLELISM':'false'}
    cmd=[sys.executable,str(ROOT/'scripts/minimax_h3_convrot_vbench51_worker.py'),'--manifest',str(mpath.resolve()),'--output',str(a.output.resolve()),'--state',str(a.state.resolve())]
    if a.smoke: cmd.append('--smoke')
    subprocess.run(cmd,check=True,env=env)
if __name__=='__main__': main()
