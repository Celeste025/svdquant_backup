#!/usr/bin/env python3
"""Copy the public blind package and replace only its playback controls.

Reads no private data. Media bytes and the public review manifest are unchanged.
The original package is never edited. No network, model, encoding, or scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'results/research/E068/blind_review'
OUTPUT = ROOT / 'results/research/E068/blind_review_v2'

CONTROLS = r'''<script>
'use strict';
(() => {
  const videos = Array.from(document.querySelectorAll('video'));
  const ignoredPauses = new Map();
  let generation = 0;
  let active = null;
  const status = (id,text) => {document.getElementById(id+'-status').textContent=text;};
  const pair = id => ['A','B'].map(label=>document.getElementById(id+'-'+label));
  const current = session => active===session && session.token===generation && !session.cancelled;
  function pauseVideo(video){
    if(!video.paused){
      ignoredPauses.set(video,(ignoredPauses.get(video)||0)+1);
      video.pause();
    }
  }
  function cancelActive(){
    const previous=active;
    active=null;
    generation+=1;
    if(previous){
      previous.cancelled=true;
      for(const cancel of Array.from(previous.pending))cancel();
    }
  }
  function stopAll(){
    cancelActive();
    videos.forEach(video=>{pauseVideo(video);video.muted=true;});
  }
  function stopCase(id,reset,message){
    if(active && active.id===id)cancelActive();
    videos.filter(video=>video.dataset.case===id).forEach(video=>{
      pauseVideo(video);
      if(reset && video.readyState>=1)video.currentTime=0;
    });
    status(id,message);
  }
  function ready(video,session){
    if(!current(session))return Promise.reject(new Error('cancelled'));
    if(video.error)return Promise.reject(new Error('media'));
    if(video.readyState>=1)return Promise.resolve();
    return new Promise((resolve,reject)=>{
      const clean=()=>{
        video.removeEventListener('loadedmetadata',ok);
        video.removeEventListener('error',bad);
        session.pending.delete(cancel);
      };
      const ok=()=>{clean();resolve();};
      const bad=()=>{clean();reject(new Error('media'));};
      const cancel=()=>{clean();reject(new Error('cancelled'));};
      session.pending.add(cancel);
      video.addEventListener('loadedmetadata',ok,{once:true});
      video.addEventListener('error',bad,{once:true});
      video.load();
    });
  }
  async function startPair(id,audible){
    const items=pair(id), oldTime=items[0].currentTime;
    stopAll();
    const session={id,items,token:generation,cancelled:false,pending:new Set(),phase:'loading'};
    active=session;
    status(id,'正在准备同步播放…');
    try {
      await Promise.all(items.map(video=>ready(video,session)));
      if(!current(session))return;
      const duration=Math.min(...items.map(video=>video.duration));
      const time=Number.isFinite(oldTime)&&oldTime<duration-0.05?oldTime:0;
      items.forEach(video=>{
        video.currentTime=time;
        video.playbackRate=1;
        video.muted=video.dataset.label!==audible;
      });
      session.phase='starting';
      await Promise.all(items.map(video=>video.play()));
      if(!current(session)){
        // Never pause media owned by a newer request for the same pair.
        items.filter(video=>!active||!active.items.includes(video)).forEach(pauseVideo);
        return;
      }
      if(items.some(video=>video.paused)){
        stopCase(id,false,'已同步暂停。');
        return;
      }
      session.phase='playing';
      status(id,audible?'同步播放，仅 '+audible+' 有声。':'同步静音播放。');
    } catch(error){
      if(!current(session))return;
      stopCase(id,false,'无法自动播放，请使用视频自身的播放控件或下载完整视频。');
    }
  }
  videos.forEach(video=>{
    video.addEventListener('pause',()=>{
      const ignored=ignoredPauses.get(video)||0;
      if(ignored){ignoredPauses.set(video,ignored-1);return;}
      if(video.paused && active && active.items.includes(video)){
        stopCase(active.id,false,video.ended?'完整播放结束。':'已同步暂停。');
      }
    });
    video.addEventListener('ended',()=>{
      if(video.ended && active && active.items.includes(video)){
        stopCase(active.id,false,'完整播放结束。');
      }
    });
    video.addEventListener('volumechange',()=>{
      if(!video.muted)videos.forEach(other=>{if(other!==video)other.muted=true;});
    });
    video.addEventListener('error',()=>{
      const message='某段视频未能加载；请核对完整性清单或下载文件。';
      if(active && active.items.includes(video))stopCase(active.id,false,message);
      else status(video.dataset.case,message);
    });
  });
  document.querySelectorAll('button[data-action]').forEach(button=>button.addEventListener('click',async()=>{
    const id=button.dataset.case, action=button.dataset.action;
    if(action==='copy'){
      const field=document.getElementById(id+'-feedback');
      field.focus();field.select();
      try{await navigator.clipboard.writeText(field.value);status(id,'已复制；请自行粘贴回复。');}
      catch(error){status(id,'文本已选中，请手动复制后回复。');}
      return;
    }
    if(action==='sync'||action==='A'||action==='B'){
      await startPair(id,action==='sync'?null:action);
      return;
    }
    stopCase(id,action==='reset',action==='reset'?'已复位到开头。':'已暂停。');
  }));
  function tick(){
    const session=active;
    if(session && current(session) && session.phase==='playing'){
      const [a,b]=session.items;
      if(a.ended||b.ended)stopCase(session.id,false,'完整播放结束。');
      else if(a.paused||b.paused)stopCase(session.id,false,'已同步暂停。');
      else if(!a.seeking&&!b.seeking&&Math.abs(a.currentTime-b.currentTime)>0.08)b.currentTime=a.currentTime;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>'''


class ReviewError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ReviewError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def build(source, output):
    source, output = source.resolve(), output.resolve()
    require(source.is_dir() and not output.exists() and output != source,
            'A public source package and a new output directory are required.')
    require(not output.is_relative_to(source), 'The new package must not be inside the original package.')
    page_path, manifest_path = source / 'index.html', source / 'review_manifest.json'
    page = page_path.read_text(encoding='utf-8')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get('status') == 'complete', 'The public source manifest is incomplete.')
    require(len(manifest['cases']) == 2 and {c['id'] for c in manifest['cases']} == {'case01', 'case02'},
            'The public case set differs.')
    clips = [clip for case in manifest['cases'] for clip in case['clips']]
    expected = {f'{case}_{label}.mp4' for case in ('case01', 'case02') for label in ('A', 'B', 'reference')}
    require(len(clips) == 6 and {clip['file'] for clip in clips} == expected, 'The neutral clip set differs.')
    require(len(re.findall(r'<script\b[^>]*>.*?</script>', page, flags=re.S | re.I)) == 1,
            'Expected exactly one embedded controls script.')
    updated = re.sub(r'<script\b[^>]*>.*?</script>', lambda match: CONTROLS, page, count=1, flags=re.S | re.I)
    require(not any(word in (updated + manifest_bytes.decode('utf-8')).lower() for word in ('restart', 'carry')),
            'The public package contains a prohibited identity token.')
    original_hashes = {'index.html': sha256(page_path), 'review_manifest.json': sha256(manifest_path)}
    for clip in clips:
        path = (source / clip['file']).resolve()
        require(path.parent == source and path.is_file() and path.stat().st_size == clip['bytes'] and
                sha256(path) == clip['source_sha256'], 'A neutral source video failed verification.')
        original_hashes[clip['file']] = clip['source_sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.blind_controls_', dir=output.parent))
    try:
        for clip in clips:
            target = stage / clip['file']
            try:
                os.link(source / clip['file'], target)
            except OSError:
                with (source / clip['file']).open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst, length=8 << 20)
            require(target.stat().st_size == clip['bytes'] and sha256(target) == clip['source_sha256'],
                    'A neutral video copy failed verification.')
        (stage / 'index.html').write_text(updated, encoding='utf-8')
        (stage / 'review_manifest.json').write_bytes(manifest_bytes)
        require(all(sha256(source / name) == value for name, value in original_hashes.items()),
                'The original public package changed during the copy.')
        update = dict(status='complete', source_page_sha256=original_hashes['index.html'],
                      source_manifest_sha256=original_hashes['review_manifest.json'],
                      page_sha256=sha256(stage / 'index.html'), only_playback_script_replaced=True,
                      original_public_files_unchanged=True, manifest_byte_exact=True, clips_byte_exact=6,
                      request_cancellation_enabled=True, paired_native_pause_and_end=True,
                      sync_drift_threshold_seconds=0.08, browser_playback_verified=False)
        save_json(stage / 'controls_update.json', update)
        summary = dict(page=str(output / 'index.html'), clips=[str(output / c['file']) for c in clips],
                       integrity=update)
        save_json(stage / 'build_summary.json', summary)
        require(not output.exists(), 'Refusing to overwrite an existing package.')
        stage.rename(output)
        return summary
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=PUBLIC)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = build(args.source_dir, args.output_dir)
    except ReviewError as error:
        print(f'Review copy not built: {error}', file=sys.stderr)
        return 1
    except Exception as error:
        print(f'Review copy not built: validation failed ({type(error).__name__}).', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
