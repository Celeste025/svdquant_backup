#!/usr/bin/env python3
"""Build the frozen E068 blind review only after independent verification.

Standard library only. No model, media conversion, scoring, network, or feedback
submission. The precommitted assignment and all source paths remain private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS = ROOT / 'results/research/E068'
ARMS = ('restart', 'carry')
CASES = (('case01', 30, 49771), ('case02', 36, 59526))
MEDIA = dict(frames=124, width=1024, height=576, fps=24.0,
             audio_streams=1, audio_sample_rate=32000)


class BuildError(Exception):
    """Messages are neutral: never include source paths or private values."""


def require(condition, message):
    if not condition:
        raise BuildError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(file=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def load(path, *, complete=False):
    require(Path(path).is_file(), 'A required verified input is unavailable.')
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'An input report has an invalid format.')
    if complete:
        require(value.get('status') == 'complete', 'Independent verification and source reports must be complete.')
    return value


def verify_record(value, expected_path=None):
    require(isinstance(value, dict) and {'file', 'sha256', 'bytes'} <= value.keys(),
            'A verified file binding is incomplete.')
    path = Path(value['file']).resolve()
    if expected_path is not None:
        require(path == Path(expected_path).resolve(), 'A verified report path differs from the requested input.')
    require(path.is_file() and path.stat().st_size == value['bytes'] and sha256(path) == value['sha256'],
            'A verified file changed or is missing.')
    return path


def write_json(path, value, *, private=False):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)


def neutral_guard(text, source_names):
    require(not any(word in text.lower() for word in ARMS), 'Public material contains a prohibited identity token.')
    require(not any(name in text for name in source_names), 'Public material contains a source filename.')


def video_markup(case_id, label, caption):
    name = f'{case_id}_{label}.mp4'
    return (f'<figure><figcaption>{html.escape(caption)}</figcaption>'
            f'<video id="{case_id}-{label}" data-case="{case_id}" data-label="{label}" '
            f'controls muted playsinline preload="metadata" aria-label="{case_id} {label}">'
            f'<source src="{name}" type="video/mp4">浏览器无法播放此视频。'
            f'</video><a href="{name}" download>下载完整视频</a></figure>')


def render_page(cases):
    sections = []
    for case in cases:
        cid = case['id']
        sections.append(f'''<section id="{cid}"><h2>{cid}</h2>
<details class="prompt"><summary>查看完整原始提示词</summary><p>{html.escape(case['prompt'])}</p></details>
<div class="pair">{video_markup(cid, 'A', 'A')}{video_markup(cid, 'B', 'B')}</div>
<div class="actions"><button data-action="sync" data-case="{cid}">同步静音播放 A / B</button>
<button data-action="A" data-case="{cid}">同步播放，仅听 A</button>
<button data-action="B" data-case="{cid}">同步播放，仅听 B</button>
<button data-action="pause" data-case="{cid}">暂停</button>
<button data-action="reset" data-case="{cid}">复位到开头</button></div>
<p class="play-status" id="{cid}-status" role="status" aria-live="polite"></p>
<details class="reference"><summary>可选：查看 BF16 参照</summary>
<p>参照仅供辅助判断；请先独立比较 A 与 B。</p>{video_markup(cid, 'reference', 'BF16 参照')}</details>
<label for="{cid}-feedback">反馈模板（可直接复制后回复）</label>
<textarea id="{cid}-feedback" rows="4" spellcheck="false">{cid}：A / B / 难分
视觉质量与运动连贯性：
声音及音画配合：
备注：</textarea>
<button class="copy" data-action="copy" data-case="{cid}">复制本组反馈</button></section>''')
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>完整视频盲评</title>
<style>
:root{color-scheme:dark;--bg:#10141b;--panel:#1a202b;--line:#394354;--text:#edf1f7;--muted:#b4bfce;--accent:#a4c7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,sans-serif}
main{max-width:1460px;margin:auto;padding:32px 24px 70px}header{max-width:900px;margin-bottom:28px}
h1{font-size:30px;line-height:1.25;margin:0 0 14px}h2{font-size:23px;margin:0 0 14px}
p{margin:10px 0}header p,.reference p,.footer{color:var(--muted)}
section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin:24px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:12px 0}figcaption{font-size:20px;font-weight:650;margin-bottom:7px}
video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border:1px solid var(--line);border-radius:7px}
a{color:var(--accent);font-size:14px}button,summary{cursor:pointer}button{font:inherit;font-size:14px;color:var(--text);background:#283447;border:1px solid #536782;border-radius:7px;padding:9px 13px}
button:hover{background:#34455f}button:focus-visible,summary:focus-visible,a:focus-visible,textarea:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
.actions{display:flex;flex-wrap:wrap;gap:9px;margin:15px 0}.play-status{min-height:1.6em;color:var(--muted);font-size:14px}
details{border:1px solid var(--line);border-radius:8px;padding:10px 13px;margin:14px 0}.prompt p{white-space:pre-wrap;overflow-wrap:anywhere}.reference figure{max-width:800px;margin:15px auto}
label{display:block;margin:20px 0 7px}textarea{display:block;width:100%;resize:vertical;border:1px solid var(--line);background:#111823;color:var(--text);border-radius:7px;padding:12px;font:15px/1.6 system-ui,sans-serif}.copy{margin-top:9px}.footer{font-size:14px;margin-top:25px}
@media(max-width:760px){main{padding:24px 12px 45px}section{padding:15px}.pair{grid-template-columns:1fr}h1{font-size:26px}}
</style></head><body><main><header><h1>完整视频盲评</h1>
<p>请看完每组的 A、B 完整视频，再回复「A」「B」或「难分」，并简述依据。A、B 为隐藏身份的候选，标签在不同组之间不保证对应同一来源。</p>
<p>每段 124 帧、1024 × 576、24 fps，保留原始完整声音。同步播放先静音；可切换只听其中一段。尚未记录任何人工评价。</p>
<p>可选的 BF16 参照折叠在每组下方。反馈仅留在本页文本框中，不会自动保存、发送或上传。</p></header>
''' + '\n'.join(sections) + '''
<p class="footer">完整性信息：<a href="review_manifest.json">中性素材清单与 SHA-256</a>。视频未重编码、裁切或自动评分。</p>
</main><script>
'use strict';
const videos = Array.from(document.querySelectorAll('video'));
const syncState = new Map();
const status = (id, text) => { document.getElementById(id + '-status').textContent = text; };
const pair = id => ['A', 'B'].map(label => document.getElementById(id + '-' + label));
function stopAll(){ videos.forEach(v => { v.pause(); v.muted = true; }); syncState.clear(); }
function ready(v){
  if(v.readyState >= 1) return Promise.resolve();
  return new Promise((resolve,reject) => {
    const clean = () => {v.removeEventListener('loadedmetadata',ok);v.removeEventListener('error',bad);};
    const ok = () => {clean();resolve();};
    const bad = () => {clean();reject(new Error('media'));};
    v.addEventListener('loadedmetadata',ok,{once:true});v.addEventListener('error',bad,{once:true});v.load();
  });
}
async function startPair(id, audible){
  const items = pair(id); const old = items[0].currentTime;
  stopAll(); status(id,'正在准备同步播放…');
  try {
    await Promise.all(items.map(ready));
    const duration = Math.min(...items.map(v => v.duration));
    const t = Number.isFinite(old) && old < duration - 0.05 ? old : 0;
    items.forEach(v => {v.currentTime=t;v.playbackRate=1;v.muted=v.dataset.label!==audible;});
    await Promise.all(items.map(v => v.play()));
    syncState.set(id,true);status(id,audible ? '同步播放，仅 ' + audible + ' 有声。' : '同步静音播放。');
  } catch(e){items.forEach(v=>v.pause());status(id,'无法自动播放，请使用视频自身的播放控件或下载完整视频。');}
}
videos.forEach(v => {
  v.addEventListener('volumechange',()=>{if(!v.muted)videos.forEach(other=>{if(other!==v)other.muted=true;});});
  v.addEventListener('error',()=>status(v.dataset.case,'某段视频未能加载；请核对完整性清单或下载文件。'));
});
document.querySelectorAll('button[data-action]').forEach(button=>button.addEventListener('click',async()=>{
  const id=button.dataset.case, action=button.dataset.action;
  if(action==='copy'){
    const field=document.getElementById(id+'-feedback');field.focus();field.select();
    try {await navigator.clipboard.writeText(field.value);status(id,'已复制；请自行粘贴回复。');}
    catch(e){status(id,'文本已选中，请手动复制后回复。');}return;
  }
  if(action==='sync'||action==='A'||action==='B'){await startPair(id,action==='sync'?null:action);return;}
  syncState.delete(id);
  videos.filter(v=>v.dataset.case===id).forEach(v=>{v.pause();if(action==='reset'&&v.readyState>=1)v.currentTime=0;});
  status(id,action==='reset'?'已复位到开头。':'已暂停。');
}));
function tick(){
  for(const id of syncState.keys()){
    const [a,b]=pair(id);
    if(a.ended){b.pause();syncState.delete(id);status(id,'完整播放结束。');continue;}
    if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>0.08)b.currentTime=a.currentTime;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
</script></body></html>
'''


def build(reports):
    reports = Path(reports).resolve()
    output = reports / 'blind_review'
    private = reports / 'private'
    key_path = private / 'blind_key.json'
    private_manifest = private / 'sources_manifest.json'
    require(not output.exists() and not private_manifest.exists(), 'Refusing to overwrite an existing review package.')
    require(private.is_dir() and not private.is_symlink() and stat.S_IMODE(private.stat().st_mode) == 0o700,
            'The private directory must have mode 0700.')
    require(key_path.is_file() and not key_path.is_symlink() and stat.S_IMODE(key_path.stat().st_mode) == 0o600,
            'The private commitment must have mode 0600.')
    independent = load(reports / 'independent.json', complete=True)
    bindings = independent.get('reports', {})
    names = ('check.json', 'launch.json', 'decode_restart.json', 'decode_carry.json')
    for name in names:
        require(name in bindings, 'Independent verification lacks a required source binding.')
        verify_record(bindings[name], reports / name)
    check = load(reports / 'check.json', complete=True)
    launch = load(reports / 'launch.json')
    key = load(key_path)
    require(sha256(key_path) == launch.get('blind_commitment_sha256') and
            key.get('created_before_generation') is True, 'The pre-generation blind commitment did not verify.')
    assignment = key.get('assignment', {})
    require(set(assignment) == {c[0] for c in CASES}, 'The blind case assignment is incomplete.')
    decoded = {arm: load(reports / f'decode_{arm}.json', complete=True) for arm in ARMS}
    references = {r['prompt_id']: r for r in check['bf16_references']}
    require(len(references) == len(check['bf16_references']) == 2, 'The reference case set changed.')
    rows = {}
    for arm, report in decoded.items():
        require(report.get('experiment') == 'E068' and report.get('arm') == arm and
                report.get('settings') == check['settings'], 'A complete source report has inconsistent identity or settings.')
        rows[arm] = {r['prompt_id']: r for r in report['cases']}
        require(len(report['cases']) == len(rows[arm]) == 2 and set(rows[arm]) == {30, 36}, 'The decoded case set changed.')
    public_cases, private_cases, source_names = [], [], []
    for cid, prompt_id, seed in CASES:
        choices = assignment[cid]
        require(set(choices) == {'A', 'B'} and set(choices.values()) == set(ARMS), 'A blind assignment is not a permutation.')
        sources = [rows[arm][prompt_id] for arm in ARMS]
        require(all(r.get('status') == 'complete' and r['seed'] == seed and r['settings'] == check['settings'] for r in sources),
                'A decoded case is incomplete or has inconsistent settings.')
        require(isinstance(sources[0]['prompt'], str) and sources[0]['prompt'] == sources[1]['prompt'], 'The original prompts differ.')
        case = dict(id=cid, prompt_id=prompt_id, seed=seed, prompt=sources[0]['prompt'], clips=[])
        private_case = dict(id=cid, sources=[])
        for label in ('A', 'B', 'reference'):
            if label == 'reference':
                source = references[prompt_id]
                require(source['seed'] == seed, 'A reference seed differs.')
                source_record = source['video']
                identity = 'bf16'
            else:
                identity = choices[label]
                source = rows[identity][prompt_id]
                source_record = record(source['video'])
                require(source_record['sha256'] == source['video_sha256'], 'A source video checksum differs.')
            path = verify_record(source_record)
            require(path.suffix.lower() == '.mp4' and source['media'] == MEDIA, 'A video has unexpected media metadata.')
            source_names.append(path.name)
            clip = dict(id=f'{cid}_{label}', file=f'{cid}_{label}.mp4',
                        source_sha256=source_record['sha256'], bytes=source_record['bytes'], media=dict(MEDIA))
            case['clips'].append(clip)
            private_case['sources'].append(dict(neutral_id=clip['id'], identity=identity, source=source_record))
        public_cases.append(case)
        private_cases.append(private_case)
    manifest = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                    cases=public_cases, settings=check['settings'],
                    integrity=dict(independent_verification_complete=True, source_sha256_verified=True,
                                   full_original_media=True, reencoded=False, trimmed=False, human_reviews_recorded=0))
    page = render_page(public_cases)
    neutral_guard(page + json.dumps(manifest, ensure_ascii=False), source_names)
    stage = Path(tempfile.mkdtemp(prefix='.blind_build_', dir=reports))
    wrote_private = False
    try:
        for case, private_case in zip(public_cases, private_cases, strict=True):
            for clip, source in zip(case['clips'], private_case['sources'], strict=True):
                target = stage / clip['file']
                try:
                    os.link(source['source']['file'], target)
                    source['materialization'] = 'hardlink'
                except OSError:
                    with open(source['source']['file'], 'rb') as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst, length=8 << 20)
                    source['materialization'] = 'copy'
                require(target.stat().st_size == clip['bytes'] and sha256(target) == clip['source_sha256'],
                        'A neutral media copy did not verify.')
        (stage / 'index.html').write_text(page, encoding='utf-8')
        write_json(stage / 'review_manifest.json', manifest)
        summary = dict(page=str(output / 'index.html'),
                       clips=[str(output / clip['file']) for case in public_cases for clip in case['clips']],
                       integrity=dict(manifest['integrity'], clips_verified=6,
                                      total_media_bytes=sum(clip['bytes'] for case in public_cases for clip in case['clips']),
                                      media=dict(MEDIA)))
        neutral_guard(json.dumps(summary, ensure_ascii=False), source_names)
        write_json(stage / 'build_summary.json', summary)
        write_json(private_manifest, dict(status='complete', builder=record(__file__),
            independent_report=record(reports / 'independent.json'), reports={name: bindings[name] for name in names},
            commitment=record(key_path), cases=private_cases,
            public_page_sha256=sha256(stage / 'index.html'),
            public_manifest_sha256=sha256(stage / 'review_manifest.json')), private=True)
        wrote_private = True
        require(not output.exists(), 'Refusing to overwrite an existing review package.')
        stage.rename(output)
        return summary
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if wrote_private:
            private_manifest.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    try:
        summary = build(args.report_dir)
    except BuildError as exc:
        print(f'Blind review not built: {exc}', file=sys.stderr)
        return 1
    except Exception as exc:
        # OS/JSON exceptions can contain original filenames or private values.
        print(f'Blind review not built: input or filesystem validation failed ({type(exc).__name__}).', file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
