#!/usr/bin/env python3
"""E038 CPU readout: same-prompt/seed attention comparisons, all 24 videos.

No model forward or composite metric. --contacts-only works before scoring.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from evaluate_h3_center_video_temporal import (
    ROOT, ARMS, DEFAULT_MANIFEST, IDENTITY_KEYS, manifest_rows, file_record, save, sha,
)
from summarize_wan_qad_videos import ASPECTS, CRITERIA, require, finite, close, mean, difference

MJ_INDICES = [0, 15, 30, 46, 61, 76, 92, 107]
CONTACT_INDICES = [0, 18, 35, 53, 70, 88, 105, 123]
PAIRS = [('global_mean', 'bf16'), ('coarse16', 'bf16'), ('coarse16', 'global_mean')]
KEYS = ('metrics', 'mj_aspects', 'mj_criteria')


def average(rows):
    require(all(set(r) == set(rows[0]) for r in rows), 'Scalar keys')
    return {k: mean([r[k] for r in rows]) for k in rows[0]}


def paired(arms):
    return {f'{a}_minus_{b}': {k: difference(arms[a][k], arms[b][k]) for k in KEYS} for a, b in PAIRS}


def recompute(t, m):
    errors = t['amt']['interpolation_reconstruction_absdiff']
    flows = t['raft']['top5percent_mean_flow_magnitudes']
    require(len(errors) == 61 and len(flows) == 41 and all(finite(v) for v in errors + flows), 'Original AMT/RAFT arrays')
    smooth = 1 - mean(errors) / 255
    close(smooth, t['scores']['motion_smoothness'], 'AMT normalization', atol=1e-7)
    require(t['amt']['input_even_frames'] == 62 and t['amt']['reference_odd_frames'] == 61
            and t['amt']['unscored_final_odd_frame'] == 123, 'AMT even-length media support')
    raft = t['raft']
    require(raft['sampled_frames'] == 42 and raft['actual_flows'] == 41, 'RAFT original stride 3')
    require(raft['threshold'] == 13.5 and raft['required_count'] == 10, 'Original RAFT formula')
    count = sum(v > 13.5 for v in flows)
    dynamic = count >= 10
    require(count == raft['above_threshold_count'] and isinstance(t['scores']['dynamic_degree'], bool)
            and t['scores']['dynamic_degree'] == dynamic, 'RAFT original decision')
    dino = t['dino']
    require(dino['frame_pairs'] == 123 and finite(dino['original_video_results_sum']), 'DINO saved sum')
    consistency = dino['original_video_results_sum'] / 123
    close(consistency, t['scores']['subject_consistency'], 'DINO normalization')
    require(set(m['criteria']) == set(CRITERIA) and set(m['aspects']) == set(ASPECTS), 'MJ 28 criteria / 5 aspects')
    require(all(finite(v) for v in [m['score'], *m['criteria'].values(), *m['aspects'].values()]), 'MJ finite outputs')
    require(m['sampled_indices'] == [MJ_INDICES] and m['patches'] == [1] * 8, 'Actual MJ frame indices')
    return dict(metrics=dict(mj_total=m['score'], mj_alignment=m['aspects']['alignment'],
                    mj_fineness=m['aspects']['fineness'], mj_coherence_consistency=m['aspects']['coherence_consistency'],
                    motion_smoothness=smooth, dynamic_degree=dynamic, subject_consistency=consistency,
                    amt_mean_absdiff_0_255=mean(errors), raft_mean_top5percent_flow=mean(flows),
                    raft_above_threshold_count=count),
                mj_aspects=m['aspects'], mj_criteria=m['criteria'], mj_sampled_indices=m['sampled_indices'],
                temporal_raw=dict(amt=errors, raft=flows, threshold=13.5, required_count=10,
                                  dino_saved_sum=dino['original_video_results_sum'], dino_terms=123))


def contacts(args, bound):
    import cv2
    from PIL import Image, ImageDraw
    manifest = json.loads(args.manifest.read_text())
    outdir = args.contact_dir or Path(manifest['data_dir']) / 'contacts'
    outdir.mkdir(parents=True, exist_ok=True)
    index = {(r['case_id'], r['arm']): r for r in bound}
    result = dict(experiment='E038', status='running', fixed_frame_indices=CONTACT_INDICES, sheets=[],
        limitation='Fixed sampled-frame researcher inspection, not full-video or blinded human evaluation. Audio unassessed.')
    for case in manifest['cases']:
        output = outdir / (case['case_id'] + '.png')
        require(not output.exists(), f'Preserve existing contact sheet: {output}')
        canvas = Image.new('RGB', (8 * 256, 70 + 3 * 176), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{case['case_id']} | seed {case['seed']} | {case['prompt']}", fill='black')
        for column, frame in enumerate(CONTACT_INDICES):
            draw.text((column * 256 + 8, 45), f'frame {frame}', fill='black')
        bindings = {}
        for row, arm in enumerate(ARMS):
            source = index[case['case_id'], arm]; bindings[arm] = source['video']
            cap = cv2.VideoCapture(source['video']['file']); require(cap.isOpened(), 'Video decode')
            selected, frame_id = {}, 0
            try:
                while True:
                    ok, image = cap.read()
                    if not ok: break
                    require(image.shape == (576, 1024, 3), 'Contact source geometry')
                    if frame_id in CONTACT_INDICES:
                        selected[frame_id] = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((256, 144))
                    frame_id += 1
            finally: cap.release()
            require(frame_id == 124 and set(selected) == set(CONTACT_INDICES), 'All source frames present')
            y = 70 + row * 176
            draw.text((8, y), f'{arm} (same native SVD linear weights)', fill='black')
            for column, frame in enumerate(CONTACT_INDICES): canvas.paste(selected[frame], (column * 256, y + 24))
        canvas.save(output)
        result['sheets'].append(dict(case_id=case['case_id'], videos=bindings, image=file_record(output)))
    result['status'] = 'complete'
    return result


def summarize(args, bound, decodes):
    temporal_path = args.temporal or args.report_dir / 'temporal_scores.json'
    mj_path = args.mjvideo or args.report_dir / 'mjvideo_scores.json'
    t = json.loads(temporal_path.read_text()); m = json.loads(mj_path.read_text())
    require(t['status'] == 'complete' and m['num_segments'] == 8, 'Scoring incomplete')
    require(t['manifest'] == file_record(args.manifest) and t['decode_reports'] == decodes, 'Bound source reports')
    for source in t['sources']:
        require(file_record(source['file']) == source, 'Executed temporal source changed')
    require(m['runtime']['evaluator_sha256'] == sha(Path(__file__).with_name('eval_mjvideo_e021.py')), 'MJ source binding')
    manifest = json.loads(args.manifest.read_text()); cases = manifest['cases']
    tids = [c['case_id'] for c in cases]
    temporal = {(r['case_id'], r['arm']): r for r in t['rows']}
    require(len(t['rows']) == len(temporal) == 24 and set(m['results']) == set(tids), 'Full 24-video scoring coverage')
    require(t['sampling'] == dict(amt_input_frames=list(range(0,124,2)), amt_reference_frames=list(range(1,122,2)),
                raft_frames=list(range(0,124,3)), raft_iterations=20, dino_frames=list(range(124))), 'Original metric sampling')
    all_rows = {}
    for row in bound:
        key = (row['case_id'], row['arm']); tr = temporal[key]
        require(all(row[k] == tr[k] for k in IDENTITY_KEYS) and row['video'] == tr['video'], 'Temporal identity/video')
        require(tr['media'] == dict(decoded_frames=124, declared_frames=124.0, fps=24.0, width=1024, height=576), 'Verified full media')
        mr = m['results'][row['case_id']]
        require(mr['prompt'] == row['prompt'] and set(mr['variants']) == set(ARMS), 'MJ prompt/arm coverage')
        mr = mr['variants'][row['arm']]
        require(mr['video'] == row['video']['file'] and mr['video_sha256'] == row['video']['sha256'], 'MJ media binding')
        all_rows[key] = dict(**{k: row[k] for k in IDENTITY_KEYS}, arm=row['arm'], video=row['video'], **recompute(tr, mr))
    for batch in t['dino_batches']:
        require(batch['batching_only_no_cross_video_similarity'], 'Within-video DINO semantics')
        values = [r['metrics']['subject_consistency'] for r in all_rows.values() if r['case_id']+'_'+r['arm'] in batch['row_ids']]
        require(len(values) == 3, 'DINO batch full three arms')
        close(mean(values), batch['original_batch_score'], 'DINO batch aggregation')
    result = dict(experiment='E038', status='running', cpu_only=True, per_case={}, per_prompt={}, arm_means={},
        limits=['Four purposively selected prompts x two seeds form a diagnostic set, not population inference or quality equivalence.',
            'All attention variants use the same native SVD linears; bf16 here names attention precision, not a BF16 whole model.',
            'MJ learned total is retained, never replaced by a mean of criteria. Eight frames may miss brief temporal defects.',
            'AMT can reward static or blurred clips; RAFT can count noise as movement; DINO within-video consistency is not semantic correctness.',
            'DINO normalization verifies saved sums; individual per-frame similarities are not saved and are not independently reconstructed.',
            'Audio is unassessed; no cross-seed similarity, composite quality index, confidence/significance claim, or result-based filtering.'],
        aggregation='Paired differences at identical prompt+seed; average the two seeds per prompt, then equally average four prompts.')
    for case in cases:
        arms = {arm: all_rows[case['case_id'], arm] for arm in ARMS}
        result['per_case'][case['case_id']] = dict(**case, arms=arms, paired_differences=paired(arms))
    pids = list(dict.fromkeys(c['prompt_id'] for c in cases))
    for pid in pids:
        selected = [c for c in cases if c['prompt_id'] == pid]
        require(len(selected) == 2 and [c['replica'] for c in selected] == [0,1], 'Fixed two seeds')
        arms = {a: {k: average([all_rows[c['case_id'],a][k] for c in selected]) for k in KEYS} for a in ARMS}
        result['per_prompt'][pid] = dict(prompt=selected[0]['prompt'], seeds=[c['seed'] for c in selected],
                case_ids=[c['case_id'] for c in selected], arms=arms, paired_differences=paired(arms))
    result['arm_means'] = {a: {k: average([p['arms'][a][k] for p in result['per_prompt'].values()]) for k in KEYS} for a in ARMS}
    result['paired_mean_differences'] = paired(result['arm_means'])
    result['paired_dynamic_transitions'] = {}
    for a,b in PAIRS:
        result['paired_dynamic_transitions'][f'{a}_versus_{b}'] = {
            f'{old}_to_{new}': [cid for cid in tids if bool(all_rows[cid,b]['metrics']['dynamic_degree']) == old
                              and bool(all_rows[cid,a]['metrics']['dynamic_degree']) == new]
            for old in (False,True) for new in (False,True)}
    result.update(status='complete', input_reports=dict(manifest=file_record(args.manifest), decode_reports=decodes,
            temporal=file_record(temporal_path), mjvideo=file_record(mj_path)), sources=[file_record(__file__),
            file_record(Path(__file__).with_name('evaluate_h3_center_video_temporal.py')),
            file_record(Path(__file__).with_name('summarize_wan_qad_videos.py'))],
            verification=dict(videos=24,prompts=4,seeds_per_prompt=2,amt_terms=61,raft_terms=41,dino_terms=123,
                              mj_actual_indices=MJ_INDICES,full_media_sha_checked=True))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--report-dir',type=Path,default=ROOT/'results/research/E038')
    p.add_argument('--temporal',type=Path);p.add_argument('--mjvideo',type=Path)
    p.add_argument('--contacts-only',action='store_true');p.add_argument('--contact-dir',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU invocation must hide CUDA')
    output=args.output or args.report_dir/('contact_sheets.json' if args.contacts_only else 'evaluation_summary.json')
    require(not output.exists(),'Preserve existing report; use --output for another attempt')
    result=dict(experiment='E038',status='failed_stop',source=file_record(__file__))
    try:
        bound,decodes=manifest_rows(args.manifest,args.report_dir)
        result=contacts(args,bound) if args.contacts_only else summarize(args,bound,decodes)
        require('torch' not in sys.modules,'No torch/model import in CPU readout')
    except BaseException:
        result['error']=traceback.format_exc();raise
    finally:
        save(output,result);print(json.dumps(dict(status=result['status'],output=str(output),sha256=sha(output))))


if __name__=='__main__':main()
