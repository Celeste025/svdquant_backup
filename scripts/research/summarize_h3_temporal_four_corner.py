#!/usr/bin/env python3
"""E057 CPU exact finite-difference attribution of two actual H3 updates."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from analyze_h3_adjacent_error_v2 import same

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E057'
OLD = ROOT/'results/research/E015'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(rec):
    path = Path(rec['file'])
    assert sha(path) == rec['sha256'] and path.stat().st_size == rec['bytes'], path
    return torch.load(path, map_location='cpu', weights_only=False)


def measures(values, centered):
    base = values['BB'].double()
    f = values['QB'].double()-base
    l = values['BQ'].double()-base
    total = values['QQ'].double()-base
    interaction = total-f-l
    if centered:
        axes = tuple(range(2, base.ndim))
        def center(x):
            return x-x.mean(axes, keepdim=True)
        f, l, total, interaction = map(center, (f,l,total,interaction))
    error = float((total-f-l-interaction).abs().max())
    assert error < 1e-12
    ff, ll, ii = (float(x.square().sum()) for x in (f,l,interaction))
    fl, fi, li = (2*float((a*b).sum()) for a,b in ((f,l),(f,interaction),(l,interaction)))
    tt = float(total.square().sum())
    assert abs(tt-(ff+ll+ii+fl+fi+li)) < 1e-9*max(tt,1)
    # Actual second-step quantization increment at the shifted input is C=L+I.
    c = l+interaction
    cc, fc = float(c.square().sum()), 2*float((f*c).sum())
    assert abs(tt-(ff+cc+fc)) < 1e-9*max(tt,1)
    return dict(first_propagated_energy=ff, second_only_energy=ll, interaction_energy=ii,
                total_energy=tt, cross_first_second=fl, cross_first_interaction=fi,
                cross_second_interaction=li,
                first_second_cosine=fl/(2*(ff*ll)**0.5),
                cross_first_second_over_diagonal=fl/(ff+ll),
                total_over_diagonal=tt/(ff+ll),
                interaction_norm_over_additive_norm=(ii/float((f+l).square().sum()))**0.5,
                shifted_second_energy=cc, cross_first_shifted_second=fc,
                first_shifted_second_cosine=fc/(2*(ff*cc)**0.5),
                cross_first_shifted_second_over_diagonal=fc/(ff+cc),
                vector_identity_max_abs=error)


def main():
    out = RD/'summary.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    start = time.monotonic()
    new_path = RD/'evaluate_v2.json'
    new = json.loads(new_path.read_text())
    assert new['status'] == 'complete' and new['complete_dit_calls'] == 5
    old = {a: json.loads((OLD/f'evaluate_{a}.json').read_text()) for a in ('bf16','plain','svd')}
    assert all(x['status'] == 'complete' for x in old.values())
    rows, sources = [], []
    assert len(new['cases']) == 4
    for row in new['cases']:
        cid, arm = row['id'], row['source_arm']
        assert arm in ('plain','svd')
        recs = dict(QB=row['artifact'])
        for key, model, modal_corner in [('BB','bf16','BB'),('BQ',arm,'BB'),('QQ',arm,'QQ')]:
            recs[key] = next(c['artifact'] for c in old[model]['cases']
                             if c['id'] == cid and c['corner'] == modal_corner)
        payloads = {k:load(r) for k,r in recs.items()}
        assert payloads['QB']['model_arm'] == 'bf16' and payloads['QB']['temporal_corner'] == 'QB'
        assert same(payloads['QB']['actual_dit_inputs'], payloads['QQ']['actual_dit_inputs'])
        assert same(payloads['BB']['actual_dit_inputs'], payloads['BQ']['actual_dit_inputs'])
        modalities = {}
        for modality in ('video','audio'):
            values = {k:p['next_endpoints'][modality] for k,p in payloads.items()}
            assert all(v.shape == values['BB'].shape and v.dtype == torch.bfloat16 and torch.isfinite(v).all()
                       for v in values.values())
            modalities[modality] = dict(raw=measures(values,False), channel_centered=measures(values,True))
        rows.append(dict(case_id=cid, source_arm=arm, modalities=modalities))
        sources.append(dict(case_id=cid, source_arm=arm, time_axis_corners=recs))
    assert {(r['case_id'], r['source_arm']) for r in rows} == {
        (c,a) for c in ('e010_p030_s05','e010_p036_s14') for a in ('plain','svd')}
    result = dict(experiment='E057', status='complete', seconds=time.monotonic()-start,
                  cuda_initialized=torch.cuda.is_initialized(), script_sha256=sha(__file__),
                  evaluation_sha256=sha(new_path), cases=rows, sources=sources,
                  corner_order='time axis: step n model, step n+1 model; not E015 modality corners',
                  limitation='Two-window actual BF16 update attribution; no video/long-horizon claim')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'rows': [
        dict(case=r['case_id'], arm=r['source_arm'], modality=m, **v['raw'])
        for r in rows for m,v in r['modalities'].items()]},indent=2))


if __name__ == '__main__':
    main()
