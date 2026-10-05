#!/usr/bin/env python3
"""Pre-result E019 auxiliary statistic: fixed modality mass, no extra arm."""
import json
import os
from pathlib import Path
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import summarize_h3_plain_baseline as util
ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E019'

def main():
    output=RD/'partition_modality_mass.json'
    util.require(not output.exists(),'Refusing overwrite')
    started=time.monotonic(); torch.set_num_threads(6); files=util.Files()
    audit,reference=files.json(RD/'partition_math_audit.json')
    report=dict(status='running',source_audit=reference,source_sha256=util.file_sha(__file__),
        methods=dict(dtype='FP64 CPU',rows='head_ids-major, query_ids-minor',
            ranges=dict(text=[0,813],audio=[813,1227],video=[1227,22539]),
            absolute_mass_bias='abs(sum_region(operator weights)-sum_region(same-score true softmax))',
            scope='Auxiliary predeclared statistic, not a new operator; no quality or modality-specific causal claim'),cases=[])
    for case in audit['cases']:
        files.record(case['matrices']); data=util.load_tensor_file(case['matrices']['file'])
        row=dict(block=case['block'],head_ids=case['head_ids'],query_ids=case['query_ids'],modalities={})
        for name,(start,end) in report['methods']['ranges'].items():
            target=data['true_weights'][:,start:end].sum(-1); arms={}
            for arm in case['arms']:
                masses={p:data[arm]['weights'][p][:,start:end].sum(-1) for p in ('full','split')}
                arms[arm]={p:dict(mass=m.tolist(),signed_mass_bias=(m-target).tolist(),
                    absolute_mass_bias=(m-target).abs().tolist(),
                    mean_absolute_mass_bias=float((m-target).abs().mean()),max_absolute_mass_bias=float((m-target).abs().max()))
                    for p,m in masses.items()}
                delta=masses['split']-masses['full']
                arms[arm]['split_minus_full']=dict(signed=delta.tolist(),mean_absolute=float(delta.abs().mean()),
                                                 max_absolute=float(delta.abs().max()))
            row['modalities'][name]=dict(true_mass=target.tolist(),arms=arms)
        report['cases'].append(row)
    util.require(not torch.cuda.is_initialized(),'CUDA initialized')
    report.update(status='complete',seconds_total=time.monotonic()-started,cuda_initialized=False,files=files.checked)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['status'],report['seconds_total'])

if __name__=='__main__':main()
