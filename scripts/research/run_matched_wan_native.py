#!/usr/bin/env python3
"""E048: thin matched-checkpoint wrapper around frozen E044 v2 native generation."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import run_vanilla_wan_native_v2 as base

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E048_matched_wan_native_manifest.json'
PLAN = ROOT/'research_state/06_experiments/E048_matched_wan_native_plan.md'


def require_completed_ptq(m):
    """Read small completed receipts before invoking the existing weight checks."""
    ptq_path = Path(m['ptq_run'])
    ptq = json.loads(ptq_path.read_text())
    base.require(ptq['status'] == 'complete' and ptq['experiment'] == 'E047', 'E047 PTQ must be complete')
    identity_path = Path(m['checkpoint_identity'])
    identity_record = base.record(identity_path)
    declared = ptq['checkpoint_identity']
    base.require(all(declared[k] == identity_record[k] for k in ('file', 'bytes', 'sha256')),
                 'E047 PTQ checkpoint identity binding differs')
    identity = json.loads(identity_path.read_text())
    base.require(identity['status'] == 'complete' and identity['experiment'] == 'E047' and
                 identity['model_dir'] == m['model_dir'], 'E047 identity/model differs')
    base.require({Path(f['file']).name for f in identity['files']} ==
                 {'model.pt', 'scale.pt', 'wgts.pt', 'branch.pt', 'smooth.pt'}, 'Five checkpoint files required')
    base.require(all(Path(f['file']).parent == Path(m['svd_checkpoint']) for f in identity['files']),
                 'Checkpoint is outside fixed E047 directory')
    return base.record(ptq_path), identity_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--arm', choices=('svdquant_nvfp4',), required=True)
    parser.add_argument('--prompt-ids', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    default_name = ('check_svdquant_nvfp4.json' if args.check_only else
                    f'worker_{args.arm}_{"-".join(map(str,args.prompt_ids))}.json')
    args.output = args.output or Path(m['report_dir'])/default_name
    base.require(not args.output.exists(), 'Preserve previous result')
    report = dict(experiment='E048', arm=args.arm, status='running', phase='check' if args.check_only else 'generate',
                  cases=[], actual_counts=dict(dit=0, native_gemm=0, dit_sdpa=0, scheduler=0,
                                             public_vae_decode=0, text_encoder=0, fastpack_checks=0))
    start = time.monotonic()
    try:
        ptq, identity = require_completed_ptq(m)
        report.update(matched_ptq_run=ptq, matched_checkpoint_identity=identity)
        base.torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        m, cases, prepared = base.prerequisites(args, report)
        report['sources'].extend(base.record(p) for p in (Path(__file__), PLAN))
        report['inherited_comparisons'] = m['inherited_comparisons']
        if args.check_only:
            base.require(not base.torch.cuda.is_available() and not base.torch.cuda.is_initialized(),
                         'CPU check requires CUDA hidden')
            report.update(status='complete', cuda_available=False, cuda_initialized=False,
                          transformer_model_weights_loaded=False, checkpoint_metadata_loaded=True,
                          prepared_case_count=len(prepared))
        else:
            base.run(args, m, cases, prepared, report)
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-start
        base.save_json(report, args.output)
        print(json.dumps(dict(status=report['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
