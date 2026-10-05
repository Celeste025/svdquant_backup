#!/usr/bin/env bash
# Resume only the post-generation eight-dimension VBench evaluation for H3 R64.
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
DATA=/data1/models/svdquant-wjq
GPU="${CUDA_VISIBLE_DEVICES:-5}"
PY="$DATA/conda-envs/svdquant-ptq/bin/python"
VBENCH_SITE=/home/wjq/.venvs/vbench/lib/python3.12/site-packages
OUT="$ROOT/results/samples/minimax_h3_svdquant_standard_r64_vbench51_seed0_calibshape"
EVAL="$ROOT/results/vbench/minimax_h3_svdquant_standard_r64_vbench51_seed0_calibshape/evaluation_8core"
DIMS=(aesthetic_quality scene imaging_quality overall_consistency background_consistency subject_consistency dynamic_degree motion_smoothness)
QUEUE="$ROOT/results/queues/minimax_h3_r64_vbench_eval_resume_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$QUEUE"
exec > >(tee -a "$QUEUE/queue.log") 2>&1
state() { "$PY" - "$QUEUE/$1.json" "$1" "$2" "${3:-}" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps({'stage':sys.argv[2],'state':sys.argv[3],'detail':sys.argv[4],'time':time.time()},indent=2)+'\n'); t.replace(p)
PY
}
trap 'rc=$?; state evaluation failed "line $LINENO exited $rc" || true; exit $rc' ERR
"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]); m=json.loads((o/'manifest.json').read_text())
for c in m['cases']:
 for v in ('bf16','w4a4','svdquant'):
  p=o/'cases'/c['case_id']/f'{v}.json'
  if not p.is_file() or json.loads(p.read_text()).get('state')!='complete': raise RuntimeError(f'incomplete {v} {c["case_id"]}')
PY
state evaluation running
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$VBENCH_SITE" "$PY" "$ROOT/scripts/eval_rcm_vbench_subset.py" --selection "$OUT/manifest.json" --generated-dir "$OUT" --output-dir "$EVAL" --prompt-metadata /home/wjq/workspace/ViDiT-Q/eval/video/Vbench/vbench/VBench_full_info.json --models bf16 w4a4 svdquant --dimensions "${DIMS[@]}"
"$PY" - "$EVAL/summary.csv" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]))); need={(m,d) for m in ('bf16','w4a4','svdquant') for d in ('aesthetic_quality','scene','imaging_quality','overall_consistency','background_consistency','subject_consistency','dynamic_degree','motion_smoothness')}
if {(x['model'],x['dimension']) for x in r} != need: raise RuntimeError('incomplete VBench summary')
PY
state evaluation complete
echo "evaluation complete $(date -Is)"
