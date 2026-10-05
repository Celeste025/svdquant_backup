#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
PY=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
VPY=/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python
GPU="${CUDA_VISIBLE_DEVICES:-6}"
OUT="$ROOT/results/samples/minimax_h3_convrot_vbench51_seed0_calibshape"
SRC="$ROOT/results/samples/minimax_h3_vbench51_seed0_calibshape"
STATE="$ROOT/results/checkpoints/minimax_h3_convrot_standard_8p64s/quant_state.pt"
EVAL="$ROOT/results/vbench/minimax_h3_convrot_vbench51_seed0_calibshape/evaluation_8core"
SITE=/home/wjq/.venvs/vbench/lib/python3.12/site-packages
QUEUE="$ROOT/results/queues/minimax_h3_convrot_vbench51_$(date +%Y%m%d_%H%M%S)"
DIMS=(aesthetic_quality scene imaging_quality overall_consistency background_consistency subject_consistency dynamic_degree motion_smoothness)
mkdir -p "$QUEUE"; exec > >(tee -a "$QUEUE/queue.log") 2>&1
state() { "$PY" - "$QUEUE/$1.json" "$1" "$2" "${3:-}" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps({'stage':sys.argv[2],'state':sys.argv[3],'detail':sys.argv[4],'time':time.time()},indent=2)+'\n'); t.replace(p)
PY
}
trap 'rc=$?; state queue failed "line $LINENO exited $rc" || true; exit $rc' ERR
echo "queue started $(date -Is), gpu=$GPU"
state smoke running
"$PY" "$ROOT/scripts/run_minimax_h3_convrot_vbench51.py" --gpu "$GPU" --output "$OUT" --source "$SRC" --state "$STATE" --smoke
"$PY" - "$OUT/cases/vbench_000/convrot.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x.get('state')=='complete' and x.get('frames')==124 and x.get('steps')==20
PY
state smoke complete; state generation running
"$PY" "$ROOT/scripts/run_minimax_h3_convrot_vbench51.py" --gpu "$GPU" --output "$OUT" --source "$SRC" --state "$STATE"
"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]); m=json.loads((o/'manifest.json').read_text())
for c in m['cases']:
 for v in ('bf16','w4a4','convrot'):
  p=o/'cases'/c['case_id']/f'{v}.json'
  if not p.is_file() or json.loads(p.read_text()).get('state')!='complete': raise RuntimeError(f'incomplete {v} {c["case_id"]}')
PY
state generation complete; state evaluation running
mkdir -p "$EVAL"
CUDA_VISIBLE_DEVICES="$GPU" HF_HOME=/home/wjq/.cache/huggingface HUGGINGFACE_HUB_CACHE=/home/wjq/.cache/huggingface/hub TRANSFORMERS_OFFLINE=1 PYTHONPATH="$SITE" "$VPY" "$ROOT/scripts/eval_rcm_vbench_subset.py" --selection "$OUT/manifest.json" --generated-dir "$OUT" --output-dir "$EVAL" --prompt-metadata /home/wjq/workspace/ViDiT-Q/eval/video/Vbench/vbench/VBench_full_info.json --models bf16 w4a4 convrot --dimensions "${DIMS[@]}"
"$VPY" - "$EVAL/summary.csv" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]))); need={(m,d) for m in ('bf16','w4a4','convrot') for d in ('aesthetic_quality','scene','imaging_quality','overall_consistency','background_consistency','subject_consistency','dynamic_degree','motion_smoothness')}
if {(x['model'],x['dimension']) for x in r} != need: raise RuntimeError('incomplete VBench summary')
PY
state evaluation complete; state queue complete
echo "queue complete $(date -Is)"
