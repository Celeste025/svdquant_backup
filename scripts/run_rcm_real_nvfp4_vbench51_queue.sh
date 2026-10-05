#!/usr/bin/env bash
# GPU-4, fail-fast queue: smoke then generate/evaluate both new rCM checkpoints.
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
DATA=/data1/models/svdquant-wjq
GPU="${CUDA_VISIBLE_DEVICES:-4}"
QUEUE="$ROOT/results/queues/rcm_real_nvfp4_vbench51_$(date +%Y%m%d_%H%M%S)"
PY="$DATA/conda-envs/svdquant-ptq/bin/python"
VBENCH_SITE=/home/wjq/.venvs/vbench/lib/python3.12/site-packages
DIMS=(aesthetic_quality scene imaging_quality overall_consistency background_consistency subject_consistency dynamic_degree motion_smoothness)
mkdir -p "$QUEUE"
exec > >(tee -a "$QUEUE/queue.log") 2>&1
state() { "$PY" - "$QUEUE/$1.json" "$1" "$2" "${3:-}" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps({'stage':sys.argv[2],'state':sys.argv[3],'detail':sys.argv[4],'time':time.time()},indent=2)+'\n'); t.replace(p)
PY
}
trap 'rc=$?; state queue failed "line $LINENO exited $rc" || true; exit $rc' ERR
run_one() {
  local tag="$1" rank="$2" grid="$3"
  local ckpt="$DATA/ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16-g${grid}-r${rank}"
  local out="$ROOT/results/samples/rcm_real_nvfp4_svdquant_${tag}_vbench51_seed0_480p77f_4step"
  local eval="$ROOT/results/vbench/rcm_real_nvfp4_svdquant_${tag}_vbench51_seed0_480p77f_4step"
  state "$tag" running
  "$PY" "$ROOT/scripts/run_rcm_real_nvfp4_vbench51.py" --gpu "$GPU" --output "$out" --checkpoint "$ckpt" --rank "$rank" --grid "$grid" --smoke
  "$PY" "$ROOT/scripts/run_rcm_real_nvfp4_vbench51.py" --gpu "$GPU" --output "$out" --checkpoint "$ckpt" --rank "$rank" --grid "$grid"
  "$PY" - "$out" <<'PY'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]); m=json.loads((o/'manifest.json').read_text())
for c in m['cases']:
 for v in ('bf16','nvfp4_svdquant'):
  p=o/'cases'/c['case_id']/f'{v}.json'
  if not p.is_file() or json.loads(p.read_text()).get('state')!='complete': raise RuntimeError(f'incomplete {v} {c["case_id"]}')
PY
  mkdir -p "$eval"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$VBENCH_SITE" "$PY" "$ROOT/scripts/eval_rcm_vbench_subset.py" --selection "$out/manifest.json" --generated-dir "$out" --output-dir "$eval" --prompt-metadata /home/wjq/workspace/ViDiT-Q/eval/video/Vbench/vbench/VBench_full_info.json --models bf16 nvfp4_svdquant --dimensions "${DIMS[@]}"
  "$PY" - "$eval/summary.csv" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]))); need={(m,d) for m in ('bf16','nvfp4_svdquant') for d in ('aesthetic_quality','scene','imaging_quality','overall_consistency','background_consistency','subject_consistency','dynamic_degree','motion_smoothness')}
if {(x['model'],x['dimension']) for x in r} != need: raise RuntimeError('incomplete VBench summary')
PY
  state "$tag" complete
}
echo "queue started $(date -Is), gpu=$GPU"
run_one g20_r32 32 20
run_one g10_r64 64 10
state queue complete
echo "queue complete $(date -Is)"
