#!/usr/bin/env bash
# Generate the two historical standard prompts with the rank-64/grid-10 state.
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
PY=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
GPU="${CUDA_VISIBLE_DEVICES:-2}"
STATE="$ROOT/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
OUT="$ROOT/results/samples/minimax_h3_svdquant_standard_r64_8p64s"
QUEUE="$ROOT/results/queues/minimax_h3_r64_p16_p26_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$QUEUE"
exec > >(tee -a "$QUEUE/queue.log") 2>&1
state() { "$PY" - "$QUEUE/$1.json" "$1" "$2" "${3:-}" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps({'stage':sys.argv[2],'state':sys.argv[3],'detail':sys.argv[4],'time':time.time()},indent=2)+'\n'); t.replace(p)
PY
}
trap 'rc=$?; state queue failed "line $LINENO exited $rc" || true; exit $rc' ERR
"$PY" - "$STATE" <<'PY'
import sys,torch
s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); c=s.get('config',{})
if s.get('format') != 'minimax-h3-svdquant-standard-v1' or len(s.get('layers',{})) != 200 or c.get('rank') != 64 or c.get('num_grids') != 10:
 raise RuntimeError(f'invalid expected rank64/grid10 state: {c}')
PY
run_case() {
  local pid="$1" seed="$2" stage="p$1"
  local dir="$OUT/$stage"
  local video="$dir/svdquant_id${pid}_seed${seed}_576x1024_124f_20steps.mp4"
  local status="$dir/svdquant.json"
  if [[ -f "$video" && -f "$status" ]] && "$PY" - "$status" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get('prompt_id') in (16,26) and x.get('frames')==124 and x.get('steps')==20 else 1)
PY
  then state "$stage" skipped "existing video"; return; fi
  state "$stage" running
  CUDA_VISIBLE_DEVICES="$GPU" DIFFSYNTH_SKIP_DOWNLOAD=True PYTHONPATH="$ROOT/scripts:/home/wjq/workspace/DiffSynth-Studio" TOKENIZERS_PARALLELISM=false \
    "$PY" "$ROOT/scripts/infer_minimax_h3_svdquant_standard.py" --case svdquant --state "$STATE" --output-dir "$dir" --prompt-id "$pid" --height 576 --width 1024 --frames 124 --steps 20
  "$PY" - "$dir/run_metadata.json" "$STATE" "$pid" "$seed" "$video" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); p.write_text(json.dumps({'state':sys.argv[2],'rank':64,'grid':10,'prompt_id':int(sys.argv[3]),'seed':int(sys.argv[4]),'settings':{'height':576,'width':1024,'frames':124,'steps':20,'cfg_scale':1.0,'tiled':True,'fps':24},'video':sys.argv[5],'completed_at':time.time()},indent=2)+'\n')
PY
  state "$stage" complete
}
echo "queue started $(date -Is), gpu=$GPU"
run_case 16 20026
run_case 26 63583
state queue complete
echo "queue complete $(date -Is)"
