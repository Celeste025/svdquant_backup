#!/usr/bin/env bash
# Durable, fail-fast execution queue for the two rCM ablations and H3 rank-64 VBench.
set -Eeuo pipefail

ROOT=/home/wjq/workspace/svdquant-exp
DATA=/data1/models/svdquant-wjq
GPU="${CUDA_VISIBLE_DEVICES:-5}"
QUEUE_ROOT="$ROOT/results/queues/rcm_h3_real_nvfp4_$(date +%Y%m%d_%H%M%S)"
LOG="$QUEUE_ROOT/queue.log"
H3_PY=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
# The historical VBench virtualenv's interpreter symlink targets the retired
# /data mount; its site-packages remain valid with the recovered PTQ runtime.
VBENCH_PY="$DATA/conda-envs/svdquant-ptq/bin/python"
VBENCH_SITE=/home/wjq/.venvs/vbench/lib/python3.12/site-packages
PTQ_PY="$DATA/conda-envs/svdquant-ptq/bin/python"
H3_STATE="$ROOT/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
H3_SOURCE="$ROOT/results/samples/minimax_h3_vbench51_seed0_calibshape"
H3_OUT="$ROOT/results/samples/minimax_h3_svdquant_standard_r64_vbench51_seed0_calibshape"
H3_EVAL="$ROOT/results/vbench/minimax_h3_svdquant_standard_r64_vbench51_seed0_calibshape/evaluation_8core"
DIMS=(aesthetic_quality scene imaging_quality overall_consistency background_consistency subject_consistency dynamic_degree motion_smoothness)

mkdir -p "$QUEUE_ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "queue started $(date -Is), gpu=$GPU"

write_state() {
  local stage="$1" state="$2" detail="${3:-}"
  "$H3_PY" - "$QUEUE_ROOT/$stage.json" "$stage" "$state" "$detail" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'stage':sys.argv[2],'state':sys.argv[3],'detail':sys.argv[4],'time':time.time()},indent=2)+'\n')
tmp.replace(p)
PY
}

on_error() {
  local rc=$?
  write_state "queue" "failed" "line $1 exited $rc" || true
  echo "queue failed at line $1 (exit $rc)" >&2
  exit "$rc"
}
trap 'on_error $LINENO' ERR

run_rcm_stage() {
  local grid="$1"
  local rank="$2"
  local tag="rcm-wan2.1-1.3b-real-nvfp4-s16-g${grid}-r${rank}"
  local ckpt="$DATA/ckpts/$tag"
  if [[ -f "$ckpt/manifest.json" ]] && "$PTQ_PY" - "$ckpt/manifest.json" "$rank" "$grid" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); s=x.get('svdquant',{})
raise SystemExit(0 if x.get('state')=='complete' and s.get('rank')==int(sys.argv[2]) and s.get('smooth_grids')==int(sys.argv[3]) else 1)
PY
  then
    echo "$tag already verified; skipping"
    write_state "$tag" "skipped" "existing verified manifest"
    return
  fi
  if [[ -e "$ckpt" ]]; then
    echo "partial or invalid checkpoint exists: $ckpt" >&2
    exit 1
  fi
  write_state "$tag" "running"
  CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/scripts/run_rcm_wan_real_nvfp4_svdquant.sh" "$grid" "$rank"
  # The PTQ runner verifies state tensors; this adds the required deterministic
  # four-step 480p/77f end-to-end smoke without decoding the VAE.
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT/third_party/deepcompressor" PATH="$DATA/conda-envs/svdquant-ptq/bin:$PATH" "$PTQ_PY" "$ROOT/scripts/infer_rcm_wan_4step.py" \
    --model "$DATA/models/Wan2.1-T2V-1.3B-Diffusers" --transformer "$DATA/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer" \
    --quant-ckpt "$ckpt" --prompt "A golden retriever runs across a sunny beach." --seed 0 --steps 4 \
    --height 480 --width 832 --frames 77 --skip-decode --latent-output "$QUEUE_ROOT/${tag}_smoke_latent.pt"
  write_state "$tag" "complete"
}

run_h3_stage() {
  test -f "$H3_STATE" && test -f "$H3_SOURCE/manifest.json"
  write_state "minimax_h3_r64_vbench" "running"
  export CUDA_VISIBLE_DEVICES="$GPU" DIFFSYNTH_SKIP_DOWNLOAD=True
  export PYTHONPATH="$ROOT/scripts:/home/wjq/workspace/DiffSynth-Studio"
  export TOKENIZERS_PARALLELISM=false
  "$H3_PY" "$ROOT/scripts/run_minimax_h3_vbench51.py" --gpu "$GPU" --output "$H3_OUT" --source "$H3_SOURCE" \
    --state "$H3_STATE" --variants svdquant --smoke
  "$H3_PY" "$ROOT/scripts/run_minimax_h3_vbench51.py" --gpu "$GPU" --output "$H3_OUT" --source "$H3_SOURCE" \
    --state "$H3_STATE" --variants svdquant
  "$H3_PY" - "$H3_OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); m=json.loads((out/'manifest.json').read_text())
for c in m['cases']:
    for v in ('bf16','w4a4','svdquant'):
        p=out/'cases'/c['case_id']/f'{v}.json'
        if not p.is_file() or json.loads(p.read_text()).get('state')!='complete':
            raise RuntimeError(f'incomplete {v}: {c["case_id"]}')
PY
  mkdir -p "$H3_EVAL"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$VBENCH_SITE" "$VBENCH_PY" "$ROOT/scripts/eval_rcm_vbench_subset.py" --selection "$H3_OUT/manifest.json" --generated-dir "$H3_OUT" --output-dir "$H3_EVAL" \
    --prompt-metadata /home/wjq/workspace/ViDiT-Q/eval/video/Vbench/vbench/VBench_full_info.json \
    --models bf16 w4a4 svdquant --dimensions "${DIMS[@]}"
  PYTHONPATH="$VBENCH_SITE" "$VBENCH_PY" - "$H3_EVAL" <<'PY'
import csv,sys
from pathlib import Path
rows=list(csv.DictReader((Path(sys.argv[1])/'summary.csv').open()))
expected={(m,d) for m in ('bf16','w4a4','svdquant') for d in ('aesthetic_quality','scene','imaging_quality','overall_consistency','background_consistency','subject_consistency','dynamic_degree','motion_smoothness')}
actual={(r['model'],r['dimension']) for r in rows}
if actual != expected: raise RuntimeError(f'VBench results incomplete: {expected-actual}')
PY
  write_state "minimax_h3_r64_vbench" "complete"
}

run_rcm_stage 20 32
run_rcm_stage 10 64
run_h3_stage
write_state "queue" "complete"
echo "queue complete $(date -Is)"
