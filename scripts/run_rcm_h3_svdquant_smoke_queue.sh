#!/usr/bin/env bash
# Fail-fast preflight for all three planned experiment stages.
set -Eeuo pipefail
ROOT=/home/wjq/workspace/svdquant-exp
DATA=/data1/models/svdquant-wjq
GPU="${CUDA_VISIBLE_DEVICES:-5}"
PTQ_PY="$DATA/conda-envs/svdquant-ptq/bin/python"
H3_PY=/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python
VBENCH_SITE=/home/wjq/.venvs/vbench/lib/python3.12/site-packages
QUEUE="$ROOT/results/queues/rcm_h3_smoke_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$QUEUE"
exec > >(tee -a "$QUEUE/queue.log") 2>&1

state() { "$H3_PY" - "$QUEUE/$1.json" "$1" "$2" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps({'stage':sys.argv[2],'state':sys.argv[3],'time':time.time()},indent=2)+'\n'); t.replace(p)
PY
}
trap 'state queue failed' ERR

smoke_rcm() {
  local grid="$1"
  local rank="$2"
  local tag="rcm-wan2.1-1.3b-real-nvfp4-s16-g${grid}-r${rank}-smoke"
  local ckpt="$DATA/ckpts/$tag"
  state "$tag" running
  CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/scripts/run_rcm_wan_real_nvfp4_smoke.sh" "$grid" "$rank"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT/third_party/deepcompressor" PATH="$DATA/conda-envs/svdquant-ptq/bin:$PATH" "$PTQ_PY" "$ROOT/scripts/infer_rcm_wan_4step.py" \
    --model "$DATA/models/Wan2.1-T2V-1.3B-Diffusers" --transformer "$DATA/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer" --quant-ckpt "$ckpt" \
    --prompt "A golden retriever runs across a sunny beach." --seed 0 --steps 4 --height 480 --width 832 --frames 77 --skip-decode \
    --latent-output "$QUEUE/${tag}_latent.pt"
  state "$tag" complete
}

smoke_h3() {
  local out="$ROOT/results/samples/minimax_h3_svdquant_standard_r64_vbench51_seed0_calibshape_smoke"
  local src="$ROOT/results/samples/minimax_h3_vbench51_seed0_calibshape"
  local ckpt="$ROOT/results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"
  local eval="$ROOT/results/vbench/minimax_h3_svdquant_standard_r64_vbench51_seed0_calibshape_smoke"
  state minimax_h3_r64_smoke running
  test ! -e "$out"
  CUDA_VISIBLE_DEVICES="$GPU" DIFFSYNTH_SKIP_DOWNLOAD=True PYTHONPATH="$ROOT/scripts:/home/wjq/workspace/DiffSynth-Studio" TOKENIZERS_PARALLELISM=false \
    "$H3_PY" "$ROOT/scripts/run_minimax_h3_vbench51.py" --gpu "$GPU" --output "$out" --source "$src" --state "$ckpt" --variants svdquant --smoke
  "$H3_PY" - "$out" "$QUEUE/h3_manifest_smoke.json" <<'PY'
import json,sys
from pathlib import Path
x=json.loads((Path(sys.argv[1])/'manifest.json').read_text()); x['cases']=x['cases'][:1]
Path(sys.argv[2]).write_text(json.dumps(x,indent=2)+'\n')
PY
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$VBENCH_SITE" "$PTQ_PY" "$ROOT/scripts/eval_rcm_vbench_subset.py" \
    --selection "$QUEUE/h3_manifest_smoke.json" --generated-dir "$out" --output-dir "$eval" --models bf16 w4a4 svdquant \
    --dimensions subject_consistency dynamic_degree motion_smoothness
  test "$(wc -l < "$eval/summary.csv")" = 10
  state minimax_h3_r64_smoke complete
}

smoke_rcm 20 32
smoke_rcm 10 64
smoke_h3
state queue complete
