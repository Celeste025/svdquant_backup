#!/usr/bin/env bash
set -euo pipefail
TASK=/data1/models/svdquant-wjq/research/20261003/E024/environment
ENV=/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003
SRC=/data1/models/svdquant-wjq/third_party/FastVideo-8444c089
REV=8444c0897a8b96848eb85b6e5750ef486f79fc92
REPORT=/home/wjq/workspace/svdquant-exp/results/research/E024
mkdir -p "$TASK"/{tmp,cache,logs} "$REPORT"
export TMPDIR="$TASK/tmp" TMP="$TASK/tmp" TEMP="$TASK/tmp"
export PIP_CACHE_DIR="$TASK/cache/pip" XDG_CACHE_HOME="$TASK/cache" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor" TRITON_CACHE_DIR="$TASK/cache/triton"
export CUDA_VISIBLE_DEVICES='' PIP_DISABLE_PIP_VERSION_CHECK=1
exec > >(tee -a "$TASK/logs/install.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$TASK/install_exit_code"; date -u +%FT%TZ > "$TASK/install_finished_utc"' EXIT
printf 'START %s\n' "$(date -u +%FT%TZ)"
if [[ ! -d "$ENV" ]]; then /home/wjq/.conda/envs/convrot-wan/bin/python3.12 -m venv "$ENV"; fi
PY="$ENV/bin/python"
if [[ ! -d "$SRC" ]]; then
 git init "$SRC"
 git -C "$SRC" remote add origin https://github.com/hao-ai-lab/FastVideo.git
 git -C "$SRC" fetch --depth 1 origin "$REV"
 git -C "$SRC" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$REV" ]]
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.12.0' torchvision torchaudio --report "$REPORT/pip_torch.json"
"$PY" -m pip install 'torch==2.12.0' 'fastvideo-kernel==0.3.5' 'flashinfer-python' 'transformers>=5.15.0' 'tokenizers>=0.20.1,<0.23' 'diffusers>=0.38.0' 'accelerate==1.0.1' 'sentencepiece>=0.2.0' 'timm>=1.0.11' 'peft>=0.15.0' 'scipy>=1.14.1' 'six>=1.16.0' 'h5py>=3.12.1' 'requests>=2.32.2' 'opencv-python>=4.10.0.84' 'pillow>=10.3.0' 'imageio>=2.36.0' 'imageio-ffmpeg>=0.5.1' einops loguru tqdm 'PyYAML>=6.0.1' 'protobuf>=5.28.3' huggingface_hub cloudpickle omegaconf 'av' 'ftfy>=6.3.1' --report "$REPORT/pip_runtime.json"
"$PY" -m pip install --no-deps -e "$SRC" --report "$REPORT/pip_fastvideo.json"
"$PY" -m pip freeze > "$REPORT/environment_freeze.txt"
"$PY" - <<'PY'
import importlib,importlib.metadata as m,json,traceback,os,sys,pathlib
import torch
r={'status':'running','python':sys.version,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'cuda_initialized':torch.cuda.is_initialized(),'packages':{d.metadata['Name']:d.version for d in m.distributions()},'imports':{}}
for name in ['fastvideo','fastvideo_kernel','fp4attn_cuda','fp4quant_cuda','attn_qat_infer','flashinfer','fastvideo.layers.quantization.nvfp4_qat_config','fastvideo.attention.backends.attn_qat_infer']:
 try: mod=importlib.import_module(name);r['imports'][name]={'status':'pass','file':getattr(mod,'__file__',None)}
 except Exception:r['imports'][name]={'status':'failed','traceback':traceback.format_exc()}
r['status']='complete' if all(x['status']=='pass' for x in r['imports'].values()) else 'import_failed'
r['cuda_initialized_after']=torch.cuda.is_initialized()
pathlib.Path('/home/wjq/workspace/svdquant-exp/results/research/E024/environment_cpu_check.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='packages'},indent=2))
assert r['status']=='complete',r['status']
assert not r['cuda_initialized_after']
PY
