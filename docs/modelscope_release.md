# ModelScope release workflow

`configs/published_artifacts.json` is the single artifact registry.  Model
packages are staged under `/data1/models/svdquant-wjq/releases/modelscope/v0.1.0`;
video datasets are under its `datasets/` child.  Those paths are intentionally
outside Git and are immutable once checksums have been written.

Each model variant contains `artifact.json`, `SHA256SUMS`, a README, a recipe
snapshot, and only the state required to reconstruct the quantized transformer.
H3 uses `quant_state.pt`; rCM and FLUX use `model.pt`, `scale.pt`, and
`wgts.pt`. The historical rCM INT4 and both FLUX variants additionally carry
`smooth.pt` and `branch.pt`, so no original PTQ cache is needed.

Download and verify an artifact (after release):

```bash
python tools/fetch_published_artifact.py rcm-wan real-nvfp4-g10-r64 \
  --namespace YOUR_NAMESPACE --output /path/to/artifacts
python scripts/infer_rcm_wan_4step.py --model /path/to/Wan2.1-T2V-1.3B \
  --quant-ckpt /path/to/artifacts/variants/real-nvfp4-g10-r64 \
  --prompt 'a short test prompt' --output test.mp4
```

For H3, pass the downloaded `quant_state.pt` to the existing runner:

```bash
python scripts/run_minimax_h3_vbench51.py --state /path/to/quant_state.pt --variants svdquant
```

Formal generated videos are published as content-addressed **ModelScope
Datasets**, rather than model packages. Download and verify one with:

```bash
python tools/fetch_published_video_dataset.py rcm-wan \
  --namespace YOUR_NAMESPACE --output /path/to/videos
```

`metadata/videos.jsonl` maps each original collection/case/variant to a file
under `videos/`; the `metadata/` tree retains the original manifests, prompts,
sidecars, and metric summaries. The MiniMax-H3 video dataset is private and
requires a ModelScope token authorized for that repository.

For FLUX, set `FLUX_MODEL_PATH` to the separately downloaded upstream model
and pass the downloaded variant directory to the existing comparison loader:

```bash
python scripts/infer_bf16_vs_w4a4_one.py --ckpt /path/to/dev-int4-r32 --only quant
```

Before public upload, the release owner must review MiniMax-H3 Community License
terms for public derived-checkpoint distribution and FLUX.1-dev's non-commercial
terms/attribution. In particular, H3's current license excludes the EU, UK,
Republic of Korea, and US; do not make it publicly reachable unless ModelScope
access controls enforce that geographic restriction. H3 packages include the
required upstream license and `NOTICE`. Set the token only in the terminal
environment (never chat):

```bash
export MODELSCOPE_API_TOKEN='...'
python tools/publish_modelscope_release.py --namespace YOUR_NAMESPACE \
  --confirm-public-release --confirm-h3-community-license \
  --confirm-h3-applicable-territory-controls \
  --confirm-flux-dev-noncommercial-license
```

If the server should use direct networking rather than a personal proxy, add
`--no-proxy`; it clears the standard upper- and lower-case proxy variables for
that publishing process only.

For H3, use `--private` unless the selected host can enforce the license's
Applicable Territory restrictions. A private repository remains usable via the
same download tool and an authenticated token.
