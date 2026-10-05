# E009 fused-up isolated environment

2026-10-02. Import and six synthetic SM120 fused GPU executions passed. This does not yet validate full H3.

- Python: `/data1/models/svdquant-wjq/research/envs/nvfp4-fused-20261002/bin/python`, new venv from convrot-wan with system-site-packages.
- `.pth` references the existing frozen nvfp4-native-20261002 site-packages (FlashInfer 0.7.0.post1, tvm-ffi0.1.14.post1, cuda-python12.9.7 / cuda-bindings12.9.9). Existing environments were not modified.
- New local packages: CuTe DSL4.7.0 and base/core/cu12/cu13 libraries4.7.0, protobuf6.33.6, nvdisasm13.4.92. Torch remains2.11.0+cu128.
- Candidate4.7.0a0 was not published in the package index; 4.7.0 release satisfies the declared version requirement. Both cu12 and cu13 libraries are required by the package's meta/extra dependencies.
- `/usr/local/cuda` nvcc13.0; no Torch or CUDA-binding upgrade. Dry-run unpinned resolution is retained only as audit evidence, **not** the installed environment.
- Wheels, actual install report, and SHA: `/data1/models/svdquant-wjq/research/wheels/E009/`.
- Dedicated JIT cache: `/data1/models/svdquant-wjq/research/cache/flashinfer-fused`.
- Exact launcher: `/data1/models/svdquant-wjq/research/20261002/E009_launch_fused_smoke.sh`.
- Results: `results/research/E009_fused_up_synthetic.json`; log/exit in `results/logs/E009_fused_up_synthetic.*`.

Six tests: two shapes × main-only/up-only/combined, same packed operands and non-power-of-two global scales. Fused-vs-FP32 reference NMSE2.72e-6–2.81e-6; combined-vs-original torch-main+BF16-up NMSE8.44e-6–8.87e-6, capturing expected rounding differences. Actual CuTe FP4 GPU kernel observed. Total test4.81s including startup, **not** benchmark latency. Synthetic script can be extended for real inputs before its final E009 result is frozen; the current result records this source version's SHA.

## Wheel hashes

```json
{
  "nvidia_cutlass_dsl_libs_base-4.7.0-cp312-cp312-manylinux_2_28_x86_64.whl": {
    "sha256": "0ee3ca427a43bd3190fc4ac06fba15fa7a923241a1cc928d3e2b4a6f480dbd39",
    "bytes": 2831724
  },
  "nvidia_cutlass_dsl_libs_cu12-4.7.0-cp312-cp312-manylinux_2_28_x86_64.whl": {
    "sha256": "eaf85bd605f41f968c483dcee1ae9e13ffd765b623f976869401f77e7255b5e1",
    "bytes": 88687558
  },
  "protobuf-6.33.6-cp39-abi3-manylinux2014_x86_64.whl": {
    "sha256": "e9db7e292e0ab79dd108d7f1a94fe31601ce1ee3f7b79e0692043423020b0593",
    "bytes": 323436
  },
  "nvidia_cutlass_dsl_libs_cu13-4.7.0-cp312-cp312-manylinux_2_28_x86_64.whl": {
    "sha256": "66dda9dd19729836b919174bb33d70aef895d785f3320dafceca695513408cef",
    "bytes": 88271940
  },
  "nvidia_cutlass_dsl_libs_core-4.7.0-py3-none-any.whl": {
    "sha256": "3c6a128ce314aca3f332eabcb0e55a2c4c3aa951e336584a7aeb2b013bed4474",
    "bytes": 1227603
  },
  "nvidia_cuda_nvdisasm-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": {
    "sha256": "f35a5b6ddb64b6758c22c0b078c5b6ff6d54f7c240cd74bf131ab5bf13b6c3fe",
    "bytes": 5203296
  },
  "nvidia_cutlass_dsl-4.7.0-py3-none-any.whl": {
    "sha256": "f96e35c1393a8afa9a20cada1aa08fdca16d673852f3274144e3e81693f13a14",
    "bytes": 10459
  }
}
```
